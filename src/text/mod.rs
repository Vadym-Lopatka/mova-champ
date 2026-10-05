//! `PText`: a persistent rope for editor-grade text.
//!
//! See `SPEC-PTEXT.md` for the design and `DESIGN.md` for the crate's
//! shared machinery this module reuses (single-allocation nodes, atomic
//! refcounts, unique-path in-place mutation, the pool allocator).
//!
//! ## Algorithmic core: `concat` + `split_at`
//!
//! Rather than hand-writing bespoke B-tree insert/delete rebalancing
//! (borrow-from-sibling, merge-with-sibling, split-on-overflow as separate
//! cases threaded through the edit logic), this implementation is built
//! from two self-rebalancing structural primitives, in the spirit of
//! purely-functional 2-3/finger trees:
//!
//! - [`concat`]: join two arbitrary well-formed trees into one well-formed
//!   tree. Equal-height operands merge directly (or split-plus-new-root if
//!   that overflows `INTERNAL_MAX`/`LEAF_MAX`); unequal-height operands
//!   descend the taller side's outer spine and recurse, propagating any
//!   overflow back up exactly like the standard B-tree join algorithm.
//! - `split_at`: descend to the child containing the split point, split it
//!   recursively (leaves split directly), then on each side wrap the
//!   untouched siblings into one same-height node (`wrap_siblings`) and
//!   join it to the recursive result with a single `concat` call. Because
//!   `concat` is already self-rebalancing, `split_at` needs no separate
//!   rebalancing logic of its own — each level's one `concat` call fixes
//!   up whatever `INTERNAL_MIN` shortfall splitting there produced. (M9,
//!   SPEC-M9-SPLITAT.md: earlier this rejoined each side via
//!   `concat_many`, a left fold over every untouched sibling — correct,
//!   but O(fanout) `concat` calls per level instead of O(1), each one
//!   re-extracting and rebuilding the whole sibling array; costly enough
//!   on a shared tree, where every extraction is an Arc-clone of every
//!   child, to dominate `slice`/persistent-`splice` wall time at scale.)
//!
//! [`PText::slice`] is directly two `split_at` calls that keep the middle —
//! this is also why `slice` shares subtrees for free: `split_at` only
//! copies nodes on the spine down to the cut point, `clone_shallow`-ing
//! every untouched sibling subtree it folds back in via `concat`.
//!
//! `splice` (the persistent edit primitive) no longer routes through
//! `concat`/`split_at` directly (M14a, SPEC-M14-TEXT-PERSIST-EQ.md — before
//! M14a its body was literally `concat(concat(split_at(a).0,
//! from_str(text)), split_at(b).1)`, and that general-purpose rebuild paid
//! full fresh-node cost on every call: 10-16x slower than the mutable
//! `ropey` competitor on the persistent-splice-heavy "undo pattern",
//! SPIKE-ROPEY-RESULTS.md). `splice` is now `self.clone().splice_owned
//! (char_range, text)`: `clone()` is an O(1) root refcount bump, so by the
//! time `splice_owned`'s descent starts, the root is already shared — its
//! own unique/shared split (below) then IS ropey's clone-then-mutate-with-
//! copy-on-shared discipline, copying only the edited spine and refcount-
//! bumping every untouched sibling instead of rebuilding it. `concat`/
//! `split_at` remain the structural core `slice`/`concat` are built from,
//! and `splice_owned`'s own multi-child-spanning-range fallback
//! (`splice_span_fallback_owned`/`_copy`) still reaches them directly for
//! whatever subtree a range wider than one child touches — just no longer
//! `splice`'s first move the way it used to be.
//!
//! [`PText::splice_owned`] (consumes `self`) is a real recursive descent
//! using take/put child-slot semantics (mirroring `src/node.rs`'s
//! `assoc_owned`/`take_node`/`put_node`): it mutates every uniquely-owned
//! node on the root-to-leaf path in place (leaf bytes included, exploiting
//! capacity slack) and only allocates/copies once it hits the first shared
//! node (or the edit crosses a leaf boundary), falling back to the general
//! `concat`/`split_at` algorithm only for the rare case of a range spanning
//! more than one child at some level. See `owned_splice_mut`'s doc comment
//! for the exact contract and NOTES-PTEXT.md for the trade-off this
//! simplification makes.

mod node;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use node::{RawNode, Summary};

// ---------------------------------------------------------------------
// PText: the safe, owning "tree handle". Exactly one refcount unit of
// `root` is owned for as long as `root` is `Some` (see `into_parts`'s doc
// comment for the one place it's transiently `None`).
// ---------------------------------------------------------------------

/// A persistent rope: an immutable, structurally-shared UTF-8 text buffer.
///
/// Cloning is O(1) (bumps the root's refcount). Every mutating operation
/// comes in a `&self` (persistent, always copies on the write path) and an
/// `_owned` (consumes `self`, mutates uniquely-owned nodes in place when
/// possible) flavor — see the module docs for the exact strategy each uses.
pub struct PText {
    root: Option<RawNode>,
    is_leaf: bool,
    /// 0 for a leaf root; for an internal root, `1 + height of its
    /// children` (so every leaf in the tree sits at depth `height`).
    height: u32,
    summary: Summary,
}

// SAFETY: `PText` only owns `Option<RawNode>` (Send/Sync unconditionally,
// see node.rs) plus plain `Copy` metadata. No interior mutability beyond
// the node layer's atomic refcounts.
unsafe impl Send for PText {}
unsafe impl Sync for PText {}

impl Drop for PText {
    fn drop(&mut self) {
        if let Some(root) = self.root.take() {
            node::drop_node(root, self.is_leaf);
        }
    }
}

impl Clone for PText {
    fn clone(&self) -> Self {
        let root = self.root.as_ref().expect("PText root always present");
        PText { root: Some(node::clone_shallow(root)), is_leaf: self.is_leaf, height: self.height, summary: self.summary }
    }
}

impl Default for PText {
    fn default() -> Self {
        PText::new()
    }
}

impl PText {
    fn from_leaf(raw: RawNode) -> PText {
        let summary = node::leaf::summary(&raw);
        PText { root: Some(raw), is_leaf: true, height: 0, summary }
    }

    fn from_internal(raw: RawNode, height: u32) -> PText {
        let summary = node::internal::total_summary(&raw);
        PText { root: Some(raw), is_leaf: false, height, summary }
    }

    /// An empty text.
    pub fn new() -> PText {
        PText::from_leaf(node::leaf::new_exact(""))
    }

    pub fn len_bytes(&self) -> usize {
        self.summary.bytes as usize
    }

    pub fn len_chars(&self) -> usize {
        self.summary.chars as usize
    }

    /// Number of `'\n'`-delimited lines (a text with `k` newlines has `k +
    /// 1` lines, matching `str::lines()`'s counting for non-empty text —
    /// see [`PText::line_to_char`]/[`PText::char_to_line`]'s docs for the
    /// exact line-start convention).
    pub fn len_lines(&self) -> usize {
        self.summary.newlines as usize + 1
    }

    pub fn is_empty(&self) -> bool {
        self.summary.bytes == 0
    }

    #[inline]
    fn root(&self) -> &RawNode {
        self.root.as_ref().expect("PText root always present")
    }

    pub fn ptr_eq(a: &PText, b: &PText) -> bool {
        match (&a.root, &b.root) {
            (Some(ra), Some(rb)) => node::ptr_eq(ra, rb),
            _ => false,
        }
    }

    /// Move `self`'s owned root out, consuming `self` without running its
    /// `Drop` impl a second time (the field is left `None`, which `Drop`
    /// treats as "nothing to release" — same pattern as
    /// `PersistentHashMap`'s `Option<NodePtr>` root).
    fn into_parts(mut self) -> (RawNode, bool, u32, Summary) {
        let root = self.root.take().expect("PText root always present until taken");
        (root, self.is_leaf, self.height, self.summary)
    }
}

impl fmt::Debug for PText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PText")
            .field("len_bytes", &self.len_bytes())
            .field("len_chars", &self.len_chars())
            .field("len_lines", &self.len_lines())
            .finish()
    }
}

// ---------------------------------------------------------------------
// Bulk construction: chunk the input directly into LEAF_MAX-sized leaves
// (never via repeated single-char insert) and build the tree bottom-up.
// ---------------------------------------------------------------------

impl From<&str> for PText {
    fn from(s: &str) -> PText {
        if s.is_empty() {
            return PText::new();
        }
        let mut leaves = Vec::with_capacity(s.len() / node::LEAF_SPLIT_TARGET + 1);
        let mut rest = s;
        while !rest.is_empty() {
            let take = chunk_boundary(rest, node::LEAF_MAX);
            let (chunk, tail) = rest.split_at(take);
            leaves.push(PText::from_leaf(node::leaf::new_exact(chunk)));
            rest = tail;
        }
        bulk_build(leaves)
    }
}

impl From<String> for PText {
    fn from(s: String) -> PText {
        PText::from(s.as_str())
    }
}

/// Largest byte count `<= max` that's still a char boundary in `s` (`s`
/// nonempty). Used by bulk construction to chunk a `&str` into leaves
/// without ever splitting a multi-byte sequence.
fn chunk_boundary(s: &str, max: usize) -> usize {
    if s.len() <= max {
        return s.len();
    }
    let mut i = max;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Bottom-up build over an arbitrary number of same-height `PText` leaves:
/// group into `<= INTERNAL_MAX`-sized, `>= INTERNAL_MIN`-sized (evenly
/// distributed, not just chunked, so the *last* group is never a
/// short-changed remainder) batches, wrap each batch in one internal node,
/// and repeat until one root remains.
fn bulk_build(mut items: Vec<PText>) -> PText {
    if items.is_empty() {
        return PText::new();
    }
    while items.len() > node::INTERNAL_MAX {
        let groups = items.len().div_ceil(node::INTERNAL_MAX);
        let base = items.len() / groups;
        let rem = items.len() % groups;
        let mut next = Vec::with_capacity(groups);
        let mut iter = items.into_iter();
        for g in 0..groups {
            let size = if g < rem { base + 1 } else { base };
            let batch: Vec<PText> = (&mut iter).take(size).collect();
            next.push(build_one_internal(batch));
        }
        items = next;
    }
    if items.len() == 1 { items.pop().unwrap() } else { build_one_internal(items) }
}

/// Wrap `children` (`1..=INTERNAL_MAX` same-height `PText`s) in one fresh
/// internal node.
fn build_one_internal(children: Vec<PText>) -> PText {
    debug_assert!(!children.is_empty() && children.len() <= node::INTERNAL_MAX);
    let leaf_kids = children[0].is_leaf;
    let child_height = children[0].height;
    let mut raws = Vec::with_capacity(children.len());
    let mut sums = Vec::with_capacity(children.len());
    for c in children {
        let (raw, is_leaf, height, summary) = c.into_parts();
        debug_assert_eq!(is_leaf, leaf_kids, "bulk_build: mixed-kind children");
        debug_assert_eq!(height, child_height, "bulk_build: mixed-height children");
        raws.push(raw);
        sums.push(summary);
    }
    let raw = node::internal::new_exact(raws, &sums, leaf_kids);
    PText::from_internal(raw, child_height + 1)
}

/// Wrap `children` (`1..=2 * INTERNAL_MAX`, same height), splitting into two
/// balanced nodes plus a fresh parent if it exceeds `INTERNAL_MAX` — the
/// single rebalancing primitive `concat`/`split_at` funnel every
/// child-count change through.
fn build_from_children(mut children: Vec<PText>) -> PText {
    debug_assert!(!children.is_empty());
    if children.len() == 1 {
        return children.pop().unwrap();
    }
    if children.len() <= node::INTERNAL_MAX {
        return build_one_internal(children);
    }
    debug_assert!(children.len() <= 2 * node::INTERNAL_MAX, "concat/split_at never produce more than 2x overflow");
    let mid = children.len() / 2;
    let right = children.split_off(mid);
    let left_node = build_one_internal(children);
    let right_node = build_one_internal(right);
    build_one_internal(vec![left_node, right_node])
}

// ---------------------------------------------------------------------
// concat: the core rebalancing primitive.
// ---------------------------------------------------------------------

/// Join two well-formed trees into one well-formed tree, always copying
/// (never mutates `left`/`right`'s nodes) — safe for the persistent path
/// and reused, unmodified, by the owned path's general fallback.
fn concat(left: PText, right: PText) -> PText {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }
    match left.height.cmp(&right.height) {
        std::cmp::Ordering::Equal => concat_equal_height(left, right),
        std::cmp::Ordering::Greater => {
            let expected_height = left.height - 1;
            let mut kids = extract_as_ptext(left);
            let last = kids.pop().expect("internal node always has >=1 child");
            let merged = concat(last, right);
            absorb(&mut kids, merged, expected_height, false);
            build_from_children(kids)
        }
        std::cmp::Ordering::Less => {
            let expected_height = right.height - 1;
            let mut kids = extract_as_ptext(right);
            let first = kids.remove(0);
            let merged = concat(left, first);
            absorb(&mut kids, merged, expected_height, true);
            build_from_children(kids)
        }
    }
}

fn concat_equal_height(left: PText, right: PText) -> PText {
    if left.is_leaf {
        let (lraw, _, _, _) = left.into_parts();
        let (rraw, _, _, _) = right.into_parts();
        let combined = node::leaf::byte_len(&lraw) + node::leaf::byte_len(&rraw);
        if combined <= node::LEAF_MAX {
            let mut s = String::with_capacity(combined);
            s.push_str(node::leaf::as_str(&lraw));
            s.push_str(node::leaf::as_str(&rraw));
            node::drop_node(lraw, true);
            node::drop_node(rraw, true);
            PText::from_leaf(node::leaf::new_exact(&s))
        } else {
            build_one_internal(vec![PText::from_leaf(lraw), PText::from_leaf(rraw)])
        }
    } else {
        let mut kids = extract_as_ptext(left);
        kids.extend(extract_as_ptext(right));
        build_from_children(kids)
    }
}

/// Insert `merged` at position `0` (`front == true`) or append it (`front
/// == false`) into `kids`. If `merged` turns out to itself be a
/// same-height-as-`kids` overflow node (the result of a nested `concat`
/// that had to split-plus-new-root), unwrap its exactly-two children
/// instead of nesting it — this is the overflow-propagation step of the
/// standard B-tree join algorithm.
fn absorb(kids: &mut Vec<PText>, merged: PText, expected_height: u32, front: bool) {
    if merged.height == expected_height {
        if front {
            kids.insert(0, merged);
        } else {
            kids.push(merged);
        }
    } else {
        debug_assert_eq!(merged.height, expected_height + 1, "concat overflow can only be one level taller");
        let mut pair = extract_as_ptext(merged);
        debug_assert_eq!(pair.len(), 2, "concat's split-plus-new-root always makes exactly 2 children");
        let b = pair.pop().unwrap();
        let a = pair.pop().unwrap();
        if front {
            kids.insert(0, b);
            kids.insert(0, a);
        } else {
            kids.push(a);
            kids.push(b);
        }
    }
}

/// Consume an internal `PText`, handing back its children as owned `PText`s
/// (respecting sharing — see `node::internal::extract_children`).
fn extract_as_ptext(node: PText) -> Vec<PText> {
    debug_assert!(!node.is_leaf);
    let child_height = node.height - 1;
    let (raw, _, _, _) = node.into_parts();
    let (children, summaries, leaf_kids) = node::internal::extract_children(raw);
    children
        .into_iter()
        .zip(summaries)
        .map(|(c, s)| PText { root: Some(c), is_leaf: leaf_kids, height: child_height, summary: s })
        .collect()
}

fn rebuild_ptext(raw: RawNode, is_leaf: bool, height: u32, summary: Summary) -> PText {
    PText { root: Some(raw), is_leaf, height, summary }
}

/// Fold an arbitrary list of valid trees into one, via repeated `concat`
/// (empty list -> empty text; `concat` itself short-circuits empty operands
/// so no filtering is needed here).
fn concat_many(items: Vec<PText>) -> PText {
    let mut iter = items.into_iter();
    let Some(mut acc) = iter.next() else {
        return PText::new();
    };
    for item in iter {
        acc = concat(acc, item);
    }
    acc
}

// ---------------------------------------------------------------------
// split_at: the second rebalancing primitive. See module docs for why it
// needs no rebalancing logic of its own — each level's one `concat` call
// (joining the recursive result to the untouched siblings, wrapped as a
// single node by `wrap_siblings`) fixes up whatever the split just did.
// ---------------------------------------------------------------------

/// Split `tree` at byte offset `byte_idx` (`<= tree.len_bytes()`, caller's
/// responsibility to land on a char boundary) into `(left, right)`,
/// structurally sharing every subtree that didn't need to be touched.
///
/// Per internal level: the untouched siblings on each side are already
/// equal-height, in-order, and `<= INTERNAL_MAX - 1` of them — exactly one
/// valid internal node's worth (`wrap_siblings`) — joined to the recursive
/// split's result with one `concat` call per side (M9, SPEC-M9-SPLITAT.md;
/// replaces an earlier `concat_many` fold that rebuilt the sibling array
/// once per fold step).
fn split_at(tree: PText, byte_idx: usize) -> (PText, PText) {
    debug_assert!(byte_idx <= tree.summary.bytes as usize);
    if tree.is_leaf {
        let (raw, _, _, _) = tree.into_parts();
        let (l, r) = node::leaf::split_at(raw, byte_idx);
        (PText::from_leaf(l), PText::from_leaf(r))
    } else {
        let mut children = extract_as_ptext(tree);
        let n = children.len();
        let mut acc = 0usize;
        let mut idx = n - 1;
        for (i, c) in children.iter().enumerate() {
            let b = c.summary.bytes as usize;
            if byte_idx < acc + b || i == n - 1 {
                idx = i;
                break;
            }
            acc += b;
        }
        let local = byte_idx - acc;
        let right_children = children.split_off(idx + 1);
        let target = children.pop().expect("idx < n, so at least one element remains");
        let (cl, cr) = split_at(target, local);
        (join_left(children, cl), join_right(cr, right_children))
    }
}

/// `prefix ++ tail`: `prefix` are `0..=INTERNAL_MAX - 1` untouched,
/// equal-height, in-order left siblings and `tail` is the recursive
/// split's left result. Replaces the old per-level `concat_many` fold of
/// up to `INTERNAL_MAX - 1` sibling-by-sibling rebuilds.
///
/// The common case — `tail` exactly one level shorter than `prefix`'s
/// siblings, which is what every non-degenerate `split_at` level produces
/// (the recursive result's height tracks the split target's height, one
/// less than its own now-wrapped siblings) — merges `tail` directly into
/// the last sibling and rebuilds once. A first cut that always went
/// through `wrap_siblings` + `concat` measured ~7.9us/op against the ≤5us
/// bar (SPEC-M9-SPLITAT.md): `sample`-profiling
/// `examples/slice_probe.rs`'s loop showed `extract_as_ptext` and
/// `build_one_internal` as the top two hot functions, because `concat`'s
/// own unequal-height branch (`left.height - right.height == 1`, true on
/// almost every call here) re-extracts the exact sibling array
/// `wrap_siblings` had just built one node from — a build-then-immediately
/// -unbuild round trip. This inlines what that `concat` branch does
/// (`extract_as_ptext` skipped entirely, since `prefix` already IS that
/// extracted array): merge `tail` into the last sibling via one
/// equal-height `concat`, `absorb` the result (same overflow-propagation
/// helper `concat` itself uses), `build_from_children` once. Any other
/// height relationship (tail same height as the siblings, or shorter by
/// more than one — both rarer, from `concat` overflow or a multi-level
/// collapse) falls back to the fully general `wrap_siblings` + `concat`,
/// which handles them correctly regardless.
fn join_left(mut prefix: Vec<PText>, tail: PText) -> PText {
    if prefix.is_empty() {
        return tail;
    }
    if tail.is_empty() {
        return wrap_siblings(prefix);
    }
    let sibling_height = prefix[0].height;
    if tail.height == sibling_height {
        let last = prefix.pop().expect("prefix checked non-empty above");
        let merged = concat(last, tail);
        absorb(&mut prefix, merged, sibling_height, false);
        return build_from_children(prefix);
    }
    concat(wrap_siblings(prefix), tail)
}

/// Symmetric to [`join_left`]: `head ++ suffix`.
fn join_right(head: PText, mut suffix: Vec<PText>) -> PText {
    if suffix.is_empty() {
        return head;
    }
    if head.is_empty() {
        return wrap_siblings(suffix);
    }
    let sibling_height = suffix[0].height;
    if head.height == sibling_height {
        let first = suffix.remove(0);
        let merged = concat(head, first);
        absorb(&mut suffix, merged, sibling_height, true);
        return build_from_children(suffix);
    }
    concat(head, wrap_siblings(suffix))
}

/// Wrap `1..=INTERNAL_MAX` same-height siblings into one node: a single
/// element is returned directly (no single-child wrapper needed — `concat`
/// doesn't care whether its operand came from a real node or a bare
/// element), `2..=INTERNAL_MAX` become exactly one fresh internal node via
/// [`build_one_internal`], which already builds a summary from its
/// children's summaries the same way every other builder here does.
fn wrap_siblings(mut siblings: Vec<PText>) -> PText {
    debug_assert!(!siblings.is_empty() && siblings.len() <= node::INTERNAL_MAX);
    if siblings.len() == 1 {
        return siblings.pop().unwrap();
    }
    build_one_internal(siblings)
}

// ---------------------------------------------------------------------
// Byte <-> char conversion (descent using per-child `Summary.chars`
// prefix sums, bottoming out in a leaf-local `str::char_indices` scan —
// the "O(log32 n + chunk)" the spec's motivation talks about).
// ---------------------------------------------------------------------

fn byte_of_char_impl(node: &RawNode, is_leaf: bool, char_idx: usize) -> usize {
    if is_leaf {
        if char_idx == 0 {
            return 0;
        }
        let summary = node::leaf::summary(node);
        // ASCII fast path: `chars == bytes` means every byte is its own
        // char, so the char index *is* the byte index — O(1) instead of
        // the O(local char offset) `char_indices().nth()` scan below.
        // Matters a lot in practice: without it, every `splice`/
        // `splice_owned`/`char_at` call pays a scan proportional to
        // however far into a (up to `LEAF_MAX`-byte) leaf the position
        // falls, on every single op, regardless of how cheap the actual
        // edit is — see NOTES-PTEXT.md's M7.5 section for the profile
        // that found this dominating (>75% of wall time) an otherwise-fast
        // owned splice.
        if summary.bytes == summary.chars {
            return char_idx.min(summary.bytes as usize);
        }
        let s = node::leaf::as_str(node);
        match s.char_indices().nth(char_idx) {
            Some((b, _)) => b,
            None => s.len(),
        }
    } else {
        let leaf_kids = node::internal::leaf_children(node);
        let (children, summaries) = node::internal::children_and_summaries(node);
        let n = children.len();
        let mut acc_bytes = 0usize;
        let mut acc_chars = 0usize;
        for i in 0..n {
            let s = summaries[i];
            let c = s.chars as usize;
            if char_idx < acc_chars + c || i == n - 1 {
                let local = char_idx - acc_chars;
                return acc_bytes + byte_of_char_impl(&children[i], leaf_kids, local);
            }
            acc_bytes += s.bytes as usize;
            acc_chars += c;
        }
        unreachable!("byte_of_char_impl: char index out of range")
    }
}

fn char_at_byte_impl(node: &RawNode, is_leaf: bool, byte_idx: usize) -> char {
    if is_leaf {
        let s = node::leaf::as_str(node);
        s[byte_idx..].chars().next().expect("char_at: index out of bounds")
    } else {
        let (idx, local) = node::internal::locate_byte(node, byte_idx);
        let leaf_kids = node::internal::leaf_children(node);
        char_at_byte_impl(node::internal::child_at(node, idx), leaf_kids, local)
    }
}

fn byte_to_char_impl(node: &RawNode, is_leaf: bool, byte_idx: usize) -> usize {
    if is_leaf {
        // ASCII fast path — see `byte_of_char_impl`'s doc comment.
        let summary = node::leaf::summary(node);
        if summary.bytes == summary.chars {
            return byte_idx;
        }
        node::leaf::as_str(node)[..byte_idx].chars().count()
    } else {
        let leaf_kids = node::internal::leaf_children(node);
        let (children, summaries) = node::internal::children_and_summaries(node);
        let n = children.len();
        let mut acc_bytes = 0usize;
        let mut acc_chars = 0usize;
        for i in 0..n {
            let s = summaries[i];
            let b = s.bytes as usize;
            if byte_idx < acc_bytes + b || i == n - 1 {
                let local = byte_idx - acc_bytes;
                return acc_chars + byte_to_char_impl(&children[i], leaf_kids, local);
            }
            acc_bytes += b;
            acc_chars += s.chars as usize;
        }
        unreachable!("byte_to_char_impl: byte index out of range")
    }
}

/// Count of `'\n'` bytes strictly before byte offset `byte_idx`.
fn newlines_before_byte_impl(node: &RawNode, is_leaf: bool, byte_idx: usize) -> usize {
    if is_leaf {
        node::leaf::as_str(node).as_bytes()[..byte_idx].iter().filter(|&&b| b == b'\n').count()
    } else {
        let leaf_kids = node::internal::leaf_children(node);
        let (children, summaries) = node::internal::children_and_summaries(node);
        let n = children.len();
        let mut acc_bytes = 0usize;
        let mut acc_nl = 0usize;
        for i in 0..n {
            let s = summaries[i];
            let b = s.bytes as usize;
            if byte_idx < acc_bytes + b || i == n - 1 {
                let local = byte_idx - acc_bytes;
                return acc_nl + newlines_before_byte_impl(&children[i], leaf_kids, local);
            }
            acc_bytes += b;
            acc_nl += s.newlines as usize;
        }
        unreachable!("newlines_before_byte_impl: byte index out of range")
    }
}

/// Byte offset of the start of `line` (0-indexed: line 0 starts at byte 0;
/// line `k > 0` starts right after the `k`-th `'\n'` byte in the text).
fn byte_of_line_start_impl(node: &RawNode, is_leaf: bool, line: usize) -> usize {
    if line == 0 {
        return 0;
    }
    if is_leaf {
        let s = node::leaf::as_str(node);
        let mut count = 0usize;
        for (i, b) in s.bytes().enumerate() {
            if b == b'\n' {
                count += 1;
                if count == line {
                    return i + 1;
                }
            }
        }
        s.len()
    } else {
        let leaf_kids = node::internal::leaf_children(node);
        let (children, summaries) = node::internal::children_and_summaries(node);
        let mut acc_bytes = 0usize;
        let mut acc_nl = 0usize;
        for (i, s) in summaries.iter().enumerate() {
            let nl = s.newlines as usize;
            if line <= acc_nl + nl {
                let local_line = line - acc_nl;
                return acc_bytes + byte_of_line_start_impl(&children[i], leaf_kids, local_line);
            }
            acc_bytes += s.bytes as usize;
            acc_nl += nl;
        }
        unreachable!("byte_of_line_start_impl: line out of range")
    }
}

// ---------------------------------------------------------------------
// M9.1 (SPEC-M9-SPLITAT.md addendum): read_range -- a direct char-range
// read into a caller-supplied String, with no tree building at all (no
// split_at, no concat, no node allocation). See PText::read_range's doc
// comment for the motivation.
// ---------------------------------------------------------------------

/// Append every byte of `node`'s subtree to `out`, leaf by leaf --
/// `read_range_impl`'s "fully inside the range" case.
fn append_all_impl(node: &RawNode, is_leaf: bool, out: &mut String) {
    if is_leaf {
        out.push_str(node::leaf::as_str(node));
    } else {
        let leaf_kids = node::internal::leaf_children(node);
        for child in node::internal::children_slice(node) {
            append_all_impl(child, leaf_kids, out);
        }
    }
}

/// Append the bytes of `local_range` (a byte range local to `node`, whose
/// own byte span is `[0, node_bytes)`) to `out`. Mirrors the descent shape
/// of `byte_of_char_impl`/`newlines_before_byte_impl` above, but instead of
/// stopping at one leaf, it collects a whole *range*: a child entirely
/// inside `local_range` is appended whole via [`append_all_impl`] (a leaf
/// walk, not a tree rebuild -- for a small target range like the ~100-char
/// case this milestone is motivated by, that touches 1-2 leaves total); a
/// child entirely outside is skipped via its cached `Summary.bytes`,
/// never descended into; a child straddling one edge of the range
/// recurses. No `split_at`/`concat`/node allocation anywhere in this path.
fn read_range_impl(node: &RawNode, is_leaf: bool, local_range: Range<usize>, node_bytes: usize, out: &mut String) {
    if local_range.start == 0 && local_range.end == node_bytes {
        append_all_impl(node, is_leaf, out);
        return;
    }
    if is_leaf {
        out.push_str(&node::leaf::as_str(node)[local_range.start..local_range.end]);
        return;
    }
    let leaf_kids = node::internal::leaf_children(node);
    let (children, summaries) = node::internal::children_and_summaries(node);
    let mut acc = 0usize;
    for (child, s) in children.iter().zip(summaries) {
        let child_bytes = s.bytes as usize;
        let child_start = acc;
        let child_end = acc + child_bytes;
        if local_range.end <= child_start {
            break; // children are in order; no further overlap possible
        }
        if local_range.start < child_end {
            let lo = local_range.start.max(child_start) - child_start;
            let hi = local_range.end.min(child_end) - child_start;
            read_range_impl(child, leaf_kids, lo..hi, child_bytes, out);
        }
        acc = child_end;
    }
}

impl PText {
    /// Byte offset of char index `idx` (`idx == len_chars()` is valid,
    /// returning `len_bytes()`).
    pub fn byte_of_char(&self, idx: usize) -> usize {
        debug_assert!(idx <= self.len_chars(), "byte_of_char: index out of bounds");
        byte_of_char_impl(self.root(), self.is_leaf, idx)
    }

    /// The `char` at char index `idx` (`idx < len_chars()`).
    pub fn char_at(&self, idx: usize) -> char {
        let b = self.byte_of_char(idx);
        char_at_byte_impl(self.root(), self.is_leaf, b)
    }

    /// Char index of the 0-indexed line containing char index `idx`
    /// (`'\n'`-delimited: see [`Self::len_lines`]).
    pub fn char_to_line(&self, idx: usize) -> usize {
        let b = self.byte_of_char(idx);
        newlines_before_byte_impl(self.root(), self.is_leaf, b)
    }

    /// Char index of the start of 0-indexed `line` (`line < len_lines()`).
    pub fn line_to_char(&self, line: usize) -> usize {
        debug_assert!(line < self.len_lines(), "line_to_char: line out of bounds");
        let b = byte_of_line_start_impl(self.root(), self.is_leaf, line);
        byte_to_char_impl(self.root(), self.is_leaf, b)
    }

    /// Delete `char_range` and insert `text` in its place — the one edit
    /// primitive (covers insert when the range is empty, delete when
    /// `text` is empty, replace otherwise). Persistent: `self` remains
    /// valid and unchanged; always copies along the edited spine, sharing
    /// every untouched subtree.
    ///
    /// M14a (SPEC-M14-TEXT-PERSIST-EQ.md): delegates straight to
    /// `self.clone().splice_owned(...)` — `clone()` is an `O(1)` root
    /// refcount bump, so the root is shared the moment `splice_owned`
    /// starts its descent, and `splice_owned`'s own unique/shared split
    /// (`owned_splice_mut`/`owned_splice_copy`) then does exactly ropey's
    /// clone-then-mutate-with-copy-on-shared: copy only the edited spine,
    /// refcount-bump every untouched sibling it passes. This used to run
    /// the general `concat`/`split_at` rebuild directly (visible in git
    /// history and in this module's docs before M14a); that path is still
    /// very much alive underneath — it's `splice_owned`'s own fallback for
    /// a range spanning more than one child (`splice_span_fallback_owned`/
    /// `_copy`) — but is no longer `splice`'s first move. See the module
    /// docs' "Algorithmic core" section and SPEC-M14-TEXT-PERSIST-EQ.md's
    /// M14a for the spike comparison (`undo_pattern_retained`) this closes
    /// the gap on.
    pub fn splice(&self, char_range: Range<usize>, text: &str) -> PText {
        self.clone().splice_owned(char_range, text)
    }

    /// Like [`Self::splice`], but consumes `self`: descends the
    /// root-to-leaf path mutating every uniquely-owned node in place
    /// (leaf bytes included, exploiting capacity slack; a leaf split is
    /// absorbed directly into a unique parent via
    /// `node::internal::unique_insert_child`), falling back to a
    /// borrowed copy *only from the first shared node encountered* —
    /// mirroring `src/node.rs`'s `assoc_mut`/`assoc_copy` split, not the
    /// all-or-nothing bail this milestone shipped with initially (see
    /// NOTES-PTEXT.md's M7.5 section for the profile that found that
    /// version silently missing the fast path on ~2/3 of random-position
    /// edits, and why). Only a range spanning more than one child at some
    /// level still falls all the way back to [`Self::splice`]'s general
    /// `concat`/`split_at` algorithm for that subtree — rare for small
    /// edits (a range wider than a `LEAF_MAX`-sized leaf), and correct
    /// either way.
    pub fn splice_owned(self, char_range: Range<usize>, text: &str) -> PText {
        let start = self.byte_of_char(char_range.start);
        let end = if char_range.end == char_range.start { start } else { self.byte_of_char(char_range.end) };
        let (raw, is_leaf, height, summary) = self.into_parts();
        match owned_splice_mut(raw, is_leaf, start..end, text, height, summary) {
            SpliceOutcome::One(r, s) => rebuild_ptext(r, is_leaf, height, s),
            SpliceOutcome::Two(a, sa, b, sb) => {
                let root = node::internal::new_exact(vec![a, b], &[sa, sb], is_leaf);
                PText::from_internal(root, height + 1)
            }
        }
    }

    /// Structural slice: `O(log n)`, shares every subtree that's entirely
    /// inside (or entirely outside) `char_range` — implemented as two
    /// `split_at` calls, same primitive `splice` is built from.
    pub fn slice(&self, char_range: Range<usize>) -> PText {
        let start = self.byte_of_char(char_range.start);
        let end = self.byte_of_char(char_range.end);
        let (_, rest) = split_at(self.clone(), start);
        let (mid, _) = split_at(rest, end - start);
        mid
    }

    /// Append `char_range`'s content directly to `out`: `O(log n + len)` --
    /// a summary-guided descent to the range's covered subtrees plus one
    /// leaf-walk over the touched leaves, with no tree building at all (no
    /// `split_at`, no `concat`, no node allocation). Motivated by M9.1
    /// (SPEC-M9-SPLITAT.md's addendum): the dominant client shape for a
    /// small range read -- an editor extracting one visible line, e.g.
    /// the host's `subs` over ~100-char spans -- never wants a `PText` result,
    /// it copies into a flat string right away, so building one first only
    /// to immediately flatten it pays `split_at`/`concat`'s tree-
    /// rebalancing cost for nothing a direct read didn't need. Same bounds
    /// contract as [`Self::slice`]: `char_range` must be valid (`start <=
    /// end <= len_chars()`), debug-checked here, not enforced in release.
    pub fn read_range(&self, char_range: Range<usize>, out: &mut String) {
        debug_assert!(char_range.start <= char_range.end && char_range.end <= self.len_chars(), "read_range: char_range out of bounds");
        if char_range.start == char_range.end {
            return;
        }
        let start = self.byte_of_char(char_range.start);
        let end = self.byte_of_char(char_range.end);
        read_range_impl(self.root(), self.is_leaf, start..end, self.summary.bytes as usize, out);
    }

    /// Convenience wrapper over [`Self::read_range`]: allocates a fresh
    /// `String` sized via `byte_of_char`-derived length up front (so the
    /// push inside `read_range` never has to grow/reallocate the buffer),
    /// then fills it.
    pub fn range_to_string(&self, char_range: Range<usize>) -> String {
        debug_assert!(char_range.start <= char_range.end && char_range.end <= self.len_chars(), "range_to_string: char_range out of bounds");
        let cap =
            if char_range.start == char_range.end { 0 } else { self.byte_of_char(char_range.end) - self.byte_of_char(char_range.start) };
        let mut out = String::with_capacity(cap);
        self.read_range(char_range, &mut out);
        out
    }

    /// Join two ropes into one, `O(log n)` — the tree-join primitive
    /// [`Self::splice`]/[`Self::slice`] are themselves built from (see the
    /// module docs' "Algorithmic core" section). `splice`'s own inserted
    /// `text` argument is always a contiguous `&str`, which is exactly
    /// right for a single small edit but can't express "join two
    /// potentially-large ropes together" without a caller first
    /// flattening one side to a `&str` (defeating the point). Exposed for
    /// callers building up a large result from several already-rope
    /// pieces (e.g. a host's n-ary string concatenation) that
    /// need to stay `O(log n)` per join instead of `O(size)`.
    pub fn concat(a: PText, b: PText) -> PText {
        concat(a, b)
    }
}

// ---------------------------------------------------------------------
// M7.5 owned splice: a real recursive descent (src/node.rs's assoc_mut/
// assoc_copy split), replacing the M7 all-or-nothing fallback. See
// NOTES-PTEXT.md's M7.5 section for the profile that motivated this and
// PText::splice_owned's doc comment for the contract.
// ---------------------------------------------------------------------

/// Outcome of a one-level splice: either the child slot's replacement is a
/// single node (the overwhelmingly common case), or the child had to split
/// (a leaf exceeding `LEAF_MAX`, or an internal node exceeding
/// `INTERNAL_MAX` after absorbing an inserted grandchild) and the caller
/// must insert the second piece as a new sibling — exactly the standard
/// B-tree insert-with-overflow-propagation shape, here expressed as a
/// return value instead of a mutable "did we split" out-parameter.
enum SpliceOutcome {
    One(RawNode, Summary),
    Two(RawNode, Summary, RawNode, Summary),
}

/// Owned (mutate-in-place) recursive descent, mirroring `src/node.rs`'s
/// `assoc_mut`: while the current node is uniquely owned, take/mutate/put
/// in place; the moment a shared node is found, delegate the rest of the
/// subtree to [`owned_splice_copy`] (this handle's one reference is then
/// released — the other owner(s) keep theirs, untouched). A leaf edit that
/// would exceed `LEAF_MAX` splits directly (`unique_splice_or_split`) and
/// is absorbed into the (unique) parent via `unique_insert_child`,
/// growing that one node's array — never the old behavior of rebuilding
/// every sibling at every level via `concat`/`split_at`. A range spanning
/// more than one child at some level still falls back to the general
/// algorithm for that subtree (rare for small edits; see
/// `splice_span_fallback_owned`).
fn owned_splice_mut(mut node: RawNode, is_leaf: bool, local_range: Range<usize>, text: &str, height: u32, current_summary: Summary) -> SpliceOutcome {
    if !node::is_unique(&node) {
        let outcome = owned_splice_copy(&node, is_leaf, local_range, text, height);
        node::drop_node(node, is_leaf);
        return outcome;
    }
    if is_leaf {
        return match node::leaf::unique_splice_or_split(node, local_range, text) {
            node::leaf::SpliceResult::One(l) => {
                let s = node::leaf::summary(&l);
                SpliceOutcome::One(l, s)
            }
            node::leaf::SpliceResult::Two(a, b) => {
                let sa = node::leaf::summary(&a);
                let sb = node::leaf::summary(&b);
                SpliceOutcome::Two(a, sa, b, sb)
            }
        };
    }
    let Some((idx, child_local)) = node::internal::locate_range(&node, &local_range) else {
        // M7.5: before falling all the way back to the general algorithm,
        // check whether this is the much cheaper "small edit straddles
        // exactly one leaf boundary" case — see `locate_span2`'s doc
        // comment for why this specific case is worth a dedicated path
        // (a `replace_all`-shaped workload hits it often enough, and each
        // *uncaught* occurrence cost as much as thousands of ordinary
        // splices combined).
        if node::internal::leaf_children(&node)
            && let Some((first_idx, local_a, local_b)) = node::internal::locate_span2(&node, &local_range)
        {
            let sum_a = node::internal::summary_at(&node, first_idx);
            let sum_b = node::internal::summary_at(&node, first_idx + 1);
            // Correctness guard (found by a targeted regression test, not
            // by luck): both leaves being merged can each independently be
            // up to `LEAF_MAX` bytes, so the merged content can be up to
            // *almost* `2 * LEAF_MAX` bytes — bigger than any single leaf,
            // sure, but also too big to always guarantee a naive 2-way
            // split keeps *both* halves within `LEAF_MAX` (pigeonhole: if
            // the merged size exceeds `2 * LEAF_MAX`, at least one half
            // must exceed `LEAF_MAX` no matter where the cut falls). This
            // is exactly the case right after a fresh bulk build (every
            // leaf starts completely full) with an edit landing near a
            // boundary — not a contrived corner case, a real one. When the
            // conservative bound below isn't met, skip this fast path
            // entirely (nothing taken yet, so nothing to undo) and fall
            // through to the general algorithm, which has no such limit.
            let would_fit_span2 = local_a + text.len() + (sum_b.bytes as usize - local_b) <= 2 * node::LEAF_MAX - 8;
            if would_fit_span2 {
                let a = node::internal::take_child(&mut node, first_idx);
                let b = node::internal::take_child(&mut node, first_idx + 1);
                if node::is_unique(&a) && node::is_unique(&b) {
                    return match node::leaf::unique_merge_splice_or_split(a, b, local_a, local_b, text) {
                        node::leaf::SpliceResult::One(merged) => {
                            let s = node::leaf::summary(&merged);
                            node::internal::put_child(&mut node, first_idx, merged, s);
                            let node = node::internal::unique_remove_taken_child(node, first_idx + 1);
                            let total = current_summary.sub(sum_a).sub(sum_b).add(s);
                            SpliceOutcome::One(node, total)
                        }
                        node::leaf::SpliceResult::Two(na, nb) => {
                            let sna = node::leaf::summary(&na);
                            let snb = node::leaf::summary(&nb);
                            node::internal::put_child(&mut node, first_idx, na, sna);
                            node::internal::put_child(&mut node, first_idx + 1, nb, snb);
                            let total = current_summary.sub(sum_a).sub(sum_b).add(sna).add(snb);
                            SpliceOutcome::One(node, total)
                        }
                    };
                }
                // Not both unique (rare — e.g. a slice/clone still holds
                // one of these two leaves): restore unchanged (no rc
                // change happened, just a peek) and fall through.
                node::internal::put_child(&mut node, first_idx, a, sum_a);
                node::internal::put_child(&mut node, first_idx + 1, b, sum_b);
            }
        }
        return splice_span_fallback_owned(node, is_leaf, local_range, text, height);
    };
    let child_is_leaf = node::internal::leaf_children(&node);
    // M7.5: the old child summary, read *before* it's overwritten, is what
    // lets the new total below be computed in O(1) — `current_summary -
    // old_child_summary + new_child_summary(/summaries)` — instead of
    // `node::internal::total_summary`'s O(n_children) rescan. Only one
    // child ever changes per splice, so re-summing all (up to
    // `INTERNAL_MAX`) siblings on every level of every op was pure waste;
    // see NOTES-PTEXT.md's M7.5 section for the profile (this rescan, at
    // every level of the descent, on every one of a replace-all's ~140k
    // splices) that found it a real contributor.
    let old_child_summary = node::internal::summary_at(&node, idx);
    let child = node::internal::take_child(&mut node, idx);
    match owned_splice_mut(child, child_is_leaf, child_local, text, height - 1, old_child_summary) {
        SpliceOutcome::One(new_child, summary) => {
            node::internal::put_child(&mut node, idx, new_child, summary);
            let total = current_summary.sub(old_child_summary).add(summary);
            SpliceOutcome::One(node, total)
        }
        SpliceOutcome::Two(a, sa, b, sb) => {
            node::internal::put_child(&mut node, idx, a, sa);
            let grown = node::internal::unique_insert_child(node, idx + 1, b, sb);
            finish_after_child_insert(grown, height)
        }
    }
}

/// Borrowed (always-copies) recursive descent, mirroring `src/node.rs`'s
/// `assoc_copy`: the twin of [`owned_splice_mut`] used once a shared node
/// is found, and for every level above it. `node` is never mutated or
/// consumed here.
fn owned_splice_copy(node: &RawNode, is_leaf: bool, local_range: Range<usize>, text: &str, height: u32) -> SpliceOutcome {
    if is_leaf {
        return match node::leaf::copy_splice_or_split(node, local_range, text) {
            node::leaf::SpliceResult::One(l) => {
                let s = node::leaf::summary(&l);
                SpliceOutcome::One(l, s)
            }
            node::leaf::SpliceResult::Two(a, b) => {
                let sa = node::leaf::summary(&a);
                let sb = node::leaf::summary(&b);
                SpliceOutcome::Two(a, sa, b, sb)
            }
        };
    }
    let Some((idx, child_local)) = node::internal::locate_range(node, &local_range) else {
        return splice_span_fallback_copy(node, is_leaf, local_range, text, height);
    };
    let child_is_leaf = node::internal::leaf_children(node);
    let child_ref = node::internal::child_at(node, idx);
    match owned_splice_copy(child_ref, child_is_leaf, child_local, text, height - 1) {
        SpliceOutcome::One(new_child, summary) => {
            let mut items = copy_children_as_ptext(node, height - 1);
            items[idx] = rebuild_ptext(new_child, child_is_leaf, height - 1, summary);
            let pt = build_one_internal(items); // same count as before: never overflows.
            let (raw, _, _, s) = pt.into_parts();
            SpliceOutcome::One(raw, s)
        }
        SpliceOutcome::Two(a, sa, b, sb) => {
            let mut items = copy_children_as_ptext(node, height - 1);
            items[idx] = rebuild_ptext(a, child_is_leaf, height - 1, sa);
            items.insert(idx + 1, rebuild_ptext(b, child_is_leaf, height - 1, sb));
            split_or_wrap_children(items)
        }
    }
}

/// `node`'s children, each `clone_shallow`'d into an owned [`PText`]
/// handle (never mutates/consumes `node` — used by the copy path, which
/// by definition operates on a shared node it must leave untouched).
fn copy_children_as_ptext(node: &RawNode, child_height: u32) -> Vec<PText> {
    let leaf_kids = node::internal::leaf_children(node);
    let (children, summaries) = node::internal::children_and_summaries(node);
    children
        .iter()
        .zip(summaries)
        .map(|(c, &s)| PText { root: Some(node::clone_shallow(c)), is_leaf: leaf_kids, height: child_height, summary: s })
        .collect()
}

/// After growing a unique internal node's child array by one
/// (`unique_insert_child`), check whether it still fits `INTERNAL_MAX`;
/// if not, split it in two — the same rare-overflow handling
/// `build_from_children` does for the general algorithm, inlined here so
/// the common (fits) case stays a single check with no extra allocation.
fn finish_after_child_insert(node: RawNode, height: u32) -> SpliceOutcome {
    let n = node::internal::n_children(&node);
    if n <= node::INTERNAL_MAX {
        let total = node::internal::total_summary(&node);
        return SpliceOutcome::One(node, total);
    }
    let leaf_kids = node::internal::leaf_children(&node);
    let (children, summaries, _) = node::internal::extract_children(node);
    let items: Vec<PText> = children
        .into_iter()
        .zip(summaries)
        .map(|(c, s)| PText { root: Some(c), is_leaf: leaf_kids, height: height - 1, summary: s })
        .collect();
    match split_or_wrap_children(items) {
        SpliceOutcome::Two(a, sa, b, sb) => SpliceOutcome::Two(a, sa, b, sb),
        SpliceOutcome::One(..) => unreachable!("n > INTERNAL_MAX implies split_or_wrap_children always splits"),
    }
}

/// `items.len()` may be `INTERNAL_MAX + 1` (one over, from an insert) —
/// wrap as a single node if it now fits (shouldn't happen given callers
/// only reach this with an overflowed count, but handled for robustness),
/// else split into two roughly-equal halves, same convention as
/// `build_from_children`'s overflow branch.
fn split_or_wrap_children(mut items: Vec<PText>) -> SpliceOutcome {
    if items.len() <= node::INTERNAL_MAX {
        let pt = build_one_internal(items);
        let (raw, _, _, s) = pt.into_parts();
        return SpliceOutcome::One(raw, s);
    }
    let mid = items.len() / 2;
    let right = items.split_off(mid);
    let l = build_one_internal(items);
    let r = build_one_internal(right);
    let (lr, _, _, ls) = l.into_parts();
    let (rr, _, _, rs) = r.into_parts();
    SpliceOutcome::Two(lr, ls, rr, rs)
}

/// Rare fallback for a range spanning more than one child at some level
/// (only possible when the edit range itself is wider than a leaf, or
/// straddles a boundary — small single-position edits essentially never
/// hit this): delegate the whole subtree to the general `concat`/
/// `split_at` algorithm, same as `PText::splice`. `node` is consumed
/// (owned path already held it uniquely, or the caller is about to drop
/// its one shared reference either way).
fn splice_span_fallback_owned(node: RawNode, is_leaf: bool, local_range: Range<usize>, text: &str, height: u32) -> SpliceOutcome {
    let summary = if is_leaf { node::leaf::summary(&node) } else { node::internal::total_summary(&node) };
    let pt = PText { root: Some(node), is_leaf, height, summary };
    let (left, rest) = split_at(pt, local_range.start);
    let (_, right) = split_at(rest, local_range.end - local_range.start);
    let result = concat_many(vec![left, PText::from(text), right]);
    ptext_to_outcome(result, height)
}

/// Copy-path twin of [`splice_span_fallback_owned`]: `node` is shared, so
/// this `clone_shallow`s it (one extra, temporary reference) rather than
/// consuming the caller's handle.
fn splice_span_fallback_copy(node: &RawNode, is_leaf: bool, local_range: Range<usize>, text: &str, height: u32) -> SpliceOutcome {
    let cloned = node::clone_shallow(node);
    splice_span_fallback_owned(cloned, is_leaf, local_range, text, height)
}

/// Adapt a [`PText`] (from the general-algorithm fallback, which may have
/// grown one level taller if it had to split-plus-new-root at the top) back
/// into a [`SpliceOutcome`] relative to `expected_height` — the height the
/// caller's own recursion is tracking for this subtree.
fn ptext_to_outcome(mut result: PText, expected_height: u32) -> SpliceOutcome {
    // Unlike `concat`'s own local overflow (bounded to *at most one level
    // taller*), the general algorithm applied to a whole SUBTREE here can
    // come back *shorter* than `expected_height` — e.g. deleting nearly
    // all of a span collapses what used to be several leaves under an
    // internal node into a single leaf. The caller is about to splice this
    // back in as one sibling among others still at `expected_height`
    // (uniform leaf depth is a real invariant elsewhere in the tree), so
    // pad it back up with single-child wrapper nodes — structurally legal
    // (the validator only requires `1..=INTERNAL_MAX` children, not `>=
    // 2`) and correct, if not maximally fanout-efficient; this path is
    // rare (a range spanning more than one leaf) so it isn't worth more
    // machinery than reusing `build_one_internal` on a one-element `Vec`.
    while result.height < expected_height {
        result = build_one_internal(vec![result]);
    }
    if result.height == expected_height {
        let (raw, _, _, s) = result.into_parts();
        return SpliceOutcome::One(raw, s);
    }
    debug_assert_eq!(result.height, expected_height + 1, "concat overflow can only be one level taller (after padding for shrink)");
    let mut items = extract_as_ptext(result);
    debug_assert_eq!(items.len(), 2, "concat's split-plus-new-root always makes exactly 2 children");
    let b = items.pop().unwrap();
    let a = items.pop().unwrap();
    let (ar, _, _, asum) = a.into_parts();
    let (br, _, _, bsum) = b.into_parts();
    SpliceOutcome::Two(ar, asum, br, bsum)
}

impl fmt::Display for PText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for chunk in self.chunks() {
            f.write_str(chunk)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// chunks()/chars(): zero-copy tree walk. Allocation-free, mirroring
// src/iter.rs's fixed-depth-array stack (a `RawNode` child slice's
// `std::slice::Iter` is cheap to hold inline, same as CHAMP's
// `(K, V)`/`NodePtr` slices).
// ---------------------------------------------------------------------

/// Generous fixed bound on tree height: with `INTERNAL_MAX = 32`-way
/// fanout and `LEAF_MAX = 2048`, even an (astronomically large) exabyte of
/// text needs on the order of `log32(2^60 / 2048) ~= 10` levels; 32 leaves
/// a wide margin without costing more than a few dozen bytes of stack.
const MAX_DEPTH: usize = 32;

struct Frame<'a> {
    children: std::slice::Iter<'a, RawNode>,
    leaf_children: bool,
}

/// Borrowing, zero-copy iterator over a [`PText`]'s leaves as `&str`
/// chunks, in left-to-right document order. This is what native scanners
/// (parsers, syntax highlighters, line scanners) should consume instead of
/// `to_string()`/[`Display`](fmt::Display), which allocate.
pub struct Chunks<'a> {
    /// `Some` only when the whole tree is a single leaf (no frames needed).
    root_leaf: Option<&'a RawNode>,
    stack: [Option<Frame<'a>>; MAX_DEPTH],
    len: u8,
}

impl<'a> Chunks<'a> {
    fn new(text: &'a PText) -> Self {
        let root = text.root();
        if text.is_leaf {
            Chunks { root_leaf: Some(root), stack: [const { None }; MAX_DEPTH], len: 0 }
        } else {
            let mut stack = [const { None }; MAX_DEPTH];
            stack[0] = Some(Frame {
                children: node::internal::children_slice(root).iter(),
                leaf_children: node::internal::leaf_children(root),
            });
            Chunks { root_leaf: None, stack, len: 1 }
        }
    }
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if let Some(leaf) = self.root_leaf.take() {
            return Some(node::leaf::as_str(leaf));
        }
        while self.len > 0 {
            let top = self.len as usize - 1;
            let frame = self.stack[top].as_mut().expect("active frame slot is always Some");
            match frame.children.next() {
                Some(child) => {
                    if frame.leaf_children {
                        return Some(node::leaf::as_str(child));
                    }
                    debug_assert!((self.len as usize) < MAX_DEPTH, "chunks iterator depth exceeded MAX_DEPTH cap");
                    self.stack[self.len as usize] = Some(Frame {
                        children: node::internal::children_slice(child).iter(),
                        leaf_children: node::internal::leaf_children(child),
                    });
                    self.len += 1;
                }
                None => {
                    self.stack[top] = None;
                    self.len -= 1;
                }
            }
        }
        None
    }
}

impl std::iter::FusedIterator for Chunks<'_> {}

impl PText {
    /// Zero-copy walk over the text's leaves as `&str` chunks. See
    /// [`Chunks`].
    pub fn chunks(&self) -> Chunks<'_> {
        Chunks::new(self)
    }

    /// Iterator over `char`s, built on [`Self::chunks`].
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        self.chunks().flat_map(str::chars)
    }
}

// ---------------------------------------------------------------------
// Equality and hashing: content-based, chunk-boundary-independent (two
// `PText`s built with different leaf splits but identical content compare
// equal and hash equal).
//
// M14b (SPEC-M14-TEXT-PERSIST-EQ.md): the slow path below is a **pruned**
// dual walk, the M13/PVector `try_eq_by` trick ported to text — see
// `EqCursor` for the two-cursor machinery it's built from.
// ---------------------------------------------------------------------

/// One frame of an [`EqCursor`]'s stack: `children`, the `leaf_children`
/// flag naming what kind they are, and `idx`, the next not-yet-visited
/// child. Deliberately index-based (not `std::slice::Iter`, unlike
/// [`Chunks`]' `Frame`) so [`EqCursor::peek`] can look at `children[idx]`
/// without consuming it — the operation the pruned walk needs that a plain
/// iterator doesn't offer.
struct EqFrame<'a> {
    children: &'a [RawNode],
    leaf_children: bool,
    idx: usize,
}

/// Dual-walk-friendly cousin of [`Chunks`]: same fixed-depth-stack shape,
/// but exposes the position at whatever granularity (leaf **or** whole
/// internal subtree) currently sits unconsumed, via [`Self::peek`] /
/// [`Self::advance`] / [`Self::descend`] — the primitives
/// [`PartialEq::eq`]'s pruned walk needs to either skip a shared subtree
/// wholesale or fall through to consuming it, **without ever looking at
/// the same child slot twice**: `peek` alone decides whether to prune, and
/// whichever of `advance`/`descend` follows reuses exactly the node
/// reference `peek` already produced (no second `trim`+index to re-derive
/// it) — the fix for an earlier version of this cursor that called `peek`
/// then unconditionally re-walked from scratch on a miss, quietly doubling
/// per-leaf navigation cost on the very `eq_disjoint` (pruning never
/// fires) case the ±5% "must be free" bar is about; see
/// SPEC-M14-TEXT-PERSIST-EQ.md's M14b bars and BASELINE-M14-EQ.txt.
struct EqCursor<'a> {
    /// `Some` only when the whole remaining tree is a single leaf not yet
    /// consumed (mirrors `Chunks::root_leaf`).
    root_leaf: Option<&'a RawNode>,
    stack: [Option<EqFrame<'a>>; MAX_DEPTH],
    len: u8,
}

impl<'a> EqCursor<'a> {
    fn new(text: &'a PText) -> Self {
        let root = text.root();
        if text.is_leaf {
            EqCursor { root_leaf: Some(root), stack: [const { None }; MAX_DEPTH], len: 0 }
        } else {
            let mut stack = [const { None }; MAX_DEPTH];
            stack[0] = Some(EqFrame { children: node::internal::children_slice(root), leaf_children: node::internal::leaf_children(root), idx: 0 });
            EqCursor { root_leaf: None, stack, len: 1 }
        }
    }

    /// Pop any frames off the top of the stack whose children are all
    /// already visited, so that (if the cursor isn't fully exhausted)
    /// `stack[len - 1]`'s `idx` names a real not-yet-visited child —
    /// the idx-based equivalent of `Chunks::next`'s inner "iterator
    /// exhausted, pop and retry" branch.
    #[inline]
    fn trim(&mut self) {
        while self.len > 0 {
            let top = self.len as usize - 1;
            let frame = self.stack[top].as_ref().expect("active frame slot is always Some");
            if frame.idx < frame.children.len() {
                break;
            }
            self.stack[top] = None;
            self.len -= 1;
        }
    }

    /// Peek the very next not-yet-consumed node together with whether it is
    /// itself a leaf (`true`, directly consumable) or an internal subtree
    /// (`false`, needs [`Self::descend`] to reach actual bytes) — without
    /// advancing past it. `None` once the cursor is exhausted (no more
    /// content on this side at all).
    #[inline]
    fn peek(&mut self) -> Option<(&'a RawNode, bool)> {
        if let Some(leaf) = self.root_leaf {
            return Some((leaf, true));
        }
        self.trim();
        if self.len == 0 {
            return None;
        }
        let frame = self.stack[self.len as usize - 1].as_ref().expect("trim() leaves an active frame with a pending child");
        Some((&frame.children[frame.idx], frame.leaf_children))
    }

    /// Advance past the node [`Self::peek`] just returned, without visiting
    /// its content — used both as the pruning shortcut (skip a
    /// `ptr_eq`-matched subtree wholesale, called on both sides) and as the
    /// bookkeeping half of actually consuming it (a leaf: this alone is
    /// enough; internal: followed by [`Self::descend`]).
    #[inline]
    fn advance(&mut self) {
        if self.root_leaf.take().is_some() {
            return;
        }
        if self.len == 0 {
            return;
        }
        let top = self.len as usize - 1;
        let frame = self.stack[top].as_mut().expect("advance: len > 0 implies an active top frame");
        frame.idx += 1;
    }

    /// Descend into the internal node [`Self::peek`] just returned (its
    /// `bool` was `false`) and the caller has already [`Self::advance`]d
    /// past — pushes `node`'s children as a new frame. Caller follows with
    /// [`Self::next_leaf`] (or another `peek`) to actually reach a leaf.
    #[inline]
    fn descend(&mut self, node: &'a RawNode) {
        debug_assert!((self.len as usize) < MAX_DEPTH, "eq cursor depth exceeded MAX_DEPTH cap");
        let leaf_kids = node::internal::leaf_children(node);
        let children = node::internal::children_slice(node);
        self.stack[self.len as usize] = Some(EqFrame { children, leaf_children: leaf_kids, idx: 0 });
        self.len += 1;
    }

    /// Next leaf chunk from scratch — `peek` + `advance` + (if needed,
    /// repeatedly) `descend`, same walk shape as [`Chunks::next`]. Used
    /// where there's no already-in-hand `peek` result to reuse (the
    /// single-sided refill case in `pruned_eq`, where the two cursors
    /// aren't aligned so no prune attempt was made this round).
    fn next_leaf(&mut self) -> Option<&'a str> {
        loop {
            let (node, is_leaf) = self.peek()?;
            self.advance();
            if is_leaf {
                return Some(node::leaf::as_str(node));
            }
            self.descend(node);
        }
    }
}

/// The pruned dual walk: [`PartialEq::eq`]'s slow path (byte-length
/// precheck and root `ptr_eq` already handled by the caller). Two
/// [`EqCursor`]s track position independently; `abuf`/`bbuf` hold whatever
/// leaf bytes have been pulled but not yet matched against the other
/// side — exactly `PartialEq`'s pre-M14b buffers, so the fallback
/// (chunk-boundary-independent byte comparison) is byte-for-byte the same
/// algorithm as before. The one addition: whenever both buffers are empty
/// (both sides sit exactly at a node boundary, having consumed the same
/// number of bytes so far — an invariant the buffer discipline below
/// maintains throughout), peek the very next node on each side; if they're
/// `node::ptr_eq` (the same allocation — necessarily byte-identical
/// content, so this can never produce a wrong answer), skip the whole
/// subtree on both sides without visiting a single byte or descending an
/// inch further, and keep doing so for as many consecutive matching
/// subtrees as line up. The two nodes being compared need not be at the
/// same tree depth or of the same kind (leaf vs. internal) on each side —
/// `ptr_eq` doesn't care, and different chunkings of equal content is
/// exactly the case the byte-walk fallback already handles when pruning
/// doesn't fire.
///
/// The one performance-critical discipline: the final (non-matching) `peek`
/// from the pruning attempt is *reused* to actually fetch the next chunk
/// (`pending_a`/`pending_b` below) rather than thrown away and re-derived
/// via a fresh `next_leaf` call — see `EqCursor`'s doc comment for why that
/// distinction is exactly the `eq_disjoint` "pruning must be free" bar.
fn pruned_eq(a: &PText, b: &PText) -> bool {
    let mut ca = EqCursor::new(a);
    let mut cb = EqCursor::new(b);
    let mut abuf: &str = "";
    let mut bbuf: &str = "";
    loop {
        let mut pending_a: Option<(&RawNode, bool)> = None;
        let mut pending_b: Option<(&RawNode, bool)> = None;
        if abuf.is_empty() && bbuf.is_empty() {
            loop {
                match (ca.peek(), cb.peek()) {
                    (Some((na, _)), Some((nb, _))) if node::ptr_eq(na, nb) => {
                        ca.advance();
                        cb.advance();
                    }
                    (pa, pb) => {
                        pending_a = pa;
                        pending_b = pb;
                        break;
                    }
                }
            }
        }
        if abuf.is_empty() {
            abuf = match pending_a {
                Some((na, true)) => {
                    ca.advance();
                    node::leaf::as_str(na)
                }
                Some((na, false)) => {
                    ca.advance();
                    ca.descend(na);
                    ca.next_leaf().unwrap_or("")
                }
                None => ca.next_leaf().unwrap_or(""),
            };
        }
        if bbuf.is_empty() {
            bbuf = match pending_b {
                Some((nb, true)) => {
                    cb.advance();
                    node::leaf::as_str(nb)
                }
                Some((nb, false)) => {
                    cb.advance();
                    cb.descend(nb);
                    cb.next_leaf().unwrap_or("")
                }
                None => cb.next_leaf().unwrap_or(""),
            };
        }
        match (abuf.is_empty(), bbuf.is_empty()) {
            (true, true) => return true,
            (true, false) | (false, true) => return false, // unreachable given the byte-length check above
            (false, false) => {}
        }
        let n = abuf.len().min(bbuf.len());
        if abuf.as_bytes()[..n] != bbuf.as_bytes()[..n] {
            return false;
        }
        abuf = &abuf[n..];
        bbuf = &bbuf[n..];
    }
}

impl PartialEq for PText {
    fn eq(&self, other: &PText) -> bool {
        if PText::ptr_eq(self, other) {
            return true;
        }
        if self.summary.bytes != other.summary.bytes {
            return false;
        }
        pruned_eq(self, other)
    }
}

impl Eq for PText {}

/// Hashes the exact same byte stream `<str as Hash>::hash` produces for the
/// equivalent flat string: the content's raw bytes (streamed chunk by
/// chunk — `Hasher::write` is defined to be call-boundary-independent, so
/// this is byte-for-byte identical to one `write` call over the whole
/// content) followed by a single `0xff` terminator byte. This exact
/// contract (bytes, then one `0xff`, no length prefix) is what std's
/// `impl Hash for str` does; matching it is what lets `PText::from(s)` and
/// `s` themselves hash equal under the same `Hasher`, independent of how
/// `PText` happens to have chunked the content into leaves.
impl Hash for PText {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for chunk in self.chunks() {
            state.write(chunk.as_bytes());
        }
        state.write_u8(0xff);
    }
}

// ---------------------------------------------------------------------
// Validator: canonical-form/invariant checker, mirroring
// `PersistentHashMap::validate`'s gating (`cfg(any(test, feature =
// "validate"))`).
// ---------------------------------------------------------------------

#[cfg(any(test, feature = "validate"))]
impl PText {
    /// Walk the tree asserting:
    /// - every leaf sits at the same depth (`self.height`);
    /// - every internal node's child count is `1..=INTERNAL_MAX` (the
    ///   `INTERNAL_MIN` target is deliberately *not* enforced here as a
    ///   hard invariant — see `node.rs`'s `INTERNAL_MAX` doc comment and
    ///   NOTES-PTEXT.md for why `concat`/`split_at` can't always guarantee
    ///   it after arbitrary edit sequences);
    /// - every leaf's byte length is `<= LEAF_MAX` and its stored
    ///   `chars`/`newlines` counts match a fresh scan of its bytes;
    /// - every internal node's per-child cached `Summary` matches that
    ///   child's own (recursively validated) authoritative summary;
    /// - the whole tree's recorded `height`/`summary` (cached in `PText`
    ///   itself for O(1) `len_*`) matches the recomputation.
    pub fn validate(&self) {
        let (leaf_depth, summary) = validate_node(self.root(), self.is_leaf, 0);
        assert_eq!(leaf_depth, self.height, "PText: recorded height does not match actual leaf depth");
        assert_eq!(summary, self.summary, "PText: recorded summary does not match recomputed content");
    }
}

#[cfg(any(test, feature = "validate"))]
fn validate_node(node: &RawNode, is_leaf: bool, depth: u32) -> (u32, Summary) {
    if is_leaf {
        let s = node::leaf::as_str(node);
        let summary = node::leaf::summary(node);
        assert!(s.len() <= node::LEAF_MAX, "PText: leaf exceeds LEAF_MAX");
        assert_eq!(summary.bytes as usize, s.len(), "PText: leaf byte-length metric stale");
        assert_eq!(summary.chars as usize, s.chars().count(), "PText: leaf char-count metric stale");
        assert_eq!(
            summary.newlines as usize,
            s.bytes().filter(|&b| b == b'\n').count(),
            "PText: leaf newline-count metric stale"
        );
        (depth, summary)
    } else {
        let n = node::internal::n_children(node);
        assert!((1..=node::INTERNAL_MAX).contains(&n), "PText: internal node child count out of bounds");
        let leaf_kids = node::internal::leaf_children(node);
        let mut total = Summary::ZERO;
        let mut leaf_depth = None;
        for i in 0..n {
            let child = node::internal::child_at(node, i);
            let (d, s) = validate_node(child, leaf_kids, depth + 1);
            match leaf_depth {
                None => leaf_depth = Some(d),
                Some(prev) => assert_eq!(prev, d, "PText: leaves not at a uniform depth"),
            }
            let stored = node::internal::summary_at(node, i);
            assert_eq!(stored, s, "PText: parent's cached child summary is stale");
            total = total.add(s);
        }
        (leaf_depth.expect("internal node always has >=1 child"), total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text() {
        let t = PText::new();
        assert!(t.is_empty());
        assert_eq!(t.len_bytes(), 0);
        assert_eq!(t.len_chars(), 0);
        assert_eq!(t.len_lines(), 1);
    }

    #[test]
    fn from_str_short() {
        let t = PText::from("hello");
        assert_eq!(t.len_bytes(), 5);
        assert_eq!(t.len_chars(), 5);
        assert_eq!(t.to_string(), "hello");
    }

    #[test]
    fn from_str_bulk_multi_leaf() {
        let s = "ab".repeat(5000); // 10_000 bytes, several leaves + internal levels
        let t = PText::from(s.as_str());
        assert_eq!(t.len_bytes(), s.len());
        assert_eq!(t.to_string(), s);
    }

    #[test]
    fn splice_insert_delete_replace() {
        let t = PText::from("hello world");
        let t2 = t.splice(5..5, ",");
        assert_eq!(t2.to_string(), "hello, world");
        assert_eq!(t.to_string(), "hello world"); // original untouched

        let t3 = t2.splice(0..6, "");
        assert_eq!(t3.to_string(), " world");

        let t4 = t3.splice(0..6, "goodbye");
        assert_eq!(t4.to_string(), "goodbye");
    }

    #[test]
    fn concat_matches_string_model() {
        let cases: &[(&str, &str)] = &[
            ("", ""),
            ("hello", ""),
            ("", "world"),
            ("hello, ", "world"),
            (&"ab".repeat(5000), &"cd".repeat(5000)), // both multi-leaf
            (&"x".repeat(3000), "y"),                 // big + tiny
            ("y", &"x".repeat(3000)),                 // tiny + big
        ];
        for &(a, b) in cases {
            let ta = PText::from(a);
            let tb = PText::from(b);
            let joined = PText::concat(ta, tb);
            joined.validate();
            assert_eq!(joined.to_string(), format!("{a}{b}"));
            assert_eq!(joined.len_chars(), a.chars().count() + b.chars().count());
        }
    }

    #[test]
    fn concat_then_splice_still_correct() {
        // The motivating M8 use case: two big slices of the same source
        // joined with a small piece in between (a rope-native "splice via
        // subs+concat" composition), then edited again afterward.
        let source = "0123456789".repeat(1000); // 10_000 chars
        let src = PText::from(source.as_str());
        let prefix = src.slice(0..4000);
        let suffix = src.slice(4001..10_000);
        let joined = PText::concat(PText::concat(prefix, PText::from("Z")), suffix);
        let mut expected = source.clone();
        expected.replace_range(4000..4001, "Z");
        assert_eq!(joined.to_string(), expected);
        joined.validate();
        let edited = joined.splice(0..0, ">>");
        assert_eq!(edited.to_string(), format!(">>{expected}"));
    }

    #[test]
    fn splice_owned_sequential_typing() {
        let mut t = PText::new();
        for (i, ch) in "hello world".chars().enumerate() {
            t = t.splice_owned(i..i, &ch.to_string());
        }
        assert_eq!(t.to_string(), "hello world");
    }

    /// Regression test for the `locate_insert_point` boundary bug: a bulk
    /// build's leaves are all exactly full, so a caret that starts exactly
    /// on a leaf boundary (deliberately engineered here, not left to
    /// chance) must not route every subsequent insert to that fresh,
    /// zero-slack neighboring leaf forever. Correctness-wise this always
    /// worked (see `splice_matches_string_model`'s broad coverage); this
    /// test instead asserts the *performance* property directly: after the
    /// one unavoidable first split, the owned path must settle into
    /// growing a single leaf with slack, not keep falling back.
    #[test]
    fn splice_owned_typing_at_leaf_boundary_uses_fast_path_after_first_split() {
        let leaf_max = 2048; // matches node::LEAF_MAX; not exported, duplicated here deliberately.
        let s = "a".repeat(leaf_max * 4); // several exactly-full, virgin leaves
        let mut t = PText::from(s.as_str());
        let start = leaf_max * 2; // exactly on a leaf boundary
        for i in 0..500 {
            t = t.splice_owned(start + i..start + i, "b");
        }
        t.validate();
        let mut expected = s.clone();
        expected.insert_str(start, &"b".repeat(500));
        assert_eq!(t.to_string(), expected);

        // The performance assertion: growing a single already-split-into
        // leaf 500 times should touch only a small, bounded number of
        // additional leaf allocations (one initial split plus a handful of
        // 256-byte growth steps), never one full-tree fallback per op. We
        // can't observe allocation counts directly here without the
        // memstats counting allocator, so this is asserted via the wall
        // clock instead: 500 in-place-capable ops must not take anywhere
        // near what 500 full split+concat fallbacks would (that would be
        // orders of magnitude slower — see BENCH-RESULTS.md).
        let start2 = t.len_chars() / 2;
        let t0 = std::time::Instant::now();
        let mut t2 = t.clone();
        for i in 0..500 {
            t2 = t2.splice_owned(start2 + i..start2 + i, "c");
        }
        let elapsed = t0.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(50),
            "500 sequential inserts took {elapsed:?} — looks like the fast path isn't engaging \
             (expect well under 1ms once warmed; 50ms is a generous CI-noise margin)"
        );
    }

    /// M7.5 regression test: a small delete+insert range straddling
    /// exactly two adjacent leaves (the `locate_span2` fast path) must
    /// produce correct content — deliberately engineered so the edit
    /// range's start is a few bytes before a leaf boundary and its end a
    /// few bytes after, both via `splice_owned` (unique path) and after a
    /// `clone()` forces the shared/not-unique branch (which must fall
    /// back to the general algorithm, not the unique-only `locate_span2`
    /// fast path — both must agree on content).
    #[test]
    fn splice_owned_span2_across_leaf_boundary() {
        let leaf_max = 2048; // matches node::LEAF_MAX; not exported, duplicated here deliberately.
        let s = "x".repeat(leaf_max * 6);
        let boundary = leaf_max * 3;

        // Range straddling the boundary by a few chars on each side, owned/unique path
        // (a fresh, uniquely-owned build — consumed here — so this exercises
        // locate_span2's truly-unique fast path, not the shared/copy fallback).
        let start = boundary - 2;
        let end = boundary + 3;
        let edited = PText::from(s.as_str()).splice_owned(start..end, "REPLACED");
        edited.validate();
        let mut expected = s.clone();
        expected.replace_range(start..end, "REPLACED");
        assert_eq!(edited.to_string(), expected);

        // Same edit, but with a clone alive first (forces the shared/
        // copy-path branch at the point locate_span2's uniqueness check
        // would otherwise fire) -- both the edited version and the
        // untouched clone must be correct.
        let t2 = PText::from(s.as_str());
        let kept = t2.clone();
        let edited2 = t2.splice_owned(start..end, "REPLACED");
        edited2.validate();
        kept.validate();
        assert_eq!(edited2.to_string(), expected);
        assert_eq!(kept.to_string(), s);

        // Exercise every leaf boundary in the document, not just one, via
        // the persistent path (always general/copy) for a cross-check.
        let t = PText::from(s.as_str());
        for i in 1..6 {
            let b = leaf_max * i;
            let (lo, hi) = (b - 2, b + 2);
            let via_persistent = t.splice(lo..hi, "Z");
            via_persistent.validate();
            let mut exp = s.clone();
            exp.replace_range(lo..hi, "Z");
            assert_eq!(via_persistent.to_string(), exp);
        }
    }

    #[test]
    fn splice_owned_across_leaf_boundary_falls_back() {
        let big = "x".repeat(5000);
        let t = PText::from(big.as_str());
        let t = t.splice_owned(0..big.len(), "y");
        assert_eq!(t.to_string(), "y");
    }

    #[test]
    fn slice_shares_and_matches_content() {
        let s = "the quick brown fox jumps over the lazy dog";
        let t = PText::from(s);
        let sl = t.slice(4..9);
        assert_eq!(sl.to_string(), "quick");
    }

    #[test]
    fn clone_and_diverge() {
        let t1 = PText::from("hello");
        let t2 = t1.clone();
        let t3 = t2.splice(5..5, " world");
        assert_eq!(t1.to_string(), "hello");
        assert_eq!(t2.to_string(), "hello");
        assert_eq!(t3.to_string(), "hello world");
    }

    #[test]
    fn char_at_and_byte_of_char_multibyte() {
        let s = "a\u{00e9}\u{1f600}b";
        let t = PText::from(s);
        assert_eq!(t.char_at(0), 'a');
        assert_eq!(t.char_at(1), '\u{00e9}');
        assert_eq!(t.char_at(2), '\u{1f600}');
        assert_eq!(t.char_at(3), 'b');
        assert_eq!(t.byte_of_char(0), 0);
        assert_eq!(t.byte_of_char(4), s.len());
    }

    #[test]
    fn ptr_eq_basic() {
        let t1 = PText::from("hello");
        let t2 = t1.clone();
        assert!(PText::ptr_eq(&t1, &t2));
        let t3 = PText::from("hello");
        assert!(!PText::ptr_eq(&t1, &t3));
    }

    #[test]
    fn empty_splice_edge_cases() {
        let t = PText::new();
        let t2 = t.splice(0..0, "x");
        assert_eq!(t2.to_string(), "x");
        let t3 = t2.splice(0..1, "");
        assert_eq!(t3.to_string(), "");
        assert!(t3.is_empty());
    }

    #[test]
    fn line_queries() {
        let s = "line0\nline1\nline2\nline3";
        let t = PText::from(s);
        assert_eq!(t.len_lines(), 4);
        assert_eq!(t.line_to_char(0), 0);
        assert_eq!(t.line_to_char(1), 6);
        assert_eq!(t.line_to_char(2), 12);
        assert_eq!(t.line_to_char(3), 18);
        assert_eq!(t.char_to_line(0), 0);
        assert_eq!(t.char_to_line(5), 0);
        assert_eq!(t.char_to_line(6), 1);
        assert_eq!(t.char_to_line(22), 3);
    }

    #[test]
    fn large_bulk_build_and_splice_across_many_leaves() {
        let s = "abcdefgh".repeat(100_000); // 800_000 bytes
        let t = PText::from(s.as_str());
        assert_eq!(t.len_bytes(), s.len());
        t.validate();
        let mid_char = t.len_chars() / 2;
        let t2 = t.splice(mid_char..mid_char, "MIDDLE");
        t2.validate();
        let mut expected = s.clone();
        expected.insert_str(mid_char, "MIDDLE");
        assert_eq!(t2.to_string(), expected);
    }

    #[test]
    fn chunks_and_chars_match_content() {
        let s = "abcdefgh".repeat(1000);
        let t = PText::from(s.as_str());
        let joined: String = t.chunks().collect();
        assert_eq!(joined, s);
        let chars: String = t.chars().collect();
        assert_eq!(chars, s);
    }

    #[test]
    fn eq_is_chunk_boundary_independent() {
        // Built two different ways (bulk vs incremental splice), so the
        // underlying leaf splits differ, but content is identical.
        let a = PText::from("hello world, this is a test of chunk-independent equality");
        let mut b = PText::new();
        for ch in "hello world, this is a test of chunk-independent equality".chars() {
            let n = b.len_chars();
            b = b.splice_owned(n..n, &ch.to_string());
        }
        assert_eq!(a, b);
        let c = PText::from("different content");
        assert_ne!(a, c);
    }

    #[test]
    fn hash_matches_str_hash() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let s = "hash me please, with \u{00e9}moji \u{1f600} too";
        let t = PText::from(s);
        let mut h1 = DefaultHasher::new();
        s.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        t.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn validator_catches_healthy_trees() {
        let t = PText::from("small");
        t.validate();
        let t2 = PText::from("x".repeat(10_000).as_str());
        t2.validate();
        let t3 = t2.splice(5..15, "inserted text here");
        t3.validate();
        let t4 = t2.slice(100..200);
        t4.validate();
    }

    /// M9.1 (SPEC-M9-SPLITAT.md addendum) differential fuzz:
    /// `range_to_string`/`read_range` against both a naive
    /// `chars().skip().take()` `String` model and `slice(r)` flattened via
    /// `chunks()`, over a multi-level doc (large enough to span several
    /// internal levels) mixing ASCII and multibyte content (Latin-1, CJK,
    /// an astral-plane emoji). 1,000 seeded-LCG random ranges plus explicit
    /// boundary coverage (`start == end`, `0..0`, `0..len`, `len..len`, and
    /// ranges landing both mid-leaf and exactly on a `LEAF_MAX` leaf
    /// boundary). Deterministic (fixed LCG seed, never OS/time-derived) so
    /// a failure always reproduces exactly.
    #[test]
    fn read_range_matches_slice_and_string_model() {
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u64 {
                // Numerical Recipes LCG constants -- matches the crate's
                // other deterministic fuzz tests (examples/slice_probe.rs,
                // tests/text_model.rs).
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                self.0
            }
            fn below(&mut self, n: usize) -> usize {
                (self.next() >> 33) as usize % n.max(1)
            }
        }

        // ASCII plus multibyte (Latin-1 supplement, CJK, an emoji outside
        // the BMP) -- deliberately echoes tests/text_model.rs's
        // FUZZ_CHARSET rather than sharing it (this test drives its own
        // LCG, not proptest's).
        let charset: &[char] = &['a', 'b', 'c', ' ', '\n', '\u{00e9}', '\u{4e2d}', '\u{1f600}'];
        let mut lcg = Lcg(0xBEEF_CAFE_0000_0001);
        // Miri is orders of magnitude slower -- shrink both the doc and the
        // iteration count (matching the crate's other Miri-scaled fuzz
        // tests, e.g. tests/text_model.rs's ops_strategy/shared-tree fuzz)
        // rather than skip this test under Miri entirely.
        let n_chars = if cfg!(miri) { 300 } else { 20_000 }; // several internal levels at LEAF_MAX=2048 outside Miri
        let iterations = if cfg!(miri) { 30 } else { 1000 };
        let model: String = (0..n_chars).map(|_| charset[lcg.below(charset.len())]).collect();
        let t = PText::from(model.as_str());
        t.validate();

        for _ in 0..iterations {
            let n = t.len_chars();
            let start = lcg.below(n + 1);
            let end = start + lcg.below(n + 1 - start);
            let expected: String = model.chars().skip(start).take(end - start).collect();

            let got = t.range_to_string(start..end);
            assert_eq!(got, expected, "range_to_string({start}..{end}) mismatch");

            let via_slice: String = t.slice(start..end).chunks().collect();
            assert_eq!(via_slice, expected, "slice({start}..{end}) flattened mismatch");

            // read_range appends to whatever's already in `out`.
            let mut buf = String::from("PREFIX-");
            t.read_range(start..end, &mut buf);
            assert_eq!(buf, format!("PREFIX-{expected}"), "read_range({start}..{end}) append mismatch");
        }

        // Explicit boundary coverage, deterministic regardless of the LCG.
        let n = t.len_chars();
        let mid = n / 2;
        for &(start, end) in &[(0, 0), (0, n), (n, n), (5, 5), (mid, mid)] {
            let expected: String = model.chars().skip(start).take(end - start).collect();
            assert_eq!(t.range_to_string(start..end), expected, "boundary range_to_string({start}..{end}) mismatch");
        }

        // Ranges landing exactly on a LEAF_MAX leaf boundary, and just past
        // one (mid-leaf), on both sides.
        let leaf_max = 2048; // matches node::LEAF_MAX; duplicated deliberately, as elsewhere in this suite.
        for b in [leaf_max, leaf_max * 2, leaf_max + leaf_max / 2, leaf_max * 3 + 17] {
            if b + 50 <= n {
                let expected: String = model.chars().skip(b).take(50).collect();
                assert_eq!(t.range_to_string(b..b + 50), expected, "leaf-boundary range_to_string({b}..{}) mismatch", b + 50);
            }
        }
    }
}
