//! `PVector<T>`: a persistent 32-ary trie + tail vector, Clojure-style.
//!
//! See `SPEC-M10-PVEC.md` for the design and `DESIGN.md`/`NOTES-M6.md` for
//! the crate's shared machinery this module reuses (single-allocation
//! nodes, atomic refcounts, unique-path in-place mutation, the pool
//! allocator). Structural sibling of `src/text/node.rs` (the rope) and
//! `src/node.rs` (CHAMP) — see `node.rs`'s module docs for the exact node
//! layout this file builds on.
//!
//! ## Shape: trie + tail + view header
//!
//! - **Trie**: a dense 32-ary tree. Every leaf that has been placed into
//!   the trie proper is always exactly `NODE_SIZE` (32) elements; every
//!   internal node is exactly `NODE_SIZE` children **except** along the
//!   rightmost spine, which may be partial (this is what lets the trie
//!   grow one leaf at a time without a full rebuild — see `push_tail`).
//!   `root: Option<RawNode<T>>` is `None` for a vector whose whole content
//!   still fits in the tail (`<= 32` elements); when `Some`, `shift`
//!   records the root's height: `shift == 0` means the root *is itself a
//!   leaf* (the trie holds exactly one full 32-element leaf and nothing
//!   more), `shift == 5*k` for `k >= 1` means `k` internal levels above the
//!   leaves. A node at recursion level `shift` covers `NODE_SIZE^(shift/5
//!   + 1)` elements when completely full.
//! - **Tail**: the rightmost `1..=32` elements, *not yet* part of the
//!   trie — a separate leaf-shaped buffer (same physical layout as a trie
//!   leaf, see `node.rs`) that carries genuine residual capacity slack
//!   after a pop (unlike trie leaves, which are always exactly full).
//!   `push_back`'s common case just grows the tail; only every 32nd push
//!   flushes it into the trie (`push_tail`) and starts a fresh
//!   single-element tail. This is the classic Clojure `PersistentVector`
//!   design, chosen (per SPEC-M10-PVEC.md) because a Clojure-dialect host calls `push_back`
//!   far more than any op that would want RRB's relaxed-node generality
//!   (`append`/`split_off`/`insert` are all dead API per the survey).
//! - **View header**: `offset`/`len` turn the handle into an O(1) window
//!   into the trie+tail (`full_len` is the *backing* store's total
//!   element count; `offset + len <= full_len`). `get(i)` reads backing
//!   index `i + offset`; `pop_front` is `offset += 1, len -= 1` — no tree
//!   touched at all, which is what keeps the host's `uncons`-driven seq-walks
//!   (the HOT path per the survey) cheap without RRB's relaxed
//!   front-restructuring. `slice` is likewise an O(1) view (same
//!   `offset`/`len` trick, Clojure `SubVector`'s design moved into the
//!   core type instead of a separate wrapper). **Caveat** (same as
//!   Clojure's `subvec`): a live view retains the *whole* backing
//!   trie/tail, so a long-lived narrow slice of a huge vector still
//!   anchors the huge vector's memory — acceptable here because the host's
//!   `uncons` chains are transient walks that drop the handle at the end
//!   (retention is walk-scoped, per the survey's op-mix), not persisted.
//!   A **non-trivial** view (`offset > 0` or `offset + len < full_len`)
//!   normalizes — rebuilds a fresh, trivial `PVector` over exactly the
//!   visible window (`O(view length)`) — before any mutating op that
//!   can't take a faster path (see below); see `Self::normalize`'s doc
//!   comment. Reads (`get`/`iter`/`chunks`/`eq`) never need to normalize.
//!
//! **Suffix-view fast paths** (SPEC-M12-SUFFIXVIEW.md, found by the M10.1
//! host-integration validation: `pop_front`/`push_back` mixed in a queue
//! pattern was hitting the full `normalize()` on every push, O(n) per op,
//! O(n²) total — un-quadratic'd as follows):
//! - A **suffix view** (`offset > 0`, `offset + len == full_len` — the
//!   visible window ends exactly at the backing's end; trivial is the
//!   `offset == 0` special case) gets the *same* O(1)-amortized fast path
//!   as a trivial view for `push_back`/`pop_back` (either flavor): the
//!   hidden prefix `[0..offset)` sits entirely below the region either op
//!   touches, so appending/popping at the backing's end never disturbs
//!   it. Only a genuinely mid/prefix view (`offset + len < full_len`)
//!   still pays `normalize()` for these ops.
//! - `set`/`set_owned` need no view-shape gate at all: a `set` touches
//!   exactly one backing slot (`offset + i`) and copies the rest
//!   unchanged, so it never normalizes, on *any* view.
//! - **Amortized memory trim**: a long-lived queue (`pop_front_owned` +
//!   `push_back_owned` in a loop) monotonically bumps `offset` while
//!   `full_len` roughly tracks `len` — left unchecked, the hidden prefix
//!   pins ever more backing memory (whole retired leaf chunks included).
//!   Policy: if `offset >= len` (the hidden prefix is at least as large
//!   as the live window — equivalently `offset * 2 >= full_len` for a
//!   suffix view), normalize *before* doing the op. Checked on
//!   `push_back_owned` only — **not** also on `pop_front_owned`, a
//!   deliberate deviation from SPEC-M12-SUFFIXVIEW.md's literal text
//!   (which names both): checking it on `pop_front_owned` too regressed
//!   the pre-existing `pop_front_walk` bench (a pure one-directional
//!   drain, no compensating pushes) from 2.08x faster than imbl to 1.28x
//!   *slower* — `normalize()` costs `O(current len)`, and a monotonic
//!   drain's shrinking `len` turns the geometric trim series into another
//!   full `O(n)` of rebuild work with no memory benefit (a one-shot drain
//!   frees its whole backing at the end regardless). The queue-churn
//!   pattern this policy targets still gets trimmed every round via
//!   `push_back_owned`'s own check, so the O(1)-amortized/`<= 2x`-live
//!   bound holds for a live queue either way; see `pop_front_owned`'s own
//!   doc comment for the measured numbers. This bounds resident backing
//!   to `<= 2x` live and is O(1) amortized (each trim roughly halves
//!   `offset`, so the total trim cost over N ops is O(N)). Deliberately
//!   **not** applied to the persistent (`&self`) flavors — they must stay
//!   cheap and
//!   non-surprising regardless of how a caller chains views.
//!
//! ## Owned (mutate-in-place) path
//!
//! `push_back_owned`/`set_owned`/`pop_back_owned` mirror `src/node.rs`'s
//! `assoc_mut`/`assoc_copy` split (also used by the rope's
//! `splice_owned`/`owned_splice_mut`/`owned_splice_copy`): descend the
//! rightmost spine (push/pop) or the index's spine (set) mutating every
//! uniquely-owned node in place via `node.rs`'s take/put primitives,
//! falling back to the borrowed/copy-path *only from the first shared node
//! encountered* — the other owner(s) keep their reference untouched.

mod node;

use std::fmt;

use node::RawNode;

/// Trie fan-out, re-exported at the module level for readability in this
/// file's arithmetic (`node.rs` is the authority; see its module docs).
const NODE_SIZE: usize = node::NODE_SIZE;
const BITS: u32 = node::BITS;

/// A persistent vector: 32-ary trie + tail, Clojure-style, with an O(1)
/// view header (`offset`/`len`) baked into the core type (see module
/// docs). Cloning is O(1) (bumps `root`'s and `tail`'s refcounts). Every
/// mutating operation comes in a `&self` (persistent, always copies on the
/// write path) and an `_owned` (consumes `self`, mutates uniquely-owned
/// nodes in place when possible) flavor.
pub struct PVector<T> {
    root: Option<RawNode<T>>,
    /// Meaningful only when `root.is_some()`. `0` means the root is
    /// itself a leaf; `5 * k` (`k >= 1`) means `k` internal levels above
    /// the leaves. See module docs.
    shift: u32,
    /// Always present (never `None`) — an empty vector's tail is simply a
    /// zero-length leaf, mirroring the rope's always-present empty-leaf
    /// root. Holds `1..=NODE_SIZE` live elements for a nonempty vector,
    /// `0` only when `full_len == 0`.
    tail: RawNode<T>,
    /// Total element count across `root` + `tail` (the *backing* store,
    /// as opposed to `len`, the *visible* window's length — see module
    /// docs' "View header" section).
    full_len: u32,
    /// View window start, `<= full_len`.
    offset: u32,
    /// View window length, `offset + len <= full_len`.
    len: u32,
}

// SAFETY: `PVector<T>` only owns `Option<RawNode<T>>`/`RawNode<T>`
// (Send/Sync exactly when `T: Send + Sync`, see `node.rs`) plus plain
// `Copy` metadata. No interior mutability beyond the node layer's atomic
// refcounts.
unsafe impl<T: Send + Sync> Send for PVector<T> {}
unsafe impl<T: Send + Sync> Sync for PVector<T> {}

impl<T> Drop for PVector<T> {
    fn drop(&mut self) {
        if let Some(root) = self.root.take() {
            node::drop_node(root, self.shift == 0);
        }
        // SAFETY: `self.tail` is a plain field (not `Option`), always
        // holding a live `RawNode<T>` — read out via `ptr::read` rather
        // than `mem::replace` specifically to avoid allocating a
        // throwaway placeholder leaf just to immediately discard it
        // (`RawNode<T>` deliberately has no `Drop` impl of its own — see
        // `node.rs`'s module docs — so `mem::replace(&mut self.tail,
        // node::leaf::empty())` would allocate a fresh empty leaf, write
        // it into `self.tail`, and then leak it: nothing ever visits
        // `self.tail` again after this function returns, since `self` is
        // being destroyed right now). `ptr::read` needs no replacement
        // value and leaves `self.tail`'s bits stale-but-unread, which is
        // sound here for the same reason: no further access ever occurs.
        let tail = unsafe { std::ptr::read(&self.tail) };
        node::drop_node(tail, true);
    }
}

impl<T> Clone for PVector<T> {
    fn clone(&self) -> Self {
        PVector {
            root: self.root.as_ref().map(node::clone_shallow),
            shift: self.shift,
            tail: node::clone_shallow(&self.tail),
            full_len: self.full_len,
            offset: self.offset,
            len: self.len,
        }
    }
}

impl<T> Default for PVector<T> {
    fn default() -> Self {
        PVector::new()
    }
}

impl<T> PVector<T> {
    /// An empty vector.
    pub fn new() -> PVector<T> {
        PVector { root: None, shift: 0, tail: node::leaf::empty(), full_len: 0, offset: 0, len: 0 }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The backing store's total element count (`root` + `tail`), as
    /// opposed to [`Self::len`] (the visible view window). Equal to `len`
    /// for every vector that hasn't gone through [`Self::slice`]/
    /// [`Self::pop_front`] — i.e. every vector built by `new`/`push_back`/
    /// `from_slice`/collecting an iterator.
    #[inline]
    fn tail_offset(&self) -> usize {
        self.full_len as usize - node::leaf::len(&self.tail)
    }

    /// A "trivial" view: `offset == 0` and the whole backing store is
    /// visible (`len == full_len`) — the common case (every vector that
    /// hasn't been through `slice`/`pop_front`). Mutating ops fast-path on
    /// this; see module docs' "View header" section.
    #[inline]
    fn is_trivial_view(&self) -> bool {
        self.offset == 0 && self.len == self.full_len
    }

    /// A "suffix" view: the visible window ends exactly at the backing
    /// store's end (`offset + len == full_len`) — a trivial view
    /// (`offset == 0`) is the special case. `push_back`/`pop_back`
    /// (either flavor) fast-path on this: appending/popping at the
    /// backing's end never touches the hidden prefix `[0..offset)`. See
    /// module docs' "Suffix-view fast paths" section.
    #[inline]
    fn is_suffix_view(&self) -> bool {
        self.offset + self.len == self.full_len
    }
}

impl<T> fmt::Debug for PVector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PVector").field("len", &self.len()).finish()
    }
}

// ---------------------------------------------------------------------
// Trie arithmetic shared by the persistent (copy) path here and the owned
// (mutate-in-place) path added in a later step. Free functions, not
// methods — they operate on bare `RawNode<T>`s plus a `shift`, mirroring
// how the rope's `concat`/`split_at` machinery is free functions
// operating on bare `RawNode`s rather than `PText` methods.
// ---------------------------------------------------------------------

/// Descend from `root` (at trie height `shift`) to the leaf containing
/// backing index `i`. Every in-trie leaf is exactly `NODE_SIZE`-aligned
/// and `NODE_SIZE` long (see module docs), so `i`'s low `BITS` bits give
/// the position *within* the returned leaf directly — no need to track a
/// separate "leaf start" offset during the descent.
fn node_for<T>(root: &RawNode<T>, shift: u32, i: usize) -> &RawNode<T> {
    let mut node = root;
    let mut level = shift;
    while level > 0 {
        let idx = (i >> level) & node::MASK;
        node = node::internal::child_at(node, idx);
        level -= BITS;
    }
    node
}

/// Does appending one more full leaf to the trie (rooted at `root`, height
/// `shift`) require growing the root a level taller first? `full_len` is
/// the backing store's element count *before* this push (with the tail
/// already full at `NODE_SIZE`, i.e. about to flush) — standard Clojure
/// `PersistentVector` formula, see `mod.rs`'s module docs' derivation.
fn overflow(full_len: usize, shift: u32) -> bool {
    (full_len >> BITS) > (1usize << shift)
}

/// Wrap `leaf` in a chain of single-child internal nodes so the resulting
/// subtree sits at trie height `shift` — used when `push_tail` needs to
/// start a brand-new subtree (the current rightmost child at some level is
/// already completely full).
fn new_path<T>(shift: u32, leaf: RawNode<T>) -> RawNode<T> {
    if shift == 0 {
        return leaf;
    }
    let child = new_path(shift - BITS, leaf);
    let leaf_children_flag = shift == BITS; // child is a leaf iff we just wrapped at the bottom level
    node::internal::new_consuming(vec![child], leaf_children_flag)
}

/// Is the subtree rooted at `node` (trie height `shift`) *completely*
/// full (`NODE_SIZE^(shift/BITS + 1)` elements)? Only ever meaningful for
/// a node reached by always following "last child" links from some
/// ancestor — see `push_tail_copy`'s doc comment for why that's the only
/// case this needs to answer.
fn is_child_full<T>(node: &RawNode<T>, shift: u32) -> bool {
    if shift == 0 {
        true // an in-trie leaf is always exactly NODE_SIZE (module docs)
    } else {
        let n = node::internal::n_children(node);
        n == NODE_SIZE && is_child_full(node::internal::child_at(node, n - 1), shift - BITS)
    }
}

/// Push-tail (persistent/copy-path): flush a freshly-full tail leaf into
/// the trie, growing the *rightmost spine only* — always copies (never
/// mutates `node`'s own nodes), safe to call on a shared tree. `node` is
/// guaranteed non-full at entry (the caller — either
/// `PVector::push_back_trivial` for the top-level call, checking
/// `overflow` first, or this function's own recursive call, having just
/// checked `is_child_full` was `false` — never hands this a completely
/// full node; see the inductive argument in this file's design notes).
/// `node` is also guaranteed internal (`shift >= BITS`): a `shift == 0`
/// root (itself a leaf) always trips `overflow` first, so this function
/// never has to handle a bare leaf.
fn push_tail_copy<T: Clone>(shift: u32, node: &RawNode<T>, tail_leaf: RawNode<T>) -> RawNode<T> {
    debug_assert!(shift >= BITS);
    let n = node::internal::n_children(node);
    if shift == BITS {
        node::internal::copy_with_child_appended(node, tail_leaf)
    } else {
        let last_idx = n - 1;
        let last_child = node::internal::child_at(node, last_idx);
        if is_child_full(last_child, shift - BITS) {
            let new_child = new_path(shift - BITS, tail_leaf);
            node::internal::copy_with_child_appended(node, new_child)
        } else {
            let new_child = push_tail_copy(shift - BITS, last_child, tail_leaf);
            node::internal::copy_with_child_replaced(node, last_idx, new_child)
        }
    }
}

/// Pop-tail (persistent/copy-path): detach the trie's rightmost leaf,
/// shrinking the rightmost spine only (the mirror image of
/// `push_tail_copy`). Returns `(remaining_subtree, detached_leaf)` —
/// `remaining_subtree` is `None` when removing the leaf empties `node`
/// entirely (caller handles root collapse, see `collapse_root`). `node`
/// is guaranteed internal (`shift >= BITS`) by the same argument as
/// `push_tail_copy` — a `shift == 0` root is handled directly by the
/// caller, never passed in here.
fn extract_rightmost_leaf_copy<T: Clone>(shift: u32, node: &RawNode<T>) -> (Option<RawNode<T>>, RawNode<T>) {
    debug_assert!(shift >= BITS);
    let n = node::internal::n_children(node);
    let last_idx = n - 1;
    if shift == BITS {
        let leaf = node::clone_shallow(node::internal::child_at(node, last_idx));
        (node::internal::copy_without_last_child(node), leaf)
    } else {
        let last_child = node::internal::child_at(node, last_idx);
        let (new_child_opt, leaf) = extract_rightmost_leaf_copy(shift - BITS, last_child);
        match new_child_opt {
            Some(new_child) => (Some(node::internal::copy_with_child_replaced(node, last_idx, new_child)), leaf),
            None => (node::internal::copy_without_last_child(node), leaf),
        }
    }
}

/// Push-tail (owned/mutate-in-place path): mirrors `push_tail_copy`
/// exactly, but descends taking/putting uniquely-owned child slots in
/// place via `node.rs`'s take/put primitives, falling back to
/// `push_tail_copy` (releasing this handle's one reference afterward) the
/// moment a shared node is found — the rope's `owned_splice_mut`/
/// `owned_splice_copy` split, specialized to "always recurse into the
/// rightmost child". Same non-full/internal preconditions as
/// `push_tail_copy`.
fn push_tail_owned<T: Clone>(shift: u32, mut node: RawNode<T>, tail_leaf: RawNode<T>) -> RawNode<T> {
    debug_assert!(shift >= BITS);
    if !node::is_unique(&node) {
        let r = push_tail_copy(shift, &node, tail_leaf);
        node::drop_node(node, false);
        return r;
    }
    let n = node::internal::n_children(&node);
    if shift == BITS {
        return node::internal::unique_append_child(node, tail_leaf);
    }
    let last_idx = n - 1;
    if is_child_full(node::internal::child_at(&node, last_idx), shift - BITS) {
        let new_child = new_path(shift - BITS, tail_leaf);
        node::internal::unique_append_child(node, new_child)
    } else {
        let child = node::internal::take_child(&mut node, last_idx);
        let new_child = push_tail_owned(shift - BITS, child, tail_leaf);
        node::internal::put_child(&mut node, last_idx, new_child);
        node
    }
}

/// Pop-tail (owned/mutate-in-place path): the owned twin of
/// `extract_rightmost_leaf_copy`, same take/put-with-copy-fallback
/// discipline as `push_tail_owned`.
fn extract_rightmost_leaf_owned<T: Clone>(shift: u32, mut node: RawNode<T>) -> (Option<RawNode<T>>, RawNode<T>) {
    debug_assert!(shift >= BITS);
    if !node::is_unique(&node) {
        let (rem, leaf) = extract_rightmost_leaf_copy(shift, &node);
        node::drop_node(node, false);
        return (rem, leaf);
    }
    if shift == BITS {
        return node::internal::unique_pop_last_child(node);
    }
    let n = node::internal::n_children(&node);
    let last_idx = n - 1;
    let child = node::internal::take_child(&mut node, last_idx);
    let (new_child_opt, leaf) = extract_rightmost_leaf_owned(shift - BITS, child);
    match new_child_opt {
        Some(new_child) => {
            node::internal::put_child(&mut node, last_idx, new_child);
            (Some(node), leaf)
        }
        None => (node::internal::finalize_after_taking_last(node, n), leaf),
    }
}

/// Owned spine descent for [`PVector::set_owned`]: mirrors CHAMP's
/// `assoc_mut`/`assoc_copy` split exactly — mutate the leaf slot in place
/// while every node on the path is uniquely owned, falling back to
/// [`set_copy`] (and releasing this handle's one reference) the moment a
/// shared node is found.
fn set_owned_node<T: Clone>(mut node: RawNode<T>, shift: u32, idx: usize, v: T) -> RawNode<T> {
    if !node::is_unique(&node) {
        let r = set_copy(&node, shift, idx, v);
        node::drop_node(node, shift == 0);
        return r;
    }
    if shift == 0 {
        node::leaf::unique_set(&mut node, idx & node::MASK, v);
        node
    } else {
        let child_idx = (idx >> shift) & node::MASK;
        let child = node::internal::take_child(&mut node, child_idx);
        let new_child = set_owned_node(child, shift - BITS, idx, v);
        node::internal::put_child(&mut node, child_idx, new_child);
        node
    }
}

/// After a `push_tail`, `root` grows exactly one level taller by wrapping
/// `[old_root, new_sibling]`; symmetrically, after a `pop_tail` that
/// leaves the (new, shrunk) root with only a single child, that child is
/// the *entire* useful content — the wrapper level is redundant and gets
/// unwrapped, shrinking `shift` back down by one level. Works identically
/// for a copy-path-built `new_root` (freshly constructed, so a
/// `clone_shallow` + `drop_node` round trip to extract its one child costs
/// nothing extra — the shell was about to be discarded either way) and an
/// owned-path one, so both pop paths share this helper.
fn unwrap_single_child<T>(node: RawNode<T>) -> RawNode<T> {
    let child = node::clone_shallow(node::internal::child_at(&node, 0));
    node::drop_node(node, false);
    child
}

/// Assemble the new `(root, shift, tail)` after a pop-tail detaches the
/// trie's rightmost leaf: `rem` is what's left of the trie (`None` if the
/// whole thing emptied), `old_shift` is the trie's height *before* this
/// pop, `leaf` is the detached leaf (becomes the vector's new tail).
fn collapse_root<T>(rem: Option<RawNode<T>>, old_shift: u32, leaf: RawNode<T>) -> (Option<RawNode<T>>, u32, RawNode<T>) {
    match rem {
        None => (None, 0, leaf),
        Some(new_root) => {
            if old_shift > BITS && node::internal::n_children(&new_root) == 1 {
                (Some(unwrap_single_child(new_root)), old_shift - BITS, leaf)
            } else {
                (Some(new_root), old_shift, leaf)
            }
        }
    }
}

/// Bottom-up trie assembly from an ordered list of same-height, `1..=
/// NODE_SIZE`-full leaves (the bulk-build primitive `from_slice`/
/// `FromIterator` chunk into, and — since our trie has a canonical shape
/// determined purely by element count, unlike the rope's history-dependent
/// leaf splits — exactly the shape sequential `push_back` calls would
/// have produced). Plain left-to-right grouping into blocks of `<=
/// NODE_SIZE`, repeated bottom-up, correctly reproduces the "only the
/// rightmost spine may be partial" invariant: grouping `[0..32),
/// [32..64), ..., [last_partial_chunk)` at each level is exactly
/// counting in base-`NODE_SIZE`.
fn build_trie<T>(leaves: Vec<RawNode<T>>) -> (Option<RawNode<T>>, u32) {
    if leaves.is_empty() {
        return (None, 0);
    }
    if leaves.len() == 1 {
        return (Some(leaves.into_iter().next().expect("checked len == 1")), 0);
    }
    let mut level = leaves;
    let mut shift = 0u32;
    let mut leaf_children_flag = true;
    while level.len() > 1 {
        let groups = level.len().div_ceil(NODE_SIZE);
        let mut next = Vec::with_capacity(groups);
        let mut it = level.into_iter();
        for _ in 0..groups {
            let chunk: Vec<RawNode<T>> = (&mut it).take(NODE_SIZE).collect();
            next.push(node::internal::new_consuming(chunk, leaf_children_flag));
        }
        level = next;
        shift += BITS;
        leaf_children_flag = false;
    }
    (Some(level.into_iter().next().expect("loop invariant: level.len() == 1 on exit")), shift)
}

/// Copy-path spine descent for [`PVector::set`]: rebuilds every node on
/// the root-to-leaf path to index `idx`, sharing every untouched sibling
/// subtree (mirrors CHAMP's `copy_with_value` spine-rebuild shape).
fn set_copy<T: Clone>(node: &RawNode<T>, shift: u32, idx: usize, v: T) -> RawNode<T> {
    if shift == 0 {
        node::leaf::copy_with_set(node, idx & node::MASK, v)
    } else {
        let child_idx = (idx >> shift) & node::MASK;
        let child = node::internal::child_at(node, child_idx);
        let new_child = set_copy(child, shift - BITS, idx, v);
        node::internal::copy_with_child_replaced(node, child_idx, new_child)
    }
}

// ---------------------------------------------------------------------
// Core ops: persistent (`&self`) flavor only — get/push_back/pop_back/
// pop_front/push_front/set/slice/from_slice/iter. The owned (`_owned`,
// consuming, in-place-when-unique) flavor is a later step.
// ---------------------------------------------------------------------

impl<T: Clone + Send + Sync> PVector<T> {
    /// `O(log32 n)` (effectively O(1) for any realistic size — a 32-ary
    /// trie is at most 7 levels deep even at 32^7 > 34 billion elements).
    pub fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len as usize {
            return None;
        }
        let idx = i + self.offset as usize;
        let tail_off = self.tail_offset();
        if idx >= tail_off {
            Some(node::leaf::get(&self.tail, idx - tail_off))
        } else {
            let leaf = node_for(self.root.as_ref().expect("idx < tail_offset implies a root exists"), self.shift, idx);
            Some(node::leaf::get(leaf, idx & node::MASK))
        }
    }

    /// Handle-identity check: `true` iff `a`/`b` are views over the exact
    /// same backing allocations at the exact same window — a cheap `O(1)`
    /// "is this literally the same version" test (stronger than
    /// [`PartialEq`], which is content equality).
    pub fn ptr_eq(a: &PVector<T>, b: &PVector<T>) -> bool {
        let tails_eq = node::ptr_eq(&a.tail, &b.tail);
        let roots_eq = match (&a.root, &b.root) {
            (None, None) => true,
            (Some(ra), Some(rb)) => node::ptr_eq(ra, rb),
            _ => false,
        };
        tails_eq && roots_eq && a.offset == b.offset && a.len == b.len
    }

    /// A **non-trivial** view (`offset > 0` or a truncated `len` — see
    /// module docs) rebuilt as a fresh, trivial `PVector` holding exactly
    /// the visible window; a trivial view is returned via an `O(1)` clone
    /// (bumped refcounts, no rebuild). `push_back`/`pop_back` (either
    /// flavor) call this for any view that ISN'T also a suffix view (see
    /// module docs' "Suffix-view fast paths" section — a suffix view
    /// takes a faster path instead); the owned flavors of `push_back`/
    /// `pop_front` also call it directly, unconditionally-checked, as the
    /// amortized memory trim. `set`/`set_owned` never call this at all —
    /// they translate the index and touch the backing directly on any
    /// view. Rationale for the ops that still fall back to a full
    /// rebuild: mutation-after-slice on a genuinely mid/prefix view is
    /// rare-to-absent in the measured op-mix; correctness over cleverness
    /// for that rare case.
    fn normalize(&self) -> PVector<T> {
        if self.is_trivial_view() {
            self.clone()
        } else {
            PVector::from_vec(self.iter().cloned().collect())
        }
    }

    /// Append `v`, persistent (`self` unaffected, always copies along the
    /// touched spine). Common case (`tail` not yet full) is `O(tail len)`
    /// (`<= NODE_SIZE`); every 32nd call additionally flushes the tail
    /// into the trie, `O(log32 n)`.
    pub fn push_back(&self, v: T) -> PVector<T> {
        if self.is_suffix_view() { self.push_back_suffix(v) } else { self.normalize().push_back_suffix(v) }
    }

    /// Fast path for any suffix view (`offset + len == full_len`,
    /// trivial included) — see module docs' "Suffix-view fast paths".
    /// `offset` is carried through unchanged; only `full_len`/`len` grow.
    fn push_back_suffix(&self, v: T) -> PVector<T> {
        debug_assert!(self.is_suffix_view());
        let tail_len = node::leaf::len(&self.tail);
        if tail_len < NODE_SIZE {
            let new_tail = node::leaf::copy_with_push(&self.tail, v);
            return PVector {
                root: self.root.as_ref().map(node::clone_shallow),
                shift: self.shift,
                tail: new_tail,
                full_len: self.full_len + 1,
                offset: self.offset,
                len: self.len + 1,
            };
        }
        // Tail full: flush it into the trie as a fresh rightmost leaf
        // (zero-copy reuse via `clone_shallow` — no need to re-clone all
        // NODE_SIZE elements, the tail's own allocation becomes the trie
        // leaf directly, both handles sharing it from here on).
        let tail_leaf = node::clone_shallow(&self.tail);
        let (new_root, new_shift) = match &self.root {
            None => (tail_leaf, 0),
            Some(root) => {
                if overflow(self.full_len as usize, self.shift) {
                    // `root`'s own children-kind (leaf vs internal) is
                    // exactly `self.shift == 0` (root itself is a leaf
                    // iff shift is 0) — the new wrapper's children are
                    // `[root, new_path(...)]`, both the SAME kind as
                    // `root` (`new_path` bottoms out at the same `shift`).
                    let nr = node::internal::new_consuming(
                        vec![node::clone_shallow(root), new_path(self.shift, tail_leaf)],
                        self.shift == 0,
                    );
                    (nr, self.shift + BITS)
                } else {
                    (push_tail_copy(self.shift, root, tail_leaf), self.shift)
                }
            }
        };
        let new_tail = node::leaf::from_iter_exact(1, std::iter::once(v));
        PVector { root: Some(new_root), shift: new_shift, tail: new_tail, full_len: self.full_len + 1, offset: self.offset, len: self.len + 1 }
    }

    /// Append `v`, consuming `self`: mutates the tail (and, on a flush,
    /// the trie's rightmost spine) in place while it's uniquely owned,
    /// falling back to the copy path from the first shared node found —
    /// see module docs' "Owned (mutate-in-place) path" section. This is
    /// what preserves the host's last-use-handle advantage (Perceus-style
    /// refcounted handles: a value with no other live reference reaches
    /// here with everything uniquely owned) for the `vlines-all`/skeleton
    /// 240k-element owned-build workload the survey measured.
    pub fn push_back_owned(self, v: T) -> PVector<T> {
        // Amortized memory trim (owned path only — see module docs'
        // "Suffix-view fast paths" section): `offset >= len` means the
        // hidden prefix pins at least as much backing memory as the live
        // window, the drifting-queue leak SPEC-M12-SUFFIXVIEW.md names.
        // Folded into the same `normalize()` call the non-suffix-view
        // fallback already needs (normalizing produces a trivial, hence
        // suffix, view either way, so one call covers both triggers).
        if self.offset >= self.len || !self.is_suffix_view() {
            self.normalize().push_back_owned_suffix(v)
        } else {
            self.push_back_owned_suffix(v)
        }
    }

    /// Owned twin of [`Self::push_back_suffix`]: mutates `self`'s fields
    /// in place, so `offset` is naturally carried through unchanged
    /// without needing to be restated anywhere in this function's body.
    ///
    /// SPEC-M11-OWNEDPUSH.md's Fix B (deriving `tail_len` from `full_len`
    /// via register arithmetic instead of this `node::leaf::len(&self.tail)`
    /// header load, to drop the load from this fast path) was implemented
    /// and investigated here, then REVERTED — it measured correct but
    /// *slower*: an interleaved, warmup-discarding A/B probe
    /// (`examples/owned_build_probe.rs`, since removed) against this exact
    /// baseline showed `owned_build`/`queue_churn` regressing ~7-9%
    /// (~1 ns/elem), reproducible across many rounds, not noise (a `Vec`
    /// baseline benched in the same session, untouched by this change,
    /// stayed flat — ruling out a thermal/environmental cause). Root
    /// cause, via disassembly (`objdump -d`, Apple M4 Pro/aarch64):
    /// removing the header load ALSO removed the only thing that made the
    /// compiler load `self.tail`'s heap pointer early; the tail's header
    /// cache line (needed moments later for `is_unique`'s atomic load and
    /// the `cap` check) was then touched for the first time later in the
    /// instruction stream, exposing real memory latency on the critical
    /// path that the original's incidental early touch had hidden. Two
    /// principled mitigations were tried — hoisting the `ptr::read` before
    /// the `tail_len` arithmetic in source, and explicitly pre-reading
    /// `cap` right after the read as an eager touch — neither changed
    /// LLVM's chosen instruction order enough to close the gap. Chasing
    /// this further would mean fighting the compiler's scheduler rather
    /// than writing the algorithm, so — per this spec's own "do NOT add
    /// speculative extra optimizations" instruction, and its "no
    /// regression anywhere in `benches/vec.rs` (±5%)" bar, which this
    /// would have failed — Fix B is not shipped. Fix A (below) stands
    /// alone; see it and [`node::leaf::from_iter_ceiling`]'s doc comment.
    fn push_back_owned_suffix(mut self, v: T) -> PVector<T> {
        debug_assert!(self.is_suffix_view());
        let tail_len = node::leaf::len(&self.tail);
        if tail_len < NODE_SIZE {
            // SAFETY: see the `Drop` impl's comment — `ptr::read` avoids
            // allocating a placeholder leaf `mem::replace` would need
            // (and would otherwise leak, `RawNode<T>` having no `Drop`);
            // `self.tail` is always overwritten with a real value before
            // this function returns, so the intervening stale-bits window
            // is never observed.
            let old_tail = unsafe { std::ptr::read(&self.tail) };
            self.tail = if node::is_unique(&old_tail) {
                node::leaf::unique_push(old_tail, v)
            } else {
                let new_tail = node::leaf::copy_with_push(&old_tail, v);
                node::drop_node(old_tail, true);
                new_tail
            };
            self.full_len += 1;
            self.len += 1;
            return self;
        }
        // Tail full: flush it into the trie. `old_tail` is always MOVED
        // (never merely `clone_shallow`'d) into the new tree — it's being
        // fully retired as `self`'s tail either way (a fresh
        // single-element tail replaces it below), so there is never a
        // reason to keep a second reference around.
        let full_len_before = self.full_len as usize;
        // SAFETY: see the `Drop` impl's comment.
        let old_tail = unsafe { std::ptr::read(&self.tail) };
        let (new_root, new_shift) = match self.root.take() {
            None => (old_tail, 0),
            Some(root) => {
                if overflow(full_len_before, self.shift) {
                    let shift = self.shift;
                    let nr = node::internal::new_consuming(vec![root, new_path(shift, old_tail)], shift == 0);
                    (nr, shift + BITS)
                } else if node::is_unique(&root) {
                    (push_tail_owned(self.shift, root, old_tail), self.shift)
                } else {
                    let r = push_tail_copy(self.shift, &root, old_tail);
                    node::drop_node(root, self.shift == 0);
                    (r, self.shift)
                }
            }
        };
        self.root = Some(new_root);
        self.shift = new_shift;
        // Fix A (SPEC-M11-OWNEDPUSH.md): ceiling-cap (cap == NODE_SIZE),
        // not exact-fit — this is the ONLY cap-1 source in the owned hot
        // loop (see the spec's re-verification addendum, item 1: the
        // empty-vector cap-0 tail already reallocs straight to the
        // ceiling on its first `unique_push`, and owned pop-refill pulls
        // exact-`NODE_SIZE` trie leaves). Without this, the fresh
        // single-element tail's very next owned push would find `n == c`
        // and pay `unique_push`'s realloc-to-ceiling slow path a second
        // time. The PERSISTENT flush's fresh tail above (`push_back_
        // suffix`, not this function) deliberately stays exact-fit via
        // `from_iter_exact` — see that constructor's doc comment.
        self.tail = node::leaf::from_iter_ceiling(1, std::iter::once(v));
        self.full_len += 1;
        self.len += 1;
        self
    }

    /// Remove and return the last element, persistent. `(self.clone(),
    /// None)` on an empty vector (`O(1)`, no-op).
    pub fn pop_back(&self) -> (PVector<T>, Option<T>) {
        if self.len == 0 {
            return (self.clone(), None);
        }
        if self.is_suffix_view() { self.pop_back_suffix() } else { self.normalize().pop_back_suffix() }
    }

    /// Fast path for any suffix view — see module docs' "Suffix-view fast
    /// paths". `full_len`/`len` shrink by one, `offset` unchanged (still
    /// `<= full_len - 1` since `len >= 1` on entry, so the suffix
    /// invariant `offset + len == full_len` is preserved). The
    /// tail-refill-from-trie arm (`extract_rightmost_leaf_copy`) only
    /// ever detaches the trie's *rightmost* leaf and is otherwise
    /// oblivious to `offset` — the hidden prefix `[0..offset)` lives at
    /// the *low* end of the backing and is never touched by a
    /// right-side detach, whether it ends up folded into the surviving
    /// spine (shift >= BITS case) or physically inside the new tail
    /// (single-leaf-root case, `shift == 0`) — either way `offset` still
    /// indexes the correct visible window afterward.
    fn pop_back_suffix(&self) -> (PVector<T>, Option<T>) {
        debug_assert!(self.is_suffix_view() && self.full_len > 0);
        let tail_len = node::leaf::len(&self.tail);
        if tail_len > 1 {
            let (new_tail, popped) = node::leaf::copy_with_pop(&self.tail);
            return (
                PVector {
                    root: self.root.as_ref().map(node::clone_shallow),
                    shift: self.shift,
                    tail: new_tail,
                    full_len: self.full_len - 1,
                    offset: self.offset,
                    len: self.len - 1,
                },
                Some(popped),
            );
        }
        // tail_len == 1: this pop empties the tail; pull a new tail out of
        // the trie (or become fully empty if there is no trie).
        let popped = node::leaf::as_slice(&self.tail)[0].clone();
        let (new_root, new_shift, new_tail) = match &self.root {
            None => (None, 0, node::leaf::empty()),
            Some(root) if self.shift == 0 => (None, 0, node::clone_shallow(root)),
            Some(root) => {
                let (rem, leaf) = extract_rightmost_leaf_copy(self.shift, root);
                collapse_root(rem, self.shift, leaf)
            }
        };
        (
            PVector { root: new_root, shift: new_shift, tail: new_tail, full_len: self.full_len - 1, offset: self.offset, len: self.len - 1 },
            Some(popped),
        )
    }

    /// Consuming twin of [`Self::pop_back`]: mutates in place while
    /// uniquely owned, copy-path fallback from the first shared node.
    /// `(self, None)` on an already-empty vector.
    pub fn pop_back_owned(self) -> (PVector<T>, Option<T>) {
        if self.len == 0 {
            return (self, None);
        }
        if self.is_suffix_view() { self.pop_back_owned_suffix() } else { self.normalize().pop_back_owned_suffix() }
    }

    /// Owned twin of [`Self::pop_back_suffix`]: field mutation in place,
    /// `offset` untouched (and hence correct) throughout. No amortized
    /// trim here — only `push_back_owned`/`pop_front_owned` grow `offset`
    /// (see module docs); this op only ever shrinks `full_len`/`len`.
    fn pop_back_owned_suffix(mut self) -> (PVector<T>, Option<T>) {
        debug_assert!(self.is_suffix_view() && self.full_len > 0);
        let tail_len = node::leaf::len(&self.tail);
        if tail_len > 1 {
            // SAFETY: see the `Drop` impl's comment — `ptr::read` avoids
            // allocating a placeholder leaf `mem::replace` would need
            // (and would otherwise leak, `RawNode<T>` having no `Drop`);
            // `self.tail` is always overwritten with a real value before
            // this function returns, so the intervening stale-bits window
            // is never observed.
            let old_tail = unsafe { std::ptr::read(&self.tail) };
            let (new_tail, popped) = if node::is_unique(&old_tail) {
                node::leaf::unique_pop(old_tail)
            } else {
                let (nt, p) = node::leaf::copy_with_pop(&old_tail);
                node::drop_node(old_tail, true);
                (nt, p)
            };
            self.tail = new_tail;
            self.full_len -= 1;
            self.len -= 1;
            return (self, Some(popped));
        }
        // tail_len == 1: this pop empties the tail; the old tail's shell
        // is being discarded either way (a leaf pulled from the trie, or
        // an empty leaf, replaces it below), so just clone the one live
        // element out and drop the shell — no benefit to a `unique_pop`
        // dance here (it would still need to be discarded right after).
        // SAFETY: see the `Drop` impl's comment.
        let old_tail = unsafe { std::ptr::read(&self.tail) };
        let popped = node::leaf::as_slice(&old_tail)[0].clone();
        node::drop_node(old_tail, true);
        let (new_root, new_shift, new_tail) = match self.root.take() {
            None => (None, 0, node::leaf::empty()),
            Some(root) if self.shift == 0 => (None, 0, root), // root IS the leaf; move directly
            Some(root) => {
                if node::is_unique(&root) {
                    let (rem, leaf) = extract_rightmost_leaf_owned(self.shift, root);
                    collapse_root(rem, self.shift, leaf)
                } else {
                    let shift = self.shift;
                    let (rem, leaf) = extract_rightmost_leaf_copy(shift, &root);
                    node::drop_node(root, false);
                    collapse_root(rem, shift, leaf)
                }
            }
        };
        self.root = new_root;
        self.shift = new_shift;
        self.tail = new_tail;
        self.full_len -= 1;
        self.len -= 1;
        (self, Some(popped))
    }

    /// Remove and return the first element, `O(1)` (offset/len only —
    /// this is what keeps the host's `uncons`-driven seq-walks cheap; see
    /// module docs). `(self.clone(), None)` on an empty vector.
    pub fn pop_front(&self) -> (PVector<T>, Option<T>) {
        if self.len == 0 {
            return (self.clone(), None);
        }
        let front = self.get(0).cloned();
        let rest = PVector {
            root: self.root.as_ref().map(node::clone_shallow),
            shift: self.shift,
            tail: node::clone_shallow(&self.tail),
            full_len: self.full_len,
            offset: self.offset + 1,
            len: self.len - 1,
        };
        (rest, front)
    }

    /// Consuming twin of [`Self::pop_front`]: no refcount traffic at all
    /// (`self` already owns its `root`/`tail` references — `offset`/`len`
    /// are just adjusted in place and handed back), unlike the persistent
    /// flavor, which must `clone_shallow` both before building `rest`
    /// (since `self` might still be observed by the caller afterward).
    /// Found via SPEC-M10-PVEC.md's own bench gate: the host's `uncons`-
    /// driven seq-walk (this milestone's motivating hot path for
    /// `pop_front`'s O(1) design) is exactly the last-use-handle shape —
    /// `v = v.pop_front().0` in a loop never needs `v`'s old value again —
    /// so paying two atomic refcount round trips per pop for a value
    /// nothing else observes was pure waste; `pop_front_walk` measured
    /// 1.74x SLOWER than imbl using the persistent flavor (missing the
    /// ">=2x faster" bar) and 2.08x FASTER using this one — see
    /// BENCH-RESULTS.md's M10 section for the measured before/after.
    ///
    /// **Deviation from SPEC-M12-SUFFIXVIEW.md's literal text**: the spec
    /// names this op (alongside `push_back_owned`) as a trim checkpoint
    /// too. Measured via `benches/vec.rs`'s pre-existing `pop_front_walk`
    /// (a pure one-directional drain — no compensating pushes): checking
    /// the trim here as well regressed it from M10's 328.75 µs (2.08x
    /// faster than imbl) to ~1.09 ms (1.28x SLOWER than imbl). Root cause:
    /// `normalize()`'s cost is `O(current len)`, and a monotonic drain's
    /// `len` shrinks every call, so the geometric series of trims it
    /// triggers (at len ~= n/2, n/4, n/8, ...) sums to another full
    /// `O(n)` of clone-and-rebuild work stacked on top of the walk's
    /// existing `O(n)` — roughly tripling total cost for a pattern that
    /// was never leaking anything (a one-shot drain frees its whole
    /// backing at the end regardless of when/whether it trims mid-walk).
    /// The queue-churn pattern this policy exists for (`pop_front_owned`
    /// alternating with `push_back_owned`) still gets the trim every
    /// round via `push_back_owned`'s own check below — moving the
    /// checkpoint there exclusively preserves the O(1)-amortized bound
    /// for a live queue while eliminating the pure-drain regression.
    pub fn pop_front_owned(mut self) -> (PVector<T>, Option<T>) {
        if self.len == 0 {
            return (self, None);
        }
        let front = self.get(0).cloned();
        self.offset += 1;
        self.len -= 1;
        (self, front)
    }

    /// Prepend `v`. `O(n)` — a dense trie (unlike RRB) has no structural
    /// trick for front-insertion, so this rebuilds a fresh vector from
    /// scratch; SPEC-M10-PVEC.md scopes RRB relaxed nodes out, and the
    /// measured op-mix never calls `push_front` in a hot loop, so
    /// simplicity wins here over engineering around it.
    pub fn push_front(&self, v: T) -> PVector<T> {
        let mut items = Vec::with_capacity(self.len() + 1);
        items.push(v);
        items.extend(self.iter().cloned());
        PVector::from_vec(items)
    }

    /// Replace the element at `i`, persistent (always copies along the
    /// touched spine). Panics if `i >= self.len()`.
    pub fn set(&self, i: usize, v: T) -> PVector<T> {
        assert!(i < self.len as usize, "PVector::set: index {i} out of bounds (len {})", self.len);
        // No view-shape gate at all (unlike push_back/pop_back): a `set`
        // touches exactly one backing slot (`offset + i`) and copies the
        // rest of the structure unchanged, so it's sound — and never
        // needs `normalize()` — on ANY view, not just suffix ones. See
        // module docs' "Suffix-view fast paths" section.
        self.set_at_backing_index(self.offset as usize + i, v)
    }

    /// `idx` is already offset-translated by the caller (`set`/
    /// `set_owned` add `self.offset`) — this operates purely on backing
    /// indices, exactly like `get`'s `tail_off`/trie-descent split, so it
    /// works correctly regardless of the view's `offset`/`len`.
    fn set_at_backing_index(&self, idx: usize, v: T) -> PVector<T> {
        let tail_off = self.tail_offset();
        if idx >= tail_off {
            let new_tail = node::leaf::copy_with_set(&self.tail, idx - tail_off, v);
            PVector {
                root: self.root.as_ref().map(node::clone_shallow),
                shift: self.shift,
                tail: new_tail,
                full_len: self.full_len,
                offset: self.offset,
                len: self.len,
            }
        } else {
            let new_root = set_copy(self.root.as_ref().expect("idx < tail_offset implies a root exists"), self.shift, idx, v);
            PVector { root: Some(new_root), shift: self.shift, tail: node::clone_shallow(&self.tail), full_len: self.full_len, offset: self.offset, len: self.len }
        }
    }

    /// Consuming twin of [`Self::set`]: mutates the target leaf slot in
    /// place while every node on its spine is uniquely owned, copy-path
    /// fallback from the first shared node. Panics if `i >= self.len()`.
    pub fn set_owned(self, i: usize, v: T) -> PVector<T> {
        assert!(i < self.len as usize, "PVector::set_owned: index {i} out of bounds (len {})", self.len);
        let idx = self.offset as usize + i;
        self.set_owned_at_backing_index(idx, v)
    }

    /// Owned twin of [`Self::set_at_backing_index`]: field mutation in
    /// place (`offset` untouched, as with every other owned op), no
    /// view-shape gate — sound on any view for the same reason as the
    /// persistent flavor.
    fn set_owned_at_backing_index(mut self, idx: usize, v: T) -> PVector<T> {
        let tail_off = self.tail_offset();
        if idx >= tail_off {
            let local = idx - tail_off;
            // SAFETY: see the `Drop` impl's comment — `ptr::read` avoids
            // allocating a placeholder leaf `mem::replace` would need
            // (and would otherwise leak, `RawNode<T>` having no `Drop`);
            // `self.tail` is always overwritten with a real value before
            // this function returns, so the intervening stale-bits window
            // is never observed.
            let old_tail = unsafe { std::ptr::read(&self.tail) };
            self.tail = if node::is_unique(&old_tail) {
                let mut t = old_tail;
                node::leaf::unique_set(&mut t, local, v);
                t
            } else {
                let nt = node::leaf::copy_with_set(&old_tail, local, v);
                node::drop_node(old_tail, true);
                nt
            };
            return self;
        }
        let old_root = self.root.take().expect("idx < tail_offset implies a root exists");
        self.root = Some(set_owned_node(old_root, self.shift, idx, v));
        self
    }

    /// Structural slice: `O(1)` — an equal-cost twin of [`Self::pop_front`]
    /// generalized to an arbitrary sub-range (see module docs' "View
    /// header" section for the shared caveat: a live slice retains the
    /// whole backing store).
    pub fn slice(&self, range: std::ops::Range<usize>) -> PVector<T> {
        assert!(range.start <= range.end && range.end <= self.len as usize, "PVector::slice: range out of bounds");
        PVector {
            root: self.root.as_ref().map(node::clone_shallow),
            shift: self.shift,
            tail: node::clone_shallow(&self.tail),
            full_len: self.full_len,
            offset: self.offset + range.start as u32,
            len: (range.end - range.start) as u32,
        }
    }

    /// Bulk build consuming `items`, chunking directly into leaves (no
    /// `push_back` loop) — the primitive [`FromIterator`] and
    /// [`Self::from_slice`] both funnel through.
    pub fn from_vec(items: Vec<T>) -> PVector<T> {
        Self::from_exact_iter(items.len(), items.into_iter())
    }

    /// Bulk build cloning every element of `items` (no `push_back` loop) —
    /// chunks `items` directly (`node::leaf::new_from_slice` per chunk)
    /// rather than routing through [`Self::from_vec`]'s iterator-`take`
    /// machinery, since a slice can be chunked with a plain `.chunks()`
    /// call.
    pub fn from_slice(items: &[T]) -> PVector<T> {
        if items.is_empty() {
            return PVector::new();
        }
        let n = items.len();
        let tail_len = if n.is_multiple_of(NODE_SIZE) { NODE_SIZE } else { n % NODE_SIZE };
        let (trie_part, tail_part) = items.split_at(n - tail_len);
        let leaves: Vec<RawNode<T>> = trie_part.chunks(NODE_SIZE).map(node::leaf::new_from_slice).collect();
        let tail = node::leaf::new_from_slice(tail_part);
        let (root, shift) = build_trie(leaves);
        PVector { root, shift, tail, full_len: n as u32, offset: 0, len: n as u32 }
    }

    fn from_exact_iter(n: usize, mut iter: impl Iterator<Item = T>) -> PVector<T> {
        if n == 0 {
            return PVector::new();
        }
        let tail_len = if n.is_multiple_of(NODE_SIZE) { NODE_SIZE } else { n % NODE_SIZE };
        let trie_len = n - tail_len;
        let n_leaves = trie_len / NODE_SIZE;
        let mut leaves: Vec<RawNode<T>> = Vec::with_capacity(n_leaves);
        for _ in 0..n_leaves {
            leaves.push(node::leaf::from_iter_exact(NODE_SIZE, iter.by_ref().take(NODE_SIZE)));
        }
        let tail = node::leaf::from_iter_exact(tail_len, iter.by_ref().take(tail_len));
        debug_assert!(iter.next().is_none(), "from_exact_iter: iterator yielded more than `n` items");
        let (root, shift) = build_trie(leaves);
        PVector { root, shift, tail, full_len: n as u32, offset: 0, len: n as u32 }
    }

    /// Zero-clone walk over the vector's leaves (and, last, the tail) as
    /// `&[T]` chunks, in order — the fast path for equality/reduce/join,
    /// mirroring the rope's `chunks()`. Works uniformly for trivial and
    /// non-trivial (sliced/popped-front) views: each `next()` call
    /// descends to the leaf covering the current position (`O(log32 n)`,
    /// since every in-trie leaf is exactly `NODE_SIZE`-aligned, no stack
    /// bookkeeping needed — see `node_for`) and trims it to both the
    /// leaf's own bounds and the view's `[offset, offset+len)` window.
    pub fn chunks(&self) -> PVecChunks<'_, T> {
        PVecChunks { v: self, pos: 0 }
    }

    /// Borrowing element iterator, `O(1)` amortized (built on
    /// [`Self::chunks`]).
    pub fn iter(&self) -> PVecIter<'_, T> {
        PVecIter { chunks: self.chunks(), cur: [].iter(), remaining: self.len() }
    }

    /// Fallible pruned equality with a caller-supplied element predicate
    /// (SPEC-M13-EQWITH.md): delivers `PartialEq`'s subtree-pruning win to
    /// callers whose notion of "equal" isn't Rust `==` (the host's script `=`
    /// has its own per-element semantics — `(= 0.0 -0.0)`, NaN, lazy
    /// forcing — that can't route through this crate's `T: PartialEq`).
    ///
    /// `eq` is called on an element pair **only** when the two sides'
    /// containing subtree is NOT the same allocation — a pointer-shared
    /// subtree (including the whole tree, the `ptr_eq` case) is skipped
    /// entirely, predicate calls included, on both sides' shared leaves
    /// and internal nodes alike. This is the whole point of the method:
    /// on the "did this change?" pattern (compare-after-one-edit), the
    /// predicate runs on a handful of elements instead of the full O(n)
    /// walk `Interp::values_equal` was paying (measured motivation: a
    /// 100k-element, 99.999%-shared pair cost the same either way through
    /// plain `==`-based comparison before this method existed — the prune
    /// never fired because there was no way to reach it without `T:
    /// PartialEq`).
    ///
    /// Alignment caveat, shared with the pruned `PartialEq` impl further
    /// down this file — same `is_trivial_view()` gate on both sides, same
    /// reasoning (see the "Equality" section comment there, above
    /// `nodes_eq`/`nodes_try_eq`, for why the two recursions' BODIES are
    /// nonetheless kept separate rather than one delegating to the
    /// other): the pruning walk is only sound when both sides are
    /// trivial views,
    /// since equal `full_len` then guarantees identical trie shape
    /// (`build_trie`'s determinism). Any other view shape (a `slice`/
    /// `pop_front` on either side shifts chunk boundaries out of
    /// alignment) falls back to paired leaf-chunk iteration via `iter()`
    /// (built on `chunks()`, never `get()` per element) — still correct,
    /// just without the pruning shortcut, exactly like `PartialEq`'s own
    /// fallback.
    ///
    /// `len` mismatch short-circuits `Ok(false)` before calling `eq` at
    /// all; a `ptr_eq` pair short-circuits `Ok(true)` (subsumes the
    /// top-level identity check the pruned `PartialEq` also does first).
    pub fn try_eq_by<E>(&self, other: &PVector<T>, mut eq: impl FnMut(&T, &T) -> Result<bool, E>) -> Result<bool, E> {
        if self.len != other.len {
            return Ok(false);
        }
        if PVector::ptr_eq(self, other) {
            return Ok(true);
        }
        if self.is_trivial_view() && other.is_trivial_view() {
            let roots_eq = match (&self.root, &other.root) {
                (None, None) => true,
                (Some(a), Some(b)) => nodes_try_eq(a, self.shift, b, other.shift, &mut eq)?,
                _ => false, // unreachable given equal full_len (see build_trie's doc comment), but handled rather than assumed
            };
            if !roots_eq {
                return Ok(false);
            }
            leaves_try_eq(&self.tail, &other.tail, &mut eq)
        } else {
            for (a, b) in self.iter().zip(other.iter()) {
                if !eq(a, b)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }

    /// Infallible convenience wrapper over [`Self::try_eq_by`] (`E =
    /// Infallible`) — falls out for free once the fallible version
    /// exists, per the spec's "don't gold-plate" instruction.
    pub fn eq_by(&self, other: &PVector<T>, mut eq: impl FnMut(&T, &T) -> bool) -> bool {
        match self.try_eq_by(other, |a, b| Ok::<bool, std::convert::Infallible>(eq(a, b))) {
            Ok(result) => result,
            Err(never) => match never {},
        }
    }
}

impl<T: Clone + Send + Sync> FromIterator<T> for PVector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        PVector::from_vec(iter.into_iter().collect())
    }
}

/// Leaf-chunk iterator over a [`PVector`] — see [`PVector::chunks`].
pub struct PVecChunks<'a, T> {
    v: &'a PVector<T>,
    /// View-local position (`0..=v.len()`).
    pos: usize,
}

impl<'a, T: Clone + Send + Sync> Iterator for PVecChunks<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<&'a [T]> {
        if self.pos >= self.v.len() {
            return None;
        }
        let backing_idx = self.pos + self.v.offset as usize;
        let tail_off = self.v.tail_offset();
        let (slice, chunk_start): (&[T], usize) = if backing_idx >= tail_off {
            (node::leaf::as_slice(&self.v.tail), tail_off)
        } else {
            let leaf = node_for(self.v.root.as_ref().expect("backing_idx < tail_offset implies a root exists"), self.v.shift, backing_idx);
            (node::leaf::as_slice(leaf), (backing_idx / NODE_SIZE) * NODE_SIZE)
        };
        let local_lo = backing_idx - chunk_start;
        let view_end = self.v.offset as usize + self.v.len as usize;
        let local_hi = slice.len().min(view_end - chunk_start);
        let out = &slice[local_lo..local_hi];
        self.pos += out.len();
        Some(out)
    }
}

impl<T: Clone + Send + Sync> std::iter::FusedIterator for PVecChunks<'_, T> {}

/// Borrowing element iterator over a [`PVector`] — see [`PVector::iter`].
pub struct PVecIter<'a, T> {
    chunks: PVecChunks<'a, T>,
    cur: std::slice::Iter<'a, T>,
    remaining: usize,
}

impl<'a, T: Clone + Send + Sync> Iterator for PVecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            if let Some(x) = self.cur.next() {
                self.remaining -= 1;
                return Some(x);
            }
            self.cur = self.chunks.next()?.iter();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Clone + Send + Sync> std::iter::FusedIterator for PVecIter<'_, T> {}
impl<T: Clone + Send + Sync> ExactSizeIterator for PVecIter<'_, T> {}

// ---------------------------------------------------------------------
// Equality: `ptr_eq` shortcut, then — for the common case of two trivial
// views (every vector that hasn't been through `slice`/`pop_front`) — a
// subtree-identity-pruned structural walk (`nodes_try_eq`): identical node
// pointers short-circuit the whole subtree as equal, free, since nodes are
// refcounted allocations (SPEC-M10-PVEC.md's design). Two trivial views
// with equal `full_len` are guaranteed the SAME trie shape (this trie's
// shape is a deterministic function of element count alone, unlike the
// rope's history-dependent leaf splits — see `build_trie`'s doc comment),
// so the walk never has to reconcile mismatched structure, only compare
// or skip corresponding subtrees. A non-trivial view falls back to a
// plain element-by-element walk over the logical window (still correct,
// just without the pruning shortcut — views are the cold path per the
// spec's survey).
//
// `nodes_eq`/`leaves_eq` (infallible, `T: PartialEq`) and
// `nodes_try_eq`/`leaves_try_eq` (any `T`, a caller-supplied fallible
// predicate — [`PVector::try_eq_by`]'s engine) share the ALIGNMENT
// reasoning that makes the pruning sound in the first place: both gate
// the pruning walk on `is_trivial_view()` on both sides (see
// `try_eq_by`'s and `PartialEq::eq`'s own bodies) before ever calling
// either recursion, so there is exactly one place that decision is made.
//
// The two recursions' BODIES are deliberately kept separate rather than
// one delegating to the other (SPEC-M13-EQWITH.md's own instruction is
// to share/extract rather than reimplement — this is a measured
// exception to that, not an oversight): an earlier version had
// `nodes_eq`/`leaves_eq` call straight into the generic
// `nodes_try_eq`/`leaves_try_eq` with `eq = |a, b| Ok(a == b)`, and it
// cost `equality/PVector_distinct_equal` **43%** (60.32 µs -> 86.43 µs
// @240k) — `leaves_eq`'s `node::leaf::as_slice(a) == node::leaf::as_
// slice(b)` reaches `<[T]>::eq`'s built-in fast path (effectively a
// `memcmp` for the primitive element types this crate's own benches use,
// SPEC-M10-PVEC.md's `>=` imbl bar this walk exists to clear), which a
// generic per-element loop through an `FnMut` closure does not reliably
// get from the optimizer. `try_eq_by`'s own predicate-skip guarantee
// (never called on a pointer-shared subtree) doesn't depend on which
// recursion runs — both check `ptr_eq` first, at every level — so
// splitting the two costs nothing correctness-wise, only avoids
// resharing a hot loop across two very different calling conventions.
// ---------------------------------------------------------------------

fn leaves_eq<T: PartialEq>(a: &RawNode<T>, b: &RawNode<T>) -> bool {
    node::ptr_eq(a, b) || node::leaf::as_slice(a) == node::leaf::as_slice(b)
}

fn nodes_eq<T: PartialEq>(a: &RawNode<T>, shift_a: u32, b: &RawNode<T>, shift_b: u32) -> bool {
    if node::ptr_eq(a, b) {
        return true; // subtree pruning: identical allocation, skip it entirely
    }
    debug_assert_eq!(shift_a, shift_b, "PVector eq: equal full_len must imply equal trie shape (see build_trie's doc comment)");
    if shift_a == 0 {
        leaves_eq(a, b)
    } else {
        let ca = node::internal::children_slice(a);
        let cb = node::internal::children_slice(b);
        debug_assert_eq!(ca.len(), cb.len(), "PVector eq: equal full_len must imply equal child counts at every level");
        ca.iter().zip(cb.iter()).all(|(x, y)| nodes_eq(x, shift_a - BITS, y, shift_b - BITS))
    }
}

fn leaves_try_eq<T, E>(a: &RawNode<T>, b: &RawNode<T>, eq: &mut impl FnMut(&T, &T) -> Result<bool, E>) -> Result<bool, E> {
    if node::ptr_eq(a, b) {
        return Ok(true); // pointer-shared leaf: `eq` is never called on any of its elements
    }
    let (sa, sb) = (node::leaf::as_slice(a), node::leaf::as_slice(b));
    if sa.len() != sb.len() {
        return Ok(false);
    }
    for (x, y) in sa.iter().zip(sb.iter()) {
        if !eq(x, y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn nodes_try_eq<T, E>(a: &RawNode<T>, shift_a: u32, b: &RawNode<T>, shift_b: u32, eq: &mut impl FnMut(&T, &T) -> Result<bool, E>) -> Result<bool, E> {
    if node::ptr_eq(a, b) {
        return Ok(true); // subtree pruning: identical allocation, skip it (and every `eq` call under it) entirely
    }
    debug_assert_eq!(shift_a, shift_b, "PVector eq: equal full_len must imply equal trie shape (see build_trie's doc comment)");
    if shift_a == 0 {
        leaves_try_eq(a, b, eq)
    } else {
        let ca = node::internal::children_slice(a);
        let cb = node::internal::children_slice(b);
        debug_assert_eq!(ca.len(), cb.len(), "PVector eq: equal full_len must imply equal child counts at every level");
        for (x, y) in ca.iter().zip(cb.iter()) {
            if !nodes_try_eq(x, shift_a - BITS, y, shift_b - BITS, eq)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl<T: Clone + Send + Sync + PartialEq> PartialEq for PVector<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        if PVector::ptr_eq(self, other) {
            return true;
        }
        if self.is_trivial_view() && other.is_trivial_view() {
            let roots_eq = match (&self.root, &other.root) {
                (None, None) => true,
                (Some(a), Some(b)) => nodes_eq(a, self.shift, b, other.shift),
                _ => false, // unreachable given equal full_len (see build_trie's doc comment), but handled rather than assumed
            };
            roots_eq && leaves_eq(&self.tail, &other.tail)
        } else {
            self.iter().eq(other.iter())
        }
    }
}

impl<T: Clone + Send + Sync + Eq> Eq for PVector<T> {}

// ---------------------------------------------------------------------
// Validator: canonical-form/invariant checker, mirroring
// `PersistentHashMap::validate`/`PText::validate`'s gating (`cfg(any(test,
// feature = "validate"))`).
// ---------------------------------------------------------------------

#[cfg(any(test, feature = "validate"))]
impl<T> PVector<T> {
    /// Walk the trie asserting:
    /// - `offset + len <= full_len`;
    /// - the tail holds `1..=NODE_SIZE` elements (nonempty vector) or
    ///   exactly `0` (empty vector, `root` also `None`);
    /// - the trie portion (`full_len - tail_len`) is a whole number of
    ///   `NODE_SIZE`-element leaves;
    /// - every in-trie leaf is exactly `NODE_SIZE`;
    /// - every non-rightmost child at every level is completely full;
    /// - every internal node's `leaf_children` flag matches its actual
    ///   height;
    /// - the recomputed trie element count matches `full_len - tail_len`.
    pub fn validate(&self) {
        assert!(self.offset + self.len <= self.full_len, "PVector: offset+len exceeds full_len");
        let tail_len = node::leaf::len(&self.tail) as u32;
        assert!(tail_len as usize <= NODE_SIZE, "PVector: tail exceeds NODE_SIZE");
        if self.full_len == 0 {
            assert!(self.root.is_none(), "PVector: empty vector must have no root");
            assert_eq!(tail_len, 0, "PVector: empty vector must have an empty tail");
        } else {
            assert!(tail_len >= 1, "PVector: nonempty backing store must have a nonempty tail");
        }
        let trie_len = self.full_len - tail_len;
        assert_eq!(trie_len as usize % NODE_SIZE, 0, "PVector: trie portion must be a whole number of NODE_SIZE leaves");
        match &self.root {
            None => assert_eq!(trie_len, 0, "PVector: root is None but trie_len != 0"),
            Some(root) => {
                let counted = validate_node(root, self.shift, self.shift == 0);
                assert_eq!(counted, trie_len as usize, "PVector: trie element count mismatch");
            }
        }
    }
}

#[cfg(any(test, feature = "validate"))]
fn validate_node<T>(node: &RawNode<T>, shift: u32, is_leaf: bool) -> usize {
    if is_leaf {
        let n = node::leaf::len(node);
        assert_eq!(n, NODE_SIZE, "PVector: an in-trie leaf must always be exactly NODE_SIZE");
        n
    } else {
        let n = node::internal::n_children(node);
        assert!((1..=NODE_SIZE).contains(&n), "PVector: internal node child count out of bounds");
        let leaf_kids = node::internal::leaf_children(node);
        assert_eq!(leaf_kids, shift == BITS, "PVector: leaf_children flag inconsistent with shift");
        let mut total = 0usize;
        for i in 0..n {
            let child = node::internal::child_at(node, i);
            let child_is_leaf = shift == BITS;
            let sub = validate_node(child, shift.saturating_sub(BITS), child_is_leaf);
            if i + 1 < n {
                assert_eq!(sub, NODE_SIZE.pow(shift / BITS), "PVector: non-rightmost child is not completely full");
            }
            total += sub;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vector() {
        let v: PVector<i32> = PVector::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn clone_bumps_refcount_not_content() {
        let v: PVector<i32> = PVector::new();
        let v2 = v.clone();
        assert_eq!(v.len(), 0);
        assert_eq!(v2.len(), 0);
    }

    #[test]
    fn default_is_empty() {
        let v: PVector<i32> = PVector::default();
        assert!(v.is_empty());
    }

    #[test]
    fn push_back_and_get_within_one_tail() {
        let mut v: PVector<i32> = PVector::new();
        for i in 0..10 {
            v = v.push_back(i);
        }
        v.validate();
        assert_eq!(v.len(), 10);
        for i in 0..10 {
            assert_eq!(*v.get(i).unwrap(), i as i32);
        }
        assert!(v.get(10).is_none());
    }

    #[test]
    fn push_back_across_tail_flush_boundary() {
        let mut v: PVector<i32> = PVector::new();
        for i in 0..33 {
            v = v.push_back(i);
            v.validate();
        }
        assert_eq!(v.len(), 33);
        for i in 0..33 {
            assert_eq!(*v.get(i).unwrap(), i as i32);
        }
    }

    #[test]
    fn push_back_many_levels() {
        let n = 100_000;
        let mut v: PVector<i32> = PVector::new();
        for i in 0..n {
            v = v.push_back(i);
        }
        v.validate();
        assert_eq!(v.len() as i32, n);
        for i in [0, 1, 31, 32, 33, 1023, 1024, 1025, n / 2, n - 1] {
            assert_eq!(*v.get(i as usize).unwrap(), i);
        }
    }

    #[test]
    fn push_back_persistent_does_not_mutate_original() {
        let v1: PVector<i32> = PVector::from_slice(&[1, 2, 3]);
        let v2 = v1.push_back(4);
        assert_eq!(v1.len(), 3);
        assert_eq!(v2.len(), 4);
        assert_eq!(*v2.get(3).unwrap(), 4);
    }

    #[test]
    fn pop_back_roundtrip() {
        let mut v: PVector<i32> = PVector::from_slice(&(0..1000).collect::<Vec<_>>());
        for expected in (0..1000).rev() {
            let (rest, popped) = v.pop_back();
            assert_eq!(popped, Some(expected));
            rest.validate();
            v = rest;
        }
        assert!(v.is_empty());
        let (still_empty, none) = v.pop_back();
        assert!(still_empty.is_empty());
        assert_eq!(none, None);
    }

    #[test]
    fn pop_back_across_leaf_and_level_boundaries() {
        for n in [1usize, 31, 32, 33, 63, 64, 1024, 1025, 1056] {
            let items: Vec<i32> = (0..n as i32).collect();
            let mut v = PVector::from_slice(&items);
            v.validate();
            let mut expected = items.clone();
            while let Some(exp) = expected.pop() {
                let (rest, popped) = v.pop_back();
                assert_eq!(popped, Some(exp), "n={n}");
                rest.validate();
                v = rest;
            }
            assert!(v.is_empty(), "n={n}");
        }
    }

    #[test]
    fn set_replaces_element_persistently() {
        let v1: PVector<i32> = PVector::from_slice(&(0..2000).collect::<Vec<_>>());
        let v2 = v1.set(1500, 99999);
        v2.validate();
        assert_eq!(*v1.get(1500).unwrap(), 1500);
        assert_eq!(*v2.get(1500).unwrap(), 99999);
        // Everything else unchanged.
        assert_eq!(*v2.get(0).unwrap(), 0);
        assert_eq!(*v2.get(1999).unwrap(), 1999);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_out_of_bounds_panics() {
        let v: PVector<i32> = PVector::from_slice(&[1, 2, 3]);
        let _ = v.set(3, 0);
    }

    #[test]
    fn slice_is_o1_view_and_reads_correct_window() {
        let items: Vec<i32> = (0..500).collect();
        let v = PVector::from_slice(&items);
        let s = v.slice(100..200);
        s.validate();
        assert_eq!(s.len(), 100);
        for i in 0..100 {
            assert_eq!(*s.get(i).unwrap(), items[100 + i]);
        }
        // Original is untouched.
        assert_eq!(v.len(), 500);
    }

    #[test]
    fn pop_front_is_view_only_and_correct() {
        let items: Vec<i32> = (0..70).collect();
        let mut v = PVector::from_slice(&items);
        for expected in items.iter() {
            let (rest, front) = v.pop_front();
            assert_eq!(front, Some(*expected));
            rest.validate();
            v = rest;
        }
        assert!(v.is_empty());
    }

    #[test]
    fn pop_front_owned_matches_persistent_and_is_view_only() {
        let items: Vec<i32> = (0..70).collect();
        let mut v = PVector::from_slice(&items);
        for expected in items.iter() {
            let (rest, front) = v.pop_front_owned();
            assert_eq!(front, Some(*expected));
            rest.validate();
            v = rest;
        }
        assert!(v.is_empty());
        let (still_empty, none) = v.pop_front_owned();
        assert!(still_empty.is_empty());
        assert_eq!(none, None);
    }

    #[test]
    fn pop_front_owned_on_shared_handle_leaves_other_handle_untouched() {
        let items: Vec<i32> = (0..100).collect();
        let base = PVector::from_slice(&items);
        let kept = base.clone();
        let (rest, front) = base.pop_front_owned();
        assert_eq!(front, Some(0));
        rest.validate();
        assert_eq!(rest.iter().copied().collect::<Vec<_>>(), items[1..]);
        assert_eq!(kept.iter().copied().collect::<Vec<_>>(), items, "clone must be unaffected by pop_front_owned");
    }

    #[test]
    fn push_front_prepends() {
        let v: PVector<i32> = PVector::from_slice(&[2, 3, 4]);
        let v2 = v.push_front(1);
        v2.validate();
        assert_eq!(v2.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn mutation_after_slice_normalizes_correctly() {
        let items: Vec<i32> = (0..200).collect();
        let v = PVector::from_slice(&items);
        let s = v.slice(50..150); // non-trivial view
        let pushed = s.push_back(9999);
        pushed.validate();
        assert_eq!(pushed.len(), 101);
        assert_eq!(*pushed.get(100).unwrap(), 9999);
        for i in 0..100 {
            assert_eq!(*pushed.get(i).unwrap(), items[50 + i]);
        }

        let set_result = s.set(0, -1);
        set_result.validate();
        assert_eq!(*set_result.get(0).unwrap(), -1);
        assert_eq!(*s.get(0).unwrap(), items[50], "original view unaffected");

        let (popped_rest, popped) = s.pop_back();
        popped_rest.validate();
        assert_eq!(popped, Some(items[149]));
        assert_eq!(popped_rest.len(), 99);
    }

    #[test]
    fn from_slice_matches_push_back_loop_shape() {
        for n in [0usize, 1, 31, 32, 33, 1000, 12_345] {
            let items: Vec<i32> = (0..n as i32).collect();
            let bulk = PVector::from_slice(&items);
            bulk.validate();
            let mut looped: PVector<i32> = PVector::new();
            for &x in &items {
                looped = looped.push_back(x);
            }
            looped.validate();
            assert_eq!(bulk.iter().copied().collect::<Vec<_>>(), looped.iter().copied().collect::<Vec<_>>(), "n={n}");
            assert_eq!(bulk.iter().copied().collect::<Vec<_>>(), items, "n={n}");
        }
    }

    #[test]
    fn from_iterator_collect() {
        let v: PVector<i32> = (0..500).collect();
        v.validate();
        assert_eq!(v.len(), 500);
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), (0..500).collect::<Vec<_>>());
    }

    #[test]
    fn ptr_eq_basic() {
        let v1: PVector<i32> = PVector::from_slice(&[1, 2, 3]);
        let v2 = v1.clone();
        assert!(PVector::ptr_eq(&v1, &v2));
        let v3: PVector<i32> = PVector::from_slice(&[1, 2, 3]);
        assert!(!PVector::ptr_eq(&v1, &v3));
        let v4 = v1.push_back(4);
        assert!(!PVector::ptr_eq(&v1, &v4));
    }

    #[test]
    fn iter_size_hint_matches_len() {
        let v: PVector<i32> = PVector::from_slice(&(0..77).collect::<Vec<_>>());
        let it = v.iter();
        assert_eq!(it.len(), 77);
    }

    #[test]
    fn validate_catches_healthy_vectors() {
        PVector::<i32>::new().validate();
        PVector::from_slice(&[1]).validate();
        PVector::from_slice(&(0..32).collect::<Vec<_>>()).validate();
        PVector::from_slice(&(0..33).collect::<Vec<_>>()).validate();
        PVector::from_slice(&(0..1024).collect::<Vec<_>>()).validate();
        PVector::from_slice(&(0..33_000).collect::<Vec<_>>()).validate();
    }

    // -------------------------------------------------------------
    // Step 3: owned path, equality, chunks().
    // -------------------------------------------------------------

    #[test]
    fn push_back_owned_matches_persistent_build() {
        for n in [0usize, 1, 31, 32, 33, 1000, 12_345] {
            let items: Vec<i32> = (0..n as i32).collect();
            let mut owned: PVector<i32> = PVector::new();
            for &x in &items {
                owned = owned.push_back_owned(x);
            }
            owned.validate();
            assert_eq!(owned.iter().copied().collect::<Vec<_>>(), items, "n={n}");
        }
    }

    #[test]
    fn push_back_owned_last_use_handle_stays_unique_across_flush() {
        // A chain of push_back_owned calls on a handle nobody else
        // references should never fall back to the copy path — this
        // doesn't directly observe that (no allocation counter here), but
        // it does exercise the tail-flush-into-trie owned path (root
        // wrap, push_tail_owned recursion) across several levels.
        let mut v: PVector<i32> = PVector::new();
        for i in 0..(32 * 32 * 3 + 17) {
            v = v.push_back_owned(i);
        }
        v.validate();
        assert_eq!(v.len(), 32 * 32 * 3 + 17);
        for i in [0, 31, 32, 1023, 1024, v.len() - 1] {
            assert_eq!(*v.get(i).unwrap(), i as i32);
        }
    }

    #[test]
    fn pop_back_owned_matches_persistent_pop() {
        for n in [1usize, 31, 32, 33, 63, 64, 1024, 1025, 1056] {
            let items: Vec<i32> = (0..n as i32).collect();
            let mut v = PVector::from_slice(&items);
            let mut expected = items.clone();
            while let Some(exp) = expected.pop() {
                let (rest, popped) = v.pop_back_owned();
                assert_eq!(popped, Some(exp), "n={n}");
                rest.validate();
                v = rest;
            }
            assert!(v.is_empty(), "n={n}");
        }
    }

    #[test]
    fn set_owned_replaces_in_place() {
        let items: Vec<i32> = (0..2000).collect();
        let v = PVector::from_slice(&items);
        let v2 = v.set_owned(1500, 99999);
        v2.validate();
        assert_eq!(*v2.get(1500).unwrap(), 99999);
        assert_eq!(*v2.get(0).unwrap(), 0);
        assert_eq!(*v2.get(1999).unwrap(), 1999);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_owned_out_of_bounds_panics() {
        let v: PVector<i32> = PVector::from_slice(&[1, 2, 3]);
        let _ = v.set_owned(3, 0);
    }

    /// Shared-handle copy-on-write assertion: when a second handle is
    /// alive, the owned path must fall back to copying (not mutate
    /// structure the other handle still observes) — the whole point of
    /// the uniqueness check. `push_back_owned`/`pop_back_owned`/
    /// `set_owned` on a *cloned* (rc > 1) vector must leave the clone's
    /// own view of the content untouched.
    #[test]
    fn owned_ops_on_shared_handle_leave_other_handle_untouched() {
        let items: Vec<i32> = (0..100).collect();
        let base = PVector::from_slice(&items);
        let kept = base.clone(); // now shared: rc bumped on tail and root

        let pushed = base.push_back_owned(999);
        pushed.validate();
        assert_eq!(pushed.len(), 101);
        assert_eq!(kept.iter().copied().collect::<Vec<_>>(), items, "clone must be unaffected by push_back_owned");

        let kept2 = pushed.clone();
        let (popped_rest, popped) = pushed.pop_back_owned();
        popped_rest.validate();
        assert_eq!(popped, Some(999));
        assert_eq!(kept2.iter().copied().collect::<Vec<_>>().last(), Some(&999), "clone must be unaffected by pop_back_owned");

        let kept3 = popped_rest.clone();
        let modified = popped_rest.set_owned(0, -1);
        modified.validate();
        assert_eq!(*modified.get(0).unwrap(), -1);
        assert_eq!(*kept3.get(0).unwrap(), 0, "clone must be unaffected by set_owned");
    }

    /// Also exercises the flush-across-a-shared-root case specifically:
    /// a clone is taken right when the tail is about to flush into the
    /// trie, so the owned push must fall back to copying the trie's
    /// rightmost spine (not the whole tree) while the clone stays intact.
    #[test]
    fn owned_push_flush_with_shared_root_leaves_clone_intact() {
        let items: Vec<i32> = (0..32).collect(); // tail exactly full, no trie yet
        let v = PVector::from_slice(&items);
        let kept = v.clone();
        let v2 = v.push_back_owned(32); // forces a flush; root becomes Some (was None) either way, but exercises the None-root branch under a shared tail
        v2.validate();
        assert_eq!(v2.len(), 33);
        assert_eq!(kept.len(), 32);
        assert_eq!(kept.iter().copied().collect::<Vec<_>>(), items);

        // Now with an actual shared trie: build past one full leaf, clone,
        // then push past the tail-flush boundary again.
        let items2: Vec<i32> = (0..64).collect();
        let v3 = PVector::from_slice(&items2);
        let kept3 = v3.clone();
        let v4 = v3.push_back_owned(64);
        v4.validate();
        assert_eq!(kept3.iter().copied().collect::<Vec<_>>(), items2);
        assert_eq!(v4.len(), 65);
    }

    #[test]
    fn chunks_cover_full_content_in_order() {
        for n in [0usize, 1, 31, 32, 33, 1000, 12_345] {
            let items: Vec<i32> = (0..n as i32).collect();
            let v = PVector::from_slice(&items);
            let flat: Vec<i32> = v.chunks().flat_map(|c| c.iter().copied()).collect();
            assert_eq!(flat, items, "n={n}");
        }
    }

    #[test]
    fn chunks_on_non_trivial_view_trims_correctly() {
        let items: Vec<i32> = (0..500).collect();
        let v = PVector::from_slice(&items);
        let s = v.slice(47..453); // deliberately off leaf/tail boundaries on both ends
        let flat: Vec<i32> = s.chunks().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(flat, items[47..453]);
    }

    #[test]
    fn partial_eq_identical_tree_is_true_and_distinct_equal_trees_match() {
        let items: Vec<i32> = (0..5000).collect();
        let a = PVector::from_slice(&items);
        let b = a.clone(); // identical tree (ptr_eq shortcut path)
        assert_eq!(a, b);

        let c = PVector::from_slice(&items); // distinct tree, equal content
        assert_eq!(a, c);
        assert!(!PVector::ptr_eq(&a, &c));

        let d = PVector::from_slice(&(0..4999).collect::<Vec<_>>());
        assert_ne!(a, d);

        let mut e_items = items.clone();
        e_items[2500] = -1;
        let e = PVector::from_slice(&e_items);
        assert_ne!(a, e);
    }

    #[test]
    fn partial_eq_shared_subtree_pruning_still_correct_after_divergence() {
        // Build a big shared vector, then diverge one copy via push_back
        // (persistent — shares almost everything with the original) and
        // confirm equality still agrees with a flat-content comparison
        // (exercises the subtree-pruned walk on a tree that's MOSTLY,
        // but not entirely, shared).
        let items: Vec<i32> = (0..5000).collect();
        let base = PVector::from_slice(&items);
        let diverged = base.set(10, -1);
        assert_ne!(base, diverged);
        assert_eq!(diverged, PVector::from_slice(&{
            let mut v = items.clone();
            v[10] = -1;
            v
        }));

        let appended = base.push_back(9999);
        assert_ne!(base, appended);
        let mut expected = items.clone();
        expected.push(9999);
        assert_eq!(appended, PVector::from_slice(&expected));
    }

    #[test]
    fn partial_eq_non_trivial_views_compare_by_content() {
        let items: Vec<i32> = (0..300).collect();
        let v = PVector::from_slice(&items);
        let s1 = v.slice(50..150);
        let s2 = PVector::from_slice(&items[50..150]);
        assert_eq!(s1, s2);
        let s3 = v.slice(50..151);
        assert_ne!(s1, s3);
    }

    // -------------------------------------------------------------
    // M12 (SPEC-M12-SUFFIXVIEW.md): suffix-view fast paths, set on any
    // view, amortized trim.
    // -------------------------------------------------------------

    #[test]
    fn suffix_view_push_back_persistent_avoids_normalize_and_stays_correct() {
        let items: Vec<i32> = (0..500).collect();
        let v = PVector::from_slice(&items);
        let (dropped_front, _) = v.pop_front(); // offset=1, suffix view (offset+len==full_len)
        let pushed = dropped_front.push_back(9999);
        pushed.validate();
        let mut expected = items[1..].to_vec();
        expected.push(9999);
        assert_eq!(pushed.iter().copied().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn suffix_view_pop_back_across_tail_and_trie_boundaries() {
        for n in [33usize, 64, 1024, 1056] {
            let items: Vec<i32> = (0..n as i32).collect();
            let v = PVector::from_slice(&items);
            let (mut suffix, _) = v.pop_front(); // offset=1, suffix view
            let mut expected = items[1..].to_vec();
            while let Some(exp) = expected.pop() {
                let (rest, popped) = suffix.pop_back();
                assert_eq!(popped, Some(exp), "n={n}");
                rest.validate();
                suffix = rest;
            }
            assert!(suffix.is_empty(), "n={n}");
        }
    }

    #[test]
    fn suffix_view_owned_push_and_pop_back_match_persistent() {
        let items: Vec<i32> = (0..2000).collect();
        let v = PVector::from_slice(&items);
        let (suffix, _) = v.pop_front(); // offset=1
        let pushed = suffix.clone().push_back_owned(-1);
        pushed.validate();
        let mut expected = items[1..].to_vec();
        expected.push(-1);
        assert_eq!(pushed.iter().copied().collect::<Vec<_>>(), expected);

        let (popped_rest, popped) = suffix.pop_back_owned();
        popped_rest.validate();
        assert_eq!(popped, Some(items[1999]));
        assert_eq!(popped_rest.iter().copied().collect::<Vec<_>>(), items[1..1999]);
    }

    #[test]
    fn mid_and_prefix_views_still_normalize_correctly() {
        // Non-suffix views (offset + len < full_len) must still fall back
        // to normalize() for push_back/pop_back — only the suffix-view
        // gate changed, not the mid/prefix-view path.
        let items: Vec<i32> = (0..300).collect();
        let v = PVector::from_slice(&items);
        let mid = v.slice(50..150); // offset=50, len=100, full_len=300: NOT a suffix view
        let pushed = mid.push_back(-1);
        pushed.validate();
        let mut expected: Vec<i32> = items[50..150].to_vec();
        expected.push(-1);
        assert_eq!(pushed.iter().copied().collect::<Vec<_>>(), expected);

        let pushed_owned = mid.clone().push_back_owned(-2);
        pushed_owned.validate();
        let mut expected2: Vec<i32> = items[50..150].to_vec();
        expected2.push(-2);
        assert_eq!(pushed_owned.iter().copied().collect::<Vec<_>>(), expected2);
    }

    #[test]
    fn set_on_suffix_and_mid_views_never_normalizes_and_is_correct() {
        let items: Vec<i32> = (0..2000).collect();
        let v = PVector::from_slice(&items);

        let (suffix, _) = v.pop_front(); // offset=1
        let s2 = suffix.set(0, -1); // touches backing index 1
        s2.validate();
        assert_eq!(*s2.get(0).unwrap(), -1);
        assert_eq!(*suffix.get(0).unwrap(), items[1], "original suffix view unaffected");

        let mid = v.slice(500..1500); // offset=500, len=1000, full_len=2000
        let m2 = mid.set(0, -2);
        m2.validate();
        assert_eq!(*m2.get(0).unwrap(), -2);
        assert_eq!(*m2.get(1).unwrap(), items[501]);

        let m3 = mid.clone().set_owned(999, -3); // last visible index -> backing index 1499
        m3.validate();
        assert_eq!(*m3.get(999).unwrap(), -3);
        assert_eq!(*mid.get(999).unwrap(), items[1499], "original mid view unaffected by set_owned");
    }

    #[test]
    fn pure_pop_front_owned_drain_stays_correct_without_trimming() {
        // A pure one-directional drain (no compensating push_back_owned)
        // deliberately does NOT trim mid-walk (see pop_front_owned's own
        // doc comment: checking the trim here regressed the pop_front_walk
        // bench, since a one-shot drain frees its whole backing at the end
        // regardless of when/whether it trims along the way) — this test
        // is a correctness check, not a memory-bound one; see
        // `examples/vec_memstats.rs`'s queue-churn section for the actual
        // memory-bound validation of the (push_back_owned-only) trim.
        let n = 4000;
        let items: Vec<i32> = (0..n).collect();
        let mut v = PVector::from_slice(&items);
        for &expected in &items {
            let (rest, popped) = v.pop_front_owned();
            assert_eq!(popped, Some(expected));
            rest.validate();
            v = rest;
        }
        assert!(v.is_empty());
        let (still_empty, none) = v.pop_front_owned();
        assert!(still_empty.is_empty());
        assert_eq!(none, None);
    }

    #[test]
    fn queue_churn_pop_front_owned_push_back_owned_matches_deque_model() {
        // The exact motivating pattern (SPEC-M12-SUFFIXVIEW.md): a queue
        // built from pop_front_owned + push_back_owned in a steady-state
        // loop. Correctness under heavy churn, including past several
        // amortized-trim triggers.
        let n = if cfg!(miri) { 20 } else { 300 };
        let rounds = if cfg!(miri) { 40 } else { 4000 };
        let items: Vec<i32> = (0..n).collect();
        let mut v = PVector::from_slice(&items);
        let mut model: std::collections::VecDeque<i32> = items.into_iter().collect();
        for i in 0..rounds {
            let (rest, popped) = v.pop_front_owned();
            let expected = model.pop_front();
            assert_eq!(popped, expected, "round {i}");
            v = rest;
            let x = 1_000_000 + i;
            model.push_back(x);
            v = v.push_back_owned(x);
            v.validate();
            assert_eq!(v.iter().copied().collect::<Vec<_>>(), model.iter().copied().collect::<Vec<_>>(), "round {i}");
        }
    }

    // -------------------------------------------------------------
    // M13 (SPEC-M13-EQWITH.md): try_eq_by / eq_by.
    // -------------------------------------------------------------

    #[test]
    fn eq_by_matches_partial_eq_semantics() {
        let a = PVector::from_slice(&(0..500).collect::<Vec<i32>>());
        let b = a.clone();
        let c = PVector::from_slice(&(0..500).collect::<Vec<i32>>());
        let d = PVector::from_slice(&(0..499).collect::<Vec<i32>>());
        assert!(a.eq_by(&b, |x, y| x == y));
        assert!(a.eq_by(&c, |x, y| x == y));
        assert!(!a.eq_by(&d, |x, y| x == y));
    }

    #[test]
    fn try_eq_by_len_mismatch_short_circuits_without_calling_predicate() {
        let a = PVector::from_slice(&[1, 2, 3]);
        let b = PVector::from_slice(&[1, 2, 3, 4]);
        let calls = std::cell::Cell::new(0usize);
        let result = a.try_eq_by(&b, |x: &i32, y: &i32| {
            calls.set(calls.get() + 1);
            Ok::<bool, std::convert::Infallible>(x == y)
        });
        assert_eq!(result, Ok(false));
        assert_eq!(calls.get(), 0, "len mismatch must short-circuit before calling the predicate at all");
    }

    #[test]
    fn try_eq_by_propagates_predicate_error() {
        let a = PVector::from_slice(&[1, 2, 3]);
        let b = PVector::from_slice(&[1, 2, 4]);
        let result = a.try_eq_by(&b, |x: &i32, y: &i32| if *x == 2 { Err("boom") } else { Ok(x == y) });
        assert_eq!(result, Err("boom"));
    }

    #[test]
    fn try_eq_by_never_calls_predicate_on_pointer_shared_subtrees() {
        let n: i32 = 240_000;
        let items: Vec<i32> = (0..n).collect();
        let v1 = PVector::from_slice(&items);
        // Diverge only the LAST element of the very first trie leaf:
        // copy-on-write rebuilds just that one leaf plus its spine to the
        // root; every sibling subtree (thousands of leaves, plus this
        // vector's whole tail) stays pointer-identical on both sides and
        // must be pruned without ever calling `eq`.
        let v2 = v1.clone().set_owned(NODE_SIZE - 1, -1);
        let calls = std::cell::Cell::new(0usize);
        let result = v1.try_eq_by(&v2, |a: &i32, b: &i32| -> Result<bool, std::convert::Infallible> {
            calls.set(calls.get() + 1);
            Ok(a == b)
        });
        assert_eq!(result, Ok(false));
        assert!(calls.get() <= 2 * NODE_SIZE, "predicate called {} times, expected <= {}", calls.get(), 2 * NODE_SIZE);
    }

    #[test]
    fn try_eq_by_ptr_eq_pair_never_calls_predicate() {
        let v1 = PVector::from_slice(&(0..1000).collect::<Vec<i32>>());
        let v2 = v1.clone();
        let calls = std::cell::Cell::new(0usize);
        let result = v1.try_eq_by(&v2, |a: &i32, b: &i32| -> Result<bool, std::convert::Infallible> {
            calls.set(calls.get() + 1);
            Ok(a == b)
        });
        assert_eq!(result, Ok(true));
        assert_eq!(calls.get(), 0, "identical-handle pair must be caught by the top-level ptr_eq check, before any subtree walk");
    }

    #[test]
    fn try_eq_by_on_non_trivial_views_falls_back_to_chunked_walk_correctly() {
        let items: Vec<i32> = (0..500).collect();
        let v = PVector::from_slice(&items);
        let s1 = v.slice(47..453); // deliberately off leaf/tail boundaries
        let s2 = PVector::from_slice(&items[47..453]);
        assert!(s1.eq_by(&s2, |x, y| x == y));
        let s3 = v.slice(47..452);
        assert!(!s1.eq_by(&s3, |x, y| x == y));
    }
}
