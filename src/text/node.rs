//! Unsafe single-allocation node layer for `PText`, the persistent rope.
//!
//! Mirrors the crate's `src/node.rs` (CHAMP) discipline exactly: all
//! `unsafe` for the text module lives here (plus reuse of `crate::pool`'s
//! choke-point allocator). Two node kinds, each one heap allocation:
//!
//! ```text
//! Leaf:     LeafHeader { rc, len, cap, chars, newlines } + cap x u8 (UTF-8 bytes)
//! Internal: InternalHeader { rc, n_children, leaf_children } + n_children x RawNode
//!           (child pointers) + n_children x Summary { bytes, chars, newlines }
//! ```
//!
//! `RawNode` is a thin `NonNull<u8>` handle that can point at either header
//! kind. Both headers put `rc: AtomicU32` as their literal first `repr(C)`
//! field, so refcount ops (`is_unique`/`clone_shallow`/`drop_node`) work
//! through a `RawNode` without needing to know which kind it is; anything
//! that needs to *interpret* the rest of the allocation (splitting,
//! scanning children, dropping recursively) is told the kind by its caller
//! — for a rope this is always statically knowable from context: a B-tree's
//! "all leaves at the same depth" invariant means every child of one
//! `Internal` node is uniformly the same kind (all leaves, or all
//! internal), so `InternalHeader` stores exactly one `leaf_children: bool`
//! for the whole node rather than tagging every child pointer individually
//! (matching the CHAMP node layer's "no per-slot tag bytes" spirit). The
//! rope's root (owned by the safe `PText` wrapper in `mod.rs`) carries its
//! own `is_leaf: bool` the same way.
//!
//! `RawNode` deliberately implements neither `Clone` nor `Drop`, same
//! reasoning as CHAMP's `NodePtr`: every refcount change is an explicit,
//! auditable `clone_shallow`/`drop_node` call, and the unique-path mutation
//! machinery below moves `RawNode` values around with take/put semantics
//! (mirroring CHAMP's `take_node`/`put_node`) that an implicit `Drop` would
//! corrupt and an implicit `Clone` would make silently expensive/incorrect.

use std::alloc::{self, Layout};
use std::ops::Range;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "pool-alloc")]
use crate::pool;

/// Maximum live bytes a leaf may ever hold. Bench candidate per the spec
/// (1024/2048/4096); 2048 shipped — see NOTES-PTEXT.md for the (informal,
/// time-boxed) comparison.
pub(crate) const LEAF_MAX: usize = 2048;
/// Target size a leaf split aims for on each side.
pub(crate) const LEAF_SPLIT_TARGET: usize = LEAF_MAX / 2;
/// Leaf capacity growth granularity (spec: "round capacity up to 256-byte
/// steps, cap LEAF_MAX").
const LEAF_GROW_STEP: usize = 256;
/// Max children per internal node (5-bit-chunk-free choice here — nothing
/// hash-derived drives this, just a wide, cache-friendly fanout). The
/// spec's `INTERNAL_MIN = 16` companion (non-root nodes should have
/// `>= 16` children) is a target `mod.rs`'s bulk builder and `concat`
/// aim for on freshly (re)built structure, not a hard invariant enforced
/// here — see `mod.rs`'s module docs and NOTES-PTEXT.md for the honest
/// writeup of why `concat`/`split_at` can't always guarantee it after
/// arbitrary edit sequences.
pub(crate) const INTERNAL_MAX: usize = 32;

// ---------------------------------------------------------------------
// Allocation choke point — reuses the crate's existing thread-local node
// pool (src/pool.rs) exactly like the CHAMP node layer does; the pool is
// generic over `Layout` and has no idea (or need to know) which crate
// module is asking.
// ---------------------------------------------------------------------

#[inline]
unsafe fn raw_alloc(layout: Layout) -> *mut u8 {
    #[cfg(feature = "pool-alloc")]
    {
        unsafe { pool::alloc_node(layout) }
    }
    #[cfg(not(feature = "pool-alloc"))]
    {
        unsafe { alloc::alloc(layout) }
    }
}

#[inline]
unsafe fn raw_dealloc(ptr: *mut u8, layout: Layout) {
    #[cfg(feature = "pool-alloc")]
    {
        unsafe { pool::dealloc_node(ptr, layout) }
    }
    #[cfg(not(feature = "pool-alloc"))]
    {
        unsafe { alloc::dealloc(ptr, layout) }
    }
}

// ---------------------------------------------------------------------
// Summary: per-child (and whole-tree) metrics.
// ---------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Summary {
    pub(crate) bytes: u32,
    pub(crate) chars: u32,
    pub(crate) newlines: u32,
}

impl Summary {
    pub(crate) const ZERO: Summary = Summary { bytes: 0, chars: 0, newlines: 0 };

    #[inline]
    pub(crate) fn add(self, other: Summary) -> Summary {
        Summary {
            bytes: self.bytes + other.bytes,
            chars: self.chars + other.chars,
            newlines: self.newlines + other.newlines,
        }
    }

    #[inline]
    pub(crate) fn sub(self, other: Summary) -> Summary {
        Summary {
            bytes: self.bytes - other.bytes,
            chars: self.chars - other.chars,
            newlines: self.newlines - other.newlines,
        }
    }
}

/// Chars + newlines in `s`, computed in one pass.
#[inline]
fn scan_str(s: &str) -> (u32, u32) {
    let mut chars = 0u32;
    let mut newlines = 0u32;
    for b in s.bytes() {
        // UTF-8 continuation bytes are `0b10xxxxxx`; every other byte
        // (ASCII or a multi-byte sequence's leader) starts a new `char`.
        if b & 0b1100_0000 != 0b1000_0000 {
            chars += 1;
        }
        if b == b'\n' {
            newlines += 1;
        }
    }
    (chars, newlines)
}

// ---------------------------------------------------------------------
// Headers.
// ---------------------------------------------------------------------

#[repr(C)]
struct LeafHeader {
    rc: AtomicU32,
    len: u32,
    cap: u32,
    chars: u32,
    newlines: u32,
}

#[repr(C)]
struct InternalHeader {
    rc: AtomicU32,
    n_children: u32,
    /// Nonzero iff every child of this node is a leaf.
    leaf_children: u32,
}

/// A thin handle to a heap-allocated leaf or internal node. See module docs.
#[repr(transparent)]
pub(crate) struct RawNode(NonNull<u8>);

// SAFETY: a `RawNode` is logically an `Arc`-like shared owner of UTF-8 byte
// data (leaves) or child `RawNode`s (internal nodes), all reached through
// atomically refcounted heap allocations containing only `u8`/`RawNode`
// (itself recursively Send+Sync under this same rule) — there is no `K`/`V`
// generic parameter here (unlike the CHAMP node layer) since text content is
// always plain bytes, so this is unconditionally sound.
unsafe impl Send for RawNode {}
unsafe impl Sync for RawNode {}

// ---------------------------------------------------------------------
// Layout math (mirrors src/node.rs's `node_layout`).
// ---------------------------------------------------------------------

#[inline]
fn leaf_layout(cap: usize) -> (Layout, usize) {
    let header = Layout::new::<LeafHeader>();
    let data = Layout::array::<u8>(cap).expect("champ/text: layout overflow");
    let (l, off) = header.extend(data).expect("champ/text: layout overflow");
    (l.pad_to_align(), off)
}

#[inline]
fn internal_layout(n: usize) -> (Layout, usize, usize) {
    let header = Layout::new::<InternalHeader>();
    let children = Layout::array::<RawNode>(n).expect("champ/text: layout overflow");
    let (l1, child_off) = header.extend(children).expect("champ/text: layout overflow");
    let summaries = Layout::array::<Summary>(n).expect("champ/text: layout overflow");
    let (l2, sum_off) = l1.extend(summaries).expect("champ/text: layout overflow");
    (l2.pad_to_align(), child_off, sum_off)
}

// ---------------------------------------------------------------------
// Shared refcount ops — work through the untyped `RawNode` since `rc` is
// always the first field of both header kinds.
// ---------------------------------------------------------------------

#[inline]
fn rc(node: &RawNode) -> &AtomicU32 {
    // SAFETY: both `LeafHeader` and `InternalHeader` are `repr(C)` with
    // `rc: AtomicU32` as their literal first field, so reading an
    // `AtomicU32` at offset 0 of a live node's allocation is valid
    // regardless of which kind it actually is.
    unsafe { &*(node.0.as_ptr() as *const AtomicU32) }
}

/// See `src/node.rs::NodePtr::is_unique`.
#[inline]
pub(crate) fn is_unique(node: &RawNode) -> bool {
    rc(node).load(Ordering::Acquire) == 1
}

/// See `src/node.rs::NodePtr::clone_shallow`.
#[inline]
pub(crate) fn clone_shallow(node: &RawNode) -> RawNode {
    let prev = rc(node).fetch_add(1, Ordering::Relaxed);
    if prev > i32::MAX as u32 {
        std::process::abort();
    }
    RawNode(node.0)
}

#[inline]
pub(crate) fn ptr_eq(a: &RawNode, b: &RawNode) -> bool {
    a.0 == b.0
}

/// Release one reference; on the last one, recursively drop children (for
/// an internal node) or just free (a leaf has no owned resources besides
/// its bytes) and deallocate. Caller supplies `is_leaf` since a bare
/// `RawNode` can't tell its own kind (see module docs).
pub(crate) fn drop_node(node: RawNode, is_leaf: bool) {
    let prev = rc(&node).fetch_sub(1, Ordering::Release);
    debug_assert!(prev >= 1, "refcount underflow");
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        if is_leaf {
            // SAFETY: last owner, no concurrent access possible from here.
            unsafe { destroy_leaf(node.0) };
        } else {
            // SAFETY: same.
            unsafe { destroy_internal(node.0) };
        }
    }
}

unsafe fn destroy_leaf(header: NonNull<u8>) {
    // SAFETY: header is live (last-owner precondition from `drop_node`).
    let h = unsafe { &*(header.as_ptr() as *const LeafHeader) };
    let cap = h.cap as usize;
    let (layout, _) = leaf_layout(cap);
    // SAFETY: `layout` matches how this allocation was sized; bytes have no
    // destructors, so nothing to drop before freeing.
    unsafe { raw_dealloc(header.as_ptr(), layout) };
}

unsafe fn destroy_internal(header: NonNull<u8>) {
    // SAFETY: header is live (last-owner precondition from `drop_node`).
    let h = unsafe { &*(header.as_ptr() as *const InternalHeader) };
    let n = h.n_children as usize;
    let leaf_children = h.leaf_children != 0;
    let (layout, child_off, _) = internal_layout(n);
    // SAFETY: offset matches how this allocation was built.
    let cptr = unsafe { header.as_ptr().add(child_off) as *mut RawNode };
    for i in 0..n {
        // SAFETY: slot `i` holds a live, initialized child `RawNode`;
        // reading it out and recursively dropping it releases exactly the
        // one reference this node's slot held.
        let child = unsafe { ptr::read(cptr.add(i)) };
        drop_node(child, leaf_children);
    }
    // SAFETY: nothing aliases this allocation any more.
    unsafe { raw_dealloc(header.as_ptr(), layout) };
}

// ---------------------------------------------------------------------
// Leaf ops.
// ---------------------------------------------------------------------

pub(crate) mod leaf {
    use super::*;

    #[inline]
    fn header(node: &RawNode) -> &LeafHeader {
        // SAFETY: caller context guarantees `node` is a leaf.
        unsafe { &*(node.0.as_ptr() as *const LeafHeader) }
    }

    #[inline]
    fn header_mut(node: &mut RawNode) -> &mut LeafHeader {
        debug_assert!(is_unique(node), "leaf header_mut requires a unique node");
        // SAFETY: unique (checked above), `&mut` prevents aliasing.
        unsafe { &mut *(node.0.as_ptr() as *mut LeafHeader) }
    }

    #[inline]
    fn data_ptr(node: &RawNode) -> *mut u8 {
        let (_, off) = leaf_layout(header(node).cap as usize);
        // SAFETY: offset matches how this allocation was built.
        unsafe { node.0.as_ptr().add(off) }
    }

    #[inline]
    pub(crate) fn byte_len(node: &RawNode) -> usize {
        header(node).len as usize
    }

    #[inline]
    pub(crate) fn cap(node: &RawNode) -> usize {
        header(node).cap as usize
    }

    #[inline]
    pub(crate) fn summary(node: &RawNode) -> Summary {
        let h = header(node);
        Summary { bytes: h.len, chars: h.chars, newlines: h.newlines }
    }

    #[inline]
    pub(crate) fn as_str(node: &RawNode) -> &str {
        let h = header(node);
        // SAFETY: `data_ptr(node)..+len` was built exclusively from `&str`
        // inputs copied verbatim (never split off a char boundary — every
        // constructor/mutator below either takes a whole `&str` or a
        // caller-asserted char-boundary byte offset), so it's valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(data_ptr(node), h.len as usize)) }
    }

    /// Fresh leaf holding exactly `s` (`s.len() <= LEAF_MAX`), exact-fit
    /// capacity (copy-path convention: exact fit, no slack).
    pub(crate) fn new_exact(s: &str) -> RawNode {
        debug_assert!(s.len() <= LEAF_MAX);
        let (chars, newlines) = scan_str(s);
        let (layout, off) = leaf_layout(s.len());
        // SAFETY: `layout` nonzero-sized (header alone is).
        let raw = unsafe { raw_alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // SAFETY: freshly allocated, correctly aligned.
        unsafe {
            (raw as *mut LeafHeader).write(LeafHeader {
                rc: AtomicU32::new(1),
                len: s.len() as u32,
                cap: s.len() as u32,
                chars,
                newlines,
            });
            ptr::copy_nonoverlapping(s.as_ptr(), raw.add(off), s.len());
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) })
    }

    /// Structural split at `byte_idx` (must be a char boundary, `<= len`).
    /// Always copies (two fresh exact-fit leaves); not the hot path — the
    /// hot path is `unique_splice_in_place` below.
    pub(crate) fn split_at(node: RawNode, byte_idx: usize) -> (RawNode, RawNode) {
        let s = as_str(&node);
        debug_assert!(s.is_char_boundary(byte_idx));
        let (l, r) = s.split_at(byte_idx);
        let left = new_exact(l);
        let right = new_exact(r);
        drop_node(node, true);
        (left, right)
    }

    /// Growth policy for the unique in-place path: round up to 256-byte
    /// steps, capped at `LEAF_MAX` (spec).
    fn grow_cap(needed: usize) -> usize {
        debug_assert!(needed <= LEAF_MAX);
        needed.next_multiple_of(LEAF_GROW_STEP).min(LEAF_MAX).max(needed)
    }

    /// Splice `local_range` (byte range, char-boundary-respecting) out of a
    /// **unique** leaf and insert `text`, in place when the result still
    /// fits `cap`, else reallocating (still take-semantics: old allocation
    /// freed raw, never dropped-and-realloc'd). Caller has already checked
    /// `new_len <= LEAF_MAX` — this is the M4-lesson "re-touch" fast path:
    /// sequential typing repeatedly hits the same unique leaf, and slack
    /// capacity from a prior grow absorbs inserts with zero allocation.
    pub(crate) fn unique_splice_in_place(mut node: RawNode, local_range: Range<usize>, text: &str) -> RawNode {
        debug_assert!(is_unique(&node));
        let cur_len = byte_len(&node);
        let cur_cap = cap(&node);
        let start = local_range.start;
        let end = local_range.end;
        debug_assert!(start <= end && end <= cur_len);
        let removed_bytes = end - start;
        let new_len = cur_len - removed_bytes + text.len();
        debug_assert!(new_len <= LEAF_MAX);

        // Scan the doomed range for its char/newline contribution before
        // it's overwritten.
        let (removed_chars, removed_newlines) = scan_str(&as_str(&node)[start..end]);
        let (ins_chars, ins_newlines) = scan_str(text);

        if new_len <= cur_cap {
            let dp = data_ptr(&node);
            if text.len() != removed_bytes {
                // SAFETY: overlapping ranges when the tail shifts (memmove,
                // not memcpy) — `ptr::copy` is exactly that. Destination
                // tail end (`start + text.len() + (cur_len - end)` ==
                // `new_len <= cur_cap`) is within the reserved capacity.
                unsafe { ptr::copy(dp.add(end), dp.add(start + text.len()), cur_len - end) };
            }
            // SAFETY: `text` and the leaf's own buffer never alias (`text`
            // is a caller-owned `&str`, not derived from this leaf).
            unsafe { ptr::copy_nonoverlapping(text.as_ptr(), dp.add(start), text.len()) };
            let h = header_mut(&mut node);
            h.len = new_len as u32;
            h.chars = h.chars - removed_chars + ins_chars;
            h.newlines = h.newlines - removed_newlines + ins_newlines;
            return node;
        }

        // Slow path: reallocate (still take-semantics, no drop/no clone of
        // byte data — bytes have no destructors to begin with).
        let new_cap = grow_cap(new_len);
        let (new_layout, new_off) = leaf_layout(new_cap);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(new_layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        let old_dp = data_ptr(&node);
        let old_h = header(&node);
        let new_chars = old_h.chars - removed_chars + ins_chars;
        let new_newlines = old_h.newlines - removed_newlines + ins_newlines;
        // SAFETY: old and new allocations never overlap.
        unsafe {
            let ndp = raw.add(new_off);
            ptr::copy_nonoverlapping(old_dp, ndp, start);
            ptr::copy_nonoverlapping(text.as_ptr(), ndp.add(start), text.len());
            ptr::copy_nonoverlapping(old_dp.add(end), ndp.add(start + text.len()), cur_len - end);
            (raw as *mut LeafHeader).write(LeafHeader {
                rc: AtomicU32::new(1),
                len: new_len as u32,
                cap: new_cap as u32,
                chars: new_chars,
                newlines: new_newlines,
            });
        }
        let (old_layout, _) = leaf_layout(cur_cap);
        // SAFETY: `node`'s allocation, exact layout it was built/grown with;
        // no live data to drop (bytes only).
        unsafe { raw_dealloc(node.0.as_ptr(), old_layout) };
        RawNode(unsafe { NonNull::new_unchecked(raw) })
    }

    /// Outcome of a splice that might not fit in one leaf.
    pub(crate) enum SpliceResult {
        One(RawNode),
        Two(RawNode, RawNode),
    }

    /// Largest byte count `<= max` that's still a char boundary in `s`
    /// (`s` nonempty when `max > 0`). Local twin of `mod.rs`'s
    /// `chunk_boundary` — kept separate rather than shared across the
    /// safe/unsafe module boundary, since it's five lines and the two
    /// call sites want it for different reasons (chunking a fresh bulk
    /// build vs. picking a leaf-split point).
    fn char_boundary_at_or_before(s: &str, max: usize) -> usize {
        let mut i = max.min(s.len());
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    /// M7.5 (SPEC-PTEXT.md follow-up): the M7 owned path's `new_len >
    /// LEAF_MAX` case used to bail all the way out to the general
    /// `concat`/`split_at` algorithm — correct, but (per the profile in
    /// NOTES-PTEXT.md's M7.5 section) responsible for ~99% of measured
    /// wall time on a random-splice workload, since it also rebuilds
    /// every internal node on the path via `extract_children`
    /// (`clone_shallow`-ing up to `INTERNAL_MAX` siblings) even when only
    /// one leaf actually needs to change shape. This is the fix: splice a
    /// **unique** leaf whose result may exceed `LEAF_MAX`, producing a
    /// `Two` (both exact-fit, fresh — same as `leaf::split_at`'s
    /// convention) instead of bailing. Consumes `node` (take-semantics,
    /// mirroring `unique_splice_in_place`).
    pub(crate) fn unique_splice_or_split(node: RawNode, local_range: Range<usize>, text: &str) -> SpliceResult {
        debug_assert!(is_unique(&node));
        let cur_len = byte_len(&node);
        let removed = local_range.end - local_range.start;
        let new_len = cur_len - removed + text.len();
        if new_len <= LEAF_MAX {
            return SpliceResult::One(unique_splice_in_place(node, local_range, text));
        }
        // Build the full resulting content (owned `String`, independent of
        // `node`'s allocation) before dropping `node` — `s` borrows from
        // `node` and must not outlive it.
        let s = as_str(&node);
        let mut result = String::with_capacity(new_len);
        result.push_str(&s[..local_range.start]);
        result.push_str(text);
        result.push_str(&s[local_range.end..]);
        drop_node(node, true);
        let split = char_boundary_at_or_before(&result, new_len / 2);
        let (l, r) = result.split_at(split);
        SpliceResult::Two(new_exact(l), new_exact(r))
    }

    /// Copy-path twin of [`unique_splice_or_split`]: `node` is shared, so
    /// this never mutates or consumes it (borrowed, not owned) — always
    /// builds fresh leaf/leaves, matching the copy-path convention
    /// (exact-fit, no slack) used throughout this module.
    pub(crate) fn copy_splice_or_split(node: &RawNode, local_range: Range<usize>, text: &str) -> SpliceResult {
        let s = as_str(node);
        let removed = local_range.end - local_range.start;
        let new_len = s.len() - removed + text.len();
        let mut result = String::with_capacity(new_len);
        result.push_str(&s[..local_range.start]);
        result.push_str(text);
        result.push_str(&s[local_range.end..]);
        if new_len <= LEAF_MAX {
            return SpliceResult::One(new_exact(&result));
        }
        let split = char_boundary_at_or_before(&result, new_len / 2);
        let (l, r) = result.split_at(split);
        SpliceResult::Two(new_exact(l), new_exact(r))
    }

    /// M7.5: splice a range that spans exactly two adjacent **unique**
    /// leaves — `a[..local_a] + text + b[local_b..]`, consuming both.
    /// Used by `mod.rs`'s `owned_splice_mut` for the "small edit straddles
    /// a leaf boundary" case (see `internal::locate_span2`'s doc comment),
    /// so that case doesn't need the general `concat`/`split_at` fallback.
    pub(crate) fn unique_merge_splice_or_split(a: RawNode, b: RawNode, local_a: usize, local_b: usize, text: &str) -> SpliceResult {
        debug_assert!(is_unique(&a) && is_unique(&b));
        let sa = as_str(&a);
        let sb = as_str(&b);
        let new_len = local_a + text.len() + (byte_len(&b) - local_b);
        let mut result = String::with_capacity(new_len);
        result.push_str(&sa[..local_a]);
        result.push_str(text);
        result.push_str(&sb[local_b..]);
        drop_node(a, true);
        drop_node(b, true);
        if new_len <= LEAF_MAX {
            return SpliceResult::One(new_exact(&result));
        }
        let split = char_boundary_at_or_before(&result, new_len / 2);
        let (l, r) = result.split_at(split);
        SpliceResult::Two(new_exact(l), new_exact(r))
    }
}

// ---------------------------------------------------------------------
// Internal ops.
// ---------------------------------------------------------------------

pub(crate) mod internal {
    use super::*;

    #[inline]
    fn header(node: &RawNode) -> &InternalHeader {
        // SAFETY: caller context guarantees `node` is internal.
        unsafe { &*(node.0.as_ptr() as *const InternalHeader) }
    }

    #[inline]
    fn child_ptr(node: &RawNode) -> *mut RawNode {
        let (_, off, _) = internal_layout(header(node).n_children as usize);
        // SAFETY: offset matches how this allocation was built.
        unsafe { node.0.as_ptr().add(off) as *mut RawNode }
    }

    #[inline]
    fn summary_ptr(node: &RawNode) -> *mut Summary {
        let (_, _, off) = internal_layout(header(node).n_children as usize);
        // SAFETY: offset matches how this allocation was built.
        unsafe { node.0.as_ptr().add(off) as *mut Summary }
    }

    #[inline]
    pub(crate) fn n_children(node: &RawNode) -> usize {
        header(node).n_children as usize
    }

    #[inline]
    pub(crate) fn leaf_children(node: &RawNode) -> bool {
        header(node).leaf_children != 0
    }

    #[inline]
    pub(crate) fn summary_at(node: &RawNode, idx: usize) -> Summary {
        debug_assert!(idx < n_children(node));
        // SAFETY: idx in bounds, slot initialized.
        unsafe { *summary_ptr(node).add(idx) }
    }

    pub(crate) fn total_summary(node: &RawNode) -> Summary {
        let n = n_children(node);
        let sp = summary_ptr(node);
        let mut total = Summary::ZERO;
        for i in 0..n {
            // SAFETY: idx in bounds, slot initialized.
            total = total.add(unsafe { *sp.add(i) });
        }
        total
    }

    /// Borrow the child `RawNode` at `idx` (does not affect refcount —
    /// caller must not drop/move out of this reference; it's a peek, used
    /// for read-only recursion such as query descents).
    pub(crate) fn child_at(node: &RawNode, idx: usize) -> &RawNode {
        debug_assert!(idx < n_children(node));
        // SAFETY: idx in bounds, slot initialized, borrow tied to `node`'s
        // own borrow so it can't outlive the node or alias a mutation.
        unsafe { &*child_ptr(node).add(idx) }
    }

    /// All children as a borrowed slice — lets callers (the `chunks()`
    /// iterator) hold a `std::slice::Iter` instead of manual index
    /// tracking, mirroring `src/node.rs`'s `node_slice()`.
    pub(crate) fn children_slice(node: &RawNode) -> &[RawNode] {
        let n = n_children(node);
        // SAFETY: `child_ptr(node)` is valid for `n` initialized `RawNode`s.
        unsafe { slice::from_raw_parts(child_ptr(node), n) }
    }

    /// All summaries as a borrowed slice — the `Summary` twin of
    /// [`children_slice`]. **Prefer this over a loop of [`summary_at`]
    /// calls**: `summary_at`/`child_at` each independently recompute this
    /// node's full `Layout` (`internal_layout`, several checked-arithmetic
    /// `Layout::array`/`extend` calls) from scratch on every call — fine
    /// for a single lookup, but a loop calling either of them `n` times
    /// pays that layout recomputation `n` times too, turning an O(n) scan
    /// into O(n^2) work. `children_slice`/`summaries_slice` compute the
    /// layout exactly once (inside `child_ptr`/`summary_ptr`, called here
    /// a single time) and hand back a slice the caller can iterate with
    /// zero further layout math. See NOTES-PTEXT.md's M7.5 section for the
    /// profile that found this dominating (`byte_of_char` alone was ~50%
    /// of a replace-all splice's wall time) every prefix-sum query in
    /// `mod.rs` before this fix.
    pub(crate) fn summaries_slice(node: &RawNode) -> &[Summary] {
        let n = n_children(node);
        // SAFETY: `summary_ptr(node)` is valid for `n` initialized `Summary`.
        unsafe { slice::from_raw_parts(summary_ptr(node), n) }
    }

    /// [`children_slice`] and [`summaries_slice`] together, computing
    /// `internal_layout` exactly once instead of once each (they'd
    /// otherwise redundantly recompute the identical layout — every
    /// caller in `mod.rs` that needs both, which is most of them, wants
    /// this instead of calling both separate accessors).
    pub(crate) fn children_and_summaries(node: &RawNode) -> (&[RawNode], &[Summary]) {
        let n = n_children(node);
        let (_, child_off, sum_off) = internal_layout(n);
        // SAFETY: offsets match how this allocation was built; both
        // slices are valid for `n` initialized elements of their type.
        unsafe {
            let base = node.0.as_ptr();
            let children = slice::from_raw_parts(base.add(child_off) as *const RawNode, n);
            let summaries = slice::from_raw_parts(base.add(sum_off) as *const Summary, n);
            (children, summaries)
        }
    }

    /// Fresh internal node consuming `children` (ownership of each element
    /// moves in — no `clone_shallow`; taking `Vec<RawNode>` by value rather
    /// than `&[RawNode]` is deliberate: it makes the transfer-of-ownership
    /// contract enforced by the type system instead of relying on
    /// `RawNode`'s lack of a `Drop` impl to make a borrow-and-bit-copy
    /// "happen to" be safe). Exact-fit, `children.len() in 1..=32`.
    pub(crate) fn new_exact(children: Vec<RawNode>, summaries: &[Summary], leaf_children_flag: bool) -> RawNode {
        let n = children.len();
        debug_assert_eq!(n, summaries.len());
        debug_assert!((1..=INTERNAL_MAX).contains(&n));
        let (layout, child_off, sum_off) = internal_layout(n);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // SAFETY: freshly allocated, correctly aligned/offset.
        unsafe {
            (raw as *mut InternalHeader).write(InternalHeader {
                rc: AtomicU32::new(1),
                n_children: n as u32,
                leaf_children: leaf_children_flag as u32,
            });
            let cp = raw.add(child_off) as *mut RawNode;
            let sp = raw.add(sum_off) as *mut Summary;
            for (i, c) in children.into_iter().enumerate() {
                cp.add(i).write(c);
            }
            for (i, s) in summaries.iter().enumerate() {
                sp.add(i).write(*s);
            }
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) })
    }

    /// Take ownership of child `idx`, logically moving it out. Caller MUST
    /// restore the slot (`put_child`) before `node` is read or dropped
    /// again. Mirrors `src/node.rs`'s `take_node`.
    pub(crate) fn take_child(node: &mut RawNode, idx: usize) -> RawNode {
        debug_assert!(is_unique(node));
        debug_assert!(idx < n_children(node));
        let cp = child_ptr(node);
        // SAFETY: idx in bounds; caller upholds the take/put contract.
        unsafe { ptr::read(cp.add(idx)) }
    }

    /// Complete a `take_child` with a (possibly different) child and an
    /// updated summary entry.
    pub(crate) fn put_child(node: &mut RawNode, idx: usize, child: RawNode, summary: Summary) {
        debug_assert!(idx < n_children(node));
        let cp = child_ptr(node);
        let sp = summary_ptr(node);
        // SAFETY: idx in bounds; slot is in the "moved-from" state left by
        // `take_child`, so a raw write does not leak/overwrite a live value.
        unsafe {
            ptr::write(cp.add(idx), child);
            ptr::write(sp.add(idx), summary);
        }
    }

    /// M7.5: insert a brand-new child at slot `idx` (shifting `idx..n`
    /// right by one), growing `node`'s child+summary arrays by one.
    /// `node` must be unique. This is what lets a leaf split
    /// ([`super::leaf::unique_splice_or_split`]'s `Two` case) be absorbed
    /// directly into a unique parent — one realloc of the parent's own
    /// array, instead of the old all-or-nothing fallback rebuilding the
    /// parent (and every one of *its* siblings, transitively, up to the
    /// root) via the general `concat`/`split_at` algorithm.
    ///
    /// Always reallocates exact-fit — internal nodes carry no capacity
    /// slack in this design (unlike leaves; see the module docs' M4-lesson
    /// discussion), since insertion here only happens on the much rarer
    /// leaf-split event, not on every edit, so there's no repeated-touch
    /// workload to amortize against. Caller is responsible for checking
    /// `n_children(&result) <= INTERNAL_MAX` afterward and splitting if it
    /// doesn't fit — this function does not enforce the cap itself, since
    /// the caller (the owned recursive splice) needs to inspect the
    /// overflowed shape either way to build the two-way split.
    pub(crate) fn unique_insert_child(node: RawNode, idx: usize, child: RawNode, summary: Summary) -> RawNode {
        debug_assert!(is_unique(&node));
        let n = n_children(&node);
        debug_assert!(idx <= n);
        let leaf_kids = leaf_children(&node);
        let (old_layout, _, _) = internal_layout(n);
        let old_cp = child_ptr(&node);
        let old_sp = summary_ptr(&node);
        let old_header = node.0;
        let new_n = n + 1;
        let (new_layout, new_child_off, new_sum_off) = internal_layout(new_n);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(new_layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        // SAFETY: old and new allocations never overlap; `idx <= n` so
        // every range below is in bounds; ownership of every existing
        // child moves from the old array into the new one (take
        // semantics — none are cloned or dropped), and the new `child`
        // moves in at its slot, completing exactly one net insertion.
        unsafe {
            (raw as *mut InternalHeader).write(InternalHeader {
                rc: AtomicU32::new(1),
                n_children: new_n as u32,
                leaf_children: leaf_kids as u32,
            });
            let new_cp = raw.add(new_child_off) as *mut RawNode;
            let new_sp = raw.add(new_sum_off) as *mut Summary;
            ptr::copy_nonoverlapping(old_cp, new_cp, idx);
            ptr::copy_nonoverlapping(old_sp, new_sp, idx);
            new_cp.add(idx).write(child);
            new_sp.add(idx).write(summary);
            ptr::copy_nonoverlapping(old_cp.add(idx), new_cp.add(idx + 1), n - idx);
            ptr::copy_nonoverlapping(old_sp.add(idx), new_sp.add(idx + 1), n - idx);
            raw_dealloc(old_header.as_ptr(), old_layout);
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) })
    }

    /// M7.5: the shrinking twin of [`unique_insert_child`] — remove child
    /// slot `idx` from a unique node, shrinking the array by one. `idx`'s
    /// slot must already be in the "moved-from" state left by a preceding
    /// [`take_child`] (this function does not drop whatever was there;
    /// the caller has already consumed it, e.g. into
    /// [`super::leaf::unique_merge_splice_or_split`]). Always reallocates
    /// exact-fit, same rationale as `unique_insert_child`. Caller ensures
    /// `n_children(&node) >= 2` before calling (a node must always keep
    /// `>= 1` child).
    pub(crate) fn unique_remove_taken_child(node: RawNode, idx: usize) -> RawNode {
        debug_assert!(is_unique(&node));
        let n = n_children(&node);
        debug_assert!(idx < n);
        let new_n = n - 1;
        debug_assert!(new_n >= 1, "a node must always keep >= 1 child");
        let leaf_kids = leaf_children(&node);
        let (old_layout, _, _) = internal_layout(n);
        let old_cp = child_ptr(&node);
        let old_sp = summary_ptr(&node);
        let old_header = node.0;
        let (new_layout, new_child_off, new_sum_off) = internal_layout(new_n);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(new_layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        // SAFETY: old and new allocations never overlap; slot `idx` holds
        // no live value (caller's take-semantics contract above), so
        // skipping it here — never reading or dropping it — is exactly
        // the one net removal; every other child's ownership moves
        // unchanged from the old array into the new one.
        unsafe {
            (raw as *mut InternalHeader).write(InternalHeader {
                rc: AtomicU32::new(1),
                n_children: new_n as u32,
                leaf_children: leaf_kids as u32,
            });
            let new_cp = raw.add(new_child_off) as *mut RawNode;
            let new_sp = raw.add(new_sum_off) as *mut Summary;
            ptr::copy_nonoverlapping(old_cp, new_cp, idx);
            ptr::copy_nonoverlapping(old_sp, new_sp, idx);
            ptr::copy_nonoverlapping(old_cp.add(idx + 1), new_cp.add(idx), n - idx - 1);
            ptr::copy_nonoverlapping(old_sp.add(idx + 1), new_sp.add(idx), n - idx - 1);
            raw_dealloc(old_header.as_ptr(), old_layout);
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) })
    }

    /// Which child (by index) covers byte offset `at` within this node's
    /// combined span, plus the offset local to that child. `at ==
    /// total_bytes` resolves to the last child (append-at-end convention).
    pub(crate) fn locate_byte(node: &RawNode, at: usize) -> (usize, usize) {
        let summaries = summaries_slice(node);
        debug_assert!(!summaries.is_empty());
        let n = summaries.len();
        let mut acc = 0usize;
        for (i, s) in summaries.iter().enumerate() {
            let b = s.bytes as usize;
            if at < acc + b || i == n - 1 {
                return (i, at - acc);
            }
            acc += b;
        }
        unreachable!("locate_byte: byte offset out of range")
    }

    /// Like [`locate_byte`], but for an *empty* (insertion) range
    /// specifically: biases toward the END of the PRECEDING child when
    /// `at` lands exactly on a child boundary, instead of the START of the
    /// following one. The two conventions are numerically equivalent — the
    /// global byte offset `at` is identical either way, only *which
    /// physical leaf* gets credited with owning it differs — so this never
    /// changes byte/char semantics, only performance: it's what lets
    /// `node_try_unique_splice`'s owned fast path keep growing (and
    /// accumulating capacity slack in) the leaf a caret just typed into,
    /// instead of perpetually landing on a fresh, never-before-touched,
    /// zero-slack neighboring leaf whenever the caret happens to sit
    /// exactly on a leaf boundary (which a freshly bulk-built or
    /// freshly-split-by-the-general-path tree does often, since every leaf
    /// it produces starts out exactly full or exact-fit).
    pub(crate) fn locate_insert_point(node: &RawNode, at: usize) -> (usize, usize) {
        let summaries = summaries_slice(node);
        debug_assert!(!summaries.is_empty());
        let mut acc = 0usize;
        for (i, s) in summaries.iter().enumerate() {
            let b = s.bytes as usize;
            if at <= acc + b {
                return (i, at - acc);
            }
            acc += b;
        }
        unreachable!("locate_insert_point: byte offset out of range")
    }

    /// Like [`locate_byte`] but for a `[start, end)` range: `Some((idx,
    /// local_start..local_end))` iff the whole range falls within one
    /// child; `None` if it spans a child boundary (caller must fall back to
    /// a more general algorithm). An empty range uses [`locate_insert_point`]
    /// (see its docs); a non-empty range must keep [`locate_byte`]'s
    /// convention — biasing its *start* toward a preceding child here would
    /// make some genuinely-single-child ranges spuriously look like they
    /// spanned a boundary (`local_end` measured from the wrong child).
    pub(crate) fn locate_range(node: &RawNode, range: &Range<usize>) -> Option<(usize, Range<usize>)> {
        if range.start == range.end {
            let (idx, local) = locate_insert_point(node, range.start);
            return Some((idx, local..local));
        }
        let (idx, local_start) = locate_byte(node, range.start);
        let child_bytes = summary_at(node, idx).bytes as usize;
        let local_end = local_start + (range.end - range.start);
        if local_end <= child_bytes { Some((idx, local_start..local_end)) } else { None }
    }

    /// M7.5: like [`locate_range`], but for the case it returns `None` —
    /// a range spanning more than one child — checks specifically whether
    /// it spans *exactly two adjacent* children, returning
    /// `Some((first_idx, local_start_in_first, local_end_in_second))`.
    /// `None` here means it spans three or more (the caller must use the
    /// general fallback). Small edits (a delete/replace of a handful of
    /// bytes/chars, never more than a few dozen — the overwhelming
    /// majority of real edits, including every match-and-replace in a
    /// `replace_all`) can only ever straddle at most one child boundary,
    /// so this covers the common "small edit happens to land on a leaf
    /// boundary" case cheaply, without [`locate_range`]'s caller falling
    /// all the way back to the general `concat`/`split_at` algorithm — see
    /// `mod.rs`'s `owned_splice_mut` and NOTES-PTEXT.md's M7.5 section for
    /// the profile (a periodic pattern across a bulk-built document, whose
    /// leaves are all packed to exactly `LEAF_MAX`, straddles a leaf
    /// boundary roughly `pattern_len / LEAF_MAX` of the time — rare per
    /// edit, but a `replace_all`-shaped workload has enough edits that the
    /// old fallback-only handling cost more wall time than every other
    /// edit in the run combined).
    pub(crate) fn locate_span2(node: &RawNode, range: &Range<usize>) -> Option<(usize, usize, usize)> {
        let (idx, local_start) = locate_byte(node, range.start);
        let first_bytes = summary_at(node, idx).bytes as usize;
        let end_local_from_first = local_start + (range.end - range.start);
        if end_local_from_first <= first_bytes {
            return None; // fits in one child; caller's locate_range already handles this case
        }
        if idx + 1 >= n_children(node) {
            return None; // no next sibling to span into
        }
        let local_end_in_second = end_local_from_first - first_bytes;
        let second_bytes = summary_at(node, idx + 1).bytes as usize;
        if local_end_in_second <= second_bytes { Some((idx, local_start, local_end_in_second)) } else { None }
    }

    /// Consume `node`, handing back its children as owned `(RawNode,
    /// Summary)` pairs plus whether they're leaves. Respects sharing: if
    /// `node` is uniquely owned, children are taken directly and only the
    /// shell is deallocated (no rc traffic on the children); if shared,
    /// each child is `clone_shallow`'d and `node`'s own one reference is
    /// released normally.
    pub(crate) fn extract_children(node: RawNode) -> (Vec<RawNode>, Vec<Summary>, bool) {
        let n = n_children(&node);
        let leaf_kids = leaf_children(&node);
        let unique = is_unique(&node);
        let cp = child_ptr(&node);
        let sp = summary_ptr(&node);
        let mut summaries = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: idx in bounds, slot initialized.
            summaries.push(unsafe { *sp.add(i) });
        }
        if unique {
            let mut children = Vec::with_capacity(n);
            for i in 0..n {
                // SAFETY: idx in bounds; ownership of each slot transfers to
                // `children`, matching the shell-only dealloc below (no
                // separate recursive drop of these children).
                children.push(unsafe { ptr::read(cp.add(i)) });
            }
            let (layout, _, _) = internal_layout(n);
            // SAFETY: `node`'s own allocation, exact layout, no data left to
            // drop (children already taken, summaries are Copy).
            unsafe { raw_dealloc(node.0.as_ptr(), layout) };
            (children, summaries, leaf_kids)
        } else {
            let mut children = Vec::with_capacity(n);
            for i in 0..n {
                // SAFETY: idx in bounds, slot initialized; bumps each
                // child's refcount, leaving the original node's own copy
                // intact for its other owner(s).
                children.push(clone_shallow(unsafe { &*cp.add(i) }));
            }
            drop_node(node, false);
            (children, summaries, leaf_kids)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_roundtrip() {
        let n = leaf::new_exact("hello");
        assert_eq!(leaf::as_str(&n), "hello");
        assert_eq!(leaf::byte_len(&n), 5);
        assert_eq!(leaf::summary(&n).chars, 5);
        assert_eq!(leaf::summary(&n).newlines, 0);
        drop_node(n, true);
    }

    #[test]
    fn leaf_multibyte_counts() {
        let s = "a\u{00e9}\u{1f600}\n"; // a, e-acute, emoji, newline
        let n = leaf::new_exact(s);
        assert_eq!(leaf::byte_len(&n), s.len());
        assert_eq!(leaf::summary(&n).chars, 4);
        assert_eq!(leaf::summary(&n).newlines, 1);
        drop_node(n, true);
    }

    #[test]
    fn refcount_shares_and_frees() {
        let n = leaf::new_exact("x");
        assert!(is_unique(&n));
        let n2 = clone_shallow(&n);
        assert!(!is_unique(&n));
        drop_node(n, true);
        assert!(is_unique(&n2));
        drop_node(n2, true);
    }

    #[test]
    fn leaf_split_at() {
        let n = leaf::new_exact("hello world");
        let (l, r) = leaf::split_at(n, 5);
        assert_eq!(leaf::as_str(&l), "hello");
        assert_eq!(leaf::as_str(&r), " world");
        drop_node(l, true);
        drop_node(r, true);
    }

    #[test]
    fn leaf_unique_insert_in_place_and_grow() {
        let n = leaf::new_exact("ac");
        let n = leaf::unique_splice_in_place(n, 1..1, "b");
        assert_eq!(leaf::as_str(&n), "abc");
        assert_eq!(leaf::cap(&n), leaf::cap(&n)); // cap is at least len
        assert!(leaf::cap(&n) >= 3);
        // Force growth beyond current cap.
        let big = "x".repeat(1000);
        let n = leaf::unique_splice_in_place(n, 3..3, &big);
        assert_eq!(leaf::byte_len(&n), 1003);
        assert_eq!(leaf::as_str(&n)[3..], big);
        drop_node(n, true);
    }

    #[test]
    fn leaf_unique_delete_in_place() {
        let n = leaf::new_exact("hello world");
        let n = leaf::unique_splice_in_place(n, 5..11, "");
        assert_eq!(leaf::as_str(&n), "hello");
        drop_node(n, true);
    }

    #[test]
    fn internal_build_and_children() {
        let a = leaf::new_exact("aa");
        let b = leaf::new_exact("bb");
        let sa = leaf::summary(&a);
        let sb = leaf::summary(&b);
        let node = internal::new_exact(vec![a, b], &[sa, sb], true);
        assert_eq!(internal::n_children(&node), 2);
        assert!(internal::leaf_children(&node));
        assert_eq!(internal::total_summary(&node), Summary { bytes: 4, chars: 4, newlines: 0 });
        assert_eq!(leaf::as_str(internal::child_at(&node, 0)), "aa");
        assert_eq!(leaf::as_str(internal::child_at(&node, 1)), "bb");
        drop_node(node, false);
    }

    #[test]
    fn internal_take_put_roundtrip() {
        let a = leaf::new_exact("aa");
        let b = leaf::new_exact("bb");
        let sa = leaf::summary(&a);
        let sb = leaf::summary(&b);
        let mut node = internal::new_exact(vec![a, b], &[sa, sb], true);
        let taken = internal::take_child(&mut node, 0);
        assert_eq!(leaf::as_str(&taken), "aa");
        let taken = leaf::unique_splice_in_place(taken, 0..0, "z");
        let new_summary = leaf::summary(&taken);
        internal::put_child(&mut node, 0, taken, new_summary);
        assert_eq!(leaf::as_str(internal::child_at(&node, 0)), "zaa");
        drop_node(node, false);
    }

    #[test]
    fn internal_locate_range() {
        let a = leaf::new_exact("aaa"); // bytes 0..3
        let b = leaf::new_exact("bbbb"); // bytes 3..7
        let sa = leaf::summary(&a);
        let sb = leaf::summary(&b);
        let node = internal::new_exact(vec![a, b], &[sa, sb], true);
        assert_eq!(internal::locate_range(&node, &(1..2)), Some((0, 1..2)));
        assert_eq!(internal::locate_range(&node, &(4..5)), Some((1, 1..2)));
        assert_eq!(internal::locate_range(&node, &(2..4)), None); // spans boundary
        assert_eq!(internal::locate_byte(&node, 7), (1, 4)); // end-of-tree convention
        drop_node(node, false);
    }

    #[test]
    fn internal_extract_children_unique() {
        let a = leaf::new_exact("aa");
        let b = leaf::new_exact("bb");
        let sa = leaf::summary(&a);
        let sb = leaf::summary(&b);
        let node = internal::new_exact(vec![a, b], &[sa, sb], true);
        let (children, summaries, leaf_kids) = internal::extract_children(node);
        assert!(leaf_kids);
        assert_eq!(summaries.len(), 2);
        assert_eq!(leaf::as_str(&children[0]), "aa");
        assert_eq!(leaf::as_str(&children[1]), "bb");
        for c in children {
            drop_node(c, true);
        }
    }

    #[test]
    fn internal_extract_children_shared() {
        let a = leaf::new_exact("aa");
        let b = leaf::new_exact("bb");
        let sa = leaf::summary(&a);
        let sb = leaf::summary(&b);
        let node = internal::new_exact(vec![a, b], &[sa, sb], true);
        let node2 = clone_shallow(&node); // now shared
        let (children, _summaries, _leaf_kids) = internal::extract_children(node);
        assert_eq!(leaf::as_str(&children[0]), "aa");
        assert_eq!(leaf::as_str(&children[1]), "bb");
        // node2 is still fully valid and independent.
        assert_eq!(leaf::as_str(internal::child_at(&node2, 0)), "aa");
        for c in children {
            drop_node(c, true);
        }
        drop_node(node2, false);
    }
}
