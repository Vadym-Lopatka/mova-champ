//! Unsafe single-allocation node layer for `PVector`, the persistent
//! 32-ary trie + tail vector. Mirrors `src/text/node.rs`'s (the rope's)
//! discipline, itself mirroring `src/node.rs` (CHAMP): all `unsafe` for the
//! vec module lives here (plus reuse of `crate::pool`'s choke-point
//! allocator). Two node kinds, each one heap allocation:
//!
//! ```text
//! Leaf:     LeafHeader { rc, len, cap } + cap x T
//! Internal: InternalHeader { rc, n_children, leaf_children } + n_children x RawNode<T>
//! ```
//!
//! Unlike the rope (whose content is always plain bytes, no generic
//! parameter) this module is generic over `T`, same as CHAMP's
//! `NodePtr<K, V>` — `RawNode<T>` carries a `PhantomData<T>` purely for
//! variance/drop-check/auto-trait purposes; the actual bytes behind a
//! `RawNode<T>` are reached through raw-pointer arithmetic exactly like the
//! rope's untyped `RawNode`. Both header kinds put `rc: AtomicU32` as their
//! literal first `repr(C)` field, so refcount ops (`is_unique`/
//! `clone_shallow`/`drop_node`) work through a `RawNode<T>` without needing
//! to know which kind it is; anything that needs to *interpret* the rest of
//! the allocation is told the kind by its caller (a `RawNode<T>` can't tell
//! its own kind, same reasoning as the rope's module docs) — for this
//! trie it's always statically knowable from context: every node at trie
//! depth `shift == 0` is a leaf, every node at `shift > 0` is internal, and
//! `InternalHeader` additionally caches a `leaf_children: bool` for
//! symmetric use where only the immediate parent (not the caller's whole
//! recursion state) is at hand (mirrors the rope's `leaf_children` field
//! exactly, same rationale).
//!
//! ## No per-node capacity slack on internal nodes; leaf residual slack only
//!
//! Interior nodes of a dense 32-ary trie are always exactly `1..=32`
//! children with no reserved-but-unused slots — SPEC-M10-PVEC.md's
//! "exact-fit" mandate — so `unique_append_child`/`unique_pop_last_child`
//! always reallocate (same policy as the rope's
//! `unique_insert_child`/`unique_remove_taken_child`, restricted here to
//! the *rightmost* slot only, since a dense trie only ever grows/shrinks
//! its rightmost spine — see `mod.rs`'s `push_tail`/`pop_tail`).
//!
//! Leaves (used both for genuine trie leaves, always exactly full at 32
//! once placed in the trie, and for the vector's separate `tail` buffer,
//! `1..=32` live elements — see `mod.rs`'s module docs for the tail
//! invariant) DO carry residual capacity slack, mirroring CHAMP's
//! `NodePtr::unique_remove_data`: `unique_pop` never shrinks `cap` when it
//! removes the last live element, so a tail that has ever grown-then-shrunk
//! (a push/pop cycle, e.g. an undo/redo or a kill-ring style workload)
//! keeps genuine slack that a later `unique_push` can reuse via its
//! in-place fast path (`len < cap`, no allocation) — same mechanism, same
//! rationale, as `src/node.rs`'s own final M4 policy (see that file's
//! module docs' "Capacity slack" section).

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "pool-alloc")]
use crate::pool;

/// Trie fan-out / max leaf size: fixed at 32 (5-bit chunks) per
/// SPEC-M10-PVEC.md's design — not a tunable, unlike the rope's
/// `LEAF_MAX`/`INTERNAL_MAX` (which were bench-picked). `mod.rs`'s shift
/// arithmetic (`>> 5`, `& 0x1f`) is hardcoded to this value throughout.
pub(crate) const BITS: u32 = 5;
pub(crate) const NODE_SIZE: usize = 1 << BITS; // 32
pub(crate) const MASK: usize = NODE_SIZE - 1; // 0x1f

// ---------------------------------------------------------------------
// Allocation choke point — reuses the crate's existing thread-local node
// pool (src/pool.rs) exactly like the CHAMP and rope node layers do.
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
// Headers.
// ---------------------------------------------------------------------

#[repr(C)]
struct LeafHeader {
    rc: AtomicU32,
    len: u32,
    cap: u32,
}

#[repr(C)]
struct InternalHeader {
    rc: AtomicU32,
    n_children: u32,
    /// Nonzero iff every child of this node is a leaf (mirrors the rope's
    /// `InternalHeader::leaf_children`).
    leaf_children: u32,
}

/// A thin handle to a heap-allocated leaf or internal node holding/pointing
/// at `T`s. See module docs.
#[repr(transparent)]
pub(crate) struct RawNode<T>(NonNull<u8>, PhantomData<T>);

// SAFETY: a `RawNode<T>` is logically an `Arc`-like shared owner of `T`
// values (leaves) or child `RawNode<T>`s (internal nodes) reachable through
// atomically refcounted heap allocations. Sound to send/share across
// threads exactly when `T` is `Send`/`Sync`, matching CHAMP's
// `NodePtr<K, V>` rule (and `Arc<T>`'s).
unsafe impl<T: Send + Sync> Send for RawNode<T> {}
unsafe impl<T: Send + Sync> Sync for RawNode<T> {}

// `RawNode<T>` deliberately implements neither `Clone` nor `Drop`, same
// reasoning as CHAMP's `NodePtr`/the rope's `RawNode`: every refcount
// change is an explicit, auditable `clone_shallow`/`drop_node` call, and
// the unique-path mutation machinery below moves `RawNode<T>` values
// around with take/put semantics that an implicit `Drop` would corrupt and
// an implicit `Clone` would make silently expensive/incorrect.

// ---------------------------------------------------------------------
// Layout math (mirrors the rope's `leaf_layout`/`internal_layout`).
// ---------------------------------------------------------------------

#[inline]
fn leaf_layout<T>(cap: usize) -> (Layout, usize) {
    let header = Layout::new::<LeafHeader>();
    let data = Layout::array::<T>(cap).expect("champ/vec: layout overflow");
    let (l, off) = header.extend(data).expect("champ/vec: layout overflow");
    (l.pad_to_align(), off)
}

#[inline]
fn internal_layout<T>(n: usize) -> (Layout, usize) {
    let header = Layout::new::<InternalHeader>();
    let children = Layout::array::<RawNode<T>>(n).expect("champ/vec: layout overflow");
    let (l, off) = header.extend(children).expect("champ/vec: layout overflow");
    (l.pad_to_align(), off)
}

// ---------------------------------------------------------------------
// Shared refcount ops — work through the untyped `NonNull<u8>` since `rc`
// is always the first field of both header kinds.
// ---------------------------------------------------------------------

#[inline]
fn rc<T>(node: &RawNode<T>) -> &AtomicU32 {
    // SAFETY: both `LeafHeader` and `InternalHeader` are `repr(C)` with
    // `rc: AtomicU32` as their literal first field, so reading an
    // `AtomicU32` at offset 0 of a live node's allocation is valid
    // regardless of which kind it actually is.
    unsafe { &*(node.0.as_ptr() as *const AtomicU32) }
}

#[inline]
pub(crate) fn is_unique<T>(node: &RawNode<T>) -> bool {
    rc(node).load(Ordering::Acquire) == 1
}

#[inline]
pub(crate) fn clone_shallow<T>(node: &RawNode<T>) -> RawNode<T> {
    let prev = rc(node).fetch_add(1, Ordering::Relaxed);
    if prev > i32::MAX as u32 {
        std::process::abort();
    }
    RawNode(node.0, PhantomData)
}

#[inline]
pub(crate) fn ptr_eq<T>(a: &RawNode<T>, b: &RawNode<T>) -> bool {
    a.0 == b.0
}

/// Release one reference; on the last one, recursively drop children (for
/// an internal node) or the live `T` elements (a leaf) and deallocate.
/// Caller supplies `is_leaf` since a bare `RawNode<T>` can't tell its own
/// kind (see module docs).
pub(crate) fn drop_node<T>(node: RawNode<T>, is_leaf: bool) {
    let prev = rc(&node).fetch_sub(1, Ordering::Release);
    debug_assert!(prev >= 1, "refcount underflow");
    if prev == 1 {
        std::sync::atomic::fence(Ordering::Acquire);
        if is_leaf {
            // SAFETY: last owner, no concurrent access possible from here.
            unsafe { destroy_leaf::<T>(node.0) };
        } else {
            // SAFETY: same.
            unsafe { destroy_internal::<T>(node.0) };
        }
    }
}

unsafe fn destroy_leaf<T>(header: NonNull<u8>) {
    // SAFETY: header is live (last-owner precondition from `drop_node`).
    let h = unsafe { &*(header.as_ptr() as *const LeafHeader) };
    let len = h.len as usize;
    let cap = h.cap as usize;
    let (layout, off) = leaf_layout::<T>(cap);
    // SAFETY: offset matches how this allocation was built.
    let data_ptr = unsafe { header.as_ptr().add(off) as *mut T };
    for i in 0..len {
        // SAFETY: slot `i` holds a live, initialized `T`; slots
        // `len..cap` are the (possibly nonempty) uninitialized slack tail
        // (see module docs' "residual slack" section) and are
        // intentionally never touched here.
        unsafe { ptr::drop_in_place(data_ptr.add(i)) };
    }
    // SAFETY: `layout` matches how this allocation was sized.
    unsafe { raw_dealloc(header.as_ptr(), layout) };
}

unsafe fn destroy_internal<T>(header: NonNull<u8>) {
    // SAFETY: header is live (last-owner precondition from `drop_node`).
    let h = unsafe { &*(header.as_ptr() as *const InternalHeader) };
    let n = h.n_children as usize;
    let leaf_children = h.leaf_children != 0;
    let (layout, child_off) = internal_layout::<T>(n);
    // SAFETY: offset matches how this allocation was built.
    let cptr = unsafe { header.as_ptr().add(child_off) as *mut RawNode<T> };
    for i in 0..n {
        // SAFETY: slot `i` holds a live, initialized child `RawNode<T>`;
        // reading it out and recursively dropping it releases exactly the
        // one reference this node's slot held.
        let child = unsafe { ptr::read(cptr.add(i)) };
        drop_node(child, leaf_children);
    }
    // SAFETY: nothing aliases this allocation any more.
    unsafe { raw_dealloc(header.as_ptr(), layout) };
}

// ---------------------------------------------------------------------
// Leaf ops. A "leaf" is used both for genuine trie leaves (always exactly
// `NODE_SIZE` full once placed in the trie — see `mod.rs`'s module docs)
// and for the vector's own `tail` buffer (`1..=NODE_SIZE` live elements,
// with residual capacity slack after a pop — see this file's module
// docs). Both roles share this one physical layout/op set.
// ---------------------------------------------------------------------

pub(crate) mod leaf {
    use super::*;

    #[inline]
    fn header<T>(node: &RawNode<T>) -> &LeafHeader {
        // SAFETY: caller context guarantees `node` is a leaf.
        unsafe { &*(node.0.as_ptr() as *const LeafHeader) }
    }

    #[inline]
    fn header_mut<T>(node: &mut RawNode<T>) -> &mut LeafHeader {
        debug_assert!(is_unique(node), "leaf header_mut requires a unique node");
        // SAFETY: unique (checked above), `&mut` prevents aliasing.
        unsafe { &mut *(node.0.as_ptr() as *mut LeafHeader) }
    }

    #[inline]
    fn data_ptr<T>(node: &RawNode<T>) -> *mut T {
        let (_, off) = leaf_layout::<T>(header(node).cap as usize);
        // SAFETY: offset matches how this allocation was built.
        unsafe { node.0.as_ptr().add(off) as *mut T }
    }

    #[inline]
    pub(crate) fn len<T>(node: &RawNode<T>) -> usize {
        header(node).len as usize
    }

    #[inline]
    pub(crate) fn cap<T>(node: &RawNode<T>) -> usize {
        header(node).cap as usize
    }

    #[inline]
    pub(crate) fn as_slice<T>(node: &RawNode<T>) -> &[T] {
        // SAFETY: `data_ptr(node)..+len` are `len` initialized `T`s.
        unsafe { slice::from_raw_parts(data_ptr(node), header(node).len as usize) }
    }

    #[inline]
    pub(crate) fn get<T>(node: &RawNode<T>, idx: usize) -> &T {
        debug_assert!(idx < len(node));
        // SAFETY: idx in bounds, slot initialized.
        unsafe { &*data_ptr(node).add(idx) }
    }

    /// A fresh, empty (`len == cap == 0`) leaf — used as the tail of a
    /// brand-new empty [`crate::PVector`], mirroring the rope's
    /// `leaf::new_exact("")`.
    pub(crate) fn empty<T>() -> RawNode<T> {
        let (layout, _) = leaf_layout::<T>(0);
        // SAFETY: `layout` is nonzero-sized (header alone is).
        let raw = unsafe { raw_alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // SAFETY: freshly allocated, correctly aligned; no data slots to
        // initialize (cap == 0).
        unsafe { (raw as *mut LeafHeader).write(LeafHeader { rc: AtomicU32::new(1), len: 0, cap: 0 }) };
        RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData)
    }

    /// Fresh leaf holding clones of every element of `items`
    /// (`items.len() <= NODE_SIZE`), exact-fit capacity (copy-path
    /// convention: exact fit, no slack — mirrors the rope's
    /// `leaf::new_exact`).
    pub(crate) fn new_from_slice<T: Clone>(items: &[T]) -> RawNode<T> {
        debug_assert!(items.len() <= NODE_SIZE);
        from_iter_exact(items.len(), items.iter().cloned())
    }

    /// Fresh exact-fit leaf consuming exactly `len` items from `iter`
    /// (take-semantics bulk build, no intermediate `Vec`/clone) — the
    /// primitive `PVector::from_vec`/`FromIterator` chunk directly into.
    pub(crate) fn from_iter_exact<T>(len: usize, iter: impl Iterator<Item = T>) -> RawNode<T> {
        debug_assert!(len <= NODE_SIZE);
        let (layout, off) = leaf_layout::<T>(len);
        // SAFETY: nonzero-sized (header alone is, even if len == 0).
        let raw = unsafe { raw_alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // SAFETY: freshly allocated, correctly aligned.
        unsafe { (raw as *mut LeafHeader).write(LeafHeader { rc: AtomicU32::new(1), len: len as u32, cap: len as u32 }) };
        let dp = unsafe { raw.add(off) as *mut T };
        let mut written = 0usize;
        for (i, item) in iter.enumerate().take(len) {
            // SAFETY: `i < len`, slot not yet written, correctly aligned.
            unsafe { dp.add(i).write(item) };
            written += 1;
        }
        debug_assert_eq!(written, len, "from_iter_exact: iterator yielded fewer than `len` items");
        RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData)
    }

    /// Fresh leaf consuming exactly `len` items from `iter`, allocated at
    /// the hard ceiling `cap == NODE_SIZE` rather than exact-fit
    /// (SPEC-M11-OWNEDPUSH.md Fix A) — identical to [`from_iter_exact`]
    /// except the header's `cap` field (`leaf_layout`'s data offset does
    /// NOT depend on `cap`, only on `T`'s alignment, so this can still
    /// place `len` live elements at the front of a `NODE_SIZE`-capacity
    /// allocation). This is exactly `unique_push`'s doc comment's
    /// grow-to-ceiling rationale — a leaf's role (trie leaf or `PVector`
    /// tail) has a *hard, known* ceiling of `NODE_SIZE` elements — applied
    /// at construction time instead of waiting for the first
    /// post-construction push to discover it. Used ONLY for the fresh
    /// single-element tail a flush leaves behind
    /// (`PVector::push_back_owned_suffix`'s tail-full arm): without this,
    /// that cap-1 tail's very next owned push finds `n == c` and pays
    /// `unique_push`'s slow path (a second alloc + copy + dealloc) on push
    /// #2 of every 32-run. Deliberately NOT used by any copy-path
    /// constructor (`from_slice`/`from_exact_iter`/`copy_with_push`/
    /// `normalize`'s rebuild) — those stay exact-fit per the copy-path
    /// convention (SPEC-M11-OWNEDPUSH.md's re-verification addendum, item
    /// 1): a persistent flush's fresh tail is built by `from_iter_exact`,
    /// and every subsequent persistent push replaces it via
    /// `copy_with_push` (exact-fit realloc each time regardless), so
    /// ceiling-cap there would only pad snapshot memory for no amortized
    /// benefit.
    pub(crate) fn from_iter_ceiling<T>(len: usize, iter: impl Iterator<Item = T>) -> RawNode<T> {
        debug_assert!(len <= NODE_SIZE);
        let cap = NODE_SIZE;
        let (layout, off) = leaf_layout::<T>(cap);
        // SAFETY: nonzero-sized (header alone is).
        let raw = unsafe { raw_alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // SAFETY: freshly allocated, correctly aligned; slots `len..cap`
        // are intentionally left uninitialized (reserved slack, exactly
        // like `unique_push`'s slow-path realloc).
        unsafe { (raw as *mut LeafHeader).write(LeafHeader { rc: AtomicU32::new(1), len: len as u32, cap: cap as u32 }) };
        let dp = unsafe { raw.add(off) as *mut T };
        let mut written = 0usize;
        for (i, item) in iter.enumerate().take(len) {
            // SAFETY: `i < len <= cap`, slot not yet written, correctly aligned.
            unsafe { dp.add(i).write(item) };
            written += 1;
        }
        debug_assert_eq!(written, len, "from_iter_ceiling: iterator yielded fewer than `len` items");
        RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData)
    }

    /// New leaf identical to `node` except the element at `idx` is
    /// replaced by `v` (persistent/copy-path set — mirrors CHAMP's
    /// `copy_with_value`).
    pub(crate) fn copy_with_set<T: Clone>(node: &RawNode<T>, idx: usize, v: T) -> RawNode<T> {
        let n = len(node);
        debug_assert!(idx < n);
        let src = as_slice(node);
        from_iter_exact(n, src.iter().enumerate().map(move |(i, x)| if i == idx { v.clone() } else { x.clone() }))
    }

    /// New leaf with `v` appended (persistent/copy-path push — mirrors
    /// CHAMP's `copy_with_data_inserted`, specialized to append).
    pub(crate) fn copy_with_push<T: Clone>(node: &RawNode<T>, v: T) -> RawNode<T> {
        let n = len(node);
        debug_assert!(n < NODE_SIZE);
        let src = as_slice(node);
        from_iter_exact(n + 1, src.iter().cloned().chain(std::iter::once(v)))
    }

    /// New leaf with its last element removed, plus a clone of that
    /// removed element (persistent/copy-path pop — `node` isn't consumed,
    /// so the popped value can't be moved out, only cloned).
    pub(crate) fn copy_with_pop<T: Clone>(node: &RawNode<T>) -> (RawNode<T>, T) {
        let n = len(node);
        debug_assert!(n >= 1);
        let src = as_slice(node);
        let popped = src[n - 1].clone();
        (from_iter_exact(n - 1, src[..n - 1].iter().cloned()), popped)
    }

    /// In-place value replace at `idx`, handing back the old value
    /// (mirrors CHAMP's `unique_replace_value`).
    pub(crate) fn unique_set<T>(node: &mut RawNode<T>, idx: usize, v: T) -> T {
        debug_assert!(is_unique(node));
        debug_assert!(idx < len(node));
        let slot = unsafe { data_ptr(node).add(idx) };
        // SAFETY: slot holds a live, initialized `T`; `ptr::replace` reads
        // the old value out and writes `v` in, exactly a `mem::replace`
        // through a raw pointer.
        unsafe { ptr::replace(slot, v) }
    }

    /// Append `v` to a **unique** leaf: in-place (no allocation) when
    /// `len < cap` (slack from a prior `unique_pop`, OR from this very
    /// function's own grow policy below), else a realloc to `cap ==
    /// NODE_SIZE` (**not** `len + 1`). Take-semantics: consumes `node`.
    ///
    /// This deliberately deviates from CHAMP's/the rope's "exact-fit,
    /// zero slack on a fresh growth realloc" policy (`src/node.rs`'s
    /// `compute_cap`): this leaf's role (both as a genuine trie leaf and
    /// as `PVector`'s own `tail` buffer — see `node.rs`'s module docs) has
    /// a *hard, known* ceiling of `NODE_SIZE` elements, unlike a CHAMP
    /// bitmap node's `cap` (which can be anywhere up to 32 but has no
    /// single canonical target). Growing to the ceiling immediately,
    /// rather than one slot at a time, isn't "arbitrary slack" in the
    /// sense CHAMP's M4 tuning measured and rejected (a memory-costing
    /// guess at how much future growth *might* happen) — it's sized to
    /// the tail's own maximum possible occupancy, so it can never
    /// over-provision beyond what the caller could ever legitimately use.
    /// Found via SPEC-M10-PVEC.md's own bench gate: an original `n + 1`
    /// exact-fit policy (mirroring CHAMP verbatim) reallocated on *every
    /// single push* while filling a tail from empty (since `cap` never
    /// exceeds `len` for a monotonically-growing, never-shrunk build) —
    /// `owned_build`/`push_back_owned` measured 2.5x SLOWER than imbl at
    /// both 20 and 240k elements, missing the "`>=` imbl's unique-path
    /// build" bar; this fix (one realloc per 32-run instead of up to 32)
    /// closes it — see BENCH-RESULTS.md's M10 section for the measured
    /// before/after.
    pub(crate) fn unique_push<T>(mut node: RawNode<T>, v: T) -> RawNode<T> {
        debug_assert!(is_unique(&node));
        let n = len(&node);
        debug_assert!(n < NODE_SIZE);
        let c = cap(&node);
        if n < c {
            let slot = unsafe { data_ptr(&node).add(n) };
            // SAFETY: slot `n` is within the reserved (but currently
            // uninitialized slack) capacity; writing `v` there is in
            // bounds and doesn't overwrite a live value.
            unsafe { slot.write(v) };
            header_mut(&mut node).len = (n + 1) as u32;
            return node;
        }
        // Slow path: reallocate to the hard ceiling (take semantics — no
        // clone/drop of `T` values, they move via `copy_nonoverlapping`).
        let new_cap = NODE_SIZE;
        let (new_layout, new_off) = leaf_layout::<T>(new_cap);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(new_layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        let old_dp = data_ptr(&node);
        let (old_layout, _) = leaf_layout::<T>(c);
        // SAFETY: old and new allocations never overlap; `n` existing
        // `T`s move via `copy_nonoverlapping` (their bits, no clone/drop),
        // then `v` is written into the newly appended slot.
        unsafe {
            let ndp = raw.add(new_off) as *mut T;
            ptr::copy_nonoverlapping(old_dp, ndp, n);
            ndp.add(n).write(v);
            // NOTE: `len` is `n + 1` (the actual live count), NOT
            // `new_cap` — slots `n + 1..new_cap` are reserved capacity,
            // not live data (this used to be the same value back when
            // `new_cap` was always exactly `n + 1`; now that `new_cap` is
            // the hard `NODE_SIZE` ceiling instead, conflating the two
            // would mark uninitialized slots as live).
            (raw as *mut LeafHeader).write(LeafHeader { rc: AtomicU32::new(1), len: (n + 1) as u32, cap: new_cap as u32 });
            raw_dealloc(node.0.as_ptr(), old_layout);
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData)
    }

    /// Remove and return the last element of a **unique** leaf, in place —
    /// never reallocates, `cap` is left untouched (mirrors CHAMP's
    /// `unique_remove_data`: this is exactly the source of the residual
    /// slack `unique_push`'s fast path later exploits). Take-semantics:
    /// consumes and returns `node`.
    pub(crate) fn unique_pop<T>(mut node: RawNode<T>) -> (RawNode<T>, T) {
        debug_assert!(is_unique(&node));
        let n = len(&node);
        debug_assert!(n >= 1);
        let slot = unsafe { data_ptr(&node).add(n - 1) };
        // SAFETY: slot `n - 1` holds a live, initialized `T`; reading it
        // out (without dropping) is exactly the one live value being
        // logically removed — `len` below is decremented so this slot is
        // no longer considered live, and it's never read again unless a
        // later `unique_push` fast-path overwrites it first.
        let popped = unsafe { ptr::read(slot) };
        header_mut(&mut node).len = (n - 1) as u32;
        (node, popped)
    }
}

// ---------------------------------------------------------------------
// Internal ops. A dense trie's internal nodes only ever grow/shrink at
// their RIGHTMOST slot (`mod.rs`'s `push_tail`/`pop_tail` — pushing always
// extends the rightmost spine, popping always retracts it), so unlike the
// rope's `internal` module (arbitrary-index insert/remove, needed for its
// general B-tree `concat`/`split_at`), this module only needs
// append-at-end / remove-last plus arbitrary-index *replace* (for
// recursing into an already-existing rightmost child, or `set`'s spine
// descent, which can target any index for reads/replacement even though
// structural growth/shrink is always rightmost-only).
// ---------------------------------------------------------------------

pub(crate) mod internal {
    use super::*;

    #[inline]
    fn header<T>(node: &RawNode<T>) -> &InternalHeader {
        // SAFETY: caller context guarantees `node` is internal.
        unsafe { &*(node.0.as_ptr() as *const InternalHeader) }
    }

    #[inline]
    fn child_ptr<T>(node: &RawNode<T>) -> *mut RawNode<T> {
        let (_, off) = internal_layout::<T>(header(node).n_children as usize);
        // SAFETY: offset matches how this allocation was built.
        unsafe { node.0.as_ptr().add(off) as *mut RawNode<T> }
    }

    #[inline]
    pub(crate) fn n_children<T>(node: &RawNode<T>) -> usize {
        header(node).n_children as usize
    }

    #[inline]
    pub(crate) fn leaf_children<T>(node: &RawNode<T>) -> bool {
        header(node).leaf_children != 0
    }

    /// Borrow the child at `idx` (does not affect refcount — caller must
    /// not drop/move out of this reference; a peek, used for read-only
    /// recursion).
    pub(crate) fn child_at<T>(node: &RawNode<T>, idx: usize) -> &RawNode<T> {
        debug_assert!(idx < n_children(node));
        // SAFETY: idx in bounds, slot initialized, borrow tied to `node`'s
        // own borrow so it can't outlive the node or alias a mutation.
        unsafe { &*child_ptr(node).add(idx) }
    }

    /// All children as a borrowed slice — the chunk/equality walkers'
    /// primitive (mirrors the rope's `children_slice`).
    pub(crate) fn children_slice<T>(node: &RawNode<T>) -> &[RawNode<T>] {
        let n = n_children(node);
        // SAFETY: `child_ptr(node)` is valid for `n` initialized `RawNode<T>`.
        unsafe { slice::from_raw_parts(child_ptr(node), n) }
    }

    /// Fresh internal node consuming `children` (ownership of each element
    /// moves in — no `clone_shallow`). Exact-fit, `children.len() in 1..=NODE_SIZE`.
    pub(crate) fn new_consuming<T>(children: Vec<RawNode<T>>, leaf_children_flag: bool) -> RawNode<T> {
        let n = children.len();
        debug_assert!((1..=NODE_SIZE).contains(&n));
        let (layout, child_off) = internal_layout::<T>(n);
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
            let cp = raw.add(child_off) as *mut RawNode<T>;
            for (i, c) in children.into_iter().enumerate() {
                cp.add(i).write(c);
            }
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData)
    }

    /// Take ownership of child `idx`, logically moving it out. Caller MUST
    /// restore the slot (`put_child`) before `node` is read or dropped
    /// again. Mirrors the rope's/CHAMP's `take_node`.
    pub(crate) fn take_child<T>(node: &mut RawNode<T>, idx: usize) -> RawNode<T> {
        debug_assert!(is_unique(node));
        debug_assert!(idx < n_children(node));
        let cp = child_ptr(node);
        // SAFETY: idx in bounds; caller upholds the take/put contract.
        unsafe { ptr::read(cp.add(idx)) }
    }

    /// Complete a `take_child` with a (possibly different) child.
    pub(crate) fn put_child<T>(node: &mut RawNode<T>, idx: usize, child: RawNode<T>) {
        debug_assert!(idx < n_children(node));
        let cp = child_ptr(node);
        // SAFETY: idx in bounds; slot is in the "moved-from" state left by
        // `take_child`, so a raw write does not leak/overwrite a live value.
        unsafe { ptr::write(cp.add(idx), child) };
    }

    /// New node identical to `node` except child `idx` is replaced
    /// (persistent/copy-path — mirrors CHAMP's `copy_with_node_replaced`).
    /// `child` is moved directly into the new node's slot (caller already
    /// owns exactly one refcount unit for it).
    pub(crate) fn copy_with_child_replaced<T>(node: &RawNode<T>, idx: usize, child: RawNode<T>) -> RawNode<T> {
        let n = n_children(node);
        debug_assert!(idx < n);
        let leaf_kids = leaf_children(node);
        let src = children_slice(node);
        let mut child_slot = Some(child);
        let children: Vec<RawNode<T>> = src
            .iter()
            .enumerate()
            .map(|(i, c)| if i == idx { child_slot.take().expect("slot written exactly once") } else { clone_shallow(c) })
            .collect();
        debug_assert!(child_slot.is_none());
        new_consuming(children, leaf_kids)
    }

    /// New node identical to `node` with `child` appended as a new
    /// rightmost slot (persistent/copy-path push overflow — mirrors the
    /// rope's `unique_insert_child` specialized to append, but always
    /// copying since this is the borrowed/shared-path twin).
    pub(crate) fn copy_with_child_appended<T>(node: &RawNode<T>, child: RawNode<T>) -> RawNode<T> {
        let n = n_children(node);
        debug_assert!(n < NODE_SIZE);
        let leaf_kids = leaf_children(node);
        let src = children_slice(node);
        let mut children: Vec<RawNode<T>> = src.iter().map(clone_shallow).collect();
        children.push(child);
        new_consuming(children, leaf_kids)
    }

    /// New node identical to `node` minus its last child (persistent/
    /// copy-path pop — the twin of `copy_with_child_appended`). Returns
    /// `None` when that would leave zero children (caller handles the
    /// "this whole node vanished" case — see `mod.rs`'s `pop_tail`).
    pub(crate) fn copy_without_last_child<T>(node: &RawNode<T>) -> Option<RawNode<T>> {
        let n = n_children(node);
        debug_assert!(n >= 1);
        if n == 1 {
            return None;
        }
        let leaf_kids = leaf_children(node);
        let src = children_slice(node);
        let children: Vec<RawNode<T>> = src[..n - 1].iter().map(clone_shallow).collect();
        Some(new_consuming(children, leaf_kids))
    }

    /// Grow a **unique** node by appending `child` as a new rightmost
    /// slot, in place — always reallocates exact-fit (internal nodes
    /// carry no capacity slack in this design, see module docs), take
    /// semantics (existing children move via `copy_nonoverlapping`, never
    /// cloned/dropped).
    pub(crate) fn unique_append_child<T>(node: RawNode<T>, child: RawNode<T>) -> RawNode<T> {
        debug_assert!(is_unique(&node));
        let n = n_children(&node);
        debug_assert!(n < NODE_SIZE);
        let leaf_kids = leaf_children(&node);
        let (old_layout, _) = internal_layout::<T>(n);
        let old_cp = child_ptr(&node);
        let old_header = node.0;
        let new_n = n + 1;
        let (new_layout, new_child_off) = internal_layout::<T>(new_n);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(new_layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        // SAFETY: old and new allocations never overlap; every existing
        // child's ownership moves unchanged (`copy_nonoverlapping`, no
        // clone/drop), then `child` is written into the new rightmost slot.
        unsafe {
            (raw as *mut InternalHeader).write(InternalHeader {
                rc: AtomicU32::new(1),
                n_children: new_n as u32,
                leaf_children: leaf_kids as u32,
            });
            let new_cp = raw.add(new_child_off) as *mut RawNode<T>;
            ptr::copy_nonoverlapping(old_cp, new_cp, n);
            new_cp.add(n).write(child);
            raw_dealloc(old_header.as_ptr(), old_layout);
        }
        RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData)
    }

    /// Shrink a **unique** node by removing its last child, in place —
    /// take-semantics (the removed child's ownership transfers to the
    /// caller, not dropped here), always reallocates exact-fit. Returns
    /// `(None, popped_child)` when removing the last child would leave
    /// zero children (caller handles the "this whole node vanished" case).
    pub(crate) fn unique_pop_last_child<T>(mut node: RawNode<T>) -> (Option<RawNode<T>>, RawNode<T>) {
        debug_assert!(is_unique(&node));
        let n = n_children(&node);
        debug_assert!(n >= 1);
        let popped = take_child(&mut node, n - 1);
        (finalize_after_taking_last(node, n), popped)
    }

    /// Second half of [`unique_pop_last_child`], factored out so
    /// `mod.rs`'s `extract_rightmost_leaf_owned` can reuse it after its
    /// own recursive `take_child`/(maybe)`put_child` dance: given a unique
    /// node whose slot `n - 1` is already in the "taken" (moved-from)
    /// state (from a preceding [`take_child`] call the caller made
    /// itself), either shrinks the child array by one (`n >= 2`, in place,
    /// exact-fit realloc) or frees the now-empty shell outright (`n ==
    /// 1`) — no data is read from or dropped at the taken slot either way.
    pub(crate) fn finalize_after_taking_last<T>(node: RawNode<T>, n: usize) -> Option<RawNode<T>> {
        debug_assert!(is_unique(&node));
        debug_assert_eq!(n_children(&node), n, "caller's `n` must be the count BEFORE the take_child that vacated the last slot");
        if n == 1 {
            // Nothing left: free the now-empty shell (no children remain
            // to drop — the sole child was already taken by the caller).
            let (layout, _) = internal_layout::<T>(1);
            // SAFETY: `node`'s own allocation, exact layout, no data left
            // (its one child slot is already moved-from).
            unsafe { raw_dealloc(node.0.as_ptr(), layout) };
            return None;
        }
        let leaf_kids = leaf_children(&node);
        let (old_layout, _) = internal_layout::<T>(n);
        let old_cp = child_ptr(&node);
        let old_header = node.0;
        let new_n = n - 1;
        let (new_layout, new_child_off) = internal_layout::<T>(new_n);
        // SAFETY: nonzero-sized.
        let raw = unsafe { raw_alloc(new_layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(new_layout);
        }
        // SAFETY: old and new allocations never overlap; slot `n - 1` was
        // already taken by the caller (moved-from, never read here), every
        // other child's ownership moves unchanged via `copy_nonoverlapping`.
        unsafe {
            (raw as *mut InternalHeader).write(InternalHeader {
                rc: AtomicU32::new(1),
                n_children: new_n as u32,
                leaf_children: leaf_kids as u32,
            });
            let new_cp = raw.add(new_child_off) as *mut RawNode<T>;
            ptr::copy_nonoverlapping(old_cp, new_cp, new_n);
            raw_dealloc(old_header.as_ptr(), old_layout);
        }
        Some(RawNode(unsafe { NonNull::new_unchecked(raw) }, PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_roundtrip() {
        let n = leaf::new_from_slice(&[1, 2, 3]);
        assert_eq!(leaf::as_slice(&n), &[1, 2, 3]);
        assert_eq!(leaf::len(&n), 3);
        assert_eq!(*leaf::get(&n, 1), 2);
        drop_node(n, true);
    }

    #[test]
    fn leaf_empty() {
        let n: RawNode<i32> = leaf::empty();
        assert_eq!(leaf::len(&n), 0);
        assert_eq!(leaf::cap(&n), 0);
        assert!(leaf::as_slice(&n).is_empty());
        drop_node(n, true);
    }

    #[test]
    fn refcount_shares_and_frees() {
        let n = leaf::new_from_slice(&[1]);
        assert!(is_unique(&n));
        let n2 = clone_shallow(&n);
        assert!(!is_unique(&n));
        drop_node(n, true);
        assert!(is_unique(&n2));
        drop_node(n2, true);
    }

    #[test]
    fn leaf_from_iter_ceiling_allocates_at_ceiling_and_reuses_slack() {
        // Fix A (SPEC-M11-OWNEDPUSH.md): a `from_iter_ceiling`-built leaf
        // has cap == NODE_SIZE up front (not exact-fit at `len`), so the
        // very next `unique_push` writes into reserved slack in place
        // (cap stays NODE_SIZE, len just bumps) rather than paying
        // `unique_push`'s realloc-to-ceiling slow path a second time.
        let n: RawNode<i32> = leaf::from_iter_ceiling(1, std::iter::once(42));
        assert_eq!(leaf::len(&n), 1);
        assert_eq!(leaf::cap(&n), NODE_SIZE, "from_iter_ceiling must allocate cap == NODE_SIZE regardless of len");
        assert_eq!(leaf::as_slice(&n), &[42]);
        let n = leaf::unique_push(n, 43);
        assert_eq!(leaf::len(&n), 2);
        assert_eq!(leaf::cap(&n), NODE_SIZE, "push into an already-ceiling-cap leaf must not realloc");
        assert_eq!(leaf::as_slice(&n), &[42, 43]);
        drop_node(n, true);
    }

    #[test]
    fn leaf_unique_push_pop_and_residual_slack() {
        // Grow-to-ceiling policy (see `unique_push`'s doc comment): the
        // very first push past `cap == 0` reallocs straight to
        // `NODE_SIZE`, not `len + 1` — so after 3 pushes from empty, `cap`
        // is already `NODE_SIZE`, and every push/pop after that is a pure
        // in-place slot write/read (no further reallocation at all until
        // the leaf is fully drained and regrown from empty again).
        let n: RawNode<i32> = leaf::empty();
        let n = leaf::unique_push(n, 1);
        assert_eq!(leaf::cap(&n), NODE_SIZE, "first grow past cap==0 should jump straight to the ceiling");
        let n = leaf::unique_push(n, 2);
        let n = leaf::unique_push(n, 3);
        assert_eq!(leaf::as_slice(&n), &[1, 2, 3]);
        assert_eq!(leaf::cap(&n), NODE_SIZE);
        let (n, popped) = leaf::unique_pop(n);
        assert_eq!(popped, 3);
        assert_eq!(leaf::as_slice(&n), &[1, 2]);
        // Residual slack: cap should still be NODE_SIZE (unique_pop never
        // shrinks cap), so the next push should be able to reuse it in
        // place.
        assert_eq!(leaf::cap(&n), NODE_SIZE);
        let n = leaf::unique_push(n, 99);
        assert_eq!(leaf::as_slice(&n), &[1, 2, 99]);
        assert_eq!(leaf::cap(&n), NODE_SIZE, "push into residual slack must not reallocate");
        drop_node(n, true);
    }

    #[test]
    fn leaf_unique_set() {
        let n = leaf::new_from_slice(&["a".to_string(), "b".to_string()]);
        let mut n = n;
        let old = leaf::unique_set(&mut n, 1, "z".to_string());
        assert_eq!(old, "b");
        assert_eq!(leaf::as_slice(&n), &["a".to_string(), "z".to_string()]);
        drop_node(n, true);
    }

    #[test]
    fn leaf_copy_ops_leave_original_untouched() {
        let n = leaf::new_from_slice(&[1, 2, 3]);
        let n2 = leaf::copy_with_set(&n, 1, 99);
        assert_eq!(leaf::as_slice(&n), &[1, 2, 3]);
        assert_eq!(leaf::as_slice(&n2), &[1, 99, 3]);
        let n3 = leaf::copy_with_push(&n, 4);
        assert_eq!(leaf::as_slice(&n3), &[1, 2, 3, 4]);
        let (n4, popped) = leaf::copy_with_pop(&n);
        assert_eq!(popped, 3);
        assert_eq!(leaf::as_slice(&n4), &[1, 2]);
        assert_eq!(leaf::as_slice(&n), &[1, 2, 3]);
        drop_node(n, true);
        drop_node(n2, true);
        drop_node(n3, true);
        drop_node(n4, true);
    }

    #[test]
    fn leaf_drops_owned_elements_exactly_once() {
        use std::rc::Rc;
        let counter = Rc::new(());
        assert_eq!(Rc::strong_count(&counter), 1);
        let items: Vec<Rc<()>> = (0..5).map(|_| counter.clone()).collect();
        assert_eq!(Rc::strong_count(&counter), 6);
        let n = leaf::new_from_slice(&items);
        assert_eq!(Rc::strong_count(&counter), 11);
        drop(items);
        assert_eq!(Rc::strong_count(&counter), 6);
        drop_node(n, true);
        assert_eq!(Rc::strong_count(&counter), 1);
    }

    #[test]
    fn internal_build_and_children() {
        let a = leaf::new_from_slice(&[1, 2]);
        let b = leaf::new_from_slice(&[3, 4]);
        let node = internal::new_consuming(vec![a, b], true);
        assert_eq!(internal::n_children(&node), 2);
        assert!(internal::leaf_children(&node));
        assert_eq!(leaf::as_slice(internal::child_at(&node, 0)), &[1, 2]);
        assert_eq!(leaf::as_slice(internal::child_at(&node, 1)), &[3, 4]);
        drop_node(node, false);
    }

    #[test]
    fn internal_take_put_roundtrip() {
        let a = leaf::new_from_slice(&[1]);
        let b = leaf::new_from_slice(&[2]);
        let mut node = internal::new_consuming(vec![a, b], true);
        let taken = internal::take_child(&mut node, 0);
        assert_eq!(leaf::as_slice(&taken), &[1]);
        let taken = leaf::unique_push(taken, 100);
        internal::put_child(&mut node, 0, taken);
        assert_eq!(leaf::as_slice(internal::child_at(&node, 0)), &[1, 100]);
        drop_node(node, false);
    }

    #[test]
    fn internal_append_and_pop_last_child() {
        let a = leaf::new_from_slice(&[1]);
        let node = internal::new_consuming(vec![a], true);
        let b = leaf::new_from_slice(&[2]);
        let node = internal::unique_append_child(node, b);
        assert_eq!(internal::n_children(&node), 2);
        let (node_opt, popped) = internal::unique_pop_last_child(node);
        assert_eq!(leaf::as_slice(&popped), &[2]);
        drop_node(popped, true);
        let node = node_opt.expect("one child remains");
        assert_eq!(internal::n_children(&node), 1);
        let (node_opt2, popped2) = internal::unique_pop_last_child(node);
        assert_eq!(leaf::as_slice(&popped2), &[1]);
        drop_node(popped2, true);
        assert!(node_opt2.is_none(), "removing the last child empties the node");
    }

    #[test]
    fn internal_copy_ops_leave_original_untouched() {
        let a = leaf::new_from_slice(&[1]);
        let b = leaf::new_from_slice(&[2]);
        let node = internal::new_consuming(vec![a, b], true);

        let replaced = internal::copy_with_child_replaced(&node, 0, leaf::new_from_slice(&[99]));
        assert_eq!(leaf::as_slice(internal::child_at(&replaced, 0)), &[99]);
        assert_eq!(leaf::as_slice(internal::child_at(&node, 0)), &[1]);

        let appended = internal::copy_with_child_appended(&node, leaf::new_from_slice(&[3]));
        assert_eq!(internal::n_children(&appended), 3);
        assert_eq!(internal::n_children(&node), 2);

        let popped = internal::copy_without_last_child(&node).expect("still has one child left");
        assert_eq!(internal::n_children(&popped), 1);
        assert_eq!(internal::n_children(&node), 2);

        drop_node(node, false);
        drop_node(replaced, false);
        drop_node(appended, false);
        drop_node(popped, false);
    }

    #[test]
    fn internal_copy_without_last_child_of_singleton_is_none() {
        let a = leaf::new_from_slice(&[1]);
        let node = internal::new_consuming(vec![a], true);
        assert!(internal::copy_without_last_child(&node).is_none());
        drop_node(node, false);
    }

    #[test]
    fn shared_internal_clone_shallow_children_independent_of_mutation() {
        // clone_shallow'd children share the underlying allocation; a
        // subsequent owned mutation on one handle must not corrupt the
        // other (standard structural-sharing sanity check).
        let a = leaf::new_from_slice(&[1, 2]);
        let node = internal::new_consuming(vec![a], true);
        let shared_child = clone_shallow(internal::child_at(&node, 0));
        assert!(!is_unique(&shared_child));
        // A copy-path mutation (not owned) must leave both intact.
        let modified = leaf::copy_with_set(&shared_child, 0, 999);
        assert_eq!(leaf::as_slice(&modified), &[999, 2]);
        assert_eq!(leaf::as_slice(internal::child_at(&node, 0)), &[1, 2]);
        drop_node(modified, true);
        drop_node(shared_child, true);
        drop_node(node, false);
    }
}
