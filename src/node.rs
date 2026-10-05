//! Unsafe single-allocation CHAMP node layer.
//!
//! All `unsafe` code in the crate lives in this file. A node is one heap
//! allocation shaped as:
//!
//! ```text
//! Header { rc: AtomicU32, datamap: u32, nodemap: u32, xlen: u32 }
//! data:  cap(xlen) x (K, V)             // bitmap node data *slots* (>= popcnt(datamap) live)
//! nodes: popcnt(nodemap) x NodePtr<K,V> // bitmap node children, always exact-fit
//! ```
//!
//! A **collision node** reuses the same physical layout but is flagged by
//! `datamap == 0 && nodemap == 0` (a bitmap node can never have both maps
//! empty, since the empty map has no root at all): its `xlen` field gives the
//! entry count (>= 2) and it has zero children. This dual use means a single
//! `layout()`/alloc/dealloc path serves both node kinds.
//!
//! ## Capacity slack (M4 tuning)
//!
//! For a **bitmap** node, `xlen` is repurposed as the node's *data-slot
//! capacity* (`cap`), a value in `count..=32` where `count == popcnt(datamap)`
//! is the number of *live* entries. Slots `count..cap` are reserved but
//! uninitialized: `destroy()` must drop only the first `count` of them, never
//! the tail. For a **collision** node there is no slack: `xlen` continues to
//! mean exactly `count` (`cap == count` always), since collision nodes have
//! no separate cap field to spend — this falls out for free below because
//! [`NodePtr::cap`] always just reads the raw `xlen` field, and for collision
//! nodes that field already *is* the exact count.
//!
//! **Growth policy (final, post-measurement): exact-fit.** [`compute_cap`]
//! — the function that sizes every *growth* realloc (`unique_insert_data`'s
//! slow path, `unique_data_to_node`, `unique_node_to_data`) — returns
//! `count` verbatim, i.e. `cap == count` immediately after any of those
//! reallocs, same as copy-path construction. An earlier version of this
//! milestone rounded growth up to `min(32, count.next_multiple_of(4))`
//! instead; see [`compute_cap`]'s doc comment and BENCH-RESULTS.md for why
//! that was reverted (a real memory-regression cost that outweighed a
//! sub-target speed gain). What's *not* reverted, and still does real work:
//! `unique_remove_data` never shrinks `cap` when it removes an entry (see
//! its own docs), so any node that has ever grown and then shrunk carries
//! genuine residual slack (`cap > count`) even under the exact-fit growth
//! policy — and `unique_insert_data`'s in-place fast path (`ptr::copy`
//! memmove, zero allocation) still fires whenever it finds `count < cap`,
//! regardless of whether that headroom came from slack-on-growth (it no
//! longer does) or from a prior removal (it still can). This residual-cap
//! mechanism is where `remove_all`'s measured 1.3-2.0x win over the pre-M4
//! baseline comes from, and it's the reason the layout/offset/dealloc math
//! below stayed fully cap-aware rather than reverting to the pre-M4
//! `cap == count`-always model.
//!
//! Refcounting follows the standard `Arc` protocol: `clone_shallow` does a
//! `Relaxed` fetch-add, `drop_node` does a `Release` fetch-sub and, on the
//! last reference, an `Acquire` fence before recursively dropping children
//! and deallocating. `NodePtr` deliberately implements neither `Clone` nor
//! `Drop`: every refcount change is an explicit, auditable call
//! (`clone_shallow` / `drop_node`), which matters because the "unique path"
//! mutation machinery in `map.rs` moves `NodePtr` values around with take/put
//! semantics that a silent `Drop` would corrupt (double frees) or a silent
//! `Clone` would make expensive/incorrect (accidental sharing).

use std::alloc::{self, Layout};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "pool-alloc")]
use crate::pool;

/// Single choke-point for every node allocation in this file (SPEC-M6-ALLOC.md
/// Step 1's "single choke-point" requirement: grep every `alloc`/`dealloc`
/// call in this module — they all go through `raw_alloc`/`raw_dealloc`).
/// Routes through the thread-local node pool when `pool-alloc` is enabled
/// (the default), else falls straight through to the system allocator.
///
/// # Safety
/// Same contract as `std::alloc::alloc`: `layout` must be nonzero-sized.
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

/// See [`raw_alloc`]. `ptr`/`layout` must be the exact pair a prior
/// `raw_alloc` call returned/was given.
///
/// # Safety
/// Same contract as `std::alloc::dealloc`.
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

/// Node header. Fixed size/layout regardless of node kind; see module docs
/// for the discrimination rule between bitmap nodes and collision nodes.
#[repr(C)]
struct Header {
    rc: AtomicU32,
    datamap: u32,
    nodemap: u32,
    /// Collision-node entry count; `0` for bitmap nodes (which derive their
    /// counts from `datamap`/`nodemap` popcounts instead).
    xlen: u32,
}

/// A thin, single-word handle to a heap-allocated CHAMP node.
///
/// Not `Clone`, not `Copy`, not `Drop` — see module docs. `K`/`V` only
/// appear via `PhantomData` for variance/drop-check purposes; the actual
/// dropping of `K`/`V` values happens explicitly in [`NodePtr::destroy`].
#[repr(transparent)]
pub(crate) struct NodePtr<K, V>(NonNull<Header>, PhantomData<(K, V)>);

// SAFETY: a `NodePtr` is logically an `Arc`-like shared owner of `(K, V)`
// data and child `NodePtr`s reachable through atomically refcounted heap
// allocations. It's sound to send/share across threads exactly when the
// data it (transitively) owns is `Send`/`Sync`, matching `Arc<T>`'s rules.
unsafe impl<K: Send + Sync, V: Send + Sync> Send for NodePtr<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for NodePtr<K, V> {}

/// `(h ^ (h >> 32)) as u32` folding of the generic hasher's 64-bit output
/// into the 32-bit hash space CHAMP operates on.
#[inline]
pub(crate) fn hash32<K: Hash + ?Sized, S: BuildHasher>(s: &S, k: &K) -> u32 {
    let h = s.hash_one(k);
    (h ^ (h >> 32)) as u32
}

/// 5-bit chunk of `hash` at tree `depth` (0..=6; depth 6 only ever yields
/// values 0..=3 since it consumes the last 2 of the 32 hash bits).
#[inline]
pub(crate) fn chunk(hash: u32, depth: u32) -> u32 {
    debug_assert!(depth <= 6, "bitmap nodes exist only at depth 0..=6");
    (hash >> (5 * depth)) & 0x1f
}

/// Bitmap bit corresponding to `chunk(hash, depth)`.
#[inline]
pub(crate) fn bit_at(hash: u32, depth: u32) -> u32 {
    1u32 << chunk(hash, depth)
}

/// Centralized layout math: header, then `cap` x `(K, V)` *reserved* data
/// slots (only the first `count <= cap` are ever live), then `n_nodes` x
/// `NodePtr<K, V>` (always exact-fit), padded to alignment. Returns the
/// overall layout plus the byte offsets of the data and node regions.
#[inline]
fn node_layout<K, V>(cap: usize, n_nodes: usize) -> (Layout, usize, usize) {
    let header = Layout::new::<Header>();
    let data = Layout::array::<(K, V)>(cap).expect("champ: layout overflow");
    let (l1, data_off) = header.extend(data).expect("champ: layout overflow");
    let nodes = Layout::array::<NodePtr<K, V>>(n_nodes).expect("champ: layout overflow");
    let (l2, node_off) = l1.extend(nodes).expect("champ: layout overflow");
    (l2.pad_to_align(), data_off, node_off)
}

#[inline]
unsafe fn raw_data_ptr<K, V>(header: NonNull<Header>, cap: usize, n_nodes: usize) -> *mut (K, V) {
    let (_, off, _) = node_layout::<K, V>(cap, n_nodes);
    unsafe { (header.as_ptr() as *mut u8).add(off) as *mut (K, V) }
}

#[inline]
unsafe fn raw_node_ptr<K, V>(header: NonNull<Header>, cap: usize, n_nodes: usize) -> *mut NodePtr<K, V> {
    let (_, _, off) = node_layout::<K, V>(cap, n_nodes);
    unsafe { (header.as_ptr() as *mut u8).add(off) as *mut NodePtr<K, V> }
}

/// The data-slot capacity granted to a freshly (re)allocated unique-path
/// node holding `count` live entries.
///
/// **Final M4 policy: exact-fit (`cap == count`), i.e. GROWTH reallocs carry
/// no slack.** An earlier version of this milestone rounded up to
/// `min(32, count.next_multiple_of(4))`, giving up to 3 free in-place
/// inserts after every growth realloc. Measured outcome (see
/// BENCH-RESULTS.md's "After M4 tuning" section): that slack bought a real
/// but modest build-speed win (1.1-1.4x, short of the ≥2x target — most
/// inserts in a sequential build touch a *freshly created* node, which
/// slack doesn't help, not a *repeatedly touched* one, which it does) at
/// the cost of a memory regression against champ's own pre-M4
/// baseline that exceeded the ≤10% budget almost everywhere and hit +28.6%
/// on the "100k independent 7-key maps" scenario — the one deliberately
/// chosen to mirror the host's actual motivating memory pain point. Given that
/// trade-off, growth slack was reverted; every other piece of M4-tuning
/// infrastructure this function's callers rely on (cap-aware layout math,
/// the in-place insert fast path, the never-realloc `unique_remove_data`)
/// is kept, because `unique_remove_data` leaving `cap` untouched means a
/// node that has ever grown-then-shrunk still carries *residual* capacity
/// slack post-revert, and the in-place insert path still exploits that —
/// see `unique_remove_data`'s docs and `remove_all`'s benchmark win, which
/// this revert does not undo. Collision nodes never call this — they stay
/// exact-fit regardless (see the module docs' "Capacity slack" section).
#[inline]
fn compute_cap(count: usize) -> usize {
    debug_assert!(count <= 32, "a bitmap node can never hold more than 32 data entries");
    count
}

/// Allocate (but do not populate data/node slots) a node header for the
/// given shape. Caller must immediately fill exactly the live data slots
/// (`0..count`, `count <= cap`) and `n_nodes` node slots via
/// `raw_data_ptr`/`raw_node_ptr` before the returned pointer is used as a
/// well-formed `NodePtr`. `cap` is written verbatim into the header's `xlen`
/// field — for a fresh copy-path node the caller passes `cap == count`
/// (exact-fit); for a unique-path realloc-with-slack node the caller passes
/// `cap == compute_cap(count)`; for a collision node `cap` is simply the
/// (always exact) entry count, since collision nodes carry no separate cap.
unsafe fn alloc_header<K, V>(datamap: u32, nodemap: u32, cap: usize, n_nodes: usize) -> NonNull<Header> {
    // The 32-slot ceiling only applies to bitmap nodes (a 5-bit chunk has
    // only 32 possible bits); a collision node's `cap` is its entry count,
    // which has no such bound (arbitrarily many keys can share one 32-bit
    // hash, e.g. under a deliberately colliding hasher).
    let is_collision = datamap == 0 && nodemap == 0;
    debug_assert!(
        is_collision || cap <= 32,
        "bitmap node cap must fit in a 5-bit chunk's worth of data slots"
    );
    let (layout, _, _) = node_layout::<K, V>(cap, n_nodes);
    // SAFETY: `layout` is nonzero-sized (Header alone is nonzero) and well-formed.
    let raw = unsafe { raw_alloc(layout) };
    if raw.is_null() {
        alloc::handle_alloc_error(layout);
    }
    let hp = raw as *mut Header;
    // SAFETY: `hp` is freshly allocated and properly aligned for `Header`.
    unsafe {
        hp.write(Header {
            rc: AtomicU32::new(1),
            datamap,
            nodemap,
            xlen: cap as u32,
        });
    }
    // SAFETY: `alloc::alloc` returned non-null.
    unsafe { NonNull::new_unchecked(hp) }
}

impl<K, V> NodePtr<K, V> {
    #[inline]
    fn header(&self) -> &Header {
        // SAFETY: `self.0` always points at a live, fully-initialized `Header`
        // for as long as this `NodePtr` exists (refcount protocol).
        unsafe { self.0.as_ref() }
    }

    #[inline]
    pub(crate) fn datamap(&self) -> u32 {
        self.header().datamap
    }

    #[inline]
    pub(crate) fn nodemap(&self) -> u32 {
        self.header().nodemap
    }

    #[inline]
    pub(crate) fn xlen(&self) -> u32 {
        self.header().xlen
    }

    /// Data-slot capacity: for a bitmap node, the number of reserved
    /// `(K, V)` slots (`>= n_data()`, see the module docs' "Capacity slack"
    /// section); for a collision node, always exactly `n_data()` (no slack).
    /// Both cases just read the raw `xlen` field — the discrimination is
    /// baked into how each node kind writes that field at construction
    /// time, not into this accessor.
    #[inline]
    pub(crate) fn cap(&self) -> usize {
        self.header().xlen as usize
    }

    /// Discrimination rule: a node is a collision node iff both bitmaps are
    /// empty (a bitmap node — even the smallest, single-entry one — always
    /// has at least one of `datamap`/`nodemap` nonzero, since the empty map
    /// has no root at all).
    #[inline]
    pub(crate) fn is_collision(&self) -> bool {
        self.datamap() == 0 && self.nodemap() == 0
    }

    #[inline]
    pub(crate) fn n_data(&self) -> usize {
        if self.is_collision() {
            self.xlen() as usize
        } else {
            self.datamap().count_ones() as usize
        }
    }

    #[inline]
    pub(crate) fn n_nodes(&self) -> usize {
        if self.is_collision() {
            0
        } else {
            self.nodemap().count_ones() as usize
        }
    }

    /// Index of the data slot for `bit` (must be set in `datamap`).
    #[inline]
    pub(crate) fn data_index(&self, bit: u32) -> usize {
        debug_assert_ne!(self.datamap() & bit, 0, "bit not present in datamap");
        (self.datamap() & (bit - 1)).count_ones() as usize
    }

    /// Index of the child slot for `bit` (must be set in `nodemap`).
    #[inline]
    pub(crate) fn node_index(&self, bit: u32) -> usize {
        debug_assert_ne!(self.nodemap() & bit, 0, "bit not present in nodemap");
        (self.nodemap() & (bit - 1)).count_ones() as usize
    }

    #[inline]
    fn data_ptr(&self) -> *mut (K, V) {
        // SAFETY: offsets computed from this node's own reserved capacity
        // (not just its live count) and node count — matches how the
        // allocation was actually sized.
        unsafe { raw_data_ptr::<K, V>(self.0, self.cap(), self.n_nodes()) }
    }

    #[inline]
    fn node_ptr(&self) -> *mut NodePtr<K, V> {
        // SAFETY: same as `data_ptr` — the node region starts after the
        // *reserved* (cap) data slots, not just the live ones.
        unsafe { raw_node_ptr::<K, V>(self.0, self.cap(), self.n_nodes()) }
    }

    #[inline]
    pub(crate) fn data_slice(&self) -> &[(K, V)] {
        // SAFETY: `data_ptr()` is valid for `n_data()` initialized `(K, V)`.
        unsafe { slice::from_raw_parts(self.data_ptr(), self.n_data()) }
    }

    #[inline]
    pub(crate) fn node_slice(&self) -> &[NodePtr<K, V>] {
        // SAFETY: `node_ptr()` is valid for `n_nodes()` initialized `NodePtr`.
        unsafe { slice::from_raw_parts(self.node_ptr(), self.n_nodes()) }
    }

    /// Mutable data access. Only sound to actually mutate through when
    /// `is_unique()` — callers (map.rs) only call this on the unique-path.
    #[inline]
    pub(crate) fn data_slice_mut(&mut self) -> &mut [(K, V)] {
        debug_assert!(self.is_unique(), "mutable data access requires a unique node");
        let n = self.n_data();
        let p = self.data_ptr();
        // SAFETY: unique node, `p` valid for `n` initialized `(K, V)`, and we
        // hold `&mut self` so no other live reference can alias this range.
        unsafe { slice::from_raw_parts_mut(p, n) }
    }

    /// Bump the refcount and hand back an independent owning handle to the
    /// same allocation (standard `Arc` clone protocol: `Relaxed` fetch-add
    /// suffices since the count itself, not the guarded data, is what needs
    /// to stay consistent).
    ///
    /// Matches `Arc`'s overflow policy: an unchecked `fetch_add` could wrap
    /// a pathologically over-cloned `u32` refcount back around to a small
    /// value, after which some clones would believe themselves the sole
    /// owner while others are still alive — a use-after-free/double-free.
    /// Since wrapping this many refs is never a legitimate program state
    /// (only a leak/bug can get here), abort the process outright rather
    /// than try to recover, exactly as `Arc::clone` does at `isize::MAX`.
    #[inline]
    pub(crate) fn clone_shallow(&self) -> NodePtr<K, V> {
        let prev = self.header().rc.fetch_add(1, Ordering::Relaxed);
        if prev > i32::MAX as u32 {
            std::process::abort();
        }
        NodePtr(self.0, PhantomData)
    }

    /// True iff this handle is the only outstanding reference to the node.
    /// Uses `Acquire` so that a `true` result synchronizes-with the `Release`
    /// decrement of every other handle that has already gone away, making it
    /// safe to treat the node as exclusively ours (Arc::get_mut protocol).
    #[inline]
    pub(crate) fn is_unique(&self) -> bool {
        self.header().rc.load(Ordering::Acquire) == 1
    }

    #[inline]
    pub(crate) fn ptr_eq(a: &NodePtr<K, V>, b: &NodePtr<K, V>) -> bool {
        a.0 == b.0
    }

    /// Mutable access to this node's header fields (`datamap`/`nodemap`),
    /// for the unique in-place mutation paths that change the live-entry
    /// bitmaps without touching (or reallocating) the underlying storage.
    /// Only sound when `is_unique()` — no other handle can be observing
    /// this header concurrently.
    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        debug_assert!(self.is_unique(), "header_mut requires a unique node");
        // SAFETY: unique node (refcount == 1, `Acquire`-checked by
        // `is_unique`), and we hold `&mut self`, so no other reference can
        // alias this header's fields while we mutate them.
        unsafe { self.0.as_mut() }
    }

    /// Take ownership of the child at `idx`, logically moving it out of this
    /// node's child array. The caller MUST overwrite the slot with
    /// [`put_node`] before this node is read or dropped again — until then
    /// the slot holds "moved-from" bits that must never be read as a
    /// `NodePtr`. Only used on the unique-mutation path.
    pub(crate) fn take_node(&mut self, idx: usize) -> NodePtr<K, V> {
        debug_assert!(self.is_unique(), "take_node requires a unique node");
        debug_assert!(idx < self.n_nodes());
        let p = self.node_ptr();
        // SAFETY: idx in bounds; caller upholds the take/put contract above.
        unsafe { ptr::read(p.add(idx)) }
    }

    /// Write a (new) child into slot `idx`, completing a preceding
    /// [`take_node`]. Does not drop any previous value at the slot — that's
    /// the point: the slot is expected to be in the "moved-from" state left
    /// by `take_node`.
    pub(crate) fn put_node(&mut self, idx: usize, val: NodePtr<K, V>) {
        debug_assert!(idx < self.n_nodes());
        let p = self.node_ptr();
        // SAFETY: idx in bounds; slot is either freshly allocated
        // (uninitialized) or freshly vacated by `take_node`, so a raw write
        // does not leak/overwrite a live value.
        unsafe { ptr::write(p.add(idx), val) }
    }

    /// Release one reference. On the last reference, recursively drops every
    /// `(K, V)` entry and every child (via recursive `drop_node`), then
    /// deallocates this node's own allocation.
    pub(crate) fn drop_node(self) {
        let prev = self.header().rc.fetch_sub(1, Ordering::Release);
        debug_assert!(prev >= 1, "refcount underflow");
        if prev == 1 {
            std::sync::atomic::fence(Ordering::Acquire);
            // SAFETY: refcount hit zero under us; we are the last owner and
            // no other handle can observe or race this allocation from here.
            unsafe { destroy::<K, V>(self.0) }
        }
    }
}

/// Drop every `(K, V)` and child, then free the allocation. Does not touch
/// the refcount (caller has already established this is the last owner).
unsafe fn destroy<K, V>(header: NonNull<Header>) {
    // SAFETY: header is live (last-owner precondition from the caller).
    let h = unsafe { header.as_ref() };
    let is_collision = h.datamap == 0 && h.nodemap == 0;
    let n_data = if is_collision {
        h.xlen as usize
    } else {
        h.datamap.count_ones() as usize
    };
    let n_nodes = if is_collision { 0 } else { h.nodemap.count_ones() as usize };
    // `cap`: for a collision node this is the same as `n_data` (no slack);
    // for a bitmap node it's the *reserved* slot count, which may exceed
    // `n_data` — that's the whole point of capacity slack. The dealloc
    // layout below MUST use `cap`, not `n_data`, to match how this
    // allocation was actually sized; the drop loop below MUST use `n_data`,
    // not `cap`, since slots `n_data..cap` are uninitialized and must never
    // be read/dropped.
    let cap = h.xlen as usize;
    debug_assert!(n_data <= cap, "cap invariant violated at destroy time");
    debug_assert!(is_collision || cap <= 32, "bitmap node cap invariant violated at destroy time");

    let (layout, _, _) = node_layout::<K, V>(cap, n_nodes);
    // SAFETY: offsets match how this allocation was built.
    let data_ptr = unsafe { raw_data_ptr::<K, V>(header, cap, n_nodes) };
    let node_ptr = unsafe { raw_node_ptr::<K, V>(header, cap, n_nodes) };

    for i in 0..n_data {
        // SAFETY: slot `i` holds a live, initialized `(K, V)`; slots
        // `n_data..cap` are the (possibly nonempty) uninitialized slack
        // tail and are intentionally never touched here.
        unsafe { ptr::drop_in_place(data_ptr.add(i)) };
    }
    for i in 0..n_nodes {
        // SAFETY: slot `i` holds a live, initialized child `NodePtr`; reading
        // it out and recursively dropping it is exactly one release of that
        // child's refcount, matching the one implicit "owned by this node's
        // slot" reference being destroyed here.
        let child: NodePtr<K, V> = unsafe { ptr::read(node_ptr.add(i)) };
        child.drop_node();
    }
    // SAFETY: `header` was allocated with exactly `layout` and nothing else
    // aliases it any more (last owner, all contents already retired above).
    unsafe { raw_dealloc(header.as_ptr() as *mut u8, layout) };
}

// ---------------------------------------------------------------------
// Constructors: build a brand-new node from scratch (used by split logic
// in map.rs when two keys diverge, or converge into a collision node).
// ---------------------------------------------------------------------

impl<K, V> NodePtr<K, V> {
    /// A bitmap node with exactly one data entry.
    pub(crate) fn new_leaf(bit: u32, k: K, v: V) -> Self {
        // SAFETY: n_data=1, n_nodes=0; we populate the single data slot below.
        unsafe {
            let header = alloc_header::<K, V>(bit, 0, 1, 0);
            raw_data_ptr::<K, V>(header, 1, 0).write((k, v));
            NodePtr(header, PhantomData)
        }
    }

    /// A bitmap node with exactly two data entries (`bit_a != bit_b`),
    /// written in ascending-bit order as the canonical layout requires.
    pub(crate) fn new_leaf2(bit_a: u32, ka: K, va: V, bit_b: u32, kb: K, vb: V) -> Self {
        debug_assert_ne!(bit_a, bit_b);
        let datamap = bit_a | bit_b;
        unsafe {
            let header = alloc_header::<K, V>(datamap, 0, 2, 0);
            let dp = raw_data_ptr::<K, V>(header, 2, 0);
            if bit_a < bit_b {
                dp.write((ka, va));
                dp.add(1).write((kb, vb));
            } else {
                dp.write((kb, vb));
                dp.add(1).write((ka, va));
            }
            NodePtr(header, PhantomData)
        }
    }

    /// A bitmap node with exactly one child (a "chain" link used while
    /// resolving a shared chunk prefix during a split).
    pub(crate) fn new_chain(bit: u32, child: Self) -> Self {
        unsafe {
            let header = alloc_header::<K, V>(0, bit, 0, 1);
            raw_node_ptr::<K, V>(header, 0, 1).write(child);
            NodePtr(header, PhantomData)
        }
    }

    /// A bitmap node with exactly one data entry and one child.
    pub(crate) fn new_leaf_and_child(data_bit: u32, k: K, v: V, node_bit: u32, child: Self) -> Self {
        debug_assert_eq!(data_bit & node_bit, 0);
        unsafe {
            let header = alloc_header::<K, V>(data_bit, node_bit, 1, 1);
            raw_data_ptr::<K, V>(header, 1, 1).write((k, v));
            raw_node_ptr::<K, V>(header, 1, 1).write(child);
            NodePtr(header, PhantomData)
        }
    }

    /// A collision node with exactly two entries (both must hash equal).
    pub(crate) fn new_collision2(ka: K, va: V, kb: K, vb: V) -> Self {
        unsafe {
            let header = alloc_header::<K, V>(0, 0, 2, 0);
            let dp = raw_data_ptr::<K, V>(header, 2, 0);
            dp.write((ka, va));
            dp.add(1).write((kb, vb));
            NodePtr(header, PhantomData)
        }
    }
}

// ---------------------------------------------------------------------
// Copy-path ops: build a new node by cloning `K`/`V` data entries and
// `clone_shallow`-ing child pointers not on the mutation path. Used
// whenever the current node is not exclusively owned by the caller.
// ---------------------------------------------------------------------

impl<K: Clone, V: Clone> NodePtr<K, V> {
    /// New node identical to `self` except the value at data slot `idx` is
    /// replaced (key at that slot is cloned, preserving Clojure's
    /// keep-existing-key-on-replace rule).
    pub(crate) fn copy_with_value(&self, idx: usize, v: V) -> Self {
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        debug_assert!(idx < n_data);
        unsafe {
            let header = alloc_header::<K, V>(self.datamap(), self.nodemap(), n_data, n_nodes);
            let dp = raw_data_ptr::<K, V>(header, n_data, n_nodes);
            for (i, (k, ev)) in self.data_slice().iter().enumerate() {
                if i == idx {
                    dp.add(i).write((k.clone(), v.clone()));
                } else {
                    dp.add(i).write((k.clone(), ev.clone()));
                }
            }
            let np = raw_node_ptr::<K, V>(header, n_data, n_nodes);
            for (i, c) in self.node_slice().iter().enumerate() {
                np.add(i).write(c.clone_shallow());
            }
            NodePtr(header, PhantomData)
        }
    }

    /// New node with a fresh data entry inserted at `bit` (must not already
    /// be set in `datamap`).
    pub(crate) fn copy_with_data_inserted(&self, bit: u32, k: K, v: V) -> Self {
        debug_assert_eq!(self.datamap() & bit, 0);
        let idx = (self.datamap() & (bit - 1)).count_ones() as usize;
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        unsafe {
            let header = alloc_header::<K, V>(self.datamap() | bit, self.nodemap(), n_data + 1, n_nodes);
            let dp = raw_data_ptr::<K, V>(header, n_data + 1, n_nodes);
            let src = self.data_slice();
            for (i, item) in src.iter().enumerate().take(idx) {
                dp.add(i).write(item.clone());
            }
            dp.add(idx).write((k, v));
            for (i, item) in src.iter().enumerate().skip(idx) {
                dp.add(i + 1).write(item.clone());
            }
            let np = raw_node_ptr::<K, V>(header, n_data + 1, n_nodes);
            for (i, c) in self.node_slice().iter().enumerate() {
                np.add(i).write(c.clone_shallow());
            }
            NodePtr(header, PhantomData)
        }
    }

    /// New node with the data entry at `bit` removed. Caller ensures the
    /// remaining shape is a legitimate node on its own: not empty (that's
    /// `Gone`, root-only), and not exactly one data entry with no children
    /// (that must canonically inline into the parent instead — see
    /// map.rs). A remaining shape of zero data entries and one child is
    /// fine (a single-child chain link, same as fresh insertion would
    /// produce for keys sharing a long chunk prefix).
    pub(crate) fn copy_with_data_removed(&self, bit: u32) -> Self {
        let idx = self.data_index(bit);
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        debug_assert!(
            !(n_data - 1 == 1 && n_nodes == 0),
            "should have canonically inlined instead"
        );
        debug_assert!(n_data - 1 + n_nodes >= 1, "should have returned Gone instead");
        unsafe {
            let header = alloc_header::<K, V>(self.datamap() & !bit, self.nodemap(), n_data - 1, n_nodes);
            let dp = raw_data_ptr::<K, V>(header, n_data - 1, n_nodes);
            let src = self.data_slice();
            for (i, item) in src.iter().enumerate().take(idx) {
                dp.add(i).write(item.clone());
            }
            for (i, item) in src.iter().enumerate().skip(idx + 1) {
                dp.add(i - 1).write(item.clone());
            }
            let np = raw_node_ptr::<K, V>(header, n_data - 1, n_nodes);
            for (i, c) in self.node_slice().iter().enumerate() {
                np.add(i).write(c.clone_shallow());
            }
            NodePtr(header, PhantomData)
        }
    }

    /// New node identical to `self` except child slot `idx` is replaced.
    /// `child` is moved directly into the new node's slot (the caller
    /// already owns exactly one refcount unit for it; no `clone_shallow`
    /// needed or wanted here).
    pub(crate) fn copy_with_node_replaced(&self, idx: usize, child: Self) -> Self {
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        debug_assert!(idx < n_nodes);
        unsafe {
            let header = alloc_header::<K, V>(self.datamap(), self.nodemap(), n_data, n_nodes);
            let dp = raw_data_ptr::<K, V>(header, n_data, n_nodes);
            for (i, kv) in self.data_slice().iter().enumerate() {
                dp.add(i).write(kv.clone());
            }
            let np = raw_node_ptr::<K, V>(header, n_data, n_nodes);
            let mut child_slot = Some(child);
            for (i, c) in self.node_slice().iter().enumerate() {
                if i == idx {
                    np.add(i).write(child_slot.take().expect("slot written once"));
                } else {
                    np.add(i).write(c.clone_shallow());
                }
            }
            debug_assert!(child_slot.is_none());
            NodePtr(header, PhantomData)
        }
    }

    /// Migrate data slot `bit` into a child slot at the same bit (canonical
    /// dissoc/assoc split: a chunk collision converts a data slot into a
    /// subtree). One allocation.
    pub(crate) fn copy_with_data_to_node(&self, bit: u32, child: Self) -> Self {
        debug_assert_ne!(self.datamap() & bit, 0);
        debug_assert_eq!(self.nodemap() & bit, 0);
        let data_idx = self.data_index(bit);
        let node_idx = self.node_index_for_insert(bit);
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        unsafe {
            let header = alloc_header::<K, V>(self.datamap() & !bit, self.nodemap() | bit, n_data - 1, n_nodes + 1);
            let dp = raw_data_ptr::<K, V>(header, n_data - 1, n_nodes + 1);
            let src = self.data_slice();
            for (i, item) in src.iter().enumerate().take(data_idx) {
                dp.add(i).write(item.clone());
            }
            for (i, item) in src.iter().enumerate().skip(data_idx + 1) {
                dp.add(i - 1).write(item.clone());
            }
            let np = raw_node_ptr::<K, V>(header, n_data - 1, n_nodes + 1);
            let src_n = self.node_slice();
            for (i, c) in src_n.iter().enumerate().take(node_idx) {
                np.add(i).write(c.clone_shallow());
            }
            np.add(node_idx).write(child);
            for (i, c) in src_n.iter().enumerate().skip(node_idx) {
                np.add(i + 1).write(c.clone_shallow());
            }
            NodePtr(header, PhantomData)
        }
    }

    /// Migrate child slot `bit` into a data slot at the same bit (canonical
    /// dissoc inlining: a child collapsed to one entry, pulled into the
    /// parent). One allocation.
    pub(crate) fn copy_with_node_to_data(&self, bit: u32, k: K, v: V) -> Self {
        debug_assert_ne!(self.nodemap() & bit, 0);
        debug_assert_eq!(self.datamap() & bit, 0);
        let node_idx = self.node_index(bit);
        let data_idx = self.data_index_for_insert(bit);
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        unsafe {
            let header = alloc_header::<K, V>(self.datamap() | bit, self.nodemap() & !bit, n_data + 1, n_nodes - 1);
            let dp = raw_data_ptr::<K, V>(header, n_data + 1, n_nodes - 1);
            let src = self.data_slice();
            for (i, item) in src.iter().enumerate().take(data_idx) {
                dp.add(i).write(item.clone());
            }
            dp.add(data_idx).write((k, v));
            for (i, item) in src.iter().enumerate().skip(data_idx) {
                dp.add(i + 1).write(item.clone());
            }
            let np = raw_node_ptr::<K, V>(header, n_data + 1, n_nodes - 1);
            let src_n = self.node_slice();
            for (i, c) in src_n.iter().enumerate().take(node_idx) {
                np.add(i).write(c.clone_shallow());
            }
            for (i, c) in src_n.iter().enumerate().skip(node_idx + 1) {
                np.add(i - 1).write(c.clone_shallow());
            }
            NodePtr(header, PhantomData)
        }
    }

    /// New collision node with one more entry appended.
    pub(crate) fn copy_with_collision_inserted(&self, k: K, v: V) -> Self {
        debug_assert!(self.is_collision());
        let n = self.n_data();
        unsafe {
            let header = alloc_header::<K, V>(0, 0, n + 1, 0);
            let dp = raw_data_ptr::<K, V>(header, n + 1, 0);
            for (i, kv) in self.data_slice().iter().enumerate() {
                dp.add(i).write(kv.clone());
            }
            dp.add(n).write((k, v));
            NodePtr(header, PhantomData)
        }
    }

    /// New collision node with entry `idx` removed. Caller ensures
    /// `n_data() - 1 >= 2` (otherwise use [`Self::extract_sole_survivor`]
    /// for canonical inlining instead).
    pub(crate) fn copy_with_collision_removed(&self, idx: usize) -> Self {
        debug_assert!(self.is_collision());
        let n = self.n_data();
        debug_assert!(n > 2);
        unsafe {
            let header = alloc_header::<K, V>(0, 0, n - 1, 0);
            let dp = raw_data_ptr::<K, V>(header, n - 1, 0);
            let src = self.data_slice();
            for (i, item) in src.iter().enumerate().take(idx) {
                dp.add(i).write(item.clone());
            }
            for (i, item) in src.iter().enumerate().skip(idx + 1) {
                dp.add(i - 1).write(item.clone());
            }
            NodePtr(header, PhantomData)
        }
    }

    /// Clone the single surviving `(K, V)` after conceptually removing entry
    /// `idx` from a 2-entry data run (either a 2-data/0-node bitmap node or a
    /// 2-entry collision node) — used for canonical inlining without ever
    /// allocating a throwaway 1-entry node.
    pub(crate) fn extract_sole_survivor(&self, idx_removed: usize) -> (K, V) {
        debug_assert_eq!(self.n_data(), 2);
        let other = 1 - idx_removed;
        self.data_slice()[other].clone()
    }
}

// index helpers used by the migration ops above, where the bit is being
// *inserted* into the map it doesn't yet belong to.
impl<K, V> NodePtr<K, V> {
    #[inline]
    fn node_index_for_insert(&self, bit: u32) -> usize {
        debug_assert_eq!(self.nodemap() & bit, 0);
        (self.nodemap() & (bit - 1)).count_ones() as usize
    }

    #[inline]
    fn data_index_for_insert(&self, bit: u32) -> usize {
        debug_assert_eq!(self.datamap() & bit, 0);
        (self.datamap() & (bit - 1)).count_ones() as usize
    }
}

// ---------------------------------------------------------------------
// Unique-path ops: `self` is consumed (caller has just verified
// `is_unique()`), so we may mutate/realloc in place using take semantics —
// `ptr::read` existing entries out of the old allocation and `dealloc` it
// WITHOUT dropping the moved-out entries, never double-dropping, never
// touching uninitialized slots.
// ---------------------------------------------------------------------

/// Deconstructed old-allocation parts handed back by [`NodePtr::take_parts`]
/// — a named struct rather than a tuple purely to keep the six-field return
/// type readable (and to dodge clippy's `type_complexity` lint on the tuple
/// form).
struct TakeParts<K, V> {
    layout: Layout,
    n_data: usize,
    n_nodes: usize,
    data_ptr: *mut (K, V),
    node_ptr: *mut NodePtr<K, V>,
}

impl<K, V> NodePtr<K, V> {
    /// In-place value replace at data slot `idx`: no realloc, drops the old
    /// value via normal assignment semantics.
    pub(crate) fn unique_set_value(&mut self, idx: usize, v: V) {
        debug_assert!(self.is_unique());
        self.data_slice_mut()[idx].1 = v;
    }

    /// In-place value replace at data slot `idx` that hands back the old
    /// value instead of dropping it (`mem::replace`, not assignment) — used
    /// by [`crate::PersistentHashMap::assoc_owned_replacing`]'s mutate path
    /// to return the previously-associated value without a second descent.
    pub(crate) fn unique_replace_value(&mut self, idx: usize, v: V) -> V {
        debug_assert!(self.is_unique());
        std::mem::replace(&mut self.data_slice_mut()[idx].1, v)
    }

    /// Deconstruct `self`'s allocation for a take-semantics rebuild, without
    /// dropping or deallocating anything yet. `layout` is computed from
    /// `old_cap` (the node's *reserved* data-slot count), NOT `n_data` (its
    /// *live* count) — using `n_data` here would be an immediate
    /// dealloc-with-wrong-layout bug for any node carrying capacity slack
    /// (`old_cap > n_data`), since the allocation was actually sized by cap.
    fn take_parts(&self) -> TakeParts<K, V> {
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        let old_cap = self.cap();
        let (layout, _, _) = node_layout::<K, V>(old_cap, n_nodes);
        TakeParts {
            layout,
            n_data,
            n_nodes,
            data_ptr: self.data_ptr(),
            node_ptr: self.node_ptr(),
        }
    }

    /// Insert a new data entry at `bit` (must be absent).
    ///
    /// Fast path (`n_data < cap`): pure in-place mutation, zero allocation —
    /// an overlapping `ptr::copy` (memmove, NOT `copy_nonoverlapping`, since
    /// source and destination ranges here alias) shifts the tail
    /// `idx..n_data` right by one slot within the *same* allocation, then
    /// the new entry is written into the vacated slot and `datamap` is
    /// updated. `cap` is unchanged; nothing is freed.
    ///
    /// Slow path (`n_data == cap`): reallocate with
    /// `new_cap = compute_cap(n_data + 1)`, moving all existing entries via
    /// `copy_nonoverlapping` (old and new allocations never overlap) —
    /// take semantics: no clone, no drop of moved values, old allocation
    /// freed raw.
    pub(crate) fn unique_insert_data(mut self, bit: u32, k: K, v: V) -> Self {
        debug_assert!(self.is_unique());
        debug_assert_eq!(self.datamap() & bit, 0);
        let idx = (self.datamap() & (bit - 1)).count_ones() as usize;
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        let cap = self.cap();
        debug_assert!(n_data <= cap && cap <= 32, "cap invariant violated on entry");

        if n_data < cap {
            let dp = self.data_ptr();
            // SAFETY: `dp.add(idx)..dp.add(n_data)` (len `n_data - idx`) and
            // `dp.add(idx + 1)..dp.add(n_data + 1)` overlap whenever
            // `n_data > idx`, so this MUST be `ptr::copy` (memmove), not
            // `copy_nonoverlapping` — Miri actively checks this distinction.
            // The destination range's tail (up to index `cap - 1`) is
            // within the node's reserved capacity, so writing into it is in
            // bounds even though slot `n_data` (0-indexed) was previously
            // uninitialized slack. `dp.add(idx)` is then written with the
            // new entry, exactly once, completing the shift with no gap.
            unsafe {
                ptr::copy(dp.add(idx), dp.add(idx + 1), n_data - idx);
                dp.add(idx).write((k, v));
            }
            self.header_mut().datamap |= bit;
            debug_assert!(self.n_data() <= self.cap(), "cap invariant violated after in-place insert");
            return self;
        }

        let new_cap = compute_cap(n_data + 1);
        let TakeParts {
            layout: old_layout,
            data_ptr: old_dp,
            node_ptr: old_np,
            ..
        } = self.take_parts();
        let old_header = self.0;
        unsafe {
            let header = alloc_header::<K, V>(self.datamap() | bit, self.nodemap(), new_cap, n_nodes);
            let dp = raw_data_ptr::<K, V>(header, new_cap, n_nodes);
            // SAFETY: `old_dp` and `dp` are distinct, non-overlapping
            // allocations (the old one is still live and un-freed until the
            // `dealloc` call below), so `copy_nonoverlapping` is sound here.
            ptr::copy_nonoverlapping(old_dp, dp, idx);
            dp.add(idx).write((k, v));
            ptr::copy_nonoverlapping(old_dp.add(idx), dp.add(idx + 1), n_data - idx);
            let np = raw_node_ptr::<K, V>(header, new_cap, n_nodes);
            ptr::copy_nonoverlapping(old_np, np, n_nodes);
            raw_dealloc(old_header.as_ptr() as *mut u8, old_layout);
            NodePtr(header, PhantomData)
        }
    }

    /// Remove the data entry at `bit`. Caller ensures the remaining shape is
    /// a legitimate node on its own — see [`Self::copy_with_data_removed`]
    /// for the exact precondition (0 data + 1 node is fine; 1 data + 0 node
    /// or fully empty are not).
    ///
    /// Always in-place now, never reallocates: an overlapping `ptr::copy`
    /// shifts the tail `idx+1..n_data` left by one slot after dropping the
    /// removed entry; `cap` is left unchanged (dissoc on a unique node never
    /// shrinks capacity, so it never allocates). The vacated slot at the
    /// (new) end of the live range is left uninitialized — never read, since
    /// `n_data()` (derived from `datamap`, updated below) no longer covers
    /// it.
    pub(crate) fn unique_remove_data(mut self, bit: u32) -> Self {
        debug_assert!(self.is_unique());
        let idx = self.data_index(bit);
        let n_data = self.n_data();
        let n_nodes = self.n_nodes();
        debug_assert!(
            !(n_data - 1 == 1 && n_nodes == 0),
            "should have canonically inlined instead"
        );
        debug_assert!(n_data - 1 + n_nodes >= 1, "should have returned Gone instead");
        let dp = self.data_ptr();
        // SAFETY: `dp.add(idx)` holds a live, initialized `(K, V)`, dropped
        // exactly once here before its slot is overwritten by the shift
        // below. The shift ranges `dp.add(idx + 1)..dp.add(n_data)` (source)
        // and `dp.add(idx)..dp.add(n_data - 1)` (dest) overlap whenever
        // `n_data - idx > 1`, so this MUST be `ptr::copy`, not
        // `copy_nonoverlapping`.
        unsafe {
            ptr::drop_in_place(dp.add(idx));
            ptr::copy(dp.add(idx + 1), dp.add(idx), n_data - idx - 1);
        }
        self.header_mut().datamap &= !bit;
        debug_assert!(self.n_data() <= self.cap(), "cap invariant violated after in-place remove");
        self
    }

    /// Migrate data slot `bit` into a child slot at the same bit, in place
    /// (the node region always changes shape here, so this still
    /// reallocates — no in-place fast path — but the new node's data region
    /// is granted `compute_cap(new_count)` slack instead of exact-fit, same
    /// as the growth path above).
    pub(crate) fn unique_data_to_node(self, bit: u32, child: Self) -> Self {
        debug_assert!(self.is_unique());
        debug_assert_ne!(self.datamap() & bit, 0);
        let data_idx = self.data_index(bit);
        let node_idx = self.node_index_for_insert(bit);
        let TakeParts {
            layout: old_layout,
            n_data,
            n_nodes,
            data_ptr: old_dp,
            node_ptr: old_np,
            ..
        } = self.take_parts();
        let old_header = self.0;
        let new_cap = compute_cap(n_data - 1);
        unsafe {
            let header =
                alloc_header::<K, V>(self.datamap() & !bit, self.nodemap() | bit, new_cap, n_nodes + 1);
            let dp = raw_data_ptr::<K, V>(header, new_cap, n_nodes + 1);
            ptr::copy_nonoverlapping(old_dp, dp, data_idx);
            // the migrated (k, v) is dropped: its payload now lives in `child`.
            ptr::drop_in_place(old_dp.add(data_idx));
            ptr::copy_nonoverlapping(old_dp.add(data_idx + 1), dp.add(data_idx), n_data - data_idx - 1);
            let np = raw_node_ptr::<K, V>(header, new_cap, n_nodes + 1);
            ptr::copy_nonoverlapping(old_np, np, node_idx);
            np.add(node_idx).write(child);
            ptr::copy_nonoverlapping(old_np.add(node_idx), np.add(node_idx + 1), n_nodes - node_idx);
            raw_dealloc(old_header.as_ptr() as *mut u8, old_layout);
            NodePtr(header, PhantomData)
        }
    }

    /// Migrate child slot `bit` into a data slot at the same bit, in place
    /// (reallocates, same as above, for the same reason — the node region
    /// shape changes — and the new data region likewise gets
    /// `compute_cap(new_count)` slack instead of exact-fit).
    ///
    /// Contract: the caller must already have consumed the child that lives
    /// at `bit` (e.g. via [`take_node`](Self::take_node) followed by a
    /// recursive dissoc call that fully retired it) before calling this —
    /// the slot is treated as already "moved-from" and is neither read nor
    /// dropped, only skipped over while the surrounding slots are shifted.
    pub(crate) fn unique_node_to_data(self, bit: u32, k: K, v: V) -> Self {
        debug_assert!(self.is_unique());
        debug_assert_ne!(self.nodemap() & bit, 0);
        let node_idx = self.node_index(bit);
        let data_idx = self.data_index_for_insert(bit);
        let TakeParts {
            layout: old_layout,
            n_data,
            n_nodes,
            data_ptr: old_dp,
            node_ptr: old_np,
            ..
        } = self.take_parts();
        let old_header = self.0;
        let new_cap = compute_cap(n_data + 1);
        unsafe {
            let header =
                alloc_header::<K, V>(self.datamap() | bit, self.nodemap() & !bit, new_cap, n_nodes - 1);
            let dp = raw_data_ptr::<K, V>(header, new_cap, n_nodes - 1);
            ptr::copy_nonoverlapping(old_dp, dp, data_idx);
            dp.add(data_idx).write((k, v));
            ptr::copy_nonoverlapping(old_dp.add(data_idx), dp.add(data_idx + 1), n_data - data_idx);
            let np = raw_node_ptr::<K, V>(header, new_cap, n_nodes - 1);
            ptr::copy_nonoverlapping(old_np, np, node_idx);
            // the migrated child pointer's ownership moves into the data
            // slot's caller-supplied (k, v); the *slot* that held the child
            // NodePtr value is not separately dropped (that value was
            // logically already consumed by whoever produced (k, v), i.e.
            // the recursive dissoc call that returned `Inline`).
            ptr::copy_nonoverlapping(old_np.add(node_idx + 1), np.add(node_idx), n_nodes - node_idx - 1);
            raw_dealloc(old_header.as_ptr() as *mut u8, old_layout);
            NodePtr(header, PhantomData)
        }
    }

    /// Grow a collision node by one entry, in place. Collision nodes carry
    /// no capacity slack (see the module docs), so this always reallocates
    /// exact-fit, same as before M4 tuning.
    pub(crate) fn unique_insert_collision(self, k: K, v: V) -> Self {
        debug_assert!(self.is_unique());
        debug_assert!(self.is_collision());
        let TakeParts {
            layout: old_layout,
            n_data: n,
            data_ptr: old_dp,
            ..
        } = self.take_parts();
        let old_header = self.0;
        unsafe {
            let header = alloc_header::<K, V>(0, 0, n + 1, 0);
            let dp = raw_data_ptr::<K, V>(header, n + 1, 0);
            ptr::copy_nonoverlapping(old_dp, dp, n);
            dp.add(n).write((k, v));
            raw_dealloc(old_header.as_ptr() as *mut u8, old_layout);
            NodePtr(header, PhantomData)
        }
    }

    /// Shrink a collision node by removing entry `idx`, in place (always
    /// exact-fit realloc, no slack — see [`Self::unique_insert_collision`]).
    /// Caller ensures `n_data() - 1 >= 2`.
    pub(crate) fn unique_remove_collision(self, idx: usize) -> Self {
        debug_assert!(self.is_unique());
        debug_assert!(self.is_collision());
        let TakeParts {
            layout: old_layout,
            n_data: n,
            data_ptr: old_dp,
            ..
        } = self.take_parts();
        debug_assert!(n > 2);
        let old_header = self.0;
        unsafe {
            let header = alloc_header::<K, V>(0, 0, n - 1, 0);
            let dp = raw_data_ptr::<K, V>(header, n - 1, 0);
            ptr::copy_nonoverlapping(old_dp, dp, idx);
            ptr::drop_in_place(old_dp.add(idx));
            ptr::copy_nonoverlapping(old_dp.add(idx + 1), dp.add(idx), n - idx - 1);
            raw_dealloc(old_header.as_ptr() as *mut u8, old_layout);
            NodePtr(header, PhantomData)
        }
    }

    /// Deallocate a unique chain-link node (0 data entries, 1 child slot)
    /// whose sole child has already been retired by the caller via
    /// [`take_node`](Self::take_node) followed by a dissoc call that fully
    /// consumed it (e.g. it returned an inlined entry, freeing itself).
    /// Frees only the raw backing memory: there is nothing left to drop
    /// (no data entries, and the one child slot is already "moved-from").
    ///
    /// Uses `self.cap()` for the dealloc layout, NOT a hardcoded `0` — a
    /// 0-data/1-node node can carry residual capacity slack here: it can
    /// arise from [`Self::unique_remove_data`] shrinking a data+node mixed
    /// node down to 0 live data entries while leaving `cap` untouched (that
    /// op never reallocates), so `cap` may be anywhere in `0..=32`, not just
    /// `0`. Using the wrong (too-small) layout here would `dealloc` fewer
    /// bytes than were actually `alloc`'d — undefined behavior.
    pub(crate) fn unique_dealloc_after_take_sole_child(self) {
        debug_assert!(self.is_unique());
        debug_assert_eq!(self.n_data(), 0);
        debug_assert_eq!(self.n_nodes(), 1);
        let cap = self.cap();
        let (layout, _, _) = node_layout::<K, V>(cap, 1);
        let header = self.0;
        unsafe {
            raw_dealloc(header.as_ptr() as *mut u8, layout);
        }
    }

    /// Take ownership of the single surviving `(K, V)` after conceptually
    /// removing entry `idx_removed` from a 2-entry data run, deallocating
    /// this node's allocation without double-dropping/double-freeing
    /// anything (the removed entry is dropped, the survivor is moved out).
    pub(crate) fn extract_sole_survivor_unique(self, idx_removed: usize) -> (K, V) {
        debug_assert!(self.is_unique());
        debug_assert_eq!(self.n_data(), 2);
        let TakeParts {
            layout: old_layout,
            data_ptr: old_dp,
            ..
        } = self.take_parts();
        let old_header = self.0;
        let other = 1 - idx_removed;
        unsafe {
            ptr::drop_in_place(old_dp.add(idx_removed));
            let survivor = ptr::read(old_dp.add(other));
            raw_dealloc(old_header.as_ptr() as *mut u8, old_layout);
            survivor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::RandomState;

    #[test]
    fn leaf_roundtrip() {
        let n = NodePtr::<i32, i32>::new_leaf(1 << 5, 1, 100);
        assert_eq!(n.n_data(), 1);
        assert_eq!(n.n_nodes(), 0);
        assert_eq!(n.data_slice(), &[(1, 100)]);
        n.drop_node();
    }

    #[test]
    fn hash32_folds() {
        let s = RandomState::new();
        let _ = hash32(&s, &"hello");
    }

    #[test]
    fn refcount_shares_and_frees() {
        let n = NodePtr::<i32, i32>::new_leaf(1, 1, 2);
        assert!(n.is_unique());
        let n2 = n.clone_shallow();
        assert!(!n.is_unique());
        assert!(!n2.is_unique());
        n.drop_node();
        assert!(n2.is_unique());
        n2.drop_node();
    }

    // -------------------------------------------------------------------
    // M4 capacity-slack targeted tests: force in-place `ptr::copy` shifts
    // at first/middle/last position, exercise the cap boundary (in-place
    // vs. realloc transition), ZST values, and a specific residual-cap
    // regression the slack design introduced (see
    // `unique_dealloc_after_take_sole_child`'s doc comment).
    //
    // Final M4 policy is exact-fit growth (`compute_cap` is the identity
    // function — see its doc comment), so none of these tests can rely on
    // growth alone to create `cap > count` headroom anymore: every one
    // below builds it the way the shipped design actually produces it —
    // grow (exact-fit), then remove one or more entries in place
    // (`unique_remove_data` never shrinks `cap`), *then* exercise the
    // resulting residual-cap in-place insert path.
    // -------------------------------------------------------------------

    /// Bitmap bit for chunk position `pos` (0..=31) — a small helper so
    /// tests below can write `bit(3)` instead of `1u32 << 3`.
    fn bit(pos: u32) -> u32 {
        1u32 << pos
    }

    fn keys<K: Copy, V>(n: &NodePtr<K, V>) -> Vec<K> {
        n.data_slice().iter().map(|(k, _)| *k).collect()
    }

    #[test]
    fn compute_cap_is_exact_fit() {
        // Final M4 policy: growth carries no slack, cap == count always
        // immediately after a growth realloc.
        for count in 0..=32 {
            assert_eq!(compute_cap(count), count);
        }
    }

    #[test]
    fn unique_insert_data_in_place_shifts_at_every_position() {
        // Build up to 5 entries via growth — exact-fit at every step under
        // the final policy, so `cap` tracks `count` throughout (no slack).
        let mut n = NodePtr::<u64, u64>::new_leaf(bit(10), 10, 100);
        for (pos, k) in [(20u32, 20u64), (15, 15), (30, 30), (25, 25)] {
            n = n.unique_insert_data(bit(pos), k, k * 10);
        }
        assert_eq!(n.cap(), 5, "exact-fit growth: cap == count == 5");
        assert_eq!(keys(&n), vec![10, 15, 20, 25, 30]);

        // Remove 3 entries, in place (cap stays 5, count drops to 2) —
        // this is what manufactures the residual cap the inserts below
        // exercise; `unique_remove_data` never reallocates or shrinks cap.
        n = n.unique_remove_data(bit(15));
        n = n.unique_remove_data(bit(25));
        n = n.unique_remove_data(bit(30));
        assert_eq!(n.cap(), 5, "remove never touches cap");
        assert_eq!(keys(&n), vec![10, 20]);

        // FIRST: insert into residual cap at idx=0 — in-place, no realloc.
        let n = n.unique_insert_data(bit(1), 1, 11);
        assert_eq!(n.cap(), 5, "still in-place: count (3) < cap (5)");
        assert_eq!(keys(&n), vec![1, 10, 20]);

        // MIDDLE: insert into residual cap at idx=2 (between 10 and 20).
        let n = n.unique_insert_data(bit(15), 15, 150);
        assert_eq!(n.cap(), 5, "still in-place: count (4) < cap (5)");
        assert_eq!(keys(&n), vec![1, 10, 15, 20]);

        // LAST: append, consuming the final slack slot — this insert call
        // itself was still in-place (count was 4 < cap 5 going in), even
        // though count == cap coming out.
        let n = n.unique_insert_data(bit(30), 30, 300);
        assert_eq!(n.cap(), 5, "cap unchanged by any of the in-place inserts above");
        assert_eq!(keys(&n), vec![1, 10, 15, 20, 30]);

        n.drop_node();
    }

    #[test]
    fn unique_remove_data_in_place_shifts_at_every_position() {
        // Build a 5-entry node — exact-fit growth, cap == count == 5.
        let mut n = NodePtr::<u64, u64>::new_leaf(bit(1), 1, 10);
        for pos in 2u32..=5 {
            n = n.unique_insert_data(bit(pos), pos as u64, pos as u64 * 10);
        }
        assert_eq!(n.cap(), 5, "exact-fit growth");
        assert_eq!(keys(&n), vec![1, 2, 3, 4, 5]);

        // MIDDLE removal (idx=2, key 3) — in-place, cap unchanged.
        let n2 = n.unique_remove_data(bit(3));
        assert_eq!(n2.cap(), 5, "remove never reallocates or changes cap");
        assert_eq!(keys(&n2), vec![1, 2, 4, 5]);
        n = n2;

        // FIRST removal (idx=0, key 1).
        let n2 = n.unique_remove_data(bit(1));
        assert_eq!(n2.cap(), 5);
        assert_eq!(keys(&n2), vec![2, 4, 5]);
        n = n2;

        // LAST removal (idx=2, key 5) — a zero-length shift (pure drop, no
        // memmove of any tail).
        let n2 = n.unique_remove_data(bit(5));
        assert_eq!(n2.cap(), 5);
        assert_eq!(keys(&n2), vec![2, 4]);

        n2.drop_node();
    }

    #[test]
    fn grows_to_32_cap_ceiling_and_no_further() {
        let mut n = NodePtr::<u32, u32>::new_leaf(bit(0), 0, 0);
        for i in 1u32..32 {
            n = n.unique_insert_data(bit(i), i, i);
        }
        assert_eq!(n.n_data(), 32);
        assert_eq!(n.cap(), 32, "exact-fit growth still hits the 32-slot hard ceiling exactly");
        n.drop_node();
    }

    #[test]
    fn zst_value_residual_cap_in_place_insert_and_remove() {
        // The set-layer's `V = ()` case: exercises the same layout math
        // (Layout::array of a zero-sized-payload tuple) through both the
        // exact-fit growth path and the residual-cap in-place path.
        let n = NodePtr::<u64, ()>::new_leaf(bit(1), 1, ());
        let n = n.unique_insert_data(bit(2), 2, ()); // exact-fit realloc, cap=2
        let n = n.unique_insert_data(bit(3), 3, ()); // exact-fit realloc, cap=3
        assert_eq!(n.cap(), 3);
        assert_eq!(keys(&n), vec![1, 2, 3]);

        let n = n.unique_remove_data(bit(2)); // in-place remove, middle; cap stays 3
        assert_eq!(n.cap(), 3, "remove never reallocates");
        assert_eq!(n.n_data(), 2);
        assert_eq!(keys(&n), vec![1, 3]);

        // Residual-cap in-place insert (count 2 < cap 3), middle position.
        let n = n.unique_insert_data(bit(2), 2, ());
        assert_eq!(n.cap(), 3, "in-place: count was 2 < cap 3 going in");
        assert_eq!(keys(&n), vec![1, 2, 3]);
        n.drop_node();
    }

    #[test]
    fn unique_dealloc_after_take_sole_child_respects_residual_cap() {
        // Regression test for a real bug the slack design introduced: a
        // 0-data/1-node "chain link" can carry nonzero capacity slack
        // inherited from before it lost its last data entry via
        // `unique_remove_data` (which deliberately never touches `cap`).
        // `unique_dealloc_after_take_sole_child` must dealloc using that
        // *residual* cap, not a hardcoded `0` — see its doc comment for the
        // full trace of how a real dissoc sequence reaches this shape. This
        // still reproduces under the final exact-fit-growth policy: growth
        // still reallocates (just without slack), and removal still leaves
        // `cap` untouched, so `cap > count` is still reachable here.
        let grandchild = NodePtr::<u64, u64>::new_leaf2(bit(1), 1, 10, bit(2), 2, 20);
        let parent = NodePtr::<u64, u64>::new_leaf_and_child(bit(3), 3, 30, bit(9), grandchild);
        assert_eq!(parent.cap(), 1, "fresh constructor is exact-fit");

        // Grow the parent's data region via realloc (exact-fit: cap tracks
        // count, no slack under the final policy).
        let mut parent = parent.unique_insert_data(bit(4), 4, 40);
        assert_eq!(parent.cap(), 2, "exact-fit growth: cap == count == 2");
        assert_eq!(parent.n_data(), 2);
        assert_eq!(parent.n_nodes(), 1);

        // Shrink back down to 0 data entries, in place — cap stays 2 (this
        // is the residual capacity, entirely from removal, not growth).
        parent = parent.unique_remove_data(bit(4));
        assert_eq!(parent.cap(), 2);
        parent = parent.unique_remove_data(bit(3));
        assert_eq!(parent.n_data(), 0);
        assert_eq!(parent.n_nodes(), 1);
        assert_eq!(parent.cap(), 2, "residual cap from removal: this is the buggy scenario before the fix");

        // Simulate the real dissoc cascade: take the sole child back out,
        // then free the now-empty parent shell.
        let child = parent.take_node(0);
        parent.unique_dealloc_after_take_sole_child(); // would UB with a hardcoded cap=0

        child.drop_node();
    }
}
