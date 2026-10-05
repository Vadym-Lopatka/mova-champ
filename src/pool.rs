//! Thread-local node-pool allocator ("magazine" allocator) — SPEC-M6-ALLOC.md
//! Step 1. Recycles freed node allocations of the same size class instead of
//! round-tripping through the system allocator on every growth/shrink
//! realloc, attacking the cost Step 0's profile confirmed: champ's own
//! `build/transient` u64 n=10k loop spends ~41% of wall time inside
//! `alloc`/`dealloc` (2.23 alloc+dealloc calls per insert; see
//! `examples/profile_build.rs` and NOTES-M6.md for the full readout).
//!
//! Design:
//! - **Size classes**: 16-byte quantization, up to `MAX_POOLED_SIZE` (2KB,
//!   128 classes). A request bigger than that falls straight through to the
//!   system allocator — a node that large already holds dozens of live
//!   entries, far past the size range where alloc/dealloc frequency (not
//!   node size) dominates cost.
//! - **Per-thread free lists**: one `thread_local!` array of intrusive
//!   singly-linked lists, one per size class. A freed block's first
//!   `size_of::<*mut u8>()` (8 on every platform this crate targets) bytes
//!   are overwritten with the "next" pointer of its class's list — sound
//!   because every pooled class size is `>= 16` bytes (`Header` alone is 16:
//!   `AtomicU32` + 3x `u32`), so there is always room for one pointer.
//! - **Rounding discipline**: every block that ever enters a free list was
//!   allocated at *exactly* its class's rounded size (`class_size(idx)`),
//!   never the caller's raw requested size. This is required for
//!   soundness — a class's free list serves *any* request that maps to that
//!   class, and those requests can have different exact sizes (e.g. a
//!   3-entry and a 4-entry bitmap node's data regions can round to the same
//!   16-byte-quantized class); if blocks were stored at their original
//!   (smaller) size, a later pop for a larger same-class request would hand
//!   back a too-small block. `alloc_node`'s system-allocator-miss path and
//!   `dealloc_node`'s system-allocator-overflow path both therefore use
//!   `class_size(idx)`, not the caller's `layout.size()`, whenever `idx` is
//!   `Some` — the two call sites always agree because both derive `idx` from
//!   the same `class_index` function applied to the same logical layout
//!   (node.rs always dealloc's a block with the identical `Layout` value it
//!   was alloc'd with, by construction of `node_layout`).
//! - **Per-class cap**: [`CLASS_CAP`] blocks. `dealloc_node` pushes onto the
//!   free list only while under cap; at/above cap it frees to the system
//!   allocator immediately. Worst-case theoretical per-thread retention is
//!   `NUM_CLASSES * CLASS_CAP * MAX_POOLED_SIZE` bytes; see NOTES-M6.md for
//!   the measured worst case in practice, which is far lower (real node-size
//!   distributions concentrate in a handful of small classes).
//! - **Cross-thread correctness**: no atomics, no shared structures. A block
//!   `alloc_node`'d on thread A and `dealloc_node`'d on thread B simply joins
//!   B's thread-local pool — sound because a pooled block is anonymous,
//!   uninitialized memory with no identity tied to the thread that first
//!   obtained it from the system allocator, and `NodePtr`'s own `Send`/`Sync`
//!   rules (see `node.rs` module docs) already govern whether moving a node
//!   across threads is legal in the first place; this module never makes
//!   that decision, it just recycles bytes after the fact.
//!
//! `unsafe` is allowed in this module (and `node.rs`) per SPEC-M6-ALLOC.md;
//! everywhere else in the crate stays safe Rust.

use std::alloc::{self, Layout};
use std::cell::RefCell;

/// Blocks up to and including this size are pool-managed; anything larger
/// falls straight through to the system allocator.
const MAX_POOLED_SIZE: usize = 2048;

/// Size-class quantum: every pooled layout's size is rounded up to a
/// multiple of this many bytes. Also the minimum alignment this pool
/// supports for a pooled block (see `alloc_node`'s debug_assert) — every
/// node layout in this crate needs at most this much.
const CLASS_QUANTUM: usize = 16;

/// Number of size classes covering `1..=MAX_POOLED_SIZE`.
const NUM_CLASSES: usize = MAX_POOLED_SIZE / CLASS_QUANTUM;

/// Per-class free-list cap. Bounds worst-case per-thread pool retention.
const CLASS_CAP: usize = 64;

/// Round `size` (`> 0`) up to its containing class index, or `None` if it's
/// too large to be pool-managed (falls through to the system allocator).
#[inline]
fn class_index(size: usize) -> Option<usize> {
    if size == 0 || size > MAX_POOLED_SIZE {
        return None;
    }
    Some((size - 1) / CLASS_QUANTUM)
}

/// The actual byte size class `idx`'s blocks are allocated at: always a
/// multiple of `CLASS_QUANTUM`, always `>=` any request that maps to `idx`.
#[inline]
fn class_size(idx: usize) -> usize {
    (idx + 1) * CLASS_QUANTUM
}

/// The `Layout` a class `idx` block is actually allocated/freed at, given
/// the caller's requested alignment (every class size is a multiple of
/// `CLASS_QUANTUM`, so this only widens alignment, never narrows it below
/// what the caller asked for).
#[inline]
fn class_layout(idx: usize, requested_align: usize) -> Layout {
    // SAFETY-relevant note: `Layout::from_size_align` only fails when `align`
    // isn't a power of two or the rounded size overflows `isize` — neither
    // can happen here (`requested_align` came from an already-valid caller
    // `Layout`, and `class_size` is bounded by `MAX_POOLED_SIZE`), but
    // `unwrap_or` degrades gracefully to an always-valid fallback rather than
    // panicking if some future caller ever violates that.
    let align = requested_align.max(CLASS_QUANTUM);
    Layout::from_size_align(class_size(idx), align)
        .unwrap_or_else(|_| Layout::from_size_align(class_size(idx), CLASS_QUANTUM).expect("class size/quantum are always valid"))
}

/// One size class's intrusive free list: a raw head pointer plus a running
/// count (so `dealloc_node` can enforce `CLASS_CAP` without walking the
/// list) and a running byte total (for `retained_bytes`).
struct FreeList {
    head: *mut u8,
    len: usize,
}

impl FreeList {
    const fn new() -> Self {
        FreeList { head: std::ptr::null_mut(), len: 0 }
    }
}

struct Pool {
    classes: [FreeList; NUM_CLASSES],
}

impl Pool {
    fn new() -> Self {
        Pool { classes: [const { FreeList::new() }; NUM_CLASSES] }
    }
}

thread_local! {
    static POOL: RefCell<Pool> = RefCell::new(Pool::new());
}

/// Allocate a block for `layout`, preferring a pooled free block of the
/// matching size class over the system allocator.
///
/// # Safety
/// Same contract as `std::alloc::alloc`: `layout` must be nonzero-sized. The
/// returned pointer, if non-null, must eventually be passed to exactly one
/// of `dealloc_node`/(nothing, if leaked) with the *same* `layout` value —
/// never `std::alloc::dealloc` directly, since a pooled block's true
/// allocated size can exceed `layout.size()`.
pub(crate) unsafe fn alloc_node(layout: Layout) -> *mut u8 {
    debug_assert!(layout.size() > 0, "champ node layouts are always nonzero-sized");
    let Some(idx) = class_index(layout.size()) else {
        // SAFETY: forwarding to the system allocator with the caller's own
        // layout, unchanged — sound by `std::alloc::alloc`'s contract, which
        // this function's contract mirrors for the too-large case (never
        // pooled, so `dealloc_node` also forwards it straight through).
        return unsafe { alloc::alloc(layout) };
    };
    debug_assert!(
        layout.align() <= CLASS_QUANTUM,
        "pool size classes are only 16-byte aligned; a larger alignment needs a dedicated path"
    );
    let popped = POOL.with(|p| {
        let mut p = p.borrow_mut();
        let fl = &mut p.classes[idx];
        if fl.head.is_null() {
            None
        } else {
            // SAFETY: `fl.head` is a block previously pushed by
            // `dealloc_node` for this exact class, which wrote a valid
            // `*mut u8` "next" pointer into its first 8 bytes before linking
            // it in (see `dealloc_node`) — reading it back here exactly
            // undoes that write. The block is otherwise untouched, unaliased
            // memory (it left the live set the moment it was freed, and pool
            // internals never leak references to it).
            let next = unsafe { *(fl.head as *mut *mut u8) };
            let block = fl.head;
            fl.head = next;
            fl.len -= 1;
            Some(block)
        }
    });
    match popped {
        Some(block) => block,
        None => {
            // Pool empty for this class: fall through to the system
            // allocator, but at the class's rounded size (see module docs'
            // "Rounding discipline") so this block is a legal source for
            // future same-class pops of any size, and so `dealloc_node`
            // later sees a layout consistent with what was actually
            // allocated.
            let cl = class_layout(idx, layout.align());
            // SAFETY: `cl` is well-formed and nonzero-sized.
            unsafe { alloc::alloc(cl) }
        }
    }
}

/// Return a block previously obtained from `alloc_node` to the pool if
/// there's room in its size class, else free it via the system allocator
/// immediately.
///
/// # Safety
/// `ptr` must have been returned by a previous call to `alloc_node` with an
/// *identical* `layout`, and must not be used (including passed to this
/// function again) afterward — the same rules as `std::alloc::dealloc`.
pub(crate) unsafe fn dealloc_node(ptr: *mut u8, layout: Layout) {
    debug_assert!(layout.size() > 0, "champ node layouts are always nonzero-sized");
    let Some(idx) = class_index(layout.size()) else {
        // SAFETY: this block was allocated via the `alloc::alloc(layout)`
        // fallback in `alloc_node` for this same (too-large) layout.
        unsafe { alloc::dealloc(ptr, layout) };
        return;
    };
    let pushed = POOL.with(|p| {
        let mut p = p.borrow_mut();
        let fl = &mut p.classes[idx];
        if fl.len >= CLASS_CAP {
            false
        } else {
            // SAFETY: `ptr` is a valid, no-longer-aliased block of at least
            // `class_size(idx) >= 16` bytes (see module docs) and at least
            // 8-byte aligned (every class size/alignment is a multiple of
            // `CLASS_QUANTUM == 16`), so writing the current list head into
            // its first 8 bytes as the intrusive "next" pointer is in bounds
            // and correctly aligned for a `*mut u8` write.
            unsafe { *(ptr as *mut *mut u8) = fl.head };
            fl.head = ptr;
            fl.len += 1;
            true
        }
    });
    if !pushed {
        // SAFETY: `class_layout(idx, layout.align())` matches what
        // `alloc_node` actually allocated this block at — pooled blocks are
        // always allocated at the rounded class size (module docs'
        // "Rounding discipline"), never the raw per-call layout — so this is
        // a matched alloc/dealloc pair from the system allocator's point of
        // view.
        let cl = class_layout(idx, layout.align());
        unsafe { alloc::dealloc(ptr, cl) };
    }
}

/// Total bytes currently parked in this thread's pool across all classes —
/// a measurement/introspection hook, not used by any allocation logic.
/// Re-exported (feature-gated) at the crate root as
/// [`crate::pool_retained_bytes`] for `examples/memstats.rs` and other
/// external callers who need to separate "pool-retained bytes" from
/// "live map bytes" when the counting allocator can't tell them apart on
/// its own (a pooled block still reads as globally-allocated memory).
pub(crate) fn retained_bytes() -> usize {
    POOL.with(|p| {
        let p = p.borrow();
        p.classes.iter().enumerate().map(|(idx, fl)| fl.len * class_size(idx)).sum()
    })
}

/// Free every block currently parked in this thread's pool back to the
/// system allocator, resetting all class lists to empty. A
/// measurement/testing hook (re-exported as [`crate::pool_drain`]) — normal
/// operation never needs this, since the pool's own `CLASS_CAP` already
/// bounds retention; it exists so `examples/memstats.rs` can force pool
/// state back to "empty" between scenarios and keep its live-byte
/// measurements apples-to-apples with a pool-off build.
pub(crate) fn drain() {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        for (idx, fl) in p.classes.iter_mut().enumerate() {
            let mut cur = fl.head;
            while !cur.is_null() {
                // SAFETY: `cur` is a live pooled block for class `idx`,
                // allocated at exactly `class_size(idx)` (module docs'
                // "Rounding discipline"); its first 8 bytes hold the next
                // pointer written by `dealloc_node`, read here before the
                // block itself is freed.
                let next = unsafe { *(cur as *mut *mut u8) };
                let cl = class_layout(idx, CLASS_QUANTUM);
                unsafe { alloc::dealloc(cur, cl) };
                cur = next;
            }
            fl.head = std::ptr::null_mut();
            fl.len = 0;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_index_and_size_round_trip_covers_requested_size() {
        for size in 1..=MAX_POOLED_SIZE {
            let idx = class_index(size).unwrap();
            assert!(class_size(idx) >= size, "class {idx} size {} must cover request {size}", class_size(idx));
            assert!(class_size(idx) - size < CLASS_QUANTUM, "rounding should never overshoot by a full quantum");
        }
        assert_eq!(class_index(MAX_POOLED_SIZE + 1), None, "just over the ceiling falls through");
        assert_eq!(class_index(1_000_000), None, "far over the ceiling falls through");
    }

    #[test]
    fn alloc_dealloc_round_trip_reuses_freed_block() {
        let layout = Layout::from_size_align(32, 8).unwrap();
        unsafe {
            let p1 = alloc_node(layout);
            assert!(!p1.is_null());
            dealloc_node(p1, layout);
            // Same class, should come back out of the free list rather than
            // hitting the system allocator again — same address.
            let p2 = alloc_node(layout);
            assert_eq!(p1, p2, "freed block should be reused from the pool");
            dealloc_node(p2, layout);
        }
        // Drain so this test leaves no pooled block behind — matters for
        // running this test module under Miri (see NOTES-M6.md): Miri's
        // leak detector fires at end of process for *any* still-pooled
        // block, since deliberate reuse-later retention is indistinguishable
        // from a real leak to a checker that doesn't know the pool's
        // contract. Every test in this module either never pools anything or
        // (like this one) explicitly drains before returning, so the whole
        // module can run Miri-clean when filtered to run alone (`cargo
        // +nightly miri test --lib pool::`), without needing
        // `-Zmiri-ignore-leaks`.
        drain();
    }

    #[test]
    fn retained_bytes_tracks_pool_state_and_drain_clears_it() {
        drain(); // isolate from any prior test on this thread
        assert_eq!(retained_bytes(), 0);
        let layout = Layout::from_size_align(48, 8).unwrap();
        unsafe {
            let p = alloc_node(layout);
            dealloc_node(p, layout);
        }
        assert!(retained_bytes() > 0, "one freed block should be parked in the pool");
        drain();
        assert_eq!(retained_bytes(), 0, "drain should free everything and reset accounting");
    }

    #[test]
    fn oversized_layout_falls_through_to_system_allocator() {
        let layout = Layout::from_size_align(MAX_POOLED_SIZE + 16, 8).unwrap();
        unsafe {
            let p = alloc_node(layout);
            assert!(!p.is_null());
            dealloc_node(p, layout);
        }
        // Oversized blocks are never pooled, so this must not have touched
        // pool accounting.
        assert_eq!(class_index(MAX_POOLED_SIZE + 16), None);
    }

    #[test]
    fn class_cap_overflow_frees_to_system_instead_of_growing_pool_forever() {
        drain();
        let layout = Layout::from_size_align(16, 8).unwrap();
        let mut ptrs = Vec::new();
        // Allocate and immediately free CLASS_CAP + a few more, one at a
        // time, so the free list fills up to its cap and the rest overflow
        // to the system allocator instead of growing the list unbounded.
        for _ in 0..(CLASS_CAP + 8) {
            let p = unsafe { alloc_node(layout) };
            ptrs.push(p);
        }
        for p in ptrs {
            unsafe { dealloc_node(p, layout) };
        }
        assert_eq!(retained_bytes(), CLASS_CAP * 16, "pool should have capped at CLASS_CAP blocks for this class");
        drain();
    }
}
