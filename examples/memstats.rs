//! Memory-per-entry harness: a counting `#[global_allocator]` measures live
//! heap bytes before/after building each collection, so the numbers below
//! are exact byte counts (not RSS estimates). See `SPEC-BENCH.md` for the
//! scenario matrix and `BENCH-RESULTS.md` for the analysis.
//!
//! Run with `cargo run --release --example memstats`.
//!
//! ## Pool honesty (M6)
//!
//! With the default-on `pool-alloc` feature, champ's node allocator
//! recycles freed node blocks through a thread-local pool instead of
//! returning them to the system allocator immediately (see `src/pool.rs`).
//! From this file's counting `#[global_allocator]`'s point of view, a
//! pooled block is still live memory — it was never `dealloc`'d — even
//! though it's logically "spare capacity" set aside for reuse, not part of
//! any retained collection. Left alone, that would inflate every live-byte
//! measurement below by however much churn a scenario's build process
//! itself produced, and would break the leak check's `after_drop ==
//! baseline` assertion outright (dropping a collection frees its nodes
//! into the pool, not back to the system allocator, so `live_bytes()`
//! would never fall back to baseline).
//!
//! `measure()` therefore calls [`drain_pool`] — a no-op when `pool-alloc`
//! is off — at the two points that matter: right after building (so the
//! reported live/peak bytes reflect only the retained value, not leftover
//! build-time churn sitting in the pool) and right after dropping (so the
//! leak check still holds). The bytes freed by the *first* drain are
//! reported separately as "pool-retained bytes" per row, so the memory
//! story stays honest in both directions: what the collection itself
//! costs, and what churn the pool was holding onto for reuse.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use champ::PersistentHashMap;
use im::HashMap as ImHashMap;
use std::collections::HashMap as StdHashMap;

// ---------------------------------------------------------------------
// Counting allocator: every alloc/dealloc/realloc updates `LIVE` (current
// live byte count) and `PEAK` (high-water mark since the last `reset_peak`
// call). Delegates the actual memory work to `System` unchanged.
// ---------------------------------------------------------------------

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                record_grow(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[inline]
fn record_grow(added: usize) {
    let new_live = LIVE.fetch_add(added, Ordering::Relaxed) + added;
    PEAK.fetch_max(new_live, Ordering::Relaxed);
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn live_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Reset the high-water mark to the current live count, so `peak_bytes()`
/// after this call reports the max live bytes *since this reset* rather
/// than since process start.
fn reset_peak() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------
// Key generation
// ---------------------------------------------------------------------

/// Exactly 16 bytes: `'k'` + 15 zero-padded decimal digits.
fn key16(i: usize) -> String {
    format!("k{i:015}")
}

// ---------------------------------------------------------------------
// Scenario runner: measures live-byte delta for `build()`, then drops the
// result and asserts the baseline is restored (leak check).
// ---------------------------------------------------------------------

/// Drain the calling thread's node pool (no-op when the `pool-alloc`
/// feature is off) — see the module docs' "Pool honesty" section.
#[cfg(feature = "pool-alloc")]
fn drain_pool() {
    champ::pool_drain();
}
#[cfg(not(feature = "pool-alloc"))]
fn drain_pool() {}

struct Measurement {
    live_delta: usize,
    peak_delta: usize,
    /// Bytes freed by the post-build pool drain: churn the build process
    /// itself generated (intermediate nodes freed by growth reallocs along
    /// the way) that the pool was holding onto for reuse rather than
    /// returning to the system allocator. Always `0` when `pool-alloc` is
    /// off. NOT part of `live_delta`/`peak_delta` — those already reflect
    /// only the retained value's own bytes, post-drain.
    pool_retained: usize,
}

fn measure<T>(build: impl FnOnce() -> T) -> Measurement {
    let baseline = live_bytes();
    reset_peak();
    let value = build();
    let peak_during = peak_bytes();
    // Drain BEFORE reading "after_build" live bytes: otherwise any node
    // freed and pooled during `build()`'s own churn (e.g. every superseded
    // node along a growth-realloc chain) would still read as live from the
    // global allocator's perspective, inflating the per-entry number for a
    // scenario that doesn't actually retain that memory as part of `value`.
    let before_drain = live_bytes();
    drain_pool();
    let after_build = live_bytes();
    let pool_retained = before_drain.saturating_sub(after_build);
    let live_delta = after_build.saturating_sub(baseline);
    let peak_delta = peak_during.saturating_sub(baseline);
    drop(value);
    // Drain again: dropping `value` itself frees nodes into the pool rather
    // than the system allocator, so without this the leak check below would
    // fail (or rather, would report a false "leak" of exactly the dropped
    // collection's own bytes, still parked for reuse).
    drain_pool();
    let after_drop = live_bytes();
    assert_eq!(
        after_drop, baseline,
        "leak detected: {} bytes still live after drop (post-drain)",
        after_drop.saturating_sub(baseline)
    );
    Measurement { live_delta, peak_delta, pool_retained }
}

// ---------------------------------------------------------------------
// Table printing
// ---------------------------------------------------------------------

struct Row {
    n: usize,
    impl_name: &'static str,
    live_bytes: usize,
    peak_bytes: usize,
    raw_kv_bytes: usize,
    /// Bytes freed by the post-build pool drain (see `measure`'s docs).
    /// Always `0` for `im`/`std` rows (they never touch champ's
    /// pool) and always `0` in a `--no-default-features` (pool-off) build.
    pool_retained: usize,
}

fn print_table(title: &str, rows: &[Row]) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!(
        "{:>10} {:<12} {:>14} {:>12} {:>14} {:>12} {:>10} {:>14}",
        "n", "impl", "live bytes", "B/entry", "peak bytes", "raw K,V B", "overhead", "pool-ret B"
    );
    println!("{}", "-".repeat(106));
    for r in rows {
        let per_entry = r.live_bytes as f64 / r.n.max(1) as f64;
        let overhead = if r.raw_kv_bytes > 0 {
            format!("{:+.0}%", (r.live_bytes as f64 / r.raw_kv_bytes as f64 - 1.0) * 100.0)
        } else {
            "n/a".to_string()
        };
        println!(
            "{:>10} {:<12} {:>14} {:>12.1} {:>14} {:>12} {:>10} {:>14}",
            r.n, r.impl_name, r.live_bytes, per_entry, r.peak_bytes, r.raw_kv_bytes, overhead, r.pool_retained
        );
    }
}

// ---------------------------------------------------------------------
// Per-key-type dataset builders
// ---------------------------------------------------------------------

fn champ_u64(n: usize) -> PersistentHashMap<u64, u64> {
    let mut t = PersistentHashMap::new().transient();
    for i in 0..n as u64 {
        t.assoc(i, i);
    }
    t.persistent()
}

fn im_u64(n: usize) -> ImHashMap<u64, u64> {
    let mut m = ImHashMap::new();
    for i in 0..n as u64 {
        m.insert(i, i);
    }
    m
}

fn std_u64(n: usize) -> StdHashMap<u64, u64> {
    let mut m = StdHashMap::new();
    for i in 0..n as u64 {
        m.insert(i, i);
    }
    m
}

fn champ_str(n: usize) -> PersistentHashMap<String, String> {
    let mut t = PersistentHashMap::new().transient();
    for i in 0..n {
        t.assoc(key16(i), key16(i));
    }
    t.persistent()
}

fn im_str(n: usize) -> ImHashMap<String, String> {
    let mut m = ImHashMap::new();
    for i in 0..n {
        m.insert(key16(i), key16(i));
    }
    m
}

fn std_str(n: usize) -> StdHashMap<String, String> {
    let mut m = StdHashMap::new();
    for i in 0..n {
        m.insert(key16(i), key16(i));
    }
    m
}

const SIZES: [usize; 7] = [1, 4, 8, 16, 100, 10_000, 100_000];

fn main() {
    // Sanity: baseline should be ~0 (or close to it — some allocator
    // bookkeeping from process startup may already be live) before any
    // scenario runs; every scenario below re-checks its own baseline is
    // restored after drop, which is the real leak check.
    println!("champ memstats — live heap bytes via a counting #[global_allocator]");
    println!("(process startup baseline: {} bytes live)", live_bytes());

    // -------------------------------------------------------------
    // u64 -> u64
    // -------------------------------------------------------------
    let mut rows = Vec::new();
    let raw_kv = std::mem::size_of::<(u64, u64)>();
    for &n in &SIZES {
        let m = measure(|| champ_u64(n));
        rows.push(Row {
            n,
            impl_name: "champ",
            live_bytes: m.live_delta,
            peak_bytes: m.peak_delta,
            raw_kv_bytes: n * raw_kv,
            pool_retained: m.pool_retained,
        });
        let m = measure(|| im_u64(n));
        rows.push(Row {
            n,
            impl_name: "im",
            live_bytes: m.live_delta,
            peak_bytes: m.peak_delta,
            raw_kv_bytes: n * raw_kv,
            pool_retained: m.pool_retained,
        });
        let m = measure(|| std_u64(n));
        rows.push(Row {
            n,
            impl_name: "std",
            live_bytes: m.live_delta,
            peak_bytes: m.peak_delta,
            raw_kv_bytes: n * raw_kv,
            pool_retained: m.pool_retained,
        });
    }
    print_table("u64 -> u64", &rows);

    // -------------------------------------------------------------
    // String(16B) -> String(16B)
    // -------------------------------------------------------------
    let mut rows = Vec::new();
    // String's own stack footprint (ptr+len+cap = 24B on 64-bit) plus its
    // 16-byte heap allocation is the fairest "raw" reference for a String
    // key/value pair; std/im/champ all pay this cost identically since
    // it's inherent to `String`, not the collection.
    let raw_kv = 2 * (std::mem::size_of::<String>() + 16);
    for &n in &SIZES {
        let m = measure(|| champ_str(n));
        rows.push(Row {
            n,
            impl_name: "champ",
            live_bytes: m.live_delta,
            peak_bytes: m.peak_delta,
            raw_kv_bytes: n * raw_kv,
            pool_retained: m.pool_retained,
        });
        let m = measure(|| im_str(n));
        rows.push(Row {
            n,
            impl_name: "im",
            live_bytes: m.live_delta,
            peak_bytes: m.peak_delta,
            raw_kv_bytes: n * raw_kv,
            pool_retained: m.pool_retained,
        });
        let m = measure(|| std_str(n));
        rows.push(Row {
            n,
            impl_name: "std",
            live_bytes: m.live_delta,
            peak_bytes: m.peak_delta,
            raw_kv_bytes: n * raw_kv,
            pool_retained: m.pool_retained,
        });
    }
    print_table("String(16B) -> String(16B)", &rows);

    // -------------------------------------------------------------
    // 100k independent 7-key (u64,u64) maps — mirrors the host's measured
    // ~8.7KB/map RSS pain point with imbl.
    // -------------------------------------------------------------
    const MAP_COUNT: usize = 100_000;
    const KEYS_PER_MAP: usize = 7;

    let m = measure(|| {
        let maps: Vec<PersistentHashMap<u64, u64>> = (0..MAP_COUNT)
            .map(|_| {
                let mut t = PersistentHashMap::new().transient();
                for k in 0..KEYS_PER_MAP as u64 {
                    t.assoc(k, k);
                }
                t.persistent()
            })
            .collect();
        maps
    });
    let champ_total = m.live_delta;
    let champ_peak = m.peak_delta;
    let champ_pool_retained = m.pool_retained;

    let m = measure(|| {
        let maps: Vec<ImHashMap<u64, u64>> = (0..MAP_COUNT)
            .map(|_| {
                let mut mm = ImHashMap::new();
                for k in 0..KEYS_PER_MAP as u64 {
                    mm.insert(k, k);
                }
                mm
            })
            .collect();
        maps
    });
    let im_total = m.live_delta;
    let im_peak = m.peak_delta;

    println!();
    let title = format!("100k independent {KEYS_PER_MAP}-key (u64,u64) maps");
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!(
        "{:>10} {:<12} {:>16} {:>14} {:>16}",
        "count", "impl", "total live B", "B/map", "peak live B"
    );
    println!("{}", "-".repeat(72));
    println!(
        "{:>10} {:<12} {:>16} {:>14.1} {:>16}",
        MAP_COUNT,
        "champ",
        champ_total,
        champ_total as f64 / MAP_COUNT as f64,
        champ_peak
    );
    println!(
        "{:>10} {:<12} {:>16} {:>14.1} {:>16}",
        MAP_COUNT,
        "im",
        im_total,
        im_total as f64 / MAP_COUNT as f64,
        im_peak
    );
    println!();
    println!(
        "the host's measured imbl baseline for this scenario: ~8.7 KB/map (RSS). champ: {:.2} KB/map (live heap bytes, not RSS).",
        champ_total as f64 / MAP_COUNT as f64 / 1024.0
    );
    println!(
        "pool-retained bytes (build-time churn recycled by the node pool, freed by the drain above; \
         0 in a --no-default-features build): {champ_pool_retained} total ({:.3} B/map)",
        champ_pool_retained as f64 / MAP_COUNT as f64
    );

    println!();
    println!("All scenarios' live-byte baseline was restored after drop (leak check passed).");
}
