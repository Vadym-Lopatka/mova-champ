//! Memory-per-element harness for `PVector` vs `imbl::Vector` vs plain
//! `Vec`, mirroring `examples/memstats.rs`'s counting-allocator pattern
//! (SPEC-M10-PVEC.md's success bars: <=9 B/elem overhead @240k dense,
//! <=120 B/elem @20-elem Big-tier for Value-sized payloads). Also reports
//! process RSS at each measurement point (via `ps`, macOS/Linux-portable
//! enough for this environment) alongside the exact counting-allocator
//! numbers, per the task's "report both" instruction — RSS includes
//! allocator fragmentation/slack the counting allocator's exact live-byte
//! count does not, so the two numbers together give the honest picture.
//!
//! Run with `cargo run --release --example vec_memstats`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use champ::PVector;

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
fn reset_peak() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

#[cfg(feature = "pool-alloc")]
fn drain_pool() {
    champ::pool_drain();
}
#[cfg(not(feature = "pool-alloc"))]
fn drain_pool() {}

/// Current process RSS in bytes, via `ps` (portable enough for macOS/
/// Linux without a `libc`/`sysinfo` dependency this crate would otherwise
/// never need).
fn rss_bytes() -> usize {
    let pid = std::process::id();
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().parse::<usize>().unwrap_or(0) * 1024, // ps reports KiB
        Err(_) => 0,
    }
}

struct Measurement {
    live_delta: usize,
    peak_delta: usize,
    rss_delta: usize,
}

fn measure<T>(build: impl FnOnce() -> T) -> (Measurement, T) {
    // A settle point before baselining: `ps` RSS reflects the OS's view,
    // which can lag/anticipate allocator behavior (freed heap pages
    // aren't always returned to the OS immediately) — baselining right
    // before `build()` and reading again right after is the best a
    // black-box `ps` probe can do; treat `rss_delta` as directional, not
    // as precise as the counting allocator's exact `live_delta`.
    let baseline_live = live_bytes();
    let baseline_rss = rss_bytes();
    reset_peak();
    let value = build();
    let peak_during = peak_bytes_helper();
    drain_pool();
    let after_build_live = live_bytes();
    let after_build_rss = rss_bytes();
    let live_delta = after_build_live.saturating_sub(baseline_live);
    let peak_delta = peak_during.saturating_sub(baseline_live);
    let rss_delta = after_build_rss.saturating_sub(baseline_rss);
    (Measurement { live_delta, peak_delta, rss_delta }, value)
}

fn peak_bytes_helper() -> usize {
    PEAK.load(Ordering::Relaxed)
}

struct Row {
    label: &'static str,
    n: usize,
    live_bytes: usize,
    peak_bytes: usize,
    rss_bytes: usize,
    payload_bytes: usize,
}

fn print_table(title: &str, rows: &[Row]) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!(
        "{:<22} {:>8} {:>14} {:>10} {:>14} {:>10} {:>14} {:>10}",
        "impl", "n", "live bytes", "B/elem", "peak bytes", "peak B/e", "rss delta B", "rss B/e"
    );
    println!("{}", "-".repeat(110));
    for r in rows {
        let per_elem = r.live_bytes as f64 / r.n.max(1) as f64;
        let peak_per = r.peak_bytes as f64 / r.n.max(1) as f64;
        let rss_per = r.rss_bytes as f64 / r.n.max(1) as f64;
        let overhead = per_elem - r.payload_bytes as f64;
        println!(
            "{:<22} {:>8} {:>14} {:>10.1} {:>14} {:>10.1} {:>14} {:>10.1}  (overhead over payload: {:+.1} B/elem)",
            r.label, r.n, r.live_bytes, per_elem, r.peak_bytes, peak_per, r.rss_bytes, rss_per, overhead
        );
    }
}

const BIG: usize = 240_000;
const SMALL: usize = 20;

fn main() {
    println!("champ PVector memstats (M10). pool-alloc feature: {}", cfg!(feature = "pool-alloc"));

    // ---- Dense i64, @240k: the "<=9 B/elem overhead" bar ----
    let mut rows_i64 = Vec::new();
    for &n in &[SMALL, BIG] {
        let items: Vec<i64> = (0..n as i64).collect();

        let (m, v) = measure(|| PVector::from_slice(&items));
        rows_i64.push(Row { label: "PVector_i64", n, live_bytes: m.live_delta, peak_bytes: m.peak_delta, rss_bytes: m.rss_delta, payload_bytes: 8 });
        drop(v);
        drain_pool();

        let (m, v) = measure(|| items.iter().copied().collect::<imbl::Vector<i64>>());
        rows_i64.push(Row { label: "imbl_i64", n, live_bytes: m.live_delta, peak_bytes: m.peak_delta, rss_bytes: m.rss_delta, payload_bytes: 8 });
        drop(v);

        let (m, v) = measure(|| items.clone());
        rows_i64.push(Row { label: "Vec_i64", n, live_bytes: m.live_delta, peak_bytes: m.peak_delta, rss_bytes: m.rss_delta, payload_bytes: 8 });
        drop(v);
    }
    print_table("Dense i64 (unboxed payload)", &rows_i64);

    // ---- Value-shaped Arc<i64>, @20 and @240k: the "<=120 B/elem @20"
    // Big-tier bar and a Value-payload cross-check at @240k. Arc<i64>'s
    // own heap box: 8 (strong) + 8 (weak) + 8 (payload) = 24 bytes each,
    // allocated once regardless of how many PVector/imbl copies share
    // the Arc (cloning an Arc bumps refcount, doesn't re-allocate) — so
    // `payload_bytes` below is 24 (the Arc box itself), matching what a
    // freshly-built, non-shared set of `n` distinct `Arc::new(i)` values
    // actually costs before any tree overhead.
    // Each row below builds its OWN fresh `Arc::new(i)` values *inside*
    // the measured closure (not shared/pre-built outside it) so the
    // measured `live_bytes` includes the Arc box cost every time — the
    // spec's own bar language ("83.1-83.9 B/elem = 72 boxed Value + 11.1
    // tree overhead") wants the full per-element cost including boxing,
    // not just the tree's marginal share of an already-boxed value.
    let mut rows_val = Vec::new();
    for &n in &[SMALL, BIG] {
        let (m, v) = measure(|| PVector::from_slice(&(0..n as i64).map(Arc::new).collect::<Vec<_>>()));
        rows_val.push(Row { label: "PVector_ArcValue", n, live_bytes: m.live_delta, peak_bytes: m.peak_delta, rss_bytes: m.rss_delta, payload_bytes: 24 });
        drop(v);
        drain_pool();

        let (m, v) = measure(|| (0..n as i64).map(Arc::new).collect::<imbl::Vector<Arc<i64>>>());
        rows_val.push(Row { label: "imbl_ArcValue", n, live_bytes: m.live_delta, peak_bytes: m.peak_delta, rss_bytes: m.rss_delta, payload_bytes: 24 });
        drop(v);

        let (m, v) = measure(|| (0..n as i64).map(Arc::new).collect::<Vec<_>>());
        rows_val.push(Row { label: "Vec_ArcValue", n, live_bytes: m.live_delta, peak_bytes: m.peak_delta, rss_bytes: m.rss_delta, payload_bytes: 24 });
        drop(v);
    }
    print_table("Value-shaped Arc<i64> (boxed/refcounted payload)", &rows_val);

    // ---- Queue churn memory bound (M12, SPEC-M12-SUFFIXVIEW.md): the
    // amortized-trim policy must keep a long-lived queue's resident
    // backing bounded to <= 2.5x the n-element baseline, not grow
    // unboundedly with the number of pop_front_owned/push_back_owned
    // rounds churned through it (the "drifting-queue leak" the policy
    // exists to close). Compares a fresh n-element vector's live bytes
    // against the SAME vector's live bytes after 10*n churn rounds.
    println!();
    println!("Queue churn memory bound (M12: amortized trim, <=2.5x n-element baseline)");
    println!("{}", "=".repeat(74));
    println!("{:<10} {:>14} {:>18} {:>10}", "n", "baseline B", "after 10n churn B", "ratio");
    println!("{}", "-".repeat(60));
    for &n in &[1_000usize, 20_000] {
        let items: Vec<i64> = (0..n as i64).collect();
        let (baseline_m, baseline_v) = measure(|| PVector::from_slice(&items));
        let baseline_live = baseline_m.live_delta;
        drop(baseline_v);
        drain_pool();

        let (churn_m, churned) = measure(|| {
            let mut v = PVector::from_slice(&items);
            for i in 0..(10 * n) {
                let (rest, _popped) = v.pop_front_owned();
                v = rest.push_back_owned(i as i64);
            }
            v
        });
        assert_eq!(churned.len(), n, "queue churn must preserve length");
        let ratio = churn_m.live_delta as f64 / baseline_live.max(1) as f64;
        println!("{n:<10} {baseline_live:>14} {:>18} {ratio:>9.2}x", churn_m.live_delta);
        drop(churned);
        drain_pool();
    }
}
