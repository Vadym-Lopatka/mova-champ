//! SPEC-M6-ALLOC.md Step 0: profile champ's own `build/transient` u64
//! n=10k loop to test the hypothesis that the a Clojure-dialect host build-loop regression
//! (`lastuse-assoc-10000` -26.2%, `reuse-assoc-10000` -13.7% vs imbl) is
//! caused by the malloc/free round-trip on every owned insert (exact-fit
//! nodes realloc on every growth), not by the copy/descent logic itself.
//!
//! Two independent measurements, both against the *build* phase only (the
//! final drop of all retained maps is deliberately left uninstrumented and
//! untimed, so it can't pollute the "cost of insert" story with "cost of
//! recursive teardown"):
//!
//! 1. **Allocator call counts** (exact, deterministic): a `#[global_allocator]`
//!    that counts every `alloc`/`dealloc` call and byte, so we get an exact
//!    allocs-per-insert and bytes-per-insert figure independent of timing
//!    noise.
//! 2. **Time spent inside `alloc`/`dealloc`** (wall-clock, `Instant`-based):
//!    the same allocator additionally accumulates `Instant::now()`-measured
//!    nanoseconds spent inside each `System::alloc`/`System::dealloc` call,
//!    compared against the wall-clock time of the whole build loop, to
//!    estimate the % of build-loop time attributable to the allocator path.
//!    This is the "at minimum a counting-allocator run" fallback named in
//!    SPEC-M6-ALLOC.md Step 0 (no `samply`/Instruments `xctrace` sampling
//!    profiler was available in this environment — see NOTES-M6.md for what
//!    was tried).
//!
//! Run with `cargo run --release --example profile_build`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use champ::PersistentHashMap;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_NS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_NS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

struct ProfilingAllocator;

unsafe impl GlobalAlloc for ProfilingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let t0 = Instant::now();
        let ptr = unsafe { System.alloc(layout) };
        let dt = t0.elapsed().as_nanos() as u64;
        ALLOC_NS.fetch_add(dt, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let t0 = Instant::now();
        let ptr = unsafe { System.alloc_zeroed(layout) };
        let dt = t0.elapsed().as_nanos() as u64;
        ALLOC_NS.fetch_add(dt, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let t0 = Instant::now();
        unsafe { System.dealloc(ptr, layout) };
        let dt = t0.elapsed().as_nanos() as u64;
        DEALLOC_NS.fetch_add(dt, Ordering::Relaxed);
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // node.rs never actually calls GlobalAlloc::realloc (it does its own
        // alloc-copy-dealloc for capacity changes), but forward correctly
        // and instrument it as an alloc+dealloc pair in case that changes.
        let t0 = Instant::now();
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        let dt = t0.elapsed().as_nanos() as u64;
        ALLOC_NS.fetch_add(dt, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        ptr
    }
}

#[global_allocator]
static ALLOC: ProfilingAllocator = ProfilingAllocator;

fn reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_NS.store(0, Ordering::Relaxed);
    DEALLOC_NS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn build(n: u64) -> PersistentHashMap<u64, u64> {
    let mut t = PersistentHashMap::new().transient();
    for i in 0..n {
        t.assoc(i, i);
    }
    t.persistent()
}

fn main() {
    const N: u64 = 10_000;
    const ROUNDS: usize = 200;

    // Warmup: also pre-faults the allocator's internal structures so the
    // first *measured* round isn't paying one-time OS/allocator setup cost.
    for _ in 0..5 {
        let m = build(N);
        std::hint::black_box(&m);
        drop(m);
    }

    // Measured rounds: time + instrument ONLY the build loop. Retain every
    // built map in a Vec so the (expensive, recursive, allocator-heavy)
    // teardown happens *after* the timed region, not during it — otherwise
    // "cost of insert" and "cost of drop" would be conflated.
    let mut maps: Vec<PersistentHashMap<u64, u64>> = Vec::with_capacity(ROUNDS);
    reset();
    let wall0 = Instant::now();
    for _ in 0..ROUNDS {
        let m = build(N);
        maps.push(std::hint::black_box(m));
    }
    let wall = wall0.elapsed();

    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_ns = ALLOC_NS.load(Ordering::Relaxed);
    let dealloc_ns = DEALLOC_NS.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    let total_inserts = ROUNDS as u64 * N;
    let alloc_path_ns = alloc_ns + dealloc_ns;
    let wall_ns = wall.as_nanos() as u64;

    println!("champ Step-0 profile: build/transient u64, n={N}, {ROUNDS} rounds ({total_inserts} total inserts)");
    println!("(drop of all {ROUNDS} retained maps happens AFTER this measurement, untimed/uninstrumented)");
    println!();
    println!("wall time (build only):     {wall:?}  ({:.2} ns/insert)", wall_ns as f64 / total_inserts as f64);
    println!();
    println!("allocator calls:");
    println!("  alloc()/alloc_zeroed()/realloc(): {allocs} calls ({:.4}/insert)", allocs as f64 / total_inserts as f64);
    println!("  dealloc():                        {deallocs} calls ({:.4}/insert)", deallocs as f64 / total_inserts as f64);
    println!(
        "  total alloc+dealloc pairs:        {} calls ({:.4}/insert)",
        allocs + deallocs,
        (allocs + deallocs) as f64 / total_inserts as f64
    );
    println!("  bytes allocated:                  {bytes} ({:.1} B/insert)", bytes as f64 / total_inserts as f64);
    println!();
    println!("time spent inside the global allocator (Instant-measured, includes call overhead):");
    println!(
        "  alloc() time:    {:>10?}  ({:.1}% of wall)",
        std::time::Duration::from_nanos(alloc_ns),
        100.0 * alloc_ns as f64 / wall_ns as f64
    );
    println!(
        "  dealloc() time:  {:>10?}  ({:.1}% of wall)",
        std::time::Duration::from_nanos(dealloc_ns),
        100.0 * dealloc_ns as f64 / wall_ns as f64
    );
    println!(
        "  alloc+dealloc:   {:>10?}  ({:.1}% of wall) <-- Step 0's headline number",
        std::time::Duration::from_nanos(alloc_path_ns),
        100.0 * alloc_path_ns as f64 / wall_ns as f64
    );
    println!();
    println!(
        "verdict: {}",
        if 100.0 * alloc_path_ns as f64 / wall_ns as f64 >= 20.0 {
            "alloc path >= ~20% of build-loop time -> hypothesis supported, proceed to build the pool (Step 1)."
        } else {
            "alloc path < ~20% of build-loop time -> STOP per spec; do not build the pool."
        }
    );

    // Keep `maps` alive until here (all measurement is done), then let it
    // drop normally at end of `main` — untimed.
    std::hint::black_box(&maps);
}
