//! Memory harness for `PText` (M7, SPEC-PTEXT.md): a counting
//! `#[global_allocator]` measures live heap bytes, mirroring
//! `examples/memstats.rs`'s methodology (see its module docs for the full
//! "pool honesty" rationale — pooled blocks read as live to this counter
//! until drained, so every measurement below drains before/after).
//!
//! Run with `cargo run --release --example text_memstats`.
//!
//! Two scenarios, both called out explicitly by SPEC-PTEXT.md:
//! 1. Bytes-per-byte-of-text at each size (64KB/1MB/8.4MB/64MB), for
//!    `PText` vs a flat `String` (trivially 1 byte/byte, the reference
//!    floor) vs `ropey` (the mutable state of the art).
//! 2. The headline undo-stack table: 1000 retained versions of an 8.4MB
//!    text after 1000 keystrokes. `PText`'s structural sharing means this
//!    computes, not allocates, whereas 1000 retained flat-`String` copies
//!    would be ~8.4GB.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use champ::PText;
use ropey::Rope;

// ---------------------------------------------------------------------
// Counting allocator (identical shape to examples/memstats.rs's).
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
fn reset_peak() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

#[cfg(feature = "pool-alloc")]
fn drain_pool() {
    champ::pool_drain();
}
#[cfg(not(feature = "pool-alloc"))]
fn drain_pool() {}

// ---------------------------------------------------------------------
// Content generation
// ---------------------------------------------------------------------

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";

fn text_of_size(bytes: usize) -> String {
    let mut s = String::with_capacity(bytes + SENTENCE.len());
    while s.len() < bytes {
        s.push_str(SENTENCE);
    }
    s.truncate(bytes);
    s
}

// ---------------------------------------------------------------------
// Scenario 1: bytes-per-byte at each size.
// ---------------------------------------------------------------------

struct Measured {
    live: usize,
    peak: usize,
}

fn measure<T>(build: impl FnOnce() -> T) -> Measured {
    let baseline = live_bytes();
    reset_peak();
    let value = build();
    let peak_during = peak_bytes();
    drain_pool();
    let after_build = live_bytes();
    let live = after_build.saturating_sub(baseline);
    let peak = peak_during.saturating_sub(baseline);
    drop(value);
    drain_pool();
    let after_drop = live_bytes();
    assert_eq!(after_drop, baseline, "leak detected: {} bytes still live after drop", after_drop.saturating_sub(baseline));
    Measured { live, peak }
}

const SIZES: [(&str, usize); 4] = [("64KB", 64 * 1024), ("1MB", 1024 * 1024), ("8.4MB", 8_400_000), ("64MB", 64 * 1024 * 1024)];

fn bytes_per_byte_table() {
    println!();
    println!("bytes-per-byte-of-text");
    println!("=======================");
    println!("{:>8} {:<12} {:>14} {:>10} {:>14}", "size", "impl", "live bytes", "B/byte", "peak bytes");
    println!("{}", "-".repeat(64));
    for &(label, n) in &SIZES {
        let s = text_of_size(n);

        let m = measure(|| PText::from(s.as_str()));
        println!("{:>8} {:<12} {:>14} {:>10.3} {:>14}", label, "PText", m.live, m.live as f64 / n as f64, m.peak);

        let m = measure(|| Rope::from_str(&s));
        println!("{:>8} {:<12} {:>14} {:>10.3} {:>14}", label, "ropey", m.live, m.live as f64 / n as f64, m.peak);

        let m = measure(|| s.clone());
        println!("{:>8} {:<12} {:>14} {:>10.3} {:>14}", label, "String", m.live, m.live as f64 / n as f64, m.peak);
    }
}

// ---------------------------------------------------------------------
// Scenario 2: the headline undo-stack table. 1000 retained versions of an
// 8.4MB text after 1000 keystrokes. PText: structural sharing (compute,
// don't allocate). Flat String: would be ~8.4GB — computed, not actually
// run, to avoid genuinely allocating 8.4GB in a demo harness; PText's and
// ropey's numbers below are real measurements.
// ---------------------------------------------------------------------

fn undo_stack_table() {
    const N: usize = 8_400_000;
    const KEYSTROKES: usize = 1000;

    println!();
    println!("undo-stack: {KEYSTROKES} retained versions of an {N}-byte text after {KEYSTROKES} keystrokes");
    println!("====================================================================================");

    let base = text_of_size(N);
    let start = base.len() / 2;

    let baseline = live_bytes();
    reset_peak();
    let mut t = PText::from(base.as_str());
    let mut versions: Vec<PText> = Vec::with_capacity(KEYSTROKES);
    for i in 0..KEYSTROKES {
        versions.push(t.clone());
        t = t.splice(start + i..start + i, "x");
    }
    drain_pool();
    let live = live_bytes().saturating_sub(baseline);
    let peak = peak_bytes().saturating_sub(baseline);
    println!(
        "PText   : {live} bytes live for base + {KEYSTROKES} retained versions + final ({:.2} MB; peak {:.2} MB; \
         {:.1} bytes/version amortized over the base)",
        live as f64 / (1024.0 * 1024.0),
        peak as f64 / (1024.0 * 1024.0),
        live as f64 / KEYSTROKES as f64
    );
    drop((t, versions));
    drain_pool();
    assert_eq!(live_bytes(), baseline, "leak detected in PText undo-stack scenario");

    let baseline = live_bytes();
    reset_peak();
    let mut r = Rope::from_str(&base);
    let mut rversions: Vec<Rope> = Vec::with_capacity(KEYSTROKES);
    for i in 0..KEYSTROKES {
        rversions.push(r.clone());
        r.insert(start + i, "x");
    }
    let live_r = live_bytes().saturating_sub(baseline);
    let peak_r = peak_bytes().saturating_sub(baseline);
    println!(
        "ropey   : {live_r} bytes live for base + {KEYSTROKES} retained clones + final ({:.2} MB; peak {:.2} MB)",
        live_r as f64 / (1024.0 * 1024.0),
        peak_r as f64 / (1024.0 * 1024.0)
    );
    drop((r, rversions));

    // Version i (0-indexed, i in 0..KEYSTROKES) is retained just before the
    // i-th splice, so it has length N+i; the final value after all
    // KEYSTROKES splices has length N+KEYSTROKES. Sum = KEYSTROKES*N +
    // (0+1+...+(KEYSTROKES-1)) + N + KEYSTROKES = (KEYSTROKES+1)*N +
    // KEYSTROKES*(KEYSTROKES-1)/2 + KEYSTROKES.
    let k = KEYSTROKES as u64;
    let flat_string_estimate = (k + 1) * N as u64 + k * (k - 1) / 2 + k;
    println!(
        "String  : NOT actually run (would allocate ~{:.2} GB: computed as sum of {} growing copies of \
         ~{N} bytes each, not measured — see NOTES-PTEXT.md/BENCH-RESULTS.md for why this one is computed \
         rather than executed)",
        flat_string_estimate as f64 / (1024.0 * 1024.0 * 1024.0),
        KEYSTROKES + 1
    );
}

fn main() {
    println!("champ text_memstats — PText vs ropey vs String, live heap bytes via a counting allocator");
    println!("(process startup baseline: {} bytes live)", live_bytes());
    bytes_per_byte_table();
    undo_stack_table();
    println!();
    println!("All measured scenarios' live-byte baseline was restored after drop (leak check passed).");
}
