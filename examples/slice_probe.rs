//! Standalone probe (NOT a criterion bench, NOT committed to the bench
//! suite): isolate the per-op cost of `PText`'s persistent (shared-tree)
//! read/slice/concat paths at ~8.5MB scale, vs. ropey, to confirm the
//! hypothesis that `PText::slice` costs ~50-70us/op dominated by
//! `split_at`'s per-level `concat_many` sibling rebuilds on the *shared*
//! path (i.e. when the document's root is reachable through more than one
//! `PText` handle, as it always is in a real editor: `doc.clone()` is kept
//! alive for the whole run below to force that path).
//!
//! Run with: `cargo run --release --bin slice_probe`

use std::hint::black_box;
use std::time::{Duration, Instant};

use champ::PText;
use ropey::Rope;

// ---------------------------------------------------------------------
// Deterministic ~8.5MB ASCII text: 240_000 lines, length 10-120 chars,
// seeded LCG (no external corpus).
// ---------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes LCG constants.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
}

const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"-()[]{}!?\n";

fn build_text(lines: usize, seed: u64) -> String {
    let mut lcg = Lcg(seed);
    // rough capacity estimate: avg line length ~65 chars
    let mut s = String::with_capacity(lines * 66);
    for _ in 0..lines {
        let len = 10 + lcg.below(111); // 10..=120
        for _ in 0..len {
            // keep it printable ASCII, not '\n' mid-line
            let idx = lcg.below(CHARSET.len() - 1); // exclude trailing '\n' entry
            s.push(CHARSET[idx] as char);
        }
        s.push('\n');
    }
    s
}

fn fmt_ns(total: Duration, ops: usize) -> f64 {
    total.as_secs_f64() * 1e9 / ops as f64
}

fn main() {
    // 240_000 lines x avg ~66 bytes/line (10-120 char body + '\n') lands at
    // ~15.8MB, not ~8.5MB -- scale the line count down to hit the ~8.5MB
    // target scale the probe is actually asked to measure at.
    const LINES: usize = 129_000;
    println!("Building ~8.5MB deterministic text ({LINES} lines, seeded LCG)...");
    let t0 = Instant::now();
    let text = build_text(LINES, 0x00C0_FFEE_1234_5678);
    println!("  text.len() = {} bytes, built in {:?}", text.len(), t0.elapsed());

    let doc = PText::from(text.as_str());
    // Keep a clone alive for the ENTIRE run so the tree is always reached
    // through a shared root -- this is the realistic editor case where the
    // document `Str`/history is always shared, forcing `split_at`'s
    // copy/extraction path rather than any unique-ownership fast path.
    let _shared = doc.clone();

    let rope = Rope::from_str(&text);

    let n_chars = doc.len_chars();
    println!("  doc.len_chars() = {}, doc.len_bytes() = {}", n_chars, doc.len_bytes());
    assert_eq!(n_chars, rope.len_chars(), "char count mismatch between PText and ropey");

    println!();
    println!("=== (a) doc.slice(i..i+100) x 10_000, LCG-spread offsets ===");
    bench_slice(&doc, &rope, n_chars);

    println!();
    println!("=== (a') doc.range_to_string(i..i+100) x 10_000, LCG-spread offsets (M9.1) ===");
    bench_range_to_string(&doc, &rope, n_chars);

    println!();
    println!("=== (b) doc.byte_of_char(i) x 100_000, LCG-spread offsets ===");
    bench_byte_of_char(&doc, &rope, n_chars);

    println!();
    println!("=== (c) doc.slice(i..i) (empty range) x 10_000 ===");
    bench_empty_slice(&doc, &rope, n_chars);

    println!();
    println!("=== (d) concat fold: 1_000 x ~100-byte pieces ===");
    bench_concat_fold();

    println!();
    println!("=== (d') concat fold: 1_000 x rope-slice pieces (doc.slice results) ===");
    bench_concat_fold_slice_pieces(&doc, &rope, n_chars);

    println!();
    println!("=== (e) doc.splice(i..i+1, \"x\") x 1_000, shared-tree ===");
    bench_splice_shared(&doc, &rope, n_chars);

    println!();
    println!("=== optional: long-running slice loop for `sample` profiling ===");
    println!("(set SLICE_PROBE_PROFILE_LOOP=1 to run a ~5s slice loop after the numbers above)");
    if std::env::var("SLICE_PROBE_PROFILE_LOOP").is_ok() {
        profile_loop(&doc, n_chars);
    }

    // keep `_shared` alive to the very end so nothing gets dropped/uniqued
    // early by the optimizer reasoning about lifetimes.
    black_box(&_shared);
}

// ---------------------------------------------------------------------
// LCG offset generator: spread across [0, n - width], deterministic.
// ---------------------------------------------------------------------

fn spread_offsets(count: usize, n: usize, width: usize, seed: u64) -> Vec<usize> {
    let mut lcg = Lcg(seed);
    let span = n.saturating_sub(width).max(1);
    (0..count).map(|_| lcg.below(span)).collect()
}

fn bench_slice(doc: &PText, rope: &Rope, n_chars: usize) {
    const REPS: usize = 10_000;
    let offsets = spread_offsets(REPS, n_chars, 100, 0xA1);

    // --- warmup ---
    for &i in offsets.iter().take(1000) {
        black_box(doc.slice(i..i + 100));
    }

    // --- PText: doc.slice(i..i+100) ---
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(doc.slice(i..i + 100));
    }
    let elapsed = t0.elapsed();
    println!("  PText  doc.slice(i..i+100)         : {:>10.1} ns/op  (total {:?}, {} ops)", fmt_ns(elapsed, REPS), elapsed, REPS);

    // --- ropey: rope.slice(i..i+100) -- lazy view, just constructing the RopeSlice ---
    for &i in offsets.iter().take(1000) {
        black_box(rope.slice(i..i + 100));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(rope.slice(i..i + 100));
    }
    let elapsed = t0.elapsed();
    println!("  ropey  rope.slice(i..i+100) [lazy view, not materialized]: {:>10.1} ns/op  (total {:?})", fmt_ns(elapsed, REPS), elapsed);

    // --- ropey: Rope::from(rope.slice(..)) -- actually materializing an owned Rope ---
    for &i in offsets.iter().take(1000) {
        black_box(Rope::from(rope.slice(i..i + 100)));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(Rope::from(rope.slice(i..i + 100)));
    }
    let elapsed = t0.elapsed();
    println!("  ropey  Rope::from(rope.slice(i..i+100)) [materialized Rope]: {:>10.1} ns/op  (total {:?})", fmt_ns(elapsed, REPS), elapsed);

    // --- ropey: "what the user actually needs" -- extract 100 chars into a String ---
    for &i in offsets.iter().take(1000) {
        let s: String = rope.slice(i..i + 100).chars().collect();
        black_box(s);
    }
    let t0 = Instant::now();
    for &i in &offsets {
        let s: String = rope.slice(i..i + 100).chars().collect();
        black_box(s);
    }
    let elapsed = t0.elapsed();
    println!("  ropey  rope.slice(i..i+100).chars().collect::<String>() [materialized String]: {:>10.1} ns/op  (total {:?})", fmt_ns(elapsed, REPS), elapsed);
}

/// M9.1 (SPEC-M9-SPLITAT.md addendum): `doc.range_to_string(i..i+100)`,
/// the direct-read alternative to `doc.slice(i..i+100)` above -- no tree
/// building at all, just a summary-guided descent plus a memcpy of the
/// range into a fresh `String`. This is the dominant real client shape
/// (the host's `subs` over a visible editor line): the caller was always
/// going to flatten the `PText` result into a small flat string anyway,
/// so skip building one.
fn bench_range_to_string(doc: &PText, rope: &Rope, n_chars: usize) {
    const REPS: usize = 10_000;
    let offsets = spread_offsets(REPS, n_chars, 100, 0xA1);

    // --- warmup ---
    for &i in offsets.iter().take(1000) {
        black_box(doc.range_to_string(i..i + 100));
    }

    // --- PText: doc.range_to_string(i..i+100) ---
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(doc.range_to_string(i..i + 100));
    }
    let elapsed = t0.elapsed();
    println!(
        "  PText  doc.range_to_string(i..i+100) [direct read, no tree build]: {:>10.1} ns/op  (total {:?}, {} ops)",
        fmt_ns(elapsed, REPS),
        elapsed,
        REPS
    );

    // --- ropey comparison, for scale reference (same materialized-String shape as (a)'s last row) ---
    for &i in offsets.iter().take(1000) {
        let s: String = rope.slice(i..i + 100).chars().collect();
        black_box(s);
    }
    let t0 = Instant::now();
    for &i in &offsets {
        let s: String = rope.slice(i..i + 100).chars().collect();
        black_box(s);
    }
    let elapsed = t0.elapsed();
    println!(
        "  ropey  rope.slice(i..i+100).chars().collect::<String>() [for scale reference]: {:>10.1} ns/op  (total {:?})",
        fmt_ns(elapsed, REPS),
        elapsed
    );
}

fn bench_byte_of_char(doc: &PText, rope: &Rope, n_chars: usize) {
    const REPS: usize = 100_000;
    let offsets = spread_offsets(REPS, n_chars, 0, 0xB2);

    for &i in offsets.iter().take(2000) {
        black_box(doc.byte_of_char(i));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(doc.byte_of_char(i));
    }
    let elapsed = t0.elapsed();
    println!("  PText  doc.byte_of_char(i)          : {:>10.1} ns/op  (total {:?}, {} ops)", fmt_ns(elapsed, REPS), elapsed, REPS);

    // ropey equivalent: char_to_byte
    for &i in offsets.iter().take(2000) {
        black_box(rope.char_to_byte(i));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(rope.char_to_byte(i));
    }
    let elapsed = t0.elapsed();
    println!("  ropey  rope.char_to_byte(i)         : {:>10.1} ns/op  (total {:?})", fmt_ns(elapsed, REPS), elapsed);
}

fn bench_empty_slice(doc: &PText, rope: &Rope, n_chars: usize) {
    const REPS: usize = 10_000;
    let offsets = spread_offsets(REPS, n_chars, 0, 0xC3);

    for &i in offsets.iter().take(1000) {
        black_box(doc.slice(i..i));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(doc.slice(i..i));
    }
    let elapsed = t0.elapsed();
    println!("  PText  doc.slice(i..i) [empty range, 2x split_at, no middle]: {:>10.1} ns/op  (total {:?}, {} ops)", fmt_ns(elapsed, REPS), elapsed, REPS);

    // ropey comparison for context (not the point of (c), but cheap to include)
    for &i in offsets.iter().take(1000) {
        black_box(rope.slice(i..i));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(rope.slice(i..i));
    }
    let elapsed = t0.elapsed();
    println!("  ropey  rope.slice(i..i) [lazy, for context]: {:>10.1} ns/op  (total {:?})", fmt_ns(elapsed, REPS), elapsed);
}

fn make_piece_text(i: usize) -> String {
    // ~100 bytes, deterministic, varies per index so it's not all identical.
    format!("piece-{i:06}-0123456789abcdefghijklmnopqrstuvwxyz-0123456789abcdefghijklmnopqrstuvwxyz-end")
}

fn bench_concat_fold() {
    const N: usize = 1_000;

    // --- PText::concat fold ---
    // warmup
    {
        let pieces: Vec<PText> = (0..N).map(|i| PText::from(make_piece_text(i).as_str())).collect();
        let result = pieces.into_iter().fold(PText::new(), PText::concat);
        black_box(result);
    }
    let pieces: Vec<PText> = (0..N).map(|i| PText::from(make_piece_text(i).as_str())).collect();
    let t0 = Instant::now();
    let result = pieces.into_iter().fold(PText::new(), PText::concat);
    let elapsed = t0.elapsed();
    black_box(&result);
    println!(
        "  PText  fold(1_000 x ~100B pieces, PText::concat): total {:?}, {:.1} ns/op ({} folds)",
        elapsed,
        fmt_ns(elapsed, N),
        N
    );
    assert_eq!(result.len_bytes(), (0..N).map(|i| make_piece_text(i).len()).sum::<usize>());

    // --- ropey: rope.append(Rope::from_str(piece)) on an accumulating rope ---
    {
        let mut acc = Rope::new();
        for i in 0..N {
            acc.append(Rope::from_str(&make_piece_text(i)));
        }
        black_box(&acc);
    }
    let mut acc = Rope::new();
    let t0 = Instant::now();
    for i in 0..N {
        acc.append(Rope::from_str(&make_piece_text(i)));
    }
    let elapsed = t0.elapsed();
    black_box(&acc);
    println!("  ropey  1_000 x rope.append(Rope::from_str(piece)): total {:?}, {:.1} ns/op", elapsed, fmt_ns(elapsed, N));
}

/// The "rope-piece concat" variant: each piece is itself a `doc.slice(i..i+100)`
/// result -- this mirrors the host's `apply str`, which folds together slices
/// pulled out of the ratchet rather than freshly-built leaf pieces.
fn bench_concat_fold_slice_pieces(doc: &PText, rope: &Rope, n_chars: usize) {
    const N: usize = 1_000;
    let offsets = spread_offsets(N, n_chars, 100, 0xD4);

    // warmup
    {
        let pieces: Vec<PText> = offsets.iter().map(|&i| doc.slice(i..i + 100)).collect();
        let result = pieces.into_iter().fold(PText::new(), PText::concat);
        black_box(result);
    }
    let pieces: Vec<PText> = offsets.iter().map(|&i| doc.slice(i..i + 100)).collect();
    let t0 = Instant::now();
    let result = pieces.into_iter().fold(PText::new(), PText::concat);
    let elapsed = t0.elapsed();
    black_box(&result);
    println!(
        "  PText  fold(1_000 x doc.slice(i..i+100) pieces, PText::concat): total {:?}, {:.1} ns/op",
        elapsed,
        fmt_ns(elapsed, N)
    );

    // ropey equivalent: accumulate rope.slice pieces via append(Rope::from(slice))
    {
        let mut acc = Rope::new();
        for &i in &offsets {
            acc.append(Rope::from(rope.slice(i..i + 100)));
        }
        black_box(&acc);
    }
    let mut acc = Rope::new();
    let t0 = Instant::now();
    for &i in &offsets {
        acc.append(Rope::from(rope.slice(i..i + 100)));
    }
    let elapsed = t0.elapsed();
    black_box(&acc);
    println!(
        "  ropey  1_000 x rope.append(Rope::from(rope.slice(i..i+100))) [rope-piece concat]: total {:?}, {:.1} ns/op",
        elapsed,
        fmt_ns(elapsed, N)
    );
}

fn bench_splice_shared(doc: &PText, rope: &Rope, n_chars: usize) {
    const REPS: usize = 1_000;
    let offsets = spread_offsets(REPS, n_chars, 1, 0xE5);

    // warmup -- doesn't mutate `doc` (splice is persistent, returns a new
    // PText each time; `doc` and the kept-alive `_shared` clone in `main`
    // remain the shared root for every call).
    for &i in offsets.iter().take(100) {
        black_box(doc.splice(i..i + 1, "x"));
    }
    let t0 = Instant::now();
    for &i in &offsets {
        black_box(doc.splice(i..i + 1, "x"));
    }
    let elapsed = t0.elapsed();
    println!("  PText  doc.splice(i..i+1, \"x\") [persistent, shared-tree]: {:>10.1} ns/op  (total {:?}, {} ops)", fmt_ns(elapsed, REPS), elapsed, REPS);

    // ropey comparison: mutable in-place remove+insert on a scratch clone
    // (ropey has no persistent-splice equivalent; this is in-place edit
    // cost on a throwaway clone per op, included for scale reference only).
    for &i in offsets.iter().take(100) {
        let mut r = rope.clone();
        r.remove(i..i + 1);
        r.insert(i, "x");
        black_box(r);
    }
    let t0 = Instant::now();
    for &i in &offsets {
        let mut r = rope.clone();
        r.remove(i..i + 1);
        r.insert(i, "x");
        black_box(r);
    }
    let elapsed = t0.elapsed();
    println!(
        "  ropey  clone+remove+insert(i, \"x\") [for scale reference only, not apples-to-apples]: {:>10.1} ns/op  (total {:?})",
        fmt_ns(elapsed, REPS),
        elapsed
    );
}

fn profile_loop(doc: &PText, n_chars: usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut lcg = Lcg(0xF00D);
    let span = n_chars.saturating_sub(100).max(1);
    let mut count = 0usize;
    println!("  running slice loop for ~5s (pid = {}), attach `sample` now...", std::process::id());
    while Instant::now() < deadline {
        let i = lcg.below(span);
        black_box(doc.slice(i..i + 100));
        count += 1;
    }
    println!("  profile loop done: {} slices", count);
}
