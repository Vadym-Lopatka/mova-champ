//! Criterion suite for `PText` vs `String` (the incumbent — what an editor
//! naively does today) and `ropey` (the mutable state-of-the-art rope, our
//! speed ceiling). See SPEC-PTEXT.md's "Benchmarks" section for the
//! scenario matrix and success bars, and BENCH-RESULTS.md's "M7: PText"
//! section for the analysis of these numbers.
//!
//! Under the `spike-ropey` feature (see src/text_ropey.rs), every group
//! also gets a `RopeyText_*` case: the same scenario driven through
//! `RopeyText`'s API (an `Arc<ropey::Rope>` wearing `PText`'s exact public
//! API) instead of raw `ropey::Rope` calls, so the `Arc`-wrapping/
//! `make_mut` overhead PText-shaped callers would actually pay is visible
//! alongside the `ropey`/`PText_owned` baselines already here.
//!
//! Sizes: 64KB, 1MB, 8.4MB (the measured motivating editor-file size, see
//! SPEC-PTEXT.md's opening), 64MB. Base content is a repeating
//! human-readable sentence (ASCII, deliberately — the timing scenarios
//! below care about structural cost, not UTF-8 decoding cost; multi-byte
//! correctness has its own dedicated coverage in tests/text_model.rs) so
//! byte and char offsets coincide, keeping the benchmark code itself simple
//! and honest about what it's measuring.

use std::time::Duration;

use champ::PText;
#[cfg(feature = "spike-ropey")]
use champ::RopeyText;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ropey::Rope;

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";

fn text_of_size(bytes: usize) -> String {
    let mut s = String::with_capacity(bytes + SENTENCE.len());
    while s.len() < bytes {
        s.push_str(SENTENCE);
    }
    s.truncate(bytes);
    // Keep it a valid char boundary (SENTENCE is pure ASCII, so any byte
    // offset already is one, but truncate() would panic otherwise).
    s
}

const SIZES: [(&str, usize); 4] = [("64KB", 64 * 1024), ("1MB", 1024 * 1024), ("8.4MB", 8_400_000), ("64MB", 64 * 1024 * 1024)];

/// Small deterministic PRNG (xorshift32) — avoids pulling in a `rand`
/// dependency for benchmark-only pseudo-randomness; determinism also makes
/// runs reproducible/comparable across competitors.
struct Xorshift32(u32);
impl Xorshift32 {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}

const OPS: usize = 1000;

// ---------------------------------------------------------------------
// 1. Sequential typing: OPS single-char inserts at a caret moving forward
//    through the middle of the document (the everyday keystroke — owned
//    path for PText, in-place `insert` for ropey/String).
// ---------------------------------------------------------------------

/// Prime the leaf(s) around `start` with capacity slack by inserting then
/// immediately deleting a byte a few hundred times before the timed loop
/// starts. Without this, a freshly bulk-built `PText` (or a freshly
/// `Rope::from_str`'d ropey rope) has *zero* slack anywhere — every leaf is
/// exactly full from bulk construction — so literally the *first* keystroke
/// of a timed run forces a full leaf split (and the next several force
/// leaf-capacity regrows) purely as an artifact of never having been edited
/// before, not as a property of "typing speed". That one-time cost is real
/// (see BENCH-RESULTS.md's honest cold-start note) but it isn't what "1-2ms
/// per keystroke today, microseconds with PText" is about — that claim is
/// about *sustained* per-keystroke cost during an editing session, which is
/// what this warm-up (untimed — it runs inside `iter_batched`'s `setup`)
/// puts the benchmark into before the clock starts.
fn warm_up_pt(mut t: PText, at: usize) -> PText {
    for _ in 0..300 {
        t = t.splice_owned(at..at, "x");
        t = t.splice_owned(at..at + 1, "");
    }
    t
}

fn warm_up_ropey(mut r: Rope, at: usize) -> Rope {
    for _ in 0..300 {
        r.insert(at, "x");
        r.remove(at..at + 1);
    }
    r
}

/// SPIKE (see src/text_ropey.rs): same warm-up discipline as `warm_up_pt`,
/// through `RopeyText`'s owned (`splice_owned`) API-level path rather than
/// raw `ropey::Rope` calls.
#[cfg(feature = "spike-ropey")]
fn warm_up_rt(mut t: RopeyText, at: usize) -> RopeyText {
    for _ in 0..300 {
        t = t.splice_owned(at..at, "x");
        t = t.splice_owned(at..at + 1, "");
    }
    t
}

fn bench_sequential_typing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_typing");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        let base = text_of_size(bytes);
        let start = base.len() / 2;

        group.bench_with_input(BenchmarkId::new("PText_owned", label), &base, |b, base| {
            b.iter_batched(
                || warm_up_pt(PText::from(base.as_str()), start),
                |mut t| {
                    for i in 0..OPS {
                        t = t.splice_owned(start + i..start + i, "x");
                    }
                    black_box(t)
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("ropey", label), &base, |b, base| {
            b.iter_batched(
                || warm_up_ropey(Rope::from_str(base), start),
                |mut r| {
                    for i in 0..OPS {
                        r.insert(start + i, "x");
                    }
                    black_box(r)
                },
                BatchSize::LargeInput,
            )
        });

        #[cfg(feature = "spike-ropey")]
        group.bench_with_input(BenchmarkId::new("RopeyText_owned", label), &base, |b, base| {
            b.iter_batched(
                || warm_up_rt(RopeyText::from(base.as_str()), start),
                |mut t| {
                    for i in 0..OPS {
                        t = t.splice_owned(start + i..start + i, "x");
                    }
                    black_box(t)
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("String", label), &base, |b, base| {
            b.iter_batched(
                || base.clone(),
                |mut s| {
                    for i in 0..OPS {
                        s.insert(start + i, 'x');
                    }
                    black_box(s)
                },
                BatchSize::LargeInput,
            )
        });
    }
}

/// Same as `bench_sequential_typing`'s `PText_owned`/`ropey` cases but
/// *without* the warm-up: the honest "first 1000 keystrokes into a
/// just-loaded, never-before-edited file" number, reported separately so
/// BENCH-RESULTS.md can show both instead of picking one. See
/// `warm_up_pt`'s doc comment for why the two numbers differ.
fn bench_sequential_typing_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_typing_cold_start");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        let base = text_of_size(bytes);
        let start = base.len() / 2;

        group.bench_with_input(BenchmarkId::new("PText_owned", label), &base, |b, base| {
            b.iter_batched(
                || PText::from(base.as_str()),
                |mut t| {
                    for i in 0..OPS {
                        t = t.splice_owned(start + i..start + i, "x");
                    }
                    black_box(t)
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("ropey", label), &base, |b, base| {
            b.iter_batched(
                || Rope::from_str(base),
                |mut r| {
                    for i in 0..OPS {
                        r.insert(start + i, "x");
                    }
                    black_box(r)
                },
                BatchSize::LargeInput,
            )
        });

        #[cfg(feature = "spike-ropey")]
        group.bench_with_input(BenchmarkId::new("RopeyText_owned", label), &base, |b, base| {
            b.iter_batched(
                || RopeyText::from(base.as_str()),
                |mut t| {
                    for i in 0..OPS {
                        t = t.splice_owned(start + i..start + i, "x");
                    }
                    black_box(t)
                },
                BatchSize::LargeInput,
            )
        });
    }
}

// ---------------------------------------------------------------------
// 2. Random-position splices (OPS x small delete+insert).
// ---------------------------------------------------------------------

fn bench_random_splices(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_position_splices");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        let base = text_of_size(bytes);

        group.bench_with_input(BenchmarkId::new("PText_owned", label), &base, |b, base| {
            b.iter_batched(
                || PText::from(base.as_str()),
                |mut t| {
                    let mut rng = Xorshift32(0xC0FFEE);
                    for _ in 0..OPS {
                        let n = t.len_chars();
                        let pos = rng.below(n.saturating_sub(4) + 1);
                        let del = rng.below(4).min(n - pos);
                        t = t.splice_owned(pos..pos + del, "abc");
                    }
                    black_box(t)
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("ropey", label), &base, |b, base| {
            b.iter_batched(
                || Rope::from_str(base),
                |mut r| {
                    let mut rng = Xorshift32(0xC0FFEE);
                    for _ in 0..OPS {
                        let n = r.len_chars();
                        let pos = rng.below(n.saturating_sub(4) + 1);
                        let del = rng.below(4).min(n - pos);
                        r.remove(pos..pos + del);
                        r.insert(pos, "abc");
                    }
                    black_box(r)
                },
                BatchSize::LargeInput,
            )
        });

        #[cfg(feature = "spike-ropey")]
        group.bench_with_input(BenchmarkId::new("RopeyText_owned", label), &base, |b, base| {
            b.iter_batched(
                || RopeyText::from(base.as_str()),
                |mut t| {
                    let mut rng = Xorshift32(0xC0FFEE);
                    for _ in 0..OPS {
                        let n = t.len_chars();
                        let pos = rng.below(n.saturating_sub(4) + 1);
                        let del = rng.below(4).min(n - pos);
                        t = t.splice_owned(pos..pos + del, "abc");
                    }
                    black_box(t)
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("String", label), &base, |b, base| {
            b.iter_batched(
                || base.clone(),
                |mut s| {
                    let mut rng = Xorshift32(0xC0FFEE);
                    for _ in 0..OPS {
                        let n = s.len();
                        let pos = rng.below(n.saturating_sub(4) + 1);
                        let del = rng.below(4).min(n - pos);
                        s.replace_range(pos..pos + del, "abc");
                    }
                    black_box(s)
                },
                BatchSize::LargeInput,
            )
        });
    }
}

// ---------------------------------------------------------------------
// 3. Undo-pattern: splice with the PREVIOUS version retained (persistent
//    path for PText; ropey has no native "keep the old version" op, so its
//    fair equivalent is clone-then-mutate — see SPEC-PTEXT.md).
// ---------------------------------------------------------------------

/// Retaining `OPS` full-size versions of a large text is exactly the point
/// of this scenario for `PText` (structural sharing makes it cheap) and
/// `ropey` (Rc-shared chunks make its clone cheap too), but `String`'s
/// retained versions are genuine independent heap copies — `OPS` of them at
/// 64MB each would be tens of GB *simultaneously resident* (all `OPS`
/// versions stay alive together until the sample's iteration finishes),
/// enough to swap or OOM a dev machine. Cap retained-version count per
/// sample so no size exceeds ~1.5GB of cumulative String bytes; still
/// enough versions at every size to see the real trend (1000 at 64KB/1MB,
/// down to ~23 at 64MB) without being a memory-safety hazard to run this
/// suite. Reported op count is called out per-size in BENCH-RESULTS.md
/// since it isn't the same `OPS` as every other scenario.
fn undo_ops_for_size(bytes: usize) -> usize {
    (1_500_000_000 / bytes.max(1)).clamp(10, OPS)
}

fn bench_undo_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("undo_pattern_retained");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        let base = text_of_size(bytes);
        let start = base.len() / 2;
        let ops = undo_ops_for_size(bytes);

        group.bench_with_input(BenchmarkId::new("PText_persistent", label), &base, |b, base| {
            b.iter_batched(
                || PText::from(base.as_str()),
                |mut t| {
                    let mut versions = Vec::with_capacity(ops);
                    for i in 0..ops {
                        versions.push(t.clone());
                        t = t.splice(start + i..start + i, "x");
                    }
                    black_box((t, versions))
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("ropey_clone_then_mutate", label), &base, |b, base| {
            b.iter_batched(
                || Rope::from_str(base),
                |mut r| {
                    let mut versions = Vec::with_capacity(ops);
                    for i in 0..ops {
                        versions.push(r.clone());
                        r.insert(start + i, "x");
                    }
                    black_box((r, versions))
                },
                BatchSize::LargeInput,
            )
        });

        #[cfg(feature = "spike-ropey")]
        group.bench_with_input(BenchmarkId::new("RopeyText_persistent", label), &base, |b, base| {
            b.iter_batched(
                || RopeyText::from(base.as_str()),
                |mut t| {
                    let mut versions = Vec::with_capacity(ops);
                    for i in 0..ops {
                        versions.push(t.clone());
                        t = t.splice(start + i..start + i, "x");
                    }
                    black_box((t, versions))
                },
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("String_full_clone", label), &base, |b, base| {
            b.iter_batched(
                || base.clone(),
                |mut s| {
                    let mut versions = Vec::with_capacity(ops);
                    for i in 0..ops {
                        versions.push(s.clone());
                        s.insert(start + i, 'x');
                    }
                    black_box((s, versions))
                },
                BatchSize::LargeInput,
            )
        });
    }
}

// ---------------------------------------------------------------------
// 4. Full scan: chunks() vs String byte iteration (the skeleton-scan
//    proxy — sum every byte value so the loop can't be optimized away).
// ---------------------------------------------------------------------

fn bench_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_scan");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        let base = text_of_size(bytes);
        let ptext = PText::from(base.as_str());
        let rope = Rope::from_str(&base);
        #[cfg(feature = "spike-ropey")]
        let ropey_text = RopeyText::from(base.as_str());

        group.bench_with_input(BenchmarkId::new("PText_chunks", label), &ptext, |b, t| {
            b.iter(|| {
                let mut sum: u64 = 0;
                for chunk in t.chunks() {
                    for byte in chunk.as_bytes() {
                        sum = sum.wrapping_add(*byte as u64);
                    }
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("ropey_chunks", label), &rope, |b, r| {
            b.iter(|| {
                let mut sum: u64 = 0;
                for chunk in r.chunks() {
                    for byte in chunk.as_bytes() {
                        sum = sum.wrapping_add(*byte as u64);
                    }
                }
                black_box(sum)
            })
        });

        #[cfg(feature = "spike-ropey")]
        group.bench_with_input(BenchmarkId::new("RopeyText_chunks", label), &ropey_text, |b, t| {
            b.iter(|| {
                let mut sum: u64 = 0;
                for chunk in t.chunks() {
                    for byte in chunk.as_bytes() {
                        sum = sum.wrapping_add(*byte as u64);
                    }
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("String_bytes", label), &base, |b, s| {
            b.iter(|| {
                let mut sum: u64 = 0;
                for byte in s.as_bytes() {
                    sum = sum.wrapping_add(*byte as u64);
                }
                black_box(sum)
            })
        });
    }
}

// ---------------------------------------------------------------------
// 5. line_to_char queries.
// ---------------------------------------------------------------------

fn bench_line_to_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_to_char");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        // Newline every ~46 bytes (SENTENCE's length) so line count scales
        // with size, giving line_to_char real tree depth to traverse.
        let base = text_of_size(bytes).replace(". ", ".\n");
        let ptext = PText::from(base.as_str());
        let rope = Rope::from_str(&base);
        let n_lines = ptext.len_lines();
        #[cfg(feature = "spike-ropey")]
        let ropey_text = RopeyText::from(base.as_str());

        group.bench_with_input(BenchmarkId::new("PText", label), &ptext, |b, t| {
            let mut rng = Xorshift32(1234);
            b.iter(|| black_box(t.line_to_char(rng.below(n_lines))))
        });

        group.bench_with_input(BenchmarkId::new("ropey", label), &rope, |b, r| {
            let mut rng = Xorshift32(1234);
            b.iter(|| black_box(r.line_to_char(rng.below(r.len_lines()))))
        });

        #[cfg(feature = "spike-ropey")]
        group.bench_with_input(BenchmarkId::new("RopeyText", label), &ropey_text, |b, t| {
            let mut rng = Xorshift32(1234);
            b.iter(|| black_box(t.line_to_char(rng.below(t.len_lines()))))
        });
    }
}

// ---------------------------------------------------------------------
// M7.5 (SPEC-PTEXT.md follow-up): replace-all. The M8 integration go/no-go
// gate — this is what the editor's real, measured `replace-all` at 8.4MB
// actually is: a storm of small splices at every match of a pattern, not
// one big edit. Competitors: PText/ropey applying one splice per match
// (owned path), and a flat `String::replace` single-pass rebuild — the
// incumbent's *actual* strategy (not a naive per-match `replace_range`,
// which no real implementation would do — `String::replace` is the fair
// "what does the incumbent really do" comparison).
// ---------------------------------------------------------------------

fn bench_replace_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("replace_all");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    let base = text_of_size(8_400_000);
    let pattern = "the ";
    let replacement = "THE ";
    debug_assert_eq!(pattern.len(), replacement.len(), "same length keeps match positions stable as edits apply in order");

    let char_positions: Vec<usize> = base
        .match_indices(pattern)
        .map(|(byte_pos, _)| base[..byte_pos].chars().count())
        .collect();
    eprintln!("replace_all: {} matches of {pattern:?} in {} bytes (~1 per {:.0} bytes)", char_positions.len(), base.len(), base.len() as f64 / char_positions.len() as f64);
    let pattern_chars = pattern.chars().count();

    group.bench_function(BenchmarkId::new("PText_owned", "8.4MB"), |b| {
        b.iter_batched(
            || PText::from(base.as_str()),
            |mut t| {
                for &pos in &char_positions {
                    t = t.splice_owned(pos..pos + pattern_chars, replacement);
                }
                black_box(t)
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function(BenchmarkId::new("ropey", "8.4MB"), |b| {
        b.iter_batched(
            || Rope::from_str(&base),
            |mut r| {
                for &pos in &char_positions {
                    r.remove(pos..pos + pattern_chars);
                    r.insert(pos, replacement);
                }
                black_box(r)
            },
            BatchSize::LargeInput,
        )
    });

    #[cfg(feature = "spike-ropey")]
    group.bench_function(BenchmarkId::new("RopeyText_owned", "8.4MB"), |b| {
        b.iter_batched(
            || RopeyText::from(base.as_str()),
            |mut t| {
                for &pos in &char_positions {
                    t = t.splice_owned(pos..pos + pattern_chars, replacement);
                }
                black_box(t)
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function(BenchmarkId::new("String_replace", "8.4MB"), |b| {
        b.iter_batched(|| base.clone(), |s| black_box(s.replace(pattern, replacement)), BatchSize::LargeInput)
    });
}

// ---------------------------------------------------------------------
// 6. M14b (SPEC-M14-TEXT-PERSIST-EQ.md): text_eq — pruned equality.
//    `eq_shared`: v1 vs v3 = v1.splice(edit).splice(inverse-edit)
//    (content-equal, sharing every subtree off the one edited spine) is the
//    "did this change?" pattern the pruned walk targets — the quantitative
//    witness that the prune actually fires. `eq_disjoint`: equal content,
//    zero shared structure (two independent `PText::from` builds of the
//    same string) — pruning must be free when it never fires. Baseline
//    (pre-M14b, full O(n) chunk-pair walk) recorded in BASELINE-M14-EQ.txt.
// ---------------------------------------------------------------------

fn bench_text_eq(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_eq");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));
    for &(label, bytes) in &SIZES {
        let base = text_of_size(bytes);
        let start = base.len() / 2;

        // eq_shared: v3 is content-equal to v1 but its root is a fresh
        // allocation (the edit spine was rebuilt) — every subtree off that
        // spine is the same shared allocation as v1's, thanks to
        // persistent `splice`'s structural sharing.
        let v1 = PText::from(base.as_str());
        let v2 = v1.splice(start..start, "Z");
        let v3 = v2.splice(start..start + 1, "");
        assert_eq!(v1.to_string(), v3.to_string(), "text_eq bench: v3 must be content-equal to v1");
        assert!(!PText::ptr_eq(&v1, &v3), "text_eq bench: v3's root must differ from v1's");
        group.bench_with_input(BenchmarkId::new("eq_shared", label), &(v1, v3), |b, (v1, v3)| b.iter(|| black_box(v1 == v3)));

        // eq_disjoint: equal content, zero shared structure.
        let d1 = PText::from(base.as_str());
        let d2 = PText::from(base.as_str());
        group.bench_with_input(BenchmarkId::new("eq_disjoint", label), &(d1, d2), |b, (d1, d2)| b.iter(|| black_box(d1 == d2)));
    }
}

criterion_group!(
    benches,
    bench_sequential_typing,
    bench_sequential_typing_cold_start,
    bench_random_splices,
    bench_undo_pattern,
    bench_replace_all,
    bench_full_scan,
    bench_line_to_char,
    bench_text_eq,
);
criterion_main!(benches);
