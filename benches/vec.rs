//! Criterion suite for `PVector` vs `imbl::Vector` (the host's actual
//! `PVec::Big` backing today — an RRB tree, our A/B baseline) and plain
//! `Vec` (the dense floor). See SPEC-M10-PVEC.md's "Success bars" section
//! for the scenario matrix and BENCH-RESULTS.md's "M10" section for the
//! analysis of these numbers.
//!
//! Two payload shapes, matching the spec's own split:
//! - `i64`: dense/unboxed, matching the survey's "@240k ints" baseline
//!   numbers (build/pop/equality bars).
//! - `Arc<i64>`: a `Send + Sync` boxed/refcounted stand-in for the host's
//!   actual `Value` type (not a champ dependency — this crate never
//!   couples to a Clojure-dialect host — but representative of "a heap-boxed dynamic
//!   value" for the `get`/memory bars the spec explicitly calls out as
//!   Value-shaped). Noted plainly in BENCH-RESULTS.md as an approximation,
//!   not a byte-exact replica of the host's `Value` enum.

use std::sync::Arc;
use std::time::Duration;

use champ::PVector;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const BIG: usize = 240_000;
const SMALL: usize = 20;

/// Small deterministic PRNG (xorshift32) — avoids a `rand` dependency,
/// determinism keeps runs comparable across competitors (matches
/// benches/text.rs's convention).
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

fn ints(n: usize) -> Vec<i64> {
    (0..n as i64).collect()
}

fn values(n: usize) -> Vec<Arc<i64>> {
    (0..n as i64).map(Arc::new).collect()
}

fn random_indices(n: usize, count: usize, seed: u32) -> Vec<usize> {
    let mut rng = Xorshift32(seed);
    (0..count).map(|_| rng.below(n)).collect()
}

// ---------------------------------------------------------------------
// 1. get: random-access reads, i64 and Arc<i64> payloads, @240k and @20.
// ---------------------------------------------------------------------

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_random");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[SMALL, BIG] {
        let idxs = random_indices(n, 10_000, 0xC0FFEE);

        let items = ints(n);
        let pv: PVector<i64> = PVector::from_slice(&items);
        let iv: imbl::Vector<i64> = items.iter().copied().collect();
        let sv: Vec<i64> = items.clone();
        group.bench_with_input(BenchmarkId::new("PVector_i64", n), &idxs, |b, idxs| {
            b.iter(|| {
                let mut acc = 0i64;
                for &i in idxs {
                    acc = acc.wrapping_add(*pv.get(i).unwrap());
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("imbl_i64", n), &idxs, |b, idxs| {
            b.iter(|| {
                let mut acc = 0i64;
                for &i in idxs {
                    acc = acc.wrapping_add(*iv.get(i).unwrap());
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("Vec_i64", n), &idxs, |b, idxs| {
            b.iter(|| {
                let mut acc = 0i64;
                for &i in idxs {
                    acc = acc.wrapping_add(sv[i]);
                }
                black_box(acc)
            })
        });

        let vitems = values(n);
        let pvv: PVector<Arc<i64>> = PVector::from_slice(&vitems);
        let ivv: imbl::Vector<Arc<i64>> = vitems.iter().cloned().collect();
        group.bench_with_input(BenchmarkId::new("PVector_ArcValue", n), &idxs, |b, idxs| {
            b.iter(|| {
                let mut acc = 0i64;
                for &i in idxs {
                    acc = acc.wrapping_add(**pvv.get(i).unwrap());
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("imbl_ArcValue", n), &idxs, |b, idxs| {
            b.iter(|| {
                let mut acc = 0i64;
                for &i in idxs {
                    acc = acc.wrapping_add(**ivv.get(i).unwrap());
                }
                black_box(acc)
            })
        });
    }
}

// ---------------------------------------------------------------------
// 2. Owned build: push_back_owned loop on a last-use handle, i64, @240k
//    and @20 (imbl's "unique-path" build is its own `push_back` on an
//    owned `Vector` — imbl doesn't expose a separate owned/shared split
//    the way champ does, so its own single-owner build already IS
//    its best case, matching the spec's "imbl's unique-path build").
// ---------------------------------------------------------------------

fn bench_owned_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("owned_build");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[SMALL, BIG] {
        let items = ints(n);
        group.bench_with_input(BenchmarkId::new("PVector_owned", n), &items, |b, items| {
            b.iter(|| {
                let mut v: PVector<i64> = PVector::new();
                for &x in items {
                    v = v.push_back_owned(x);
                }
                black_box(v)
            })
        });
        group.bench_with_input(BenchmarkId::new("imbl_owned", n), &items, |b, items| {
            b.iter(|| {
                let mut v: imbl::Vector<i64> = imbl::Vector::new();
                for &x in items {
                    v.push_back(x);
                }
                black_box(v)
            })
        });
        group.bench_with_input(BenchmarkId::new("Vec", n), &items, |b, items| {
            b.iter(|| {
                let mut v: Vec<i64> = Vec::new();
                for &x in items {
                    v.push(x);
                }
                black_box(v)
            })
        });
    }
}

// ---------------------------------------------------------------------
// 3. Bulk from_slice / FromIterator, i64, @240k and @20.
// ---------------------------------------------------------------------

fn bench_bulk_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_build");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[SMALL, BIG] {
        let items = ints(n);
        group.bench_with_input(BenchmarkId::new("PVector_from_slice", n), &items, |b, items| {
            b.iter(|| black_box(PVector::from_slice(items)))
        });
        group.bench_with_input(BenchmarkId::new("imbl_from_iter", n), &items, |b, items| {
            b.iter(|| black_box(items.iter().copied().collect::<imbl::Vector<i64>>()))
        });
        group.bench_with_input(BenchmarkId::new("Vec_from_iter", n), &items, |b, items| {
            b.iter(|| black_box(items.to_vec()))
        });
    }
}

// ---------------------------------------------------------------------
// 4. pop_front walk (the uncons shape): drain a 240k vector one
//    pop_front at a time, i64.
// ---------------------------------------------------------------------

fn bench_pop_front_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_front_walk");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    let items = ints(BIG);

    group.bench_function("PVector_persistent", |b| {
        b.iter_batched(
            || PVector::from_slice(&items),
            |mut v| {
                let mut acc = 0i64;
                loop {
                    let (rest, popped) = v.pop_front();
                    match popped {
                        Some(x) => acc = acc.wrapping_add(x),
                        None => break,
                    }
                    v = rest;
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });
    // The realistic `uncons`-walk shape: a last-use handle, nobody holds
    // `v`'s old value after each pop — see `PVector::pop_front_owned`'s
    // doc comment for why this avoids all refcount traffic the
    // persistent flavor above pays.
    group.bench_function("PVector_owned", |b| {
        b.iter_batched(
            || PVector::from_slice(&items),
            |mut v| {
                let mut acc = 0i64;
                loop {
                    let (rest, popped) = v.pop_front_owned();
                    match popped {
                        Some(x) => acc = acc.wrapping_add(x),
                        None => break,
                    }
                    v = rest;
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("imbl", |b| {
        b.iter_batched(
            || items.iter().copied().collect::<imbl::Vector<i64>>(),
            |mut v| {
                let mut acc = 0i64;
                while let Some(x) = v.pop_front() {
                    acc = acc.wrapping_add(x);
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });
}

// ---------------------------------------------------------------------
// 5. Equality: identical-tree (ptr shortcut) and distinct-equal, i64 @240k.
// ---------------------------------------------------------------------

fn bench_equality(c: &mut Criterion) {
    let mut group = c.benchmark_group("equality");
    group.measurement_time(Duration::from_secs(3));
    let items = ints(BIG);

    let pv_a = PVector::from_slice(&items);
    let pv_identical = pv_a.clone(); // same tree, ptr_eq shortcut
    let pv_distinct = PVector::from_slice(&items); // distinct tree, equal content
    group.bench_function("PVector_identical_tree", |b| b.iter(|| black_box(pv_a == pv_identical)));
    group.bench_function("PVector_distinct_equal", |b| b.iter(|| black_box(pv_a == pv_distinct)));

    let iv_a: imbl::Vector<i64> = items.iter().copied().collect();
    let iv_identical = iv_a.clone();
    let iv_distinct: imbl::Vector<i64> = items.iter().copied().collect();
    group.bench_function("imbl_identical_tree", |b| b.iter(|| black_box(iv_a == iv_identical)));
    group.bench_function("imbl_distinct_equal", |b| b.iter(|| black_box(iv_a == iv_distinct)));
}

// ---------------------------------------------------------------------
// 6. Queue churn (SPEC-M12-SUFFIXVIEW.md): the pop-front-then-push-back
//    pattern that was O(n) per op before the suffix-view fast path
//    (every push normalized the whole backing). Steady-state: the queue's
//    length never changes across the ROUNDS iterations, only its content
//    slides forward — this is exactly the host's `PersistentQueue`/mailbox
//    shape. Same ROUNDS count at both `n` values is the point: per-op
//    cost flat in `n` demonstrates O(1) amortized (an O(n) per-op bug
//    would show total time scaling with `n` at fixed ROUNDS).
// ---------------------------------------------------------------------

const QUEUE_CHURN_ROUNDS: usize = 2_000;

fn bench_queue_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_churn");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &n in &[1_000usize, BIG] {
        let items = ints(n);

        group.bench_with_input(BenchmarkId::new("PVector_owned", n), &items, |b, items| {
            b.iter_batched(
                || PVector::from_slice(items),
                |mut v| {
                    for i in 0..QUEUE_CHURN_ROUNDS {
                        let (rest, _popped) = v.pop_front_owned();
                        v = rest.push_back_owned(i as i64);
                    }
                    black_box(v)
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("imbl", n), &items, |b, items| {
            b.iter_batched(
                || items.iter().copied().collect::<imbl::Vector<i64>>(),
                |mut v| {
                    for i in 0..QUEUE_CHURN_ROUNDS {
                        v.pop_front();
                        v.push_back(i as i64);
                    }
                    black_box(v)
                },
                BatchSize::LargeInput,
            )
        });
    }
}

// ---------------------------------------------------------------------
// 7. try_eq_by (SPEC-M13-EQWITH.md): shared-subtree (one-trailing-edit)
//    pair vs a disjoint-but-equal pair, both @240k, compared against
//    plain `==` (the full, unpruned walk `Interp::values_equal` used to
//    pay every time, before `try_eq_by` gave it a way to reach the
//    prune).
// ---------------------------------------------------------------------

fn bench_try_eq_by(c: &mut Criterion) {
    let mut group = c.benchmark_group("try_eq_by");
    group.measurement_time(Duration::from_secs(3));
    let items = ints(BIG);

    let base = PVector::from_slice(&items);
    let edited = base.clone().set_owned(BIG - 1, -1); // one trailing edit, rest pointer-shared
    let disjoint = PVector::from_slice(&items); // distinct tree, equal content

    group.bench_function("PVector_shared_one_edit_eq_by", |b| {
        b.iter(|| black_box(black_box(&base).eq_by(black_box(&edited), |x, y| x == y)))
    });
    group.bench_function("PVector_shared_one_edit_partial_eq", |b| b.iter(|| black_box(black_box(&base) == black_box(&edited))));
    group.bench_function("PVector_disjoint_equal_eq_by", |b| b.iter(|| black_box(black_box(&base).eq_by(black_box(&disjoint), |x, y| x == y))));
    group.bench_function("PVector_disjoint_equal_partial_eq", |b| b.iter(|| black_box(black_box(&base) == black_box(&disjoint))));
}

criterion_group!(benches, bench_get, bench_owned_build, bench_bulk_build, bench_pop_front_walk, bench_equality, bench_queue_churn, bench_try_eq_by);
criterion_main!(benches);
