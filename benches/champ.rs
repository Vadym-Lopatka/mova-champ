//! Criterion suite for champ vs `im::HashMap` vs
//! `std::collections::HashMap`. See `SPEC-BENCH.md` for the full matrix
//! rationale and `BENCH-RESULTS.md` for the analysis of these numbers.
//!
//! Hasher choice: champ uses its own `DefaultBuildHasher` (the
//! crate's shipping default); `im`/`std` use their own shipping default,
//! `RandomState`. Each competitor runs with the config its own users would
//! actually reach for — that is the honest apples-to-apples comparison, not
//! forcing everyone onto the same hasher. `im`'s equality is content-based
//! (not hash-bitmap based), so `RandomState`'s per-instance random seed
//! doesn't break its `==` — see `hash.rs`'s module docs for why champ
//! itself cannot default to `RandomState` for the same reason.
//!
//! Timing budget: criterion's defaults (100 samples / 3s warm-up / 5s
//! measurement) applied uniformly across the full matrix below would run
//! well past the ~25 minute target (252 benchmark configurations). Instead
//! of shrinking the size matrix, [`tune`] scales sample count and
//! warm-up/measurement duration down as `n` grows — big-`n` benchmarks are
//! individually more expensive per sample, so they need fewer samples to
//! reach a stable estimate anyway. This is noted here rather than removing
//! any (size, key type, competitor) combination from the spec's matrix.

use std::cell::Cell;
use std::collections::HashMap as StdHashMap;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use champ::PersistentHashMap;
use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use im::HashMap as ImHashMap;

// ---------------------------------------------------------------------
// Size matrix
// ---------------------------------------------------------------------

/// Sizes for every non-`get` benchmark category.
const SIZES: [usize; 4] = [8, 64, 1_000, 100_000];
/// `get` is O(1) regardless of map size, so it gets an extra data point
/// (100) per the spec.
const GET_SIZES: [usize; 5] = [8, 64, 100, 1_000, 100_000];

/// Scale criterion's sample count / warm-up / measurement time down as `n`
/// grows, so the O(n)-per-op categories (build, remove_all, iterate, eq)
/// don't blow the suite's time budget at n=100_000 while still using
/// criterion's own statistical machinery (not a hand-rolled timer).
fn tune(group: &mut BenchmarkGroup<'_, WallTime>, n: usize) {
    let (samples, warm_up_ms, measurement_ms) = if n >= 100_000 {
        (15, 400, 1200)
    } else if n >= 1_000 {
        (30, 700, 1500)
    } else {
        (50, 700, 1500)
    };
    group.sample_size(samples);
    group.warm_up_time(Duration::from_millis(warm_up_ms));
    group.measurement_time(Duration::from_millis(measurement_ms));
}

// ---------------------------------------------------------------------
// Key generation, shared by every benchmark category.
// ---------------------------------------------------------------------

/// A benchmark key type: something we can deterministically manufacture N
/// of, distinct, from an index.
trait BenchKey: Clone + Eq + Hash + Send + Sync + 'static {
    fn from_index(i: usize) -> Self;
    fn label() -> &'static str;
}

impl BenchKey for u64 {
    fn from_index(i: usize) -> Self {
        i as u64
    }
    fn label() -> &'static str {
        "u64"
    }
}

impl BenchKey for String {
    /// Exactly 16 bytes: `'k'` + 15 zero-padded decimal digits.
    fn from_index(i: usize) -> Self {
        format!("k{i:015}")
    }
    fn label() -> &'static str {
        "string16"
    }
}

/// Keys guaranteed disjoint from `0..n` (used for `get/miss` and to
/// generate a genuinely-new key for `assoc/persistent`).
const MISS_OFFSET: usize = 10_000_000;

fn std_hash_of<K: Hash>(k: &K) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    k.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------
// Dataset builders (untimed setup, called outside `b.iter`).
// ---------------------------------------------------------------------

fn champ_dataset<K: BenchKey>(n: usize) -> PersistentHashMap<K, K> {
    let mut t = PersistentHashMap::new().transient();
    for i in 0..n {
        let k = K::from_index(i);
        t.assoc(k.clone(), k);
    }
    t.persistent()
}

fn im_dataset<K: BenchKey>(n: usize) -> ImHashMap<K, K> {
    let mut m = ImHashMap::new();
    for i in 0..n {
        let k = K::from_index(i);
        m.insert(k.clone(), k);
    }
    m
}

fn std_dataset<K: BenchKey>(n: usize) -> StdHashMap<K, K> {
    let mut m = StdHashMap::new();
    for i in 0..n {
        let k = K::from_index(i);
        m.insert(k.clone(), k);
    }
    m
}

// ---------------------------------------------------------------------
// get/hit, get/miss
// ---------------------------------------------------------------------
//
// Single lookup per criterion iteration, key rotated through a
// pre-shuffled cycling set via a `Cell` index — matches "criterion iter
// over a cycling key set" from the spec without paying O(n) per iteration.

fn bench_get<K: BenchKey>(c: &mut Criterion) {
    for &n in &GET_SIZES {
        let hit_keys: Vec<K> = (0..n).map(K::from_index).collect();
        let miss_keys: Vec<K> = (0..n.max(1)).map(|i| K::from_index(i + MISS_OFFSET)).collect();

        let champ = champ_dataset::<K>(n);
        let im = im_dataset::<K>(n);
        let std_m = std_dataset::<K>(n);

        {
            let mut group = c.benchmark_group(format!("get_hit/{}", K::label()));
            tune(&mut group, n);
            let idx = Cell::new(0usize);
            group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
                b.iter(|| {
                    let i = idx.get();
                    idx.set((i + 1) % hit_keys.len());
                    black_box(champ.get(black_box(&hit_keys[i])))
                })
            });
            let idx = Cell::new(0usize);
            group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
                b.iter(|| {
                    let i = idx.get();
                    idx.set((i + 1) % hit_keys.len());
                    black_box(im.get(black_box(&hit_keys[i])))
                })
            });
            let idx = Cell::new(0usize);
            group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
                b.iter(|| {
                    let i = idx.get();
                    idx.set((i + 1) % hit_keys.len());
                    black_box(std_m.get(black_box(&hit_keys[i])))
                })
            });
        }

        {
            let mut group = c.benchmark_group(format!("get_miss/{}", K::label()));
            tune(&mut group, n);
            let idx = Cell::new(0usize);
            group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
                b.iter(|| {
                    let i = idx.get();
                    idx.set((i + 1) % miss_keys.len());
                    black_box(champ.get(black_box(&miss_keys[i])))
                })
            });
            let idx = Cell::new(0usize);
            group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
                b.iter(|| {
                    let i = idx.get();
                    idx.set((i + 1) % miss_keys.len());
                    black_box(im.get(black_box(&miss_keys[i])))
                })
            });
            let idx = Cell::new(0usize);
            group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
                b.iter(|| {
                    let i = idx.get();
                    idx.set((i + 1) % miss_keys.len());
                    black_box(std_m.get(black_box(&miss_keys[i])))
                })
            });
        }
    }
}

// ---------------------------------------------------------------------
// assoc/persistent — single op on a retained size-N map, forces path copy.
// ---------------------------------------------------------------------

fn bench_assoc_persistent<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("assoc_persistent/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let champ = champ_dataset::<K>(n);
        let im = im_dataset::<K>(n);
        let std_m = std_dataset::<K>(n);
        // A genuinely new key: forces a real insert (not champ's
        // pointer-identical same-key-same-value fast path).
        let new_key = K::from_index(n + MISS_OFFSET);

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter(|| black_box(champ.assoc(black_box(new_key.clone()), black_box(new_key.clone()))))
        });
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
            b.iter(|| black_box(im.update(black_box(new_key.clone()), black_box(new_key.clone()))))
        });
        // std has no persistent update; the only way to "keep the old
        // version" is a full clone + insert. Included per spec's
        // "std-where-sensible" as the ceiling-cost reference for what
        // persistence would cost you if you had to fake it — not a fair
        // apples-to-apples "assoc", and called out as such in the report.
        group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
            b.iter_batched(
                || std_m.clone(),
                |mut m| {
                    m.insert(black_box(new_key.clone()), black_box(new_key.clone()));
                    black_box(m)
                },
                BatchSize::SmallInput,
            )
        });
    }
}

// ---------------------------------------------------------------------
// build/persistent — fold N inserts keeping only the newest version.
// ---------------------------------------------------------------------

fn bench_build_persistent<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("build_persistent/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let keys: Vec<K> = (0..n).map(K::from_index).collect();

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter(|| {
                let mut m = PersistentHashMap::new();
                for k in &keys {
                    m = m.assoc_owned(k.clone(), k.clone());
                }
                black_box(m)
            })
        });
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
            b.iter(|| {
                let mut m = ImHashMap::new();
                for k in &keys {
                    m = m.update(k.clone(), k.clone());
                }
                black_box(m)
            })
        });
        // std has no persistent-fold notion (its "persistent" build is
        // identical to its transient build); intentionally omitted here,
        // present in build/transient instead.
    }
}

// ---------------------------------------------------------------------
// build/transient — N inserts through an owned builder.
// ---------------------------------------------------------------------

fn bench_build_transient<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("build_transient/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let keys: Vec<K> = (0..n).map(K::from_index).collect();

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter(|| {
                let mut t = PersistentHashMap::new().transient();
                for k in &keys {
                    t.assoc(k.clone(), k.clone());
                }
                black_box(t.persistent())
            })
        });
        // im has no transient type; its plain `insert(&mut self, ..)` on
        // an owned map is the closest analog (per SPEC-BENCH.md).
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
            b.iter(|| {
                let mut m = ImHashMap::new();
                for k in &keys {
                    m.insert(k.clone(), k.clone());
                }
                black_box(m)
            })
        });
        group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
            b.iter(|| {
                let mut m = StdHashMap::new();
                for k in &keys {
                    m.insert(k.clone(), k.clone());
                }
                black_box(m)
            })
        });
    }
}

// ---------------------------------------------------------------------
// dissoc/persistent — remove one key from a retained size-N map.
// ---------------------------------------------------------------------

fn bench_dissoc_persistent<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("dissoc_persistent/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let champ = champ_dataset::<K>(n);
        let im = im_dataset::<K>(n);
        let victim = K::from_index(n / 2);

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter(|| black_box(champ.dissoc(black_box(&victim))))
        });
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
            b.iter(|| black_box(im.without(black_box(&victim))))
        });
        // std excluded: no persistent (retain-old) removal exists to
        // compare; see remove_all for std's owned/mutable removal cost.
    }
}

// ---------------------------------------------------------------------
// remove_all — dissoc_owned N keys in random order down to empty.
// ---------------------------------------------------------------------

/// Deterministic xorshift shuffle — no extra dependency needed for "random
/// order", and reproducible across runs.
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut state = seed.max(1);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in (1..v.len()).rev() {
        let j = (next() as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

fn bench_remove_all<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("remove_all/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let champ = champ_dataset::<K>(n);
        let im = im_dataset::<K>(n);
        let std_m = std_dataset::<K>(n);
        let order: Vec<K> = shuffled_indices(n, 0x9E37_79B9)
            .into_iter()
            .map(K::from_index)
            .collect();

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter_batched(
                || champ.clone(),
                |mut m| {
                    for k in &order {
                        m = m.dissoc_owned(k);
                    }
                    black_box(m)
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
            b.iter_batched(
                || im.clone(),
                |mut m| {
                    for k in &order {
                        m.remove(k);
                    }
                    black_box(m)
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
            b.iter_batched(
                || std_m.clone(),
                |mut m| {
                    for k in &order {
                        m.remove(k);
                    }
                    black_box(m)
                },
                BatchSize::SmallInput,
            )
        });
    }
}

// ---------------------------------------------------------------------
// iterate — full traversal summing key hashes.
// ---------------------------------------------------------------------

fn bench_iterate<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("iterate/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let champ = champ_dataset::<K>(n);
        let im = im_dataset::<K>(n);
        let std_m = std_dataset::<K>(n);

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter(|| {
                let mut acc = 0u64;
                for k in champ.keys() {
                    acc ^= std_hash_of(black_box(k));
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
            b.iter(|| {
                let mut acc = 0u64;
                for k in im.keys() {
                    acc ^= std_hash_of(black_box(k));
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
            b.iter(|| {
                let mut acc = 0u64;
                for k in std_m.keys() {
                    acc ^= std_hash_of(black_box(k));
                }
                black_box(acc)
            })
        });
    }
}

// ---------------------------------------------------------------------
// eq/equal — two structurally equal maps, built separately (no shared
// nodes). eq/shared — clone + 1 change (near-O(1) for champ).
// ---------------------------------------------------------------------

fn bench_eq<K: BenchKey>(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group(format!("eq_equal/{}", K::label()));
        for &n in &SIZES {
            tune(&mut group, n);
            let champ_a = champ_dataset::<K>(n);
            let champ_b = champ_dataset::<K>(n);
            let im_a = im_dataset::<K>(n);
            let im_b = im_dataset::<K>(n);
            let std_a = std_dataset::<K>(n);
            let std_b = std_dataset::<K>(n);

            group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
                b.iter(|| black_box(black_box(&champ_a) == black_box(&champ_b)))
            });
            group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
                b.iter(|| black_box(black_box(&im_a) == black_box(&im_b)))
            });
            group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
                b.iter(|| black_box(black_box(&std_a) == black_box(&std_b)))
            });
        }
    }

    {
        let mut group = c.benchmark_group(format!("eq_shared/{}", K::label()));
        for &n in &SIZES {
            tune(&mut group, n);
            // Same key set as `_a` (count-preserving), one value changed at
            // an existing key — a true "clone + 1 change". Using a brand
            // new key here would be wrong: it changes `count`/`len`, and
            // both champ's `PartialEq` and im's `test_eq` fast-reject
            // on a length mismatch before ever touching the tree, which
            // would make this benchmark measure an integer comparison
            // instead of the intended pointer-sharing short-circuit.
            let champ_a = champ_dataset::<K>(n);
            let victim = K::from_index(n / 2);
            let new_value = K::from_index(n / 2 + 1);
            let champ_b = champ_a.assoc(victim.clone(), new_value.clone());

            let im_a = im_dataset::<K>(n);
            let im_b = im_a.update(victim, new_value);

            group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
                b.iter(|| black_box(black_box(&champ_a) == black_box(&champ_b)))
            });
            group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| {
                b.iter(|| black_box(black_box(&im_a) == black_box(&im_b)))
            });
            // std has no structural sharing to short-circuit on; omitted
            // (its eq/equal number above is the relevant std reference).
        }
    }
}

// ---------------------------------------------------------------------
// clone — O(1) sanity check for champ/im; std's O(n) clone is included
// as the calibration reference.
// ---------------------------------------------------------------------

fn bench_clone<K: BenchKey>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("clone/{}", K::label()));
    for &n in &SIZES {
        tune(&mut group, n);
        let champ = champ_dataset::<K>(n);
        let im = im_dataset::<K>(n);
        let std_m = std_dataset::<K>(n);

        group.bench_with_input(BenchmarkId::new("champ", n), &n, |b, _| {
            b.iter(|| black_box(champ.clone()))
        });
        group.bench_with_input(BenchmarkId::new("im", n), &n, |b, _| b.iter(|| black_box(im.clone())));
        group.bench_with_input(BenchmarkId::new("std", n), &n, |b, _| {
            b.iter(|| black_box(std_m.clone()))
        });
    }
}

// ---------------------------------------------------------------------
// Top-level per-key-type entry points (criterion_group targets must be
// plain `fn(&mut Criterion)`, so monomorphize here).
// ---------------------------------------------------------------------

fn get_u64(c: &mut Criterion) {
    bench_get::<u64>(c);
}
fn get_string(c: &mut Criterion) {
    bench_get::<String>(c);
}
fn assoc_persistent_u64(c: &mut Criterion) {
    bench_assoc_persistent::<u64>(c);
}
fn assoc_persistent_string(c: &mut Criterion) {
    bench_assoc_persistent::<String>(c);
}
fn build_persistent_u64(c: &mut Criterion) {
    bench_build_persistent::<u64>(c);
}
fn build_persistent_string(c: &mut Criterion) {
    bench_build_persistent::<String>(c);
}
fn build_transient_u64(c: &mut Criterion) {
    bench_build_transient::<u64>(c);
}
fn build_transient_string(c: &mut Criterion) {
    bench_build_transient::<String>(c);
}
fn dissoc_persistent_u64(c: &mut Criterion) {
    bench_dissoc_persistent::<u64>(c);
}
fn dissoc_persistent_string(c: &mut Criterion) {
    bench_dissoc_persistent::<String>(c);
}
fn remove_all_u64(c: &mut Criterion) {
    bench_remove_all::<u64>(c);
}
fn remove_all_string(c: &mut Criterion) {
    bench_remove_all::<String>(c);
}
fn iterate_u64(c: &mut Criterion) {
    bench_iterate::<u64>(c);
}
fn iterate_string(c: &mut Criterion) {
    bench_iterate::<String>(c);
}
fn eq_u64(c: &mut Criterion) {
    bench_eq::<u64>(c);
}
fn eq_string(c: &mut Criterion) {
    bench_eq::<String>(c);
}
fn clone_u64(c: &mut Criterion) {
    bench_clone::<u64>(c);
}
fn clone_string(c: &mut Criterion) {
    bench_clone::<String>(c);
}

criterion_group!(
    benches,
    get_u64,
    get_string,
    assoc_persistent_u64,
    assoc_persistent_string,
    build_persistent_u64,
    build_persistent_string,
    build_transient_u64,
    build_transient_string,
    dissoc_persistent_u64,
    dissoc_persistent_string,
    remove_all_u64,
    remove_all_string,
    iterate_u64,
    iterate_string,
    eq_u64,
    eq_string,
    clone_u64,
    clone_string,
);
criterion_main!(benches);
