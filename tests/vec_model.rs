//! Property tests: `PVector<i32>` against `Vec<i32>` as a model, over
//! random op sequences (push/pop both ends, set, get, slice views,
//! iteration, equality, mixed sequences on SHARED handles with a live
//! clone held) plus a dedicated deterministic LCG fuzz forcing the
//! copy-path fallback specifically. Mirrors `tests/text_model.rs`'s
//! structure (SPEC-M10-PVEC.md gate 2).

use champ::PVector;
use proptest::prelude::*;

// ---------------------------------------------------------------------
// Boundary suite (mandatory per the spec: len 0/1/32/33/1024/33k, plus
// view-then-mutate normalization paths). Plain #[test]s, not proptest —
// specific cases the fuzzer might not reliably hit, checked
// deterministically.
// ---------------------------------------------------------------------

/// `PVector`'s own `NODE_SIZE` isn't public; 32 per SPEC-M10-PVEC.md (5-bit
/// trie chunks).
const NODE_SIZE: usize = 32;

fn assert_matches_model(v: &PVector<i32>, model: &[i32]) {
    v.validate();
    assert_eq!(v.len(), model.len(), "len mismatch");
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), model, "content mismatch");
    assert_eq!(v.chunks().flat_map(|c| c.iter().copied()).collect::<Vec<_>>(), model, "chunks() mismatch");
    if !model.is_empty() {
        let n = model.len();
        for &spot in &[0usize, n / 3, n / 2, n - 1] {
            assert_eq!(*v.get(spot).unwrap(), model[spot], "get({spot}) mismatch");
        }
    }
    assert!(v.get(model.len()).is_none(), "get(len) must be None");
}

fn boundary_len(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

#[test]
fn boundary_empty() {
    let v: PVector<i32> = PVector::new();
    assert_matches_model(&v, &[]);
    let (still, popped) = v.pop_back();
    assert_matches_model(&still, &[]);
    assert_eq!(popped, None);
    let (still2, popped2) = v.pop_front();
    assert_matches_model(&still2, &[]);
    assert_eq!(popped2, None);
}

#[test]
fn boundary_lens_0_1_32_33_1024_33000() {
    // Miri is orders of magnitude slower; keep the largest size much
    // smaller (still exercises several trie levels: NODE_SIZE^2 == 1024)
    // rather than skip this test under Miri entirely, matching the
    // crate's other Miri-scaled fuzz/boundary tests (e.g. text_model.rs).
    let sizes: &[usize] = if cfg!(miri) { &[0, 1, NODE_SIZE, NODE_SIZE + 1, 1024] } else { &[0, 1, NODE_SIZE, NODE_SIZE + 1, 1024, 33_000] };
    for &n in sizes {
        let items = boundary_len(n);
        let v = PVector::from_slice(&items);
        assert_matches_model(&v, &items);
        let v2: PVector<i32> = PVector::from_vec(items.clone());
        assert_matches_model(&v2, &items);
        let v3: PVector<i32> = items.iter().copied().collect();
        assert_matches_model(&v3, &items);

        // push_back one more, both flavors.
        let pushed = v.push_back(999);
        let mut expected = items.clone();
        expected.push(999);
        assert_matches_model(&pushed, &expected);
        let pushed_owned = v.clone().push_back_owned(999);
        assert_matches_model(&pushed_owned, &expected);

        if n > 0 {
            // pop_back, both flavors.
            let (rest, popped) = v.pop_back();
            let mut exp = items.clone();
            let last = exp.pop();
            assert_eq!(popped, last);
            assert_matches_model(&rest, &exp);
            let (rest_o, popped_o) = v.clone().pop_back_owned();
            assert_eq!(popped_o, last);
            assert_matches_model(&rest_o, &exp);

            // pop_front.
            let (rest_f, popped_f) = v.pop_front();
            let mut exp_f = items.clone();
            let first = exp_f.remove(0);
            assert_eq!(popped_f, Some(first));
            assert_matches_model(&rest_f, &exp_f);

            // set, both flavors, at position 0 and len-1.
            for &pos in &[0usize, n - 1] {
                let set_v = v.set(pos, -1);
                let mut exp_s = items.clone();
                exp_s[pos] = -1;
                assert_matches_model(&set_v, &exp_s);
                let set_o = v.clone().set_owned(pos, -1);
                assert_matches_model(&set_o, &exp_s);
            }
        }
    }
}

#[test]
fn view_then_mutate_normalizes_correctly_across_sizes() {
    let sizes: &[usize] = if cfg!(miri) { &[1, NODE_SIZE, NODE_SIZE + 1, 1024] } else { &[1, NODE_SIZE, NODE_SIZE + 1, 1024, 1025, 5000] };
    for &n in sizes {
        let items = boundary_len(n);
        let v = PVector::from_slice(&items);
        // A non-trivial view: drop the first and last quarter.
        let q = n / 4;
        let view = v.slice(q..n - q);
        let expected_view = &items[q..n - q];
        assert_matches_model(&view, expected_view);

        let pushed = view.push_back(-1);
        let mut exp1 = expected_view.to_vec();
        exp1.push(-1);
        assert_matches_model(&pushed, &exp1);

        let pushed_owned = view.clone().push_back_owned(-2);
        let mut exp2 = expected_view.to_vec();
        exp2.push(-2);
        assert_matches_model(&pushed_owned, &exp2);

        if !expected_view.is_empty() {
            let (popped_rest, popped) = view.pop_back();
            let mut exp3 = expected_view.to_vec();
            let last = exp3.pop();
            assert_eq!(popped, last);
            assert_matches_model(&popped_rest, &exp3);

            let set_v = view.set(0, -3);
            let mut exp4 = expected_view.to_vec();
            exp4[0] = -3;
            assert_matches_model(&set_v, &exp4);
        }
        // Original view itself must remain untouched by all the above.
        assert_matches_model(&view, expected_view);
        assert_matches_model(&v, &items);
    }
}

#[test]
fn refcount_leak_stress_clone_slice_shuffle_drop() {
    // Build a moderately large vector, take many overlapping slices and
    // clones, shuffle-drop them in a non-nested order (deterministic
    // pseudo-shuffle, not OS/time-derived), and confirm every surviving
    // handle still reads back correctly at the end. Exercises refcount
    // bookkeeping under a churny access pattern (a real leak or
    // underflow would either abort via the debug_assert in `drop_node`
    // or corrupt content read back by a surviving handle).
    let n = if cfg!(miri) { 300 } else { 3000 };
    let items: Vec<i32> = (0..n).collect();
    let base = PVector::from_slice(&items);
    let mut handles: Vec<(PVector<i32>, Vec<i32>)> = Vec::new();
    let rounds = if cfg!(miri) { 10 } else { 50 };
    for i in 0..rounds {
        let start = (i * 37) % items.len();
        let end = (start + 1 + (i * 53) % (items.len() - start)).min(items.len());
        let s = base.slice(start..end);
        handles.push((s, items[start..end].to_vec()));
    }
    // Deterministic pseudo-shuffle: drop every 3rd, then every 2nd of what's left.
    let mut kept: Vec<(PVector<i32>, Vec<i32>)> = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        if i % 3 != 0 {
            kept.push(h);
        }
        // else: dropped here
    }
    let mut kept2: Vec<(PVector<i32>, Vec<i32>)> = Vec::new();
    for (i, h) in kept.into_iter().enumerate() {
        if i % 2 != 0 {
            kept2.push(h);
        }
    }
    for (v, expected) in &kept2 {
        assert_matches_model(v, expected);
    }
    assert_matches_model(&base, &items);
}

// ---------------------------------------------------------------------
// Random op sequences vs `Vec<i32>` model.
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Op {
    PushBack(i32),
    PopBack,
    PopFront,
    PushFront(i32),
    Set(f64, i32),
    Slice(f64, f64),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => any::<i32>().prop_map(Op::PushBack),
        2 => Just(Op::PopBack),
        2 => Just(Op::PopFront),
        1 => any::<i32>().prop_map(Op::PushFront),
        3 => (0.0..1.0f64, any::<i32>()).prop_map(|(f, x)| Op::Set(f, x)),
        1 => (0.0..1.0f64, 0.0..1.0f64).prop_map(|(a, b)| Op::Slice(a, b)),
    ]
}

fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
    let max_len = if cfg!(miri) { 15 } else { 200 };
    prop::collection::vec(op_strategy(), 0..max_len)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 300 }))]

    /// Random push/pop/set/slice sequences, alternating the persistent
    /// and owned flavor of every op that has one, checked against a
    /// `Vec<i32>` model after every op. Every 7th op also snapshots
    /// (clone) both structures; all snapshots are re-verified at the end
    /// (the undo-stack pattern — retaining an old `PVector` version must
    /// not let later edits, via either path, leak into it).
    #[test]
    fn ops_match_vec_model(ops in ops_strategy()) {
        let mut v: PVector<i32> = PVector::new();
        let mut model: Vec<i32> = Vec::new();
        let mut retained: Vec<(PVector<i32>, Vec<i32>)> = Vec::new();

        for (i, op) in ops.into_iter().enumerate() {
            match op {
                Op::PushBack(x) => {
                    model.push(x);
                    v = if i % 2 == 0 { v.push_back(x) } else { v.push_back_owned(x) };
                }
                Op::PopBack => {
                    let expected = model.pop();
                    let (rest, popped) = if i % 2 == 0 { v.pop_back() } else { v.pop_back_owned() };
                    prop_assert_eq!(popped, expected);
                    v = rest;
                }
                Op::PopFront => {
                    let expected = if model.is_empty() { None } else { Some(model.remove(0)) };
                    // Alternate persistent/owned like every other op with
                    // two flavors — the M10 version of this test only ever
                    // exercised the persistent `pop_front`, so the owned
                    // path's suffix-view/amortized-trim arithmetic
                    // (SPEC-M12-SUFFIXVIEW.md) was never interleaved with
                    // push_back/set/slice under the model here.
                    let (rest, popped) = if i % 2 == 0 { v.pop_front() } else { v.pop_front_owned() };
                    prop_assert_eq!(popped, expected);
                    v = rest;
                }
                Op::PushFront(x) => {
                    model.insert(0, x);
                    v = v.push_front(x);
                }
                Op::Set(frac, x) => {
                    if !model.is_empty() {
                        let idx = ((frac * model.len() as f64) as usize).min(model.len() - 1);
                        model[idx] = x;
                        v = if i % 2 == 0 { v.set(idx, x) } else { v.set_owned(idx, x) };
                    }
                }
                Op::Slice(sf, ef) => {
                    let n = model.len();
                    let start = ((sf * n as f64) as usize).min(n);
                    let end = (start + ((ef * (n - start) as f64) as usize)).min(n);
                    model = model[start..end].to_vec();
                    v = v.slice(start..end);
                }
            }
            v.validate();
            prop_assert_eq!(v.len(), model.len());
            prop_assert_eq!(v.iter().copied().collect::<Vec<_>>(), model.clone());

            if i % 7 == 0 {
                retained.push((v.clone(), model.clone()));
            }
        }

        for (rv, rm) in &retained {
            rv.validate();
            prop_assert_eq!(rv.iter().copied().collect::<Vec<_>>(), rm.clone());
        }
        // The live (final) version must still be independently correct
        // after every retained clone has been checked.
        v.validate();
        prop_assert_eq!(v.iter().copied().collect::<Vec<_>>(), model);
    }

    /// Equality (`PartialEq`) differential: two vectors built different
    /// ways (bulk `from_slice` vs an incremental `push_back` loop) with
    /// identical content must compare equal; content that diverges by
    /// exactly one element must compare unequal.
    #[test]
    fn equality_matches_content_regardless_of_build_path(items in prop::collection::vec(any::<i32>(), 0..(if cfg!(miri) { 40 } else { 500 }))) {
        let bulk = PVector::from_slice(&items);
        let mut looped: PVector<i32> = PVector::new();
        for &x in &items {
            looped = looped.push_back(x);
        }
        prop_assert_eq!(&bulk, &looped);
        prop_assert_eq!(bulk.clone(), bulk.clone()); // ptr_eq shortcut path (same value, cloned)

        if !items.is_empty() {
            let mut other = items.clone();
            let mid = other.len() / 2;
            other[mid] = other[mid].wrapping_add(1);
            let diverged = PVector::from_slice(&other);
            prop_assert_ne!(&bulk, &diverged);
        }
    }
}

// ---------------------------------------------------------------------
// Dedicated deterministic LCG fuzz: mixed owned + persistent ops on a
// handle `v` while a second clone `_shared` is held alive for the WHOLE
// run — every owned op on `v` is therefore forced to discover shared
// (non-unique) structure at some point and fall back to the copy path
// (mirrors `tests/text_model.rs`'s `shared_tree_slice_splice_fuzz_
// matches_string_model`'s `_shared`-held pattern, applied to the owned
// path specifically, which is this milestone's most safety-critical
// property: an owned op must never mutate structure a live sibling
// handle still observes). Deterministic (fixed LCG seed, never OS/
// time-derived) so a failure always reproduces exactly.
// ---------------------------------------------------------------------

struct FuzzLcg(u64);
impl FuzzLcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes LCG constants (matches text_model.rs/
        // examples/slice_probe.rs's convention).
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
    fn i32(&mut self) -> i32 {
        self.next() as i32
    }
}

const FUZZ_SEED: u64 = 0x5EED_C0DE_5EED_C0DE;

#[test]
fn shared_handle_owned_ops_fuzz_matches_vec_model() {
    let sizes: &[usize] = if cfg!(miri) {
        &[0, 1, 37]
    } else {
        &[0, 1, NODE_SIZE, NODE_SIZE + 1, 1024, 1025, 33_000]
    };
    let ops_per_size: usize = if cfg!(miri) { 20 } else { 3000 };

    let mut lcg = FuzzLcg(FUZZ_SEED);
    let mut total_ops = 0usize;

    for &size in sizes {
        let original_model: Vec<i32> = (0..size).map(|_| lcg.i32()).collect();
        let mut model = original_model.clone();
        let mut v = PVector::from_slice(&model);
        // Held alive for this whole size's run of ops — forces every
        // owned op below to eventually hit shared structure.
        let shared = v.clone();
        v.validate();

        for _ in 0..ops_per_size {
            total_ops += 1;
            let n = model.len();
            let choice = lcg.below(6);
            match choice {
                0 => {
                    let x = lcg.i32();
                    model.push(x);
                    v = if lcg.below(2) == 0 { v.push_back(x) } else { v.push_back_owned(x) };
                }
                1 => {
                    let expected = model.pop();
                    let (rest, popped) = if lcg.below(2) == 0 { v.pop_back() } else { v.pop_back_owned() };
                    assert_eq!(popped, expected, "size={size}");
                    v = rest;
                }
                2 => {
                    let expected = if model.is_empty() { None } else { Some(model.remove(0)) };
                    // Owned pop_front's amortized trim (SPEC-M12-
                    // SUFFIXVIEW.md) must also be exercised here, where a
                    // second `shared` handle stays alive for the whole
                    // run — the trim's `normalize()` rebuild must never
                    // let `shared`'s own (older) backing get corrupted.
                    let (rest, popped) = if lcg.below(2) == 0 { v.pop_front() } else { v.pop_front_owned() };
                    assert_eq!(popped, expected, "size={size}");
                    v = rest;
                }
                3 if n > 0 => {
                    let idx = lcg.below(n);
                    let x = lcg.i32();
                    model[idx] = x;
                    v = if lcg.below(2) == 0 { v.set(idx, x) } else { v.set_owned(idx, x) };
                }
                4 if n > 0 => {
                    let start = lcg.below(n + 1);
                    let end = start + lcg.below(n + 1 - start);
                    model = model[start..end].to_vec();
                    v = v.slice(start..end);
                }
                _ => {
                    let x = lcg.i32();
                    model.insert(0, x);
                    v = v.push_front(x);
                }
            }
            v.validate();
            assert_eq!(v.iter().copied().collect::<Vec<_>>(), model, "size={size}, op={choice}");
        }

        v.validate();
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), model, "final mismatch, size={size}");
        // The untouched `shared` clone must still match the size's
        // ORIGINAL content — structural sharing must copy-on-write,
        // never let an owned mutation on `v` leak into `shared`.
        shared.validate();
        assert_eq!(shared.iter().copied().collect::<Vec<_>>(), original_model, "shared clone corrupted, size={size}");
    }

    assert!(total_ops >= if cfg!(miri) { 1 } else { 10_000 }, "fuzz op count too low: {total_ops}");
}

// ---------------------------------------------------------------------
// SPEC-M12-SUFFIXVIEW.md: the queue pattern (pop_front_owned +
// push_back_owned in a steady-state loop) that motivated the suffix-view
// fast path + amortized trim, checked against a `VecDeque` model over a
// mix of sizes (small enough to stay under one leaf, and large enough to
// cross several trie levels and several trim triggers).
// ---------------------------------------------------------------------

#[test]
fn queue_churn_matches_vecdeque_model() {
    use std::collections::VecDeque;

    let sizes: &[usize] = if cfg!(miri) { &[0, 1, 20] } else { &[0, 1, NODE_SIZE, 500, 5000] };
    let rounds = if cfg!(miri) { 30 } else { 6000 };

    for &n in sizes {
        let items = boundary_len(n);
        let mut v = PVector::from_slice(&items);
        let mut model: VecDeque<i32> = items.into_iter().collect();
        for i in 0..rounds {
            let (rest, popped) = v.pop_front_owned();
            let expected = model.pop_front();
            assert_eq!(popped, expected, "n={n}, round={i}");
            v = rest;

            let x = 10_000_000 + i;
            model.push_back(x);
            v = v.push_back_owned(x);

            v.validate();
            assert_eq!(v.len(), model.len(), "n={n}, round={i}");
            assert_eq!(v.iter().copied().collect::<Vec<_>>(), model.iter().copied().collect::<Vec<_>>(), "n={n}, round={i}");
        }
    }
}

// ---------------------------------------------------------------------
// SPEC-M13-EQWITH.md: `try_eq_by` against the obvious `zip` + predicate
// reference, across a matrix of view shapes (trivial / suffix / mid) and
// shared/disjoint backing trees.
// ---------------------------------------------------------------------

fn reference_eq_by(a: &PVector<i32>, b: &PVector<i32>, mut eq: impl FnMut(i32, i32) -> bool) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| eq(x, y))
}

/// `kind % 4`: 0 = trivial (plain clone), 1 = trivial via a full-range
/// `slice`, 2 = suffix view (`offset > 0`, window reaches the end), 3 =
/// mid view (`offset > 0`, window ends before the backing's end).
fn make_view(v: &PVector<i32>, kind: u8) -> PVector<i32> {
    let n = v.len();
    match kind % 4 {
        0 => v.clone(),
        1 => v.slice(0..n),
        2 => v.slice(n / 3..n),
        _ => v.slice(n / 4..n - n / 4),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 200 }))]

    #[test]
    fn try_eq_by_matches_reference_across_view_matrix(
        items in prop::collection::vec(any::<i32>(), 0..(if cfg!(miri) { 60 } else { 400 })),
        view_kind_a in 0u8..4,
        view_kind_b in 0u8..4,
        share in any::<bool>(),
        diverge in any::<bool>(),
    ) {
        let base_a = PVector::from_slice(&items);
        let mut other_items = items.clone();
        if diverge && !other_items.is_empty() {
            let idx = other_items.len() / 2;
            other_items[idx] = other_items[idx].wrapping_add(1);
        }
        // `share`: b's backing is a clone of a's (pointer-shared subtrees
        // exercise the pruning path); otherwise an independently built,
        // content-equal (unless `diverge`) tree.
        let base_b = if share && !diverge { base_a.clone() } else { PVector::from_slice(&other_items) };

        let a = make_view(&base_a, view_kind_a);
        let b = make_view(&base_b, view_kind_b);

        let expected = reference_eq_by(&a, &b, |x, y| x == y);
        let got = a.try_eq_by(&b, |x: &i32, y: &i32| Ok::<bool, std::convert::Infallible>(x == y)).unwrap();
        prop_assert_eq!(got, expected);
        // `try_eq_by` with a `==` predicate must agree with `PartialEq`.
        prop_assert_eq!(got, a == b);
    }
}
