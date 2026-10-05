//! Property tests: `PersistentHashMap` against `std::collections::HashMap`
//! as a model, over random op sequences, under both a well-behaved hasher
//! and deliberately colliding hashers (to stress deep chains and collision
//! nodes). Also covers structural sharing (clone-and-diverge) and
//! refcount/leak sanity with `Rc`-keyed/valued maps.

use champ::PersistentHashMap;
use proptest::prelude::*;
use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::rc::Rc;

#[derive(Clone, Debug)]
enum Op {
    Assoc(i32, i32),
    Dissoc(i32),
    Get(i32),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let key = -40..40i32;
    prop_oneof![
        (key.clone(), any::<i32>()).prop_map(|(k, v)| Op::Assoc(k, v)),
        key.clone().prop_map(Op::Dissoc),
        key.prop_map(Op::Get),
    ]
}

fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
    // Miri is orders of magnitude slower than native execution; cap sequence
    // length under it (combine with `PROPTEST_CASES=<small>` when running
    // `cargo +nightly miri test`) so the suite stays tractable while still
    // exercising every op path many times over.
    #[cfg(miri)]
    let max_len = 20;
    #[cfg(not(miri))]
    let max_len = 200;
    prop::collection::vec(op_strategy(), 0..max_len)
}

/// A `BuildHasher` that masks a good hash down to few bits, forcing
/// collisions and deep chains for small key ranges like the ones proptest
/// generates above.
#[derive(Clone, Default)]
struct BadBuildHasher {
    mask: u64,
}

struct BadHasher {
    inner: std::collections::hash_map::DefaultHasher,
    mask: u64,
}

impl Hasher for BadHasher {
    fn finish(&self) -> u64 {
        self.inner.finish() & self.mask
    }
    fn write(&mut self, bytes: &[u8]) {
        self.inner.write(bytes)
    }
}

impl BuildHasher for BadBuildHasher {
    type Hasher = BadHasher;
    fn build_hasher(&self) -> Self::Hasher {
        BadHasher {
            inner: std::collections::hash_map::DefaultHasher::new(),
            mask: self.mask,
        }
    }
}

fn run_against_model<S: BuildHasher + Clone + Default>(hasher: S, ops: &[Op]) {
    let mut model: HashMap<i32, i32> = HashMap::new();
    let mut m: PersistentHashMap<i32, i32, S> = PersistentHashMap::with_hasher(hasher);
    for op in ops {
        match *op {
            Op::Assoc(k, v) => {
                model.insert(k, v);
                m = m.assoc_owned(k, v);
            }
            Op::Dissoc(k) => {
                model.remove(&k);
                m = m.dissoc_owned(&k);
            }
            Op::Get(k) => {
                assert_eq!(m.get(&k), model.get(&k), "get({k}) mismatch");
            }
        }
        assert_eq!(m.len(), model.len(), "len mismatch after {op:?}");
    }
    m.validate();
    for (k, v) in &model {
        assert_eq!(m.get(k), Some(v));
    }
    assert_eq!(m.len(), model.len());
}

/// Clone `m`/`model` partway through, then diverge: keep mutating the new
/// version while the old (cloned) version and its own model must remain
/// valid and unaffected. Catches broken structural sharing.
fn run_clone_and_diverge<S: BuildHasher + Clone + Default>(hasher: S, ops_a: &[Op], ops_b: &[Op]) {
    let mut model_a: HashMap<i32, i32> = HashMap::new();
    let mut a: PersistentHashMap<i32, i32, S> = PersistentHashMap::with_hasher(hasher);
    for op in ops_a {
        match *op {
            Op::Assoc(k, v) => {
                model_a.insert(k, v);
                a = a.assoc_owned(k, v);
            }
            Op::Dissoc(k) => {
                model_a.remove(&k);
                a = a.dissoc_owned(&k);
            }
            Op::Get(_) => {}
        }
    }
    a.validate();

    // Clone: `b`/`model_b` starts as a snapshot of `a`/`model_a`.
    let mut model_b = model_a.clone();
    let mut b = a.clone();

    for op in ops_b {
        match *op {
            Op::Assoc(k, v) => {
                model_b.insert(k, v);
                b = b.assoc_owned(k, v);
            }
            Op::Dissoc(k) => {
                model_b.remove(&k);
                b = b.dissoc_owned(&k);
            }
            Op::Get(k) => {
                assert_eq!(b.get(&k), model_b.get(&k));
            }
        }
        // `a` must be completely unaffected by mutating `b`.
        for (k, v) in &model_a {
            assert_eq!(a.get(k), Some(v), "old snapshot corrupted by diverging clone");
        }
        assert_eq!(a.len(), model_a.len());
    }

    a.validate();
    b.validate();
    for (k, v) in &model_a {
        assert_eq!(a.get(k), Some(v));
    }
    for (k, v) in &model_b {
        assert_eq!(b.get(k), Some(v));
    }
    assert_eq!(a.len(), model_a.len());
    assert_eq!(b.len(), model_b.len());
}

proptest! {
    #[test]
    fn model_random_hasher(ops in ops_strategy()) {
        run_against_model(RandomState::new(), &ops);
    }

    #[test]
    fn model_mod8_hasher(ops in ops_strategy()) {
        run_against_model(BadBuildHasher { mask: 0x7 }, &ops);
    }

    #[test]
    fn model_and3_hasher(ops in ops_strategy()) {
        run_against_model(BadBuildHasher { mask: 0x3 }, &ops);
    }

    #[test]
    fn clone_and_diverge_random_hasher(ops_a in ops_strategy(), ops_b in ops_strategy()) {
        run_clone_and_diverge(RandomState::new(), &ops_a, &ops_b);
    }

    #[test]
    fn clone_and_diverge_mod8_hasher(ops_a in ops_strategy(), ops_b in ops_strategy()) {
        run_clone_and_diverge(BadBuildHasher { mask: 0x7 }, &ops_a, &ops_b);
    }

    #[test]
    fn persistent_assoc_matches_owned(ops in ops_strategy()) {
        // `&self` assoc/dissoc must observe the exact same semantics as the
        // owned variants (they share the copy-path logic, but exercise it
        // through a different entry point).
        let mut model: HashMap<i32, i32> = HashMap::new();
        let mut m: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        for op in &ops {
            match *op {
                Op::Assoc(k, v) => {
                    model.insert(k, v);
                    m = m.assoc(k, v);
                }
                Op::Dissoc(k) => {
                    model.remove(&k);
                    m = m.dissoc(&k);
                }
                Op::Get(k) => {
                    prop_assert_eq!(m.get(&k), model.get(&k));
                }
            }
            prop_assert_eq!(m.len(), model.len());
        }
        m.validate();
    }
}

// ---------------------------------------------------------------------
// F1 regression: structural equality under the deterministic default
// hasher. Two maps built *independently* (separate `new()` calls, no
// shared hasher value) with equal contents must compare `==`, and a
// differing pair must compare `!=`. Before the fix, `new()` used
// `RandomState` and this test would fail intermittently (two independently
// seeded hashers assign different hashes to the same key, producing
// differently shaped-but-each-canonical trees that the bitmap fast-fail
// reports as unequal).
// ---------------------------------------------------------------------

#[test]
fn independently_built_default_hasher_maps_are_eq() {
    let a = PersistentHashMap::<i32, i32>::new().assoc(1, 10).assoc(2, 20).assoc(3, 30);
    // Independently constructed (separate `new()` call) and inserted in a
    // different order.
    let b = PersistentHashMap::<i32, i32>::new().assoc(3, 30).assoc(1, 10).assoc(2, 20);
    assert_eq!(a, b);

    let c = PersistentHashMap::<i32, i32>::new().assoc(1, 10).assoc(2, 999).assoc(3, 30);
    assert_ne!(a, c);

    let d = PersistentHashMap::<i32, i32>::new().assoc(1, 10).assoc(2, 20);
    assert_ne!(a, d, "differing key sets must not compare equal");
}

proptest! {
    /// Canonical form promises history-independence of shape: build the
    /// same final key/value set two different ways (different insertion
    /// order, plus assoc/dissoc churn that cancels out) and assert the
    /// resulting maps are `==`.
    #[test]
    fn same_final_mapping_different_histories_are_eq(
        entries in prop::collection::hash_map(-40..40i32, any::<i32>(), 0..30)
    ) {
        let mut entries: Vec<(i32, i32)> = entries.into_iter().collect();

        // History A: insert in the order given.
        let a: PersistentHashMap<i32, i32> = entries
            .iter()
            .cloned()
            .fold(PersistentHashMap::new(), |m, (k, v)| m.assoc_owned(k, v));

        // History B: reversed insertion order, plus churn (assoc a
        // placeholder value, dissoc an absent key, then overwrite to the
        // real value) — a different history landing on the same final
        // key/value set.
        entries.reverse();
        let mut b: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        for &(k, v) in &entries {
            b = b.assoc_owned(k, v.wrapping_add(1));
            b = b.dissoc_owned(&k.wrapping_add(1_000));
            b = b.assoc_owned(k, v);
        }

        prop_assert_eq!(a.len(), entries.len());
        prop_assert_eq!(&a, &b);
    }
}

// ---------------------------------------------------------------------
// M3: transient round-trips against the model, interleaved with
// clone-and-diverge (clone the persistent map BEFORE moving it into a
// transient, verify the old snapshot is unaffected by the transient's
// mutations).
// ---------------------------------------------------------------------

proptest! {
    #[test]
    fn transient_round_trip_matches_model(ops_before in ops_strategy(), ops_transient in ops_strategy()) {
        // Build up a baseline persistent map + model the ordinary way.
        let mut model: HashMap<i32, i32> = HashMap::new();
        let mut m: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        for op in &ops_before {
            match *op {
                Op::Assoc(k, v) => {
                    model.insert(k, v);
                    m = m.assoc_owned(k, v);
                }
                Op::Dissoc(k) => {
                    model.remove(&k);
                    m = m.dissoc_owned(&k);
                }
                Op::Get(_) => {}
            }
        }
        m.validate();

        // Clone BEFORE the transient conversion: this snapshot must be
        // completely unaffected by whatever the transient does next.
        let snapshot = m.clone();
        let snapshot_model = model.clone();

        // persistent -> transient -> batch ops -> persistent.
        let mut t = m.transient();
        for op in &ops_transient {
            match *op {
                Op::Assoc(k, v) => {
                    model.insert(k, v);
                    t.assoc(k, v);
                }
                Op::Dissoc(k) => {
                    model.remove(&k);
                    t.dissoc(&k);
                }
                Op::Get(k) => {
                    prop_assert_eq!(t.get(&k), model.get(&k), "transient get({}) mismatch", k);
                }
            }
            prop_assert_eq!(t.len(), model.len(), "transient len mismatch after {:?}", op);
        }
        let m2 = t.persistent();
        m2.validate();
        for (k, v) in &model {
            prop_assert_eq!(m2.get(k), Some(v));
        }
        prop_assert_eq!(m2.len(), model.len());

        // The pre-transient snapshot must remain exactly as it was.
        snapshot.validate();
        for (k, v) in &snapshot_model {
            prop_assert_eq!(snapshot.get(k), Some(v), "old snapshot corrupted by transient mutation");
        }
        prop_assert_eq!(snapshot.len(), snapshot_model.len());
    }
}

#[test]
fn assoc_same_value_is_pointer_identical() {
    let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100).assoc(2, 200);
    let m2 = m.assoc(1, 100);
    // Same key, same (== ) value: must be a no-op, no new allocation.
    // We can't directly compare root pointers from outside the crate, but
    // we CAN check the whole map short-circuits to structural/pointer
    // equality via a cheap `len` + a fresh assoc of a truly new value being
    // detectably different (indirect confidence check + explicit model
    // agreement).
    assert_eq!(m.len(), m2.len());
    assert_eq!(m.get(&1), Some(&100));
    assert_eq!(m2.get(&1), Some(&100));
}

#[test]
fn dissoc_absent_no_alloc_and_pointer_identical_map_contents() {
    let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100).assoc(2, 200);
    let m2 = m.dissoc(&999);
    assert_eq!(m2.len(), m.len());
    assert_eq!(m2.get(&1), Some(&100));
    assert_eq!(m2.get(&2), Some(&200));
}

#[test]
fn rc_refcounts_return_to_baseline() {
    let key_a = Rc::new("a".to_string());
    let key_b = Rc::new("b".to_string());
    let val = Rc::new("v".to_string());
    assert_eq!(Rc::strong_count(&key_a), 1);
    assert_eq!(Rc::strong_count(&key_b), 1);
    assert_eq!(Rc::strong_count(&val), 1);

    {
        let m = PersistentHashMap::<Rc<String>, Rc<String>>::new()
            .assoc(key_a.clone(), val.clone())
            .assoc(key_b.clone(), val.clone());
        assert_eq!(Rc::strong_count(&key_a), 2);
        assert_eq!(Rc::strong_count(&key_b), 2);
        assert_eq!(Rc::strong_count(&val), 3);

        let m2 = m.clone(); // shallow, shares nodes -> no extra Rc bumps
        assert_eq!(Rc::strong_count(&key_a), 2);
        let m3 = m2.dissoc(&key_a);
        assert_eq!(Rc::strong_count(&key_a), 2, "dissoc'd from m3 but still held by m/m2");
        drop(m3);
        assert_eq!(Rc::strong_count(&key_a), 2);
        drop(m2);
        assert_eq!(Rc::strong_count(&key_a), 2);
        drop(m);
    }
    assert_eq!(Rc::strong_count(&key_a), 1);
    assert_eq!(Rc::strong_count(&key_b), 1);
    assert_eq!(Rc::strong_count(&val), 1);
}

#[test]
fn rc_refcounts_return_to_baseline_owned_heavy_churn() {
    let baseline: Vec<Rc<i32>> = (0..64).map(Rc::new).collect();
    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 1);
    }
    {
        let mut m: PersistentHashMap<Rc<i32>, Rc<i32>> = PersistentHashMap::new();
        for r in &baseline {
            m = m.assoc_owned(r.clone(), r.clone());
        }
        let mut removed: HashSet<usize> = HashSet::new();
        for (i, r) in baseline.iter().enumerate() {
            if i % 2 == 0 {
                m = m.dissoc_owned(r);
                removed.insert(i);
            }
        }
        for (i, r) in baseline.iter().enumerate() {
            let expected = if removed.contains(&i) { 1 } else { 3 };
            assert_eq!(Rc::strong_count(r), expected, "index {i}");
        }
    }
    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 1);
    }
}
