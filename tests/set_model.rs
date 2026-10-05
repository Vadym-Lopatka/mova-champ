//! Property tests: `PersistentHashSet` against `std::collections::HashSet`
//! as a model — mirrors `tests/model.rs`'s map coverage (random op
//! sequences under both the default hasher and deliberately colliding
//! hashers, plus transient round-trips interleaved with clone-and-diverge).

use champ::{DefaultBuildHasher, PersistentHashSet};
use proptest::prelude::*;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};

#[derive(Clone, Debug)]
enum Op {
    Insert(i32),
    Remove(i32),
    Contains(i32),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let key = -40..40i32;
    prop_oneof![
        key.clone().prop_map(Op::Insert),
        key.clone().prop_map(Op::Remove),
        key.prop_map(Op::Contains),
    ]
}

fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
    // Miri is orders of magnitude slower than native execution; cap
    // sequence length under it (combine with `PROPTEST_CASES=<small>` when
    // running `cargo +nightly miri test`).
    #[cfg(miri)]
    let max_len = 20;
    #[cfg(not(miri))]
    let max_len = 200;
    prop::collection::vec(op_strategy(), 0..max_len)
}

/// A `BuildHasher` that masks a good hash down to few bits, forcing
/// collisions and deep chains for small key ranges like the ones proptest
/// generates above (same shape as the one in `tests/model.rs`, duplicated
/// per that file's existing per-integration-test-binary convention).
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
    let mut model: HashSet<i32> = HashSet::new();
    let mut s: PersistentHashSet<i32, S> = PersistentHashSet::with_hasher(hasher);
    for op in ops {
        match *op {
            Op::Insert(k) => {
                model.insert(k);
                s = s.insert_owned(k);
            }
            Op::Remove(k) => {
                model.remove(&k);
                s = s.remove_owned(&k);
            }
            Op::Contains(k) => {
                assert_eq!(s.contains(&k), model.contains(&k), "contains({k}) mismatch");
            }
        }
        assert_eq!(s.len(), model.len(), "len mismatch after {op:?}");
    }
    s.validate();
    for k in &model {
        assert!(s.contains(k));
    }
    assert_eq!(s.len(), model.len());
}

proptest! {
    #[test]
    fn model_default_hasher(ops in ops_strategy()) {
        run_against_model(DefaultBuildHasher, &ops);
    }

    #[test]
    fn model_mod8_hasher(ops in ops_strategy()) {
        run_against_model(BadBuildHasher { mask: 0x7 }, &ops);
    }

    #[test]
    fn model_and3_hasher(ops in ops_strategy()) {
        run_against_model(BadBuildHasher { mask: 0x3 }, &ops);
    }

    /// Mirrors `tests/model.rs`'s `transient_round_trip_matches_model`:
    /// persistent -> transient -> batch ops -> persistent stays in sync
    /// with the model, and a set cloned BEFORE the transient conversion
    /// is unaffected by the transient's mutations.
    #[test]
    fn transient_round_trip_matches_model(ops_before in ops_strategy(), ops_transient in ops_strategy()) {
        let mut model: HashSet<i32> = HashSet::new();
        let mut s: PersistentHashSet<i32> = PersistentHashSet::new();
        for op in &ops_before {
            match *op {
                Op::Insert(k) => {
                    model.insert(k);
                    s = s.insert_owned(k);
                }
                Op::Remove(k) => {
                    model.remove(&k);
                    s = s.remove_owned(&k);
                }
                Op::Contains(_) => {}
            }
        }
        s.validate();

        let snapshot = s.clone();
        let snapshot_model = model.clone();

        let mut t = s.transient();
        for op in &ops_transient {
            match *op {
                Op::Insert(k) => {
                    model.insert(k);
                    t.insert(k);
                }
                Op::Remove(k) => {
                    model.remove(&k);
                    t.remove(&k);
                }
                Op::Contains(k) => {
                    prop_assert_eq!(t.contains(&k), model.contains(&k));
                }
            }
            prop_assert_eq!(t.len(), model.len());
        }
        let s2 = t.persistent();
        s2.validate();
        for k in &model {
            prop_assert!(s2.contains(k));
        }
        prop_assert_eq!(s2.len(), model.len());

        snapshot.validate();
        for k in &snapshot_model {
            prop_assert!(snapshot.contains(k), "old snapshot corrupted by transient mutation");
        }
        prop_assert_eq!(snapshot.len(), snapshot_model.len());
    }
}
