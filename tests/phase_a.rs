//! M5 Phase A integration API tests (SPEC-INTEGRATION.md):
//! `ptr_eq` (map + set), `assoc_owned_replacing`, and the consuming
//! `IntoIterator` impls (map + set).

use champ::{PersistentHashMap, PersistentHashSet};
use proptest::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

// ---------------------------------------------------------------------
// assoc_owned_replacing — model test against std::collections::HashMap's
// `insert` return contract (Option<V> = the value previously associated
// with the key, if any) — the exact semantics this method is meant to
// match, per SPEC-INTEGRATION.md.
// ---------------------------------------------------------------------

proptest! {
    #[test]
    fn assoc_owned_replacing_matches_std_insert_return(
        entries in prop::collection::vec((-40..40i32, any::<i32>()), 0..200)
    ) {
        let mut model: HashMap<i32, i32> = HashMap::new();
        let mut m: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        for (k, v) in entries {
            let model_old = model.insert(k, v);
            let (m2, old) = m.assoc_owned_replacing(k, v);
            m = m2;
            prop_assert_eq!(old, model_old, "assoc_owned_replacing({}, {}) old-value mismatch", k, v);
            prop_assert_eq!(m.len(), model.len());
            prop_assert_eq!(m.get(&k), Some(&v));
        }
        m.validate();
        for (k, v) in &model {
            prop_assert_eq!(m.get(k), Some(v));
        }
    }
}

#[test]
fn assoc_owned_replacing_returns_old_value_and_none_for_fresh_key() {
    let m = PersistentHashMap::<i32, i32>::new();

    let (m, old) = m.assoc_owned_replacing(1, 100);
    assert_eq!(old, None, "fresh key: no old value");
    assert_eq!(m.get(&1), Some(&100));

    let (m, old) = m.assoc_owned_replacing(1, 200);
    assert_eq!(old, Some(100), "existing key, changed value: old value returned");
    assert_eq!(m.get(&1), Some(&200));

    // Same key, same value re-inserted: the map is observably unchanged,
    // but the old value must still come back (per spec's "Unchanged
    // shortcut returns a clone of the existing value").
    let (m, old) = m.assoc_owned_replacing(1, 200);
    assert_eq!(old, Some(200), "unchanged re-insert still returns the (unchanged) old value");
    assert_eq!(m.get(&1), Some(&200));
    assert_eq!(m.len(), 1);
}

#[test]
fn assoc_owned_replacing_copy_path_when_shared() {
    // Force the non-unique (copy-path) branch: `shared` holds a second
    // reference to `m`'s root, so `m`'s own subsequent
    // `assoc_owned_replacing` call cannot mutate in place.
    let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100).assoc(2, 200);
    let shared = m.clone();

    let (m2, old) = m.assoc_owned_replacing(1, 999);
    assert_eq!(old, Some(100));
    assert_eq!(m2.get(&1), Some(&999));
    assert_eq!(m2.get(&2), Some(&200));
    assert_eq!(m2.len(), 2);

    // The snapshot taken before the mutation must be completely unaffected.
    assert_eq!(shared.get(&1), Some(&100));
    assert_eq!(shared.get(&2), Some(&200));
}

#[test]
fn assoc_owned_replacing_new_key_via_copy_path() {
    let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100);
    let _shared = m.clone();
    let (m2, old) = m.assoc_owned_replacing(2, 200);
    assert_eq!(old, None);
    assert_eq!(m2.len(), 2);
    assert_eq!(m2.get(&1), Some(&100));
    assert_eq!(m2.get(&2), Some(&200));
}

// ---------------------------------------------------------------------
// ptr_eq — true/false/empty cases, map and set.
// ---------------------------------------------------------------------

#[test]
fn map_ptr_eq_cases() {
    let empty_a: PersistentHashMap<i32, i32> = PersistentHashMap::new();
    let empty_b: PersistentHashMap<i32, i32> = PersistentHashMap::new();
    assert!(empty_a.ptr_eq(&empty_b), "both-empty counts as true");

    let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100);
    let m_clone = m.clone(); // O(1) shallow clone: shares the root allocation
    assert!(m.ptr_eq(&m_clone), "clone shares the same root");

    let m_unchanged = m.assoc(1, 100); // same key+value: pointer-identical fast path
    assert!(m.ptr_eq(&m_unchanged), "no-op assoc returns the same root");

    let m2 = m.assoc(2, 200); // a real structural change: new root
    assert!(!m.ptr_eq(&m2), "structurally different maps must not ptr_eq");

    assert!(!m.ptr_eq(&empty_a), "non-empty vs empty must not ptr_eq");
    assert!(!empty_a.ptr_eq(&m), "asymmetric case, other direction");
}

#[test]
fn set_ptr_eq_cases() {
    let empty_a: PersistentHashSet<i32> = PersistentHashSet::new();
    let empty_b: PersistentHashSet<i32> = PersistentHashSet::new();
    assert!(empty_a.ptr_eq(&empty_b), "both-empty counts as true");

    let s = PersistentHashSet::<i32>::new().insert(1);
    let s_clone = s.clone();
    assert!(s.ptr_eq(&s_clone), "clone shares the same root");

    let s_unchanged = s.insert(1); // already present: pointer-identical fast path
    assert!(s.ptr_eq(&s_unchanged));

    let s2 = s.insert(2);
    assert!(!s.ptr_eq(&s2), "structurally different sets must not ptr_eq");
    assert!(!s.ptr_eq(&empty_a));
}

// ---------------------------------------------------------------------
// Consuming iterator contents == borrowing iterator contents (map + set),
// in the same (deterministic, hash-order) sequence.
// ---------------------------------------------------------------------

proptest! {
    #[test]
    fn map_consuming_iter_matches_borrowing_iter(
        entries in prop::collection::hash_map(-100..100i32, any::<i32>(), 0..150)
    ) {
        let m: PersistentHashMap<i32, i32> = entries
            .into_iter()
            .fold(PersistentHashMap::new(), |m, (k, v)| m.assoc_owned(k, v));

        let borrowed: Vec<(i32, i32)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        let consumed: Vec<(i32, i32)> = m.clone().into_iter().collect();
        prop_assert_eq!(borrowed, consumed);
    }

    #[test]
    fn set_consuming_iter_matches_borrowing_iter(
        entries in prop::collection::hash_set(-100..100i32, 0..150)
    ) {
        let s: PersistentHashSet<i32> = entries
            .into_iter()
            .fold(PersistentHashSet::new(), |s, k| s.insert(k));

        let borrowed: Vec<i32> = s.iter().copied().collect();
        let consumed: Vec<i32> = s.clone().into_iter().collect();
        prop_assert_eq!(borrowed, consumed);
    }
}

#[test]
fn map_into_iter_empty() {
    let m = PersistentHashMap::<i32, i32>::new();
    let collected: Vec<(i32, i32)> = m.into_iter().collect();
    assert!(collected.is_empty());
}

#[test]
fn set_into_iter_empty() {
    let s = PersistentHashSet::<i32>::new();
    let collected: Vec<i32> = s.into_iter().collect();
    assert!(collected.is_empty());
}

// ---------------------------------------------------------------------
// Refcount-leak tests with `Rc` values for the consuming iterator: full
// consumption, and early-drop (abandoned mid-iteration) — the latter is
// the case that specifically exercises `IntoIter`'s `Drop` impl, since
// `NodePtr` itself implements neither `Clone` nor `Drop` (every release
// must be an explicit `drop_node()` call somewhere).
// ---------------------------------------------------------------------

#[test]
fn consuming_iter_rc_refcounts_full_consumption() {
    let baseline: Vec<Rc<i32>> = (0..40).map(Rc::new).collect();
    let mut m: PersistentHashMap<Rc<i32>, Rc<i32>> = PersistentHashMap::new();
    for r in &baseline {
        m = m.assoc_owned(r.clone(), r.clone());
    }
    // baseline vec's own clone + the map's key slot + the map's value slot.
    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 3);
    }

    let collected: Vec<(Rc<i32>, Rc<i32>)> = m.into_iter().collect();
    assert_eq!(collected.len(), baseline.len());
    // The map itself is fully consumed and dropped by `into_iter()`
    // reaching exhaustion; `collected` holds its own fresh clones.
    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 3, "baseline vec + collected key clone + collected value clone");
    }

    drop(collected);
    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 1, "back to baseline after dropping the collected clones");
    }
}

#[test]
fn consuming_iter_rc_refcounts_early_drop_does_not_leak() {
    let baseline: Vec<Rc<i32>> = (0..40).map(Rc::new).collect();
    let mut m: PersistentHashMap<Rc<i32>, Rc<i32>> = PersistentHashMap::new();
    for r in &baseline {
        m = m.assoc_owned(r.clone(), r.clone());
    }

    let mut it = m.into_iter();
    // Consume a handful of entries (each immediately dropped) and then
    // abandon the iterator without exhausting it -- every not-yet-visited
    // node's still-owned (K, V) pairs must be released via `IntoIter`'s
    // `Drop` impl, recursively, through `drop_node()`, not leaked.
    for _ in 0..5 {
        let _ = it.next();
    }
    drop(it);

    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 1, "index {r}: no leak from early-dropped consuming iterator");
    }
}

#[test]
fn consuming_iter_rc_refcounts_immediate_drop_does_not_leak() {
    // The extreme case of "early drop": never call `next()` at all.
    let baseline: Vec<Rc<i32>> = (0..16).map(Rc::new).collect();
    let mut m: PersistentHashMap<Rc<i32>, Rc<i32>> = PersistentHashMap::new();
    for r in &baseline {
        m = m.assoc_owned(r.clone(), r.clone());
    }
    let it = m.into_iter();
    drop(it);
    for r in &baseline {
        assert_eq!(Rc::strong_count(r), 1);
    }
}
