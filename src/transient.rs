//! [`TransientMap`]: an owned, mutable builder for [`PersistentHashMap`].
//!
//! Clojure's transients exist to fake single-ownership on the JVM: `(transient
//! m)` hands back a mutable view and marks `m` itself unusable (any further
//! use of the original persistent handle throws at runtime), because the JVM
//! has no static way to prove nobody else still holds `m`. Rust doesn't have
//! that problem — ownership is real and static. [`PersistentHashMap::transient`]
//! *moves* `self`: there is no leftover persistent handle to accidentally
//! keep using, so there is nothing to invalidate and no runtime check is
//! needed. This is strictly safer than Clojure's contract (a use-after-transient
//! bug is a compile error here, not a runtime exception) at zero extra cost —
//! the move is O(1), no allocation.
//!
//! Internally, `TransientMap` simply owns a [`PersistentHashMap`] and drives
//! every op through the existing `assoc_owned`/`dissoc_owned` machinery: a
//! transient's root always starts out uniquely owned (nothing else can
//! reference it — `self` was moved in), so any node touched by an update is
//! either already refcount-1 (mutated in place) or gets copied exactly once
//! on first touch, after which every subsequent touch mutates in place. A
//! transient built from a freshly `clone()`d map (refcounts > 1 throughout)
//! naturally copies-on-first-write and then mutates freely — that copy-once
//! behavior *is* Clojure's transient semantic, it just falls out of the
//! refcount machinery for free instead of needing a separate edit-token
//! check on every op.

use std::hash::{BuildHasher, Hash};

use crate::PersistentHashMap;

/// An owned, mutable builder for [`PersistentHashMap`]. See the module docs
/// for how this relates to (and improves on) Clojure's transients.
///
/// Build one via [`PersistentHashMap::transient`], mutate with
/// [`assoc`](Self::assoc)/[`dissoc`](Self::dissoc), and finish with
/// [`persistent`](Self::persistent) — both conversions are O(1) moves.
///
/// This milestone does not add capacity slack (over-allocation headroom for
/// future growth): each `assoc`/`dissoc` sizes nodes exactly, same as the
/// persistent API. A capacity-slack strategy is left for a possible future
/// milestone if benchmarks show it's worth the extra bookkeeping.
pub struct TransientMap<K, V, S = crate::DefaultBuildHasher> {
    // Always `Some` except momentarily inside a method body, while the
    // owned map is handed off to `assoc_owned`/`dissoc_owned` (which take
    // `self` by value and return a new `Self`) and not yet put back.
    inner: Option<PersistentHashMap<K, V, S>>,
}

const INVARIANT: &str = "TransientMap invariant: `inner` is always `Some` between method calls";

impl<K, V, S> TransientMap<K, V, S> {
    pub(crate) fn from_persistent(m: PersistentHashMap<K, V, S>) -> Self {
        TransientMap { inner: Some(m) }
    }

    #[inline]
    fn as_map(&self) -> &PersistentHashMap<K, V, S> {
        self.inner.as_ref().expect(INVARIANT)
    }

    /// Finish building: an O(1) move back into an ordinary
    /// [`PersistentHashMap`].
    pub fn persistent(mut self) -> PersistentHashMap<K, V, S> {
        self.inner.take().expect(INVARIANT)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.as_map().len()
    }

    /// Whether the builder currently holds no entries.
    pub fn is_empty(&self) -> bool {
        self.as_map().is_empty()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> TransientMap<K, V, S> {
    /// Look up `k`.
    pub fn get(&self, k: &K) -> Option<&V> {
        self.as_map().get(k)
    }

    /// Whether `k` is present.
    pub fn contains_key(&self, k: &K) -> bool {
        self.as_map().contains_key(k)
    }
}

impl<K, V, S> TransientMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    /// Insert-or-replace, mutating in place (delegates to
    /// [`PersistentHashMap::assoc_owned`] — see the module docs for why
    /// that's always in-place or copy-once-then-in-place here). Same
    /// keep-existing-key-on-replace rule as [`PersistentHashMap::assoc`].
    pub fn assoc(&mut self, k: K, v: V) {
        let m = self.inner.take().expect(INVARIANT);
        self.inner = Some(m.assoc_owned(k, v));
    }
}

impl<K, V, S> TransientMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Remove `k`, mutating in place (delegates to
    /// [`PersistentHashMap::dissoc_owned`]). Returns whether `k` was
    /// present.
    pub fn dissoc(&mut self, k: &K) -> bool {
        let m = self.inner.take().expect(INVARIANT);
        let before = m.len();
        let m = m.dissoc_owned(k);
        let removed = m.len() != before;
        self.inner = Some(m);
        removed
    }
}

// ---------------------------------------------------------------------
// TransientSet: mirrors TransientMap exactly, wrapping a
// `TransientMap<K, (), S>` the same way `PersistentHashSet` wraps a
// `PersistentHashMap<K, (), S>`.
// ---------------------------------------------------------------------

use crate::set::PersistentHashSet;

/// An owned, mutable builder for [`PersistentHashSet`]. Mirrors
/// [`TransientMap`] exactly — see its docs for the semantics (in
/// particular: how this differs from, and improves on, Clojure's
/// edit-token transients).
///
/// Build one via [`PersistentHashSet::transient`], mutate with
/// [`insert`](Self::insert)/[`remove`](Self::remove), and finish with
/// [`persistent`](Self::persistent) — both conversions are O(1) moves.
pub struct TransientSet<K, S = crate::DefaultBuildHasher> {
    inner: TransientMap<K, (), S>,
}

impl<K, S> TransientSet<K, S> {
    pub(crate) fn from_persistent(inner: TransientMap<K, (), S>) -> Self {
        TransientSet { inner }
    }

    /// Finish building: an O(1) move back into an ordinary
    /// [`PersistentHashSet`].
    pub fn persistent(self) -> PersistentHashSet<K, S> {
        PersistentHashSet::from_map(self.inner.persistent())
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the builder currently holds no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K: Hash + Eq, S: BuildHasher> TransientSet<K, S> {
    /// Whether `k` is a member.
    pub fn contains(&self, k: &K) -> bool {
        self.inner.contains_key(k)
    }
}

impl<K, S> TransientSet<K, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Insert `k`, mutating in place.
    pub fn insert(&mut self, k: K) {
        self.inner.assoc(k, ());
    }

    /// Remove `k`, mutating in place. Returns whether `k` was present.
    pub fn remove(&mut self, k: &K) -> bool {
        self.inner.dissoc(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let m: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        let t = m.transient();
        assert_eq!(t.len(), 0);
        let m2 = t.persistent();
        assert!(m2.is_empty());
    }

    #[test]
    fn assoc_and_dissoc() {
        let mut t = PersistentHashMap::<i32, i32>::new().transient();
        t.assoc(1, 10);
        t.assoc(2, 20);
        assert_eq!(t.get(&1), Some(&10));
        assert_eq!(t.len(), 2);
        assert!(t.dissoc(&1));
        assert!(!t.dissoc(&1));
        assert_eq!(t.get(&1), None);
        assert_eq!(t.len(), 1);
        let m = t.persistent();
        assert_eq!(m.get(&2), Some(&20));
    }

    #[test]
    fn transient_from_shared_map_does_not_mutate_original() {
        let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100);
        let clone = m.clone();
        let mut t = m.transient();
        t.assoc(1, 999);
        t.assoc(2, 200);
        assert_eq!(clone.get(&1), Some(&100), "old snapshot must be unaffected");
        assert_eq!(clone.get(&2), None);
        let m2 = t.persistent();
        assert_eq!(m2.get(&1), Some(&999));
        assert_eq!(m2.get(&2), Some(&200));
    }

    #[test]
    fn set_round_trip_and_diverge() {
        let s = PersistentHashSet::<i32>::new().insert(1);
        let clone = s.clone();
        let mut t = s.transient();
        t.insert(2);
        assert!(t.remove(&1));
        assert!(!t.remove(&1));
        assert_eq!(t.len(), 1);
        assert!(clone.contains(&1), "old snapshot must be unaffected");
        assert!(!clone.contains(&2));
        let s2 = t.persistent();
        assert!(!s2.contains(&1));
        assert!(s2.contains(&2));
    }
}
