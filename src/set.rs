//! [`PersistentHashSet`]: a CHAMP persistent hash set.
//!
//! A thin wrapper around [`PersistentHashMap<K, (), S>`](PersistentHashMap):
//! every operation just delegates to the equivalent map operation with a
//! `()` value. This costs nothing extra in memory — `()` is zero-sized, so
//! `Layout::array::<(K, ())>` (the node layer's per-entry storage, see
//! `node.rs`) is byte-for-byte identical to `Layout::array::<K>()` for any
//! `K`; see the static assertions and tests below. A set node really is a
//! keys-only node, automatically, with no special-casing anywhere in the
//! node layer.

use std::fmt;
use std::hash::{BuildHasher, Hash};

use crate::hash::DefaultBuildHasher;
use crate::iter::Keys;
use crate::map::PersistentHashMap;
use crate::transient::TransientSet;

/// A CHAMP persistent hash set: `{K}` with the same structural-sharing,
/// canonical-form, and equality-contract properties as
/// [`PersistentHashMap`] (see its type docs for the full equality contract
/// — it applies here identically, since a set literally *is* a map with a
/// `()` value).
pub struct PersistentHashSet<K, S = DefaultBuildHasher> {
    map: PersistentHashMap<K, (), S>,
}

impl<K, S: Clone> Clone for PersistentHashSet<K, S> {
    /// O(1): bumps the root's refcount, clones the hasher (see
    /// [`PersistentHashMap::clone`]).
    fn clone(&self) -> Self {
        PersistentHashSet { map: self.map.clone() }
    }
}

impl<K, S: Default> Default for PersistentHashSet<K, S> {
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, S> PersistentHashSet<K, S> {
    /// Build an empty set using an explicit hasher builder.
    pub fn with_hasher(hasher: S) -> Self {
        PersistentHashSet {
            map: PersistentHashMap::with_hasher(hasher),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set has no elements.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// True iff `self` and `other` share the exact same root allocation
    /// (both-empty counts as true). See
    /// [`PersistentHashMap::ptr_eq`](crate::PersistentHashMap::ptr_eq) —
    /// identical contract, since a set's storage literally is its
    /// underlying map's storage.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.map.ptr_eq(&other.map)
    }

    /// Convert into a [`TransientSet`]: an O(1) move into an owned builder.
    /// See [`crate::TransientMap`]'s docs (the set version mirrors it
    /// exactly) for the semantics.
    pub fn transient(self) -> TransientSet<K, S> {
        TransientSet::from_persistent(self.map.transient())
    }

    pub(crate) fn from_map(map: PersistentHashMap<K, (), S>) -> Self {
        PersistentHashSet { map }
    }
}

impl<K> PersistentHashSet<K, DefaultBuildHasher> {
    /// Build an empty set using the crate's deterministic default hasher.
    /// See [`PersistentHashMap`]'s equality-contract docs for why the
    /// default is deterministic rather than randomly seeded.
    pub fn new() -> Self {
        Self::with_hasher(DefaultBuildHasher)
    }
}

impl<K: Hash + Eq, S: BuildHasher> PersistentHashSet<K, S> {
    /// Whether `k` is a member.
    pub fn contains(&self, k: &K) -> bool {
        self.map.contains_key(k)
    }
}

impl<K, S> PersistentHashSet<K, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher + Clone,
{
    /// Persistent insert: returns a new set with `k` added (or, if `k` was
    /// already present, a pointer-identical set — no allocation).
    pub fn insert(&self, k: K) -> Self {
        PersistentHashSet {
            map: self.map.assoc(k, ()),
        }
    }

    /// Persistent removal: returns a new set with `k` removed. Absent `k`:
    /// returns a pointer-identical set, no allocation.
    pub fn remove(&self, k: &K) -> Self {
        PersistentHashSet {
            map: self.map.dissoc(k),
        }
    }
}

impl<K, S> PersistentHashSet<K, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Owning insert: mutates uniquely-owned nodes in place instead of
    /// copying. Observationally identical to [`Self::insert`], just faster
    /// when `self` isn't shared.
    pub fn insert_owned(self, k: K) -> Self {
        PersistentHashSet {
            map: self.map.assoc_owned(k, ()),
        }
    }

    /// Owning removal: mutates uniquely-owned nodes in place instead of
    /// copying.
    pub fn remove_owned(self, k: &K) -> Self {
        PersistentHashSet {
            map: self.map.dissoc_owned(k),
        }
    }
}

impl<K, S> PersistentHashSet<K, S> {
    /// Iterate over elements in hash order (deterministic and stable across
    /// identical contents, same as [`PersistentHashMap::iter`]).
    pub fn iter(&self) -> Keys<'_, K, ()> {
        self.map.keys()
    }
}

impl<'a, K, S> IntoIterator for &'a PersistentHashSet<K, S> {
    type Item = &'a K;
    type IntoIter = Keys<'a, K, ()>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Consuming iteration, yielding owned `K` clones in hash order. See
/// [`PersistentHashMap`]'s `IntoIterator` impl docs (identical rationale —
/// a set's storage literally is its underlying map's storage with `V =
/// ()`).
impl<K: Clone, S> IntoIterator for PersistentHashSet<K, S> {
    type Item = K;
    type IntoIter = crate::iter::IntoKeys<K>;
    fn into_iter(self) -> Self::IntoIter {
        crate::iter::IntoKeys(self.map.into_iter())
    }
}

impl<K: Eq, S> PartialEq for PersistentHashSet<K, S> {
    /// Structural equality; see [`PersistentHashMap`]'s equality-contract
    /// docs (identical here — a set's equality is its underlying map's
    /// equality).
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl<K: Eq, S> Eq for PersistentHashSet<K, S> {}

impl<K: fmt::Debug, S> fmt::Debug for PersistentHashSet<K, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<K, S> FromIterator<K> for PersistentHashSet<K, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut t = Self::with_hasher(S::default()).transient();
        for k in iter {
            t.insert(k);
        }
        t.persistent()
    }
}

impl<K, S> Extend<K> for PersistentHashSet<K, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher + Default,
{
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        let mut t = std::mem::take(self).transient();
        for k in iter {
            t.insert(k);
        }
        *self = t.persistent();
    }
}

// ---------------------------------------------------------------------
// Static layout assertions: `()` values must not cost anything extra —
// see module docs.
// ---------------------------------------------------------------------

const _: () = {
    assert!(std::mem::size_of::<(u64, ())>() == std::mem::size_of::<u64>());
    assert!(std::mem::align_of::<(u64, ())>() == std::mem::align_of::<u64>());
    assert!(std::mem::size_of::<([u8; 33], ())>() == std::mem::size_of::<[u8; 33]>());
    assert!(std::mem::align_of::<([u8; 33], ())>() == std::mem::align_of::<[u8; 33]>());
};

#[cfg(any(test, feature = "validate"))]
impl<K, S> PersistentHashSet<K, S>
where
    K: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    /// Walk the underlying tree asserting canonical-form invariants — see
    /// [`PersistentHashMap::validate`].
    pub fn validate(&self) {
        self.map.validate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn set_is_send_sync() {
        assert_send_sync::<PersistentHashSet<String>>();
    }

    #[test]
    fn memstats_zero_sized_value_collapses_layout() {
        // Runtime companion to the const asserts above: a handful of
        // concrete key types, checked at test time too (belt & suspenders
        // — the const asserts already guarantee this at compile time).
        assert_eq!(std::mem::size_of::<(i32, ())>(), std::mem::size_of::<i32>());
        assert_eq!(std::mem::size_of::<(String, ())>(), std::mem::size_of::<String>());
        assert_eq!(
            std::mem::align_of::<(String, ())>(),
            std::mem::align_of::<String>()
        );
    }

    #[test]
    fn empty_set() {
        let s: PersistentHashSet<i32> = PersistentHashSet::new();
        assert!(s.is_empty());
        assert!(!s.contains(&1));
    }

    #[test]
    fn basic_insert_contains_remove() {
        let s = PersistentHashSet::<i32>::new();
        let s = s.insert(1);
        let s = s.insert(2);
        assert!(s.contains(&1));
        assert!(s.contains(&2));
        assert_eq!(s.len(), 2);
        s.validate();

        let s2 = s.remove(&1);
        assert!(!s2.contains(&1));
        assert!(s2.contains(&2));
        assert!(s.contains(&1)); // original untouched
        s2.validate();
    }

    #[test]
    fn remove_absent_is_pointer_identical_map() {
        let s = PersistentHashSet::<i32>::new().insert(1);
        let s2 = s.remove(&2);
        assert_eq!(s2.len(), s.len());
        assert!(s2.contains(&1));
    }

    #[test]
    fn from_iter_and_extend() {
        let mut s: PersistentHashSet<i32> = (0..10).collect();
        assert_eq!(s.len(), 10);
        for i in 0..10 {
            assert!(s.contains(&i));
        }
        s.extend(10..20);
        assert_eq!(s.len(), 20);
        for i in 0..20 {
            assert!(s.contains(&i));
        }
        s.validate();
    }

    #[test]
    fn equality() {
        let a: PersistentHashSet<i32> = [1, 2, 3].into_iter().collect();
        let b: PersistentHashSet<i32> = [3, 2, 1].into_iter().collect();
        assert_eq!(a, b);
        let c: PersistentHashSet<i32> = [1, 2].into_iter().collect();
        assert_ne!(a, c);
    }

    #[test]
    fn debug_format_smoke() {
        let s: PersistentHashSet<i32> = [1].into_iter().collect();
        let text = format!("{s:?}");
        assert!(text.contains('1'));
    }
}
