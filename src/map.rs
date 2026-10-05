//! `PersistentHashMap`: the public CHAMP map type.
//!
//! Every mutating operation comes in two flavors:
//! - `&self` (e.g. [`PersistentHashMap::assoc`]): persistent, `self` is
//!   untouched and remains valid for the caller. Because `self` survives the
//!   call, no node reachable from `self.root` may ever be mutated in place —
//!   not even the root itself, even if its refcount happens to read as 1 —
//!   since `self` is a second, ongoing observer of that exact tree. This
//!   path is a pure path-copy.
//! - `_owned` (e.g. [`PersistentHashMap::assoc_owned`]): `self` is consumed.
//!   Since no one else can observe the old value once it's moved in, any
//!   node on the update path whose refcount reads as 1 is exclusively ours
//!   to mutate/realloc in place; the first node found to be shared (refcount
//!   greater than 1) forces a fall back to the same path-copy logic as the
//!   `&self` variant for the remainder of the path.
//!
//! Both flavors are implemented by pairing a borrowed, always-copies
//! recursive function with an owned, checks-uniqueness-then-falls-back
//! recursive function, rather than by threading a `may_mutate: bool`
//! through one unified function — the two functions naturally have
//! different ownership shapes (borrow vs. take), and letting that show
//! keeps each one simple.

use std::fmt;
use std::hash::{BuildHasher, Hash};

use crate::hash::DefaultBuildHasher;
use crate::iter::{IntoIter, Iter, Keys, Values};
use crate::node::{self, NodePtr};

/// A CHAMP persistent hash map.
///
/// Cloning is O(1) and shallow (bumps the root's refcount); structural
/// sharing means related versions of a map can share most of their nodes.
///
/// ## Equality contract
///
/// `PartialEq`/`Eq` are **structural**: two maps compare equal by walking
/// (and short-circuiting on) their trees, not by re-hashing every key. This
/// is only a valid implementation of `=` — i.e. only guaranteed to agree
/// with "same key set, same values" — when the two maps' hashers assign
/// identical hashes to equal keys. That holds for any single
/// stateless/deterministic `BuildHasher` value used by both maps (the
/// default, [`DefaultBuildHasher`], always qualifies: it's a unit struct,
/// every instance behaves identically). It does **not** hold for randomly
/// seeded hashers such as `std::collections::hash_map::RandomState`, where
/// two independently constructed instances hash the same key differently —
/// comparing two maps built with independently seeded `RandomState`s (or
/// any two hashers that disagree on some key's hash) gives a meaningless
/// result: CHAMP's canonical form means differing hashes produce differing
/// (but each individually valid) tree shapes, which this equality's
/// bitmap-mismatch fast path reports as unequal even when the logical
/// key/value contents are identical. If you need randomized hashing,
/// construct both sides of any comparison with the *same* hasher value
/// (`with_hasher`), not independently seeded ones.
pub struct PersistentHashMap<K, V, S = DefaultBuildHasher> {
    root: Option<NodePtr<K, V>>,
    count: usize,
    hasher: S,
}

// SAFETY: `PersistentHashMap` only owns `Option<NodePtr<K, V>>` (Send/Sync
// exactly when K, V are, per node.rs) plus a plain `S`. No interior
// mutability beyond the node layer's atomic refcounts.
unsafe impl<K: Send + Sync, V: Send + Sync, S: Send> Send for PersistentHashMap<K, V, S> {}
unsafe impl<K: Send + Sync, V: Send + Sync, S: Sync> Sync for PersistentHashMap<K, V, S> {}

impl<K, V, S> Drop for PersistentHashMap<K, V, S> {
    fn drop(&mut self) {
        if let Some(root) = self.root.take() {
            root.drop_node();
        }
    }
}

impl<K, V, S: Clone> Clone for PersistentHashMap<K, V, S> {
    /// O(1): bumps the root's refcount, copies `count`, clones the hasher.
    fn clone(&self) -> Self {
        PersistentHashMap {
            root: self.root.as_ref().map(NodePtr::clone_shallow),
            count: self.count,
            hasher: self.hasher.clone(),
        }
    }
}

impl<K, V, S: Default> Default for PersistentHashMap<K, V, S> {
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> PersistentHashMap<K, V, S> {
    /// Build an empty map using an explicit hasher builder.
    pub fn with_hasher(hasher: S) -> Self {
        PersistentHashMap {
            root: None,
            count: 0,
            hasher,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// True iff `self` and `other` share the exact same root allocation
    /// (both-empty counts as true). O(1), no traversal — a strictly
    /// stronger check than [`PartialEq`] (which walks structurally-equal
    /// but not-necessarily-identical trees too). Matches the host's
    /// `PMap::ptr_eq` contract: it's the fast "is this literally the same
    /// version of the map" check a caller can use to skip a full `==` when
    /// it already expects pointer identity to be common (e.g. after an
    /// unchanged `assoc`).
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.root, &other.root) {
            (None, None) => true,
            (Some(a), Some(b)) => NodePtr::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl<K, V> PersistentHashMap<K, V, DefaultBuildHasher> {
    /// Build an empty map using the crate's deterministic default hasher
    /// ([`DefaultBuildHasher`]). See the type-level equality contract docs
    /// on [`PersistentHashMap`] for why the default is deterministic rather
    /// than randomly seeded.
    pub fn new() -> Self {
        Self::with_hasher(DefaultBuildHasher)
    }
}

// ---------------------------------------------------------------------
// get / contains_key — iterative, no recursion, no allocation.
// ---------------------------------------------------------------------

/// Shared lookup core for both [`PersistentHashMap::get`] and
/// [`crate::transient::TransientMap::get`] — an iterative, no-recursion,
/// no-allocation descent from `root`.
pub(crate) fn get_in<'a, K, V, S>(root: Option<&'a NodePtr<K, V>>, hasher: &S, k: &K) -> Option<&'a V>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    let hash = node::hash32(hasher, k);
    let mut cur = root?;
    let mut depth = 0u32;
    loop {
        if cur.is_collision() {
            return cur.data_slice().iter().find(|(ek, _)| ek == k).map(|(_, ev)| ev);
        }
        let bit = node::bit_at(hash, depth);
        if cur.datamap() & bit != 0 {
            let idx = cur.data_index(bit);
            let (ek, ev) = &cur.data_slice()[idx];
            return if ek == k { Some(ev) } else { None };
        } else if cur.nodemap() & bit != 0 {
            let idx = cur.node_index(bit);
            cur = &cur.node_slice()[idx];
            depth += 1;
        } else {
            return None;
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> PersistentHashMap<K, V, S> {
    /// Look up `k`. `O(log32 n)` (effectively O(1) for realistic sizes).
    pub fn get(&self, k: &K) -> Option<&V> {
        get_in(self.root.as_ref(), &self.hasher, k)
    }

    /// Whether `k` is present.
    pub fn contains_key(&self, k: &K) -> bool {
        self.get(k).is_some()
    }
}

impl<K, V, S> PersistentHashMap<K, V, S> {
    /// Convert into a [`TransientMap`](crate::TransientMap): an O(1) move
    /// (no allocation, no copying) into an owned builder that mutates
    /// uniquely-owned nodes in place. See [`TransientMap`](crate::TransientMap)'s
    /// docs for the full semantics, including how this differs from
    /// Clojure's edit-token transients.
    pub fn transient(self) -> crate::transient::TransientMap<K, V, S> {
        crate::transient::TransientMap::from_persistent(self)
    }
}

// ---------------------------------------------------------------------
// assoc — split logic shared by both the copy-path and mutate-path.
// ---------------------------------------------------------------------

/// Outcome of a copy-path (borrowed) assoc recursion. `Unchanged` carries no
/// node at all — the whole point is that detecting "nothing to do" costs
/// nothing beyond the descent itself (no allocation, no refcount traffic).
enum AssocDelta<K, V> {
    Unchanged,
    Changed { node: NodePtr<K, V>, inserted: bool },
}

/// Build the replacement subtree for two entries that land in the same
/// chunk at `depth` (an existing leaf entry `(ek, ev)` with hash `ehash`,
/// and a new entry `(k, v)` with hash `hash`). Per the canonical-form rule,
/// if the two hashes are fully equal this becomes a collision node right
/// here (a collision node may appear at any depth, not only depth 7);
/// otherwise a chain of single-child nodes is built down to the first
/// differing chunk, capped by a two-entry data node.
fn split_leaf<K, V>(depth: u32, ehash: u32, ek: K, ev: V, hash: u32, k: K, v: V) -> NodePtr<K, V> {
    if ehash == hash {
        return NodePtr::new_collision2(ek, ev, k, v);
    }
    debug_assert!(depth <= 6, "distinct 32-bit hashes must diverge by depth 6");
    let ec = node::chunk(ehash, depth);
    let nc = node::chunk(hash, depth);
    if ec == nc {
        let child = split_leaf(depth + 1, ehash, ek, ev, hash, k, v);
        NodePtr::new_chain(1 << ec, child)
    } else {
        NodePtr::new_leaf2(1 << ec, ek, ev, 1 << nc, k, v)
    }
}

/// Same idea as [`split_leaf`], but the existing side is a whole collision
/// subtree (`coll`, whose entries all share `coll_hash`) rather than a
/// single entry — used when a new key's hash differs from an existing
/// collision node's hash even though they share a chunk prefix.
fn split_collision<K, V>(
    depth: u32,
    coll_hash: u32,
    coll: NodePtr<K, V>,
    hash: u32,
    k: K,
    v: V,
) -> NodePtr<K, V> {
    debug_assert_ne!(coll_hash, hash);
    debug_assert!(depth <= 6, "distinct 32-bit hashes must diverge by depth 6");
    let cc = node::chunk(coll_hash, depth);
    let nc = node::chunk(hash, depth);
    if cc == nc {
        let child = split_collision(depth + 1, coll_hash, coll, hash, k, v);
        NodePtr::new_chain(1 << cc, child)
    } else {
        NodePtr::new_leaf_and_child(1 << nc, k, v, 1 << cc, coll)
    }
}

/// Copy-path assoc: `node` is borrowed, never mutated; always produces a
/// brand-new owned node on `Changed`, or nothing at all on `Unchanged`.
fn assoc_copy<K, V, S>(node: &NodePtr<K, V>, hash: u32, depth: u32, key: K, value: V, hasher: &S) -> AssocDelta<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    if node.is_collision() {
        let entries = node.data_slice();
        if let Some(idx) = entries.iter().position(|(ek, _)| *ek == key) {
            if entries[idx].1 == value {
                return AssocDelta::Unchanged;
            }
            return AssocDelta::Changed {
                node: node.copy_with_value(idx, value),
                inserted: false,
            };
        }
        let coll_hash = node::hash32(hasher, &entries[0].0);
        if coll_hash == hash {
            return AssocDelta::Changed {
                node: node.copy_with_collision_inserted(key, value),
                inserted: true,
            };
        }
        let new_subtree = split_collision(depth, coll_hash, node.clone_shallow(), hash, key, value);
        return AssocDelta::Changed {
            node: new_subtree,
            inserted: true,
        };
    }

    let bit = node::bit_at(hash, depth);
    if node.datamap() & bit != 0 {
        let idx = node.data_index(bit);
        let (ek, ev) = &node.data_slice()[idx];
        if *ek == key {
            if *ev == value {
                return AssocDelta::Unchanged;
            }
            return AssocDelta::Changed {
                node: node.copy_with_value(idx, value),
                inserted: false,
            };
        }
        let ehash = node::hash32(hasher, ek);
        let (ek2, ev2) = (ek.clone(), ev.clone());
        let child = split_leaf(depth + 1, ehash, ek2, ev2, hash, key, value);
        AssocDelta::Changed {
            node: node.copy_with_data_to_node(bit, child),
            inserted: true,
        }
    } else if node.nodemap() & bit != 0 {
        let idx = node.node_index(bit);
        let child = &node.node_slice()[idx];
        match assoc_copy(child, hash, depth + 1, key, value, hasher) {
            AssocDelta::Unchanged => AssocDelta::Unchanged,
            AssocDelta::Changed { node: new_child, inserted } => AssocDelta::Changed {
                node: node.copy_with_node_replaced(idx, new_child),
                inserted,
            },
        }
    } else {
        AssocDelta::Changed {
            node: node.copy_with_data_inserted(bit, key, value),
            inserted: true,
        }
    }
}

/// Owned/mutate-path assoc: `node` is consumed. Mutates/reallocs in place
/// while `is_unique()` holds; falls back to [`assoc_copy`] (and discards its
/// own now-redundant owned handle) the first time a shared node is hit.
fn assoc_mut<K, V, S>(mut node: NodePtr<K, V>, hash: u32, depth: u32, key: K, value: V, hasher: &S) -> (NodePtr<K, V>, bool)
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    if !node.is_unique() {
        return match assoc_copy(&node, hash, depth, key, value, hasher) {
            AssocDelta::Unchanged => (node, false),
            AssocDelta::Changed { node: new_node, inserted } => {
                node.drop_node();
                (new_node, inserted)
            }
        };
    }

    if node.is_collision() {
        if let Some(idx) = node.data_slice().iter().position(|(ek, _)| *ek == key) {
            if node.data_slice()[idx].1 == value {
                return (node, false);
            }
            node.unique_set_value(idx, value);
            return (node, false);
        }
        let coll_hash = node::hash32(hasher, &node.data_slice()[0].0);
        if coll_hash == hash {
            return (node.unique_insert_collision(key, value), true);
        }
        let new_subtree = split_collision(depth, coll_hash, node, hash, key, value);
        return (new_subtree, true);
    }

    let bit = node::bit_at(hash, depth);
    if node.datamap() & bit != 0 {
        let idx = node.data_index(bit);
        if node.data_slice()[idx].0 == key {
            if node.data_slice()[idx].1 == value {
                return (node, false);
            }
            node.unique_set_value(idx, value);
            return (node, false);
        }
        let (ek, ev) = node.data_slice()[idx].clone();
        let ehash = node::hash32(hasher, &ek);
        let child = split_leaf(depth + 1, ehash, ek, ev, hash, key, value);
        (node.unique_data_to_node(bit, child), true)
    } else if node.nodemap() & bit != 0 {
        let idx = node.node_index(bit);
        let child = node.take_node(idx);
        let (new_child, inserted) = assoc_mut(child, hash, depth + 1, key, value, hasher);
        node.put_node(idx, new_child);
        (node, inserted)
    } else {
        (node.unique_insert_data(bit, key, value), true)
    }
}

impl<K, V, S> PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher + Clone,
{
    /// Persistent insert-or-replace. On a key match, the *existing* key
    /// instance is kept (only the value slot is overwritten) — this matches
    /// Clojure's `assoc` contract. If both the key and value are unchanged
    /// (`V: PartialEq`), returns a pointer-identical map with no allocation.
    pub fn assoc(&self, k: K, v: V) -> Self {
        let hash = node::hash32(&self.hasher, &k);
        match &self.root {
            None => PersistentHashMap {
                root: Some(NodePtr::new_leaf(node::bit_at(hash, 0), k, v)),
                count: 1,
                hasher: self.hasher.clone(),
            },
            Some(root) => match assoc_copy(root, hash, 0, k, v, &self.hasher) {
                AssocDelta::Unchanged => self.clone(),
                AssocDelta::Changed { node, inserted } => PersistentHashMap {
                    root: Some(node),
                    count: self.count + inserted as usize,
                    hasher: self.hasher.clone(),
                },
            },
        }
    }
}

impl<K, V, S> PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    /// Owning insert-or-replace: mutates uniquely-owned nodes along the
    /// update path in place instead of copying. Observationally identical
    /// to [`Self::assoc`], just faster when `self` isn't shared.
    pub fn assoc_owned(mut self, k: K, v: V) -> Self {
        let hash = node::hash32(&self.hasher, &k);
        match self.root.take() {
            None => {
                self.root = Some(NodePtr::new_leaf(node::bit_at(hash, 0), k, v));
                self.count = 1;
            }
            Some(root) => {
                let (new_root, inserted) = assoc_mut(root, hash, 0, k, v, &self.hasher);
                self.root = Some(new_root);
                if inserted {
                    self.count += 1;
                }
            }
        }
        self
    }
}

// ---------------------------------------------------------------------
// assoc_owned_replacing — like assoc_owned, but returns the value `k` was
// previously associated with (`None` for a fresh key). A single descent:
// the mutate path swaps the old value out via `mem::replace`
// (`unique_replace_value`); the copy-path fallback clones the old value
// before overwriting it. Needed by callers (e.g. the host's `PMap::insert`)
// that want the replaced value without a second lookup. Kept entirely
// separate from `assoc`/`assoc_owned`/`assoc_copy`/`assoc_mut` rather than
// threading an "also return the old value" flag through them, so the
// common (old value not needed) path keeps its zero-clone `Unchanged`
// shortcut — this variant can never take that shortcut for free, since it
// must produce a clone of the existing value even when nothing else
// changes.
// ---------------------------------------------------------------------

/// Outcome of a copy-path (borrowed) `assoc_owned_replacing` recursion.
/// Unlike [`AssocDelta`], `Unchanged` still carries the (cloned) old value
/// — that clone is unavoidable here, since the caller always needs it.
enum AssocReplacingDelta<K, V> {
    Unchanged { old: V },
    Changed { node: NodePtr<K, V>, old: Option<V> },
}

fn assoc_copy_replacing<K, V, S>(
    node: &NodePtr<K, V>,
    hash: u32,
    depth: u32,
    key: K,
    value: V,
    hasher: &S,
) -> AssocReplacingDelta<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    if node.is_collision() {
        let entries = node.data_slice();
        if let Some(idx) = entries.iter().position(|(ek, _)| *ek == key) {
            let old = entries[idx].1.clone();
            if old == value {
                return AssocReplacingDelta::Unchanged { old };
            }
            return AssocReplacingDelta::Changed {
                node: node.copy_with_value(idx, value),
                old: Some(old),
            };
        }
        let coll_hash = node::hash32(hasher, &entries[0].0);
        if coll_hash == hash {
            return AssocReplacingDelta::Changed {
                node: node.copy_with_collision_inserted(key, value),
                old: None,
            };
        }
        let new_subtree = split_collision(depth, coll_hash, node.clone_shallow(), hash, key, value);
        return AssocReplacingDelta::Changed { node: new_subtree, old: None };
    }

    let bit = node::bit_at(hash, depth);
    if node.datamap() & bit != 0 {
        let idx = node.data_index(bit);
        let (ek, ev) = &node.data_slice()[idx];
        if *ek == key {
            let old = ev.clone();
            if old == value {
                return AssocReplacingDelta::Unchanged { old };
            }
            return AssocReplacingDelta::Changed {
                node: node.copy_with_value(idx, value),
                old: Some(old),
            };
        }
        let ehash = node::hash32(hasher, ek);
        let (ek2, ev2) = (ek.clone(), ev.clone());
        let child = split_leaf(depth + 1, ehash, ek2, ev2, hash, key, value);
        AssocReplacingDelta::Changed {
            node: node.copy_with_data_to_node(bit, child),
            old: None,
        }
    } else if node.nodemap() & bit != 0 {
        let idx = node.node_index(bit);
        let child = &node.node_slice()[idx];
        match assoc_copy_replacing(child, hash, depth + 1, key, value, hasher) {
            AssocReplacingDelta::Unchanged { old } => AssocReplacingDelta::Unchanged { old },
            AssocReplacingDelta::Changed { node: new_child, old } => AssocReplacingDelta::Changed {
                node: node.copy_with_node_replaced(idx, new_child),
                old,
            },
        }
    } else {
        AssocReplacingDelta::Changed {
            node: node.copy_with_data_inserted(bit, key, value),
            old: None,
        }
    }
}

fn assoc_mut_replacing<K, V, S>(
    mut node: NodePtr<K, V>,
    hash: u32,
    depth: u32,
    key: K,
    value: V,
    hasher: &S,
) -> (NodePtr<K, V>, Option<V>)
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    if !node.is_unique() {
        return match assoc_copy_replacing(&node, hash, depth, key, value, hasher) {
            AssocReplacingDelta::Unchanged { old } => (node, Some(old)),
            AssocReplacingDelta::Changed { node: new_node, old } => {
                node.drop_node();
                (new_node, old)
            }
        };
    }

    if node.is_collision() {
        if let Some(idx) = node.data_slice().iter().position(|(ek, _)| *ek == key) {
            if node.data_slice()[idx].1 == value {
                let old = node.data_slice()[idx].1.clone();
                return (node, Some(old));
            }
            let old = node.unique_replace_value(idx, value);
            return (node, Some(old));
        }
        let coll_hash = node::hash32(hasher, &node.data_slice()[0].0);
        if coll_hash == hash {
            return (node.unique_insert_collision(key, value), None);
        }
        let new_subtree = split_collision(depth, coll_hash, node, hash, key, value);
        return (new_subtree, None);
    }

    let bit = node::bit_at(hash, depth);
    if node.datamap() & bit != 0 {
        let idx = node.data_index(bit);
        if node.data_slice()[idx].0 == key {
            if node.data_slice()[idx].1 == value {
                let old = node.data_slice()[idx].1.clone();
                return (node, Some(old));
            }
            let old = node.unique_replace_value(idx, value);
            return (node, Some(old));
        }
        let (ek, ev) = node.data_slice()[idx].clone();
        let ehash = node::hash32(hasher, &ek);
        let child = split_leaf(depth + 1, ehash, ek, ev, hash, key, value);
        (node.unique_data_to_node(bit, child), None)
    } else if node.nodemap() & bit != 0 {
        let idx = node.node_index(bit);
        let child = node.take_node(idx);
        let (new_child, old) = assoc_mut_replacing(child, hash, depth + 1, key, value, hasher);
        node.put_node(idx, new_child);
        (node, old)
    } else {
        (node.unique_insert_data(bit, key, value), None)
    }
}

impl<K, V, S> PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher,
{
    /// Owning insert-or-replace that also returns the value previously
    /// associated with `k`, if any (`None` for a fresh key). Same
    /// keep-existing-key-on-replace / in-place-when-unique semantics as
    /// [`Self::assoc_owned`] — this is the single-descent version for
    /// callers (e.g. the host's `PMap::insert`) that need the old value
    /// without a second lookup. Prefer [`Self::assoc_owned`] when the old
    /// value isn't needed: this variant always clones the existing value
    /// when the key is already present (even if the value is unchanged),
    /// so it never gets `assoc`/`assoc_owned`'s zero-allocation,
    /// zero-clone `Unchanged` shortcut.
    pub fn assoc_owned_replacing(mut self, k: K, v: V) -> (Self, Option<V>) {
        let hash = node::hash32(&self.hasher, &k);
        match self.root.take() {
            None => {
                self.root = Some(NodePtr::new_leaf(node::bit_at(hash, 0), k, v));
                self.count = 1;
                (self, None)
            }
            Some(root) => {
                let (new_root, old) = assoc_mut_replacing(root, hash, 0, k, v, &self.hasher);
                self.root = Some(new_root);
                if old.is_none() {
                    self.count += 1;
                }
                (self, old)
            }
        }
    }
}

// ---------------------------------------------------------------------
// dissoc — bottom-up removal with canonical inlining.
// ---------------------------------------------------------------------

/// Outcome of a copy-path (borrowed) dissoc recursion.
enum DissocResult<K, V> {
    /// Key absent in this subtree; propagates all the way up with zero
    /// allocation.
    NotFound,
    /// Key removed; the resulting node still holds >= 2 total slots.
    Removed(NodePtr<K, V>),
    /// The (root) subtree became fully empty.
    Gone,
    /// The subtree collapsed to exactly one data entry and no children;
    /// the parent must pull it up into a data slot (canonical inlining).
    Inline(K, V),
}

/// Outcome of an owned/mutate-path dissoc recursion. Unlike [`DissocResult`],
/// `NotFound` must hand back the untouched owned node.
enum DissocMutResult<K, V> {
    NotFound(NodePtr<K, V>),
    Removed(NodePtr<K, V>),
    Gone,
    Inline(K, V),
}

fn dissoc_copy<K, V>(node: &NodePtr<K, V>, hash: u32, depth: u32, key: &K) -> DissocResult<K, V>
where
    K: Eq + Clone,
    V: Clone,
{
    if node.is_collision() {
        let idx = match node.data_slice().iter().position(|(ek, _)| ek == key) {
            Some(i) => i,
            None => return DissocResult::NotFound,
        };
        if node.n_data() == 2 {
            let (k, v) = node.extract_sole_survivor(idx);
            return DissocResult::Inline(k, v);
        }
        return DissocResult::Removed(node.copy_with_collision_removed(idx));
    }

    let bit = node::bit_at(hash, depth);
    if node.datamap() & bit != 0 {
        let idx = node.data_index(bit);
        if node.data_slice()[idx].0 != *key {
            return DissocResult::NotFound;
        }
        let n_data = node.n_data();
        let n_nodes = node.n_nodes();
        if n_data == 2 && n_nodes == 0 {
            let (k, v) = node.extract_sole_survivor(idx);
            return DissocResult::Inline(k, v);
        }
        if n_data == 1 && n_nodes == 0 {
            return DissocResult::Gone;
        }
        DissocResult::Removed(node.copy_with_data_removed(bit))
    } else if node.nodemap() & bit != 0 {
        let idx = node.node_index(bit);
        let child = &node.node_slice()[idx];
        match dissoc_copy(child, hash, depth + 1, key) {
            DissocResult::NotFound => DissocResult::NotFound,
            DissocResult::Removed(new_child) => DissocResult::Removed(node.copy_with_node_replaced(idx, new_child)),
            DissocResult::Gone => {
                unreachable!("non-root subtrees always retain >= 2 entries by canonical form")
            }
            DissocResult::Inline(k, v) => {
                if node.n_data() == 0 && node.n_nodes() == 1 {
                    // This node is itself a pure single-child chain link
                    // that just lost its only child to inlining: cascade
                    // the inline further up without allocating anything.
                    DissocResult::Inline(k, v)
                } else {
                    DissocResult::Removed(node.copy_with_node_to_data(bit, k, v))
                }
            }
        }
    } else {
        DissocResult::NotFound
    }
}

fn dissoc_dispatch<K, V>(node: NodePtr<K, V>, hash: u32, depth: u32, key: &K) -> DissocMutResult<K, V>
where
    K: Eq + Clone,
    V: Clone,
{
    if node.is_unique() {
        dissoc_mut(node, hash, depth, key)
    } else {
        match dissoc_copy(&node, hash, depth, key) {
            DissocResult::NotFound => DissocMutResult::NotFound(node),
            DissocResult::Removed(n) => {
                node.drop_node();
                DissocMutResult::Removed(n)
            }
            DissocResult::Gone => {
                node.drop_node();
                DissocMutResult::Gone
            }
            DissocResult::Inline(k, v) => {
                node.drop_node();
                DissocMutResult::Inline(k, v)
            }
        }
    }
}

fn dissoc_mut<K, V>(mut node: NodePtr<K, V>, hash: u32, depth: u32, key: &K) -> DissocMutResult<K, V>
where
    K: Eq + Clone,
    V: Clone,
{
    debug_assert!(node.is_unique());

    if node.is_collision() {
        let idx = match node.data_slice().iter().position(|(ek, _)| ek == key) {
            Some(i) => i,
            None => return DissocMutResult::NotFound(node),
        };
        if node.n_data() == 2 {
            let (k, v) = node.extract_sole_survivor_unique(idx);
            return DissocMutResult::Inline(k, v);
        }
        return DissocMutResult::Removed(node.unique_remove_collision(idx));
    }

    let bit = node::bit_at(hash, depth);
    if node.datamap() & bit != 0 {
        let idx = node.data_index(bit);
        if node.data_slice()[idx].0 != *key {
            return DissocMutResult::NotFound(node);
        }
        let n_data = node.n_data();
        let n_nodes = node.n_nodes();
        if n_data == 2 && n_nodes == 0 {
            let (k, v) = node.extract_sole_survivor_unique(idx);
            return DissocMutResult::Inline(k, v);
        }
        if n_data == 1 && n_nodes == 0 {
            // The whole node becomes empty: dropping it drops its one
            // remaining (the-one-being-removed) entry and deallocates.
            node.drop_node();
            return DissocMutResult::Gone;
        }
        DissocMutResult::Removed(node.unique_remove_data(bit))
    } else if node.nodemap() & bit != 0 {
        let idx = node.node_index(bit);
        let n_data0 = node.n_data();
        let n_nodes0 = node.n_nodes();
        let child = node.take_node(idx);
        match dissoc_dispatch(child, hash, depth + 1, key) {
            DissocMutResult::NotFound(child_back) => {
                node.put_node(idx, child_back);
                DissocMutResult::NotFound(node)
            }
            DissocMutResult::Removed(new_child) => {
                node.put_node(idx, new_child);
                DissocMutResult::Removed(node)
            }
            DissocMutResult::Gone => {
                unreachable!("non-root subtrees always retain >= 2 entries by canonical form")
            }
            DissocMutResult::Inline(k, v) => {
                if n_data0 == 0 && n_nodes0 == 1 {
                    // Pure single-child chain link losing its only child:
                    // free this now-empty shell and cascade the inline up.
                    node.unique_dealloc_after_take_sole_child();
                    DissocMutResult::Inline(k, v)
                } else {
                    DissocMutResult::Removed(node.unique_node_to_data(bit, k, v))
                }
            }
        }
    } else {
        DissocMutResult::NotFound(node)
    }
}

impl<K, V, S> PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
{
    /// Persistent removal. Absent key: returns a pointer-identical map, no
    /// allocation. Present key: canonicalizes on the way back up so the
    /// result is exactly the tree fresh insertion of the remaining entries
    /// would have produced.
    pub fn dissoc(&self, k: &K) -> Self {
        let Some(root) = self.root.as_ref() else {
            return self.clone();
        };
        let hash = node::hash32(&self.hasher, k);
        match dissoc_copy(root, hash, 0, k) {
            DissocResult::NotFound => self.clone(),
            DissocResult::Removed(new_root) => PersistentHashMap {
                root: Some(new_root),
                count: self.count - 1,
                hasher: self.hasher.clone(),
            },
            DissocResult::Gone => PersistentHashMap {
                root: None,
                count: 0,
                hasher: self.hasher.clone(),
            },
            DissocResult::Inline(nk, nv) => {
                let nk_hash = node::hash32(&self.hasher, &nk);
                PersistentHashMap {
                    root: Some(NodePtr::new_leaf(node::bit_at(nk_hash, 0), nk, nv)),
                    count: self.count - 1,
                    hasher: self.hasher.clone(),
                }
            }
        }
    }
}

impl<K, V, S> PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Owning removal: mutates uniquely-owned nodes along the removal path
    /// in place instead of copying.
    pub fn dissoc_owned(mut self, k: &K) -> Self {
        let Some(root) = self.root.take() else {
            return self;
        };
        let hash = node::hash32(&self.hasher, k);
        match dissoc_dispatch(root, hash, 0, k) {
            DissocMutResult::NotFound(root) => {
                self.root = Some(root);
            }
            DissocMutResult::Removed(new_root) => {
                self.root = Some(new_root);
                self.count -= 1;
            }
            DissocMutResult::Gone => {
                self.count = 0;
            }
            DissocMutResult::Inline(nk, nv) => {
                let nk_hash = node::hash32(&self.hasher, &nk);
                self.root = Some(NodePtr::new_leaf(node::bit_at(nk_hash, 0), nk, nv));
                self.count -= 1;
            }
        }
        self
    }
}

// ---------------------------------------------------------------------
// Iteration, equality, Debug, FromIterator/Extend.
// ---------------------------------------------------------------------

impl<K, V, S> PersistentHashMap<K, V, S> {
    /// Iterate over `(&K, &V)` pairs in hash order (deterministic and
    /// stable across identical contents).
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.root.as_ref())
    }

    /// Iterate over keys in hash order.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// Iterate over values in hash order.
    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }
}

impl<'a, K, V, S> IntoIterator for &'a PersistentHashMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Consuming iteration, yielding owned `(K, V)` clones in hash order (see
/// [`IntoIter`]'s docs for why clones, not moves — cloning matches what
/// callers like the host's small-map iterator already do, and a
/// move-out-when-unique optimization is future work). `self.root.take()`
/// leaves `self`'s own `Drop` impl a no-op when it falls out of scope right
/// after — the taken root is what [`IntoIter`] now owns and will release.
impl<K, V, S> IntoIterator for PersistentHashMap<K, V, S>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(mut self) -> Self::IntoIter {
        IntoIter::new(self.root.take())
    }
}

/// Recursive node equality: pointer-equal nodes short-circuit (skip the
/// subtree entirely), a bitmap mismatch fails fast, data arrays compare
/// index-wise (canonical form makes the order deterministic), and collision
/// nodes compare as unordered multisets.
fn nodes_eq<K: Eq, V: PartialEq>(a: &NodePtr<K, V>, b: &NodePtr<K, V>) -> bool {
    if NodePtr::ptr_eq(a, b) {
        return true;
    }
    if a.datamap() != b.datamap() || a.nodemap() != b.nodemap() {
        return false;
    }
    if a.is_collision() {
        if a.n_data() != b.n_data() {
            return false;
        }
        let bd = b.data_slice();
        'outer: for (ka, va) in a.data_slice() {
            for (kb, vb) in bd {
                if ka == kb {
                    if va != vb {
                        return false;
                    }
                    continue 'outer;
                }
            }
            return false;
        }
        return true;
    }
    for ((ka, va), (kb, vb)) in a.data_slice().iter().zip(b.data_slice()) {
        if ka != kb || va != vb {
            return false;
        }
    }
    for (ca, cb) in a.node_slice().iter().zip(b.node_slice()) {
        if !nodes_eq(ca, cb) {
            return false;
        }
    }
    true
}

/// Structural equality. **Only meaningful when `self` and `other`'s hashers
/// agree on every key's hash** — see the "Equality contract" section on
/// [`PersistentHashMap`]'s type docs. In short: fine for the default
/// [`crate::DefaultBuildHasher`] or any other single deterministic hasher
/// value shared by both sides; meaningless for independently seeded
/// randomized hashers.
impl<K: Eq, V: PartialEq, S> PartialEq for PersistentHashMap<K, V, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.count != other.count {
            return false;
        }
        match (&self.root, &other.root) {
            (None, None) => true,
            (Some(a), Some(b)) => nodes_eq(a, b),
            _ => false,
        }
    }
}

impl<K: Eq, V: Eq, S> Eq for PersistentHashMap<K, V, S> {}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for PersistentHashMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> FromIterator<(K, V)> for PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut t = Self::with_hasher(S::default()).transient();
        for (k, v) in iter {
            t.assoc(k, v);
        }
        t.persistent()
    }
}

impl<K, V, S> Extend<(K, V)> for PersistentHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
    S: BuildHasher + Default,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let mut t = std::mem::take(self).transient();
        for (k, v) in iter {
            t.assoc(k, v);
        }
        *self = t.persistent();
    }
}

// ---------------------------------------------------------------------
// Canonical-form validator, only compiled for tests or the `validate`
// feature (see tests/invariants.rs).
// ---------------------------------------------------------------------

#[cfg(any(test, feature = "validate"))]
fn validate_node<K, V, S>(node: &NodePtr<K, V>, hasher: &S, depth: u32) -> usize
where
    K: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    assert_eq!(node.datamap() & node.nodemap(), 0, "datamap/nodemap overlap");
    // M4 capacity-slack invariant: `count <= cap` always, for both node
    // kinds. The tighter `cap <= 32` ceiling only applies to *bitmap* nodes
    // (`cap` there is a reserved data-slot count within a single 5-bit
    // chunk's 32 possible bits); a *collision* node's `cap` is simply its
    // (unbounded — arbitrarily many keys can share one 32-bit hash) entry
    // count, so that ceiling would be a false positive there.
    let cap = node.cap();
    assert!(node.n_data() <= cap, "node count {} exceeds its own cap {cap}", node.n_data());
    if node.is_collision() {
        assert!(node.n_data() >= 2, "collision node with < 2 entries");
        let h0 = node::hash32(hasher, &node.data_slice()[0].0);
        for (k, _) in node.data_slice() {
            assert_eq!(
                node::hash32(hasher, k),
                h0,
                "collision node entry {k:?} has a different hash than its siblings"
            );
        }
        return node.n_data();
    }
    assert!(cap <= 32, "bitmap node cap {cap} exceeds the 32-slot hard ceiling");
    for (i, (k, _)) in node.data_slice().iter().enumerate() {
        let h = node::hash32(hasher, k);
        let bit = node::bit_at(h, depth);
        assert_eq!(
            node.datamap() & bit,
            bit,
            "entry {k:?} not addressed by its own hash chunk at depth {depth}"
        );
        assert_eq!(node.data_index(bit), i, "entry {k:?} stored at the wrong data slot");
    }
    let mut total = node.n_data();
    for child in node.node_slice() {
        let sub = validate_node(child, hasher, depth + 1);
        assert!(sub >= 2, "non-root subtree with < 2 entries (non-canonical)");
        total += sub;
    }
    total
}

#[cfg(any(test, feature = "validate"))]
impl<K, V, S> PersistentHashMap<K, V, S>
where
    K: Hash + Eq + fmt::Debug,
    V: PartialEq + fmt::Debug,
    S: BuildHasher,
{
    /// Walk the tree asserting canonical-form invariants:
    /// - `datamap & nodemap == 0` on every bitmap node;
    /// - every non-root subtree has >= 2 entries (no degenerate nodes);
    /// - every collision node has >= 2 entries, all sharing one hash;
    /// - every entry is reachable under its own hash path (checked by
    ///   cross-referencing against [`Self::get`]);
    /// - the tree's total entry count matches `self.len()`.
    pub fn validate(&self) {
        let total = match &self.root {
            Some(root) => validate_node(root, &self.hasher, 0),
            None => 0,
        };
        assert_eq!(total, self.count, "map count does not match tree contents");
        for (k, v) in self.iter() {
            assert_eq!(self.get(k), Some(v), "entry {k:?} not reachable under its own hash path");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn map_is_send_sync() {
        assert_send_sync::<PersistentHashMap<String, String>>();
    }

    #[test]
    fn empty_map() {
        let m: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        assert!(m.is_empty());
        assert_eq!(m.get(&1), None);
    }

    #[test]
    fn basic_assoc_get_dissoc() {
        let m = PersistentHashMap::<i32, i32>::new();
        let m = m.assoc(1, 100);
        let m = m.assoc(2, 200);
        assert_eq!(m.get(&1), Some(&100));
        assert_eq!(m.get(&2), Some(&200));
        assert_eq!(m.len(), 2);
        m.validate();

        let m2 = m.dissoc(&1);
        assert_eq!(m2.get(&1), None);
        assert_eq!(m2.get(&2), Some(&200));
        assert_eq!(m.get(&1), Some(&100)); // original untouched
        m2.validate();
        m.validate();
    }

    #[test]
    fn dissoc_absent_is_pointer_identical() {
        let m = PersistentHashMap::<i32, i32>::new().assoc(1, 100);
        let m2 = m.dissoc(&2);
        assert!(NodePtr::ptr_eq(m.root.as_ref().unwrap(), m2.root.as_ref().unwrap()));
    }

    #[test]
    fn assoc_owned_in_place_grows() {
        let mut m = PersistentHashMap::<i32, i32>::new();
        for i in 0..2000 {
            m = m.assoc_owned(i, i * 2);
        }
        assert_eq!(m.len(), 2000);
        for i in 0..2000 {
            assert_eq!(m.get(&i), Some(&(i * 2)));
        }
        m.validate();
    }

    #[test]
    fn structural_sharing_clone_and_diverge() {
        let mut a = PersistentHashMap::<i32, i32>::new();
        for i in 0..500 {
            a = a.assoc_owned(i, i);
        }
        let b = a.clone();
        let mut a = a;
        for i in 500..1000 {
            a = a.assoc_owned(i, i);
        }
        for i in 0..500 {
            assert_eq!(b.get(&i), Some(&i));
            assert_eq!(a.get(&i), Some(&i));
        }
        for i in 500..1000 {
            assert_eq!(b.get(&i), None);
            assert_eq!(a.get(&i), Some(&i));
        }
        a.validate();
        b.validate();
    }
}
