//! Canonical-form invariant checks: `PersistentHashMap::validate` (exposed
//! outside unit tests via the `validate` feature, see Cargo.toml's
//! self-referencing dev-dependency) walks the tree asserting:
//! - `datamap & nodemap == 0`;
//! - every non-root subtree has >= 2 entries (no degenerate nodes left
//!   behind by dissoc's canonical inlining);
//! - every collision node has >= 2 entries, all sharing one hash;
//! - every entry is reachable under its own hash path;
//! - the tree's total entry count matches `len()`.
//!
//! Run under both a well-behaved hasher and deliberately colliding hashers
//! to exercise deep chains and collision nodes.

use champ::{PersistentHashMap, PersistentHashSet};
use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};

/// A `BuildHasher` that masks the underlying (good) hash down to very few
/// bits, forcing deep chains and collision nodes even for small key sets.
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

fn mod8_hasher() -> BadBuildHasher {
    BadBuildHasher { mask: 0x7 }
}

fn and3_hasher() -> BadBuildHasher {
    BadBuildHasher { mask: 0x3 }
}

fn dense_scripted_test<S: BuildHasher + Clone + Default>(hasher: S, n: i64, seed: u64) {
    // Deterministic xorshift-ish PRNG (no external dep needed).
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut m: PersistentHashMap<i64, i64, S> = PersistentHashMap::with_hasher(hasher);
    for i in 0..n {
        m = m.assoc_owned(i, i * 2);
    }
    m.validate();
    assert_eq!(m.len(), n as usize);

    // Remove half, in random order.
    let mut order: Vec<i64> = (0..n).collect();
    for i in (1..order.len()).rev() {
        let j = (next() as usize) % (i + 1);
        order.swap(i, j);
    }
    for &k in order.iter().take((n / 2) as usize) {
        m = m.dissoc_owned(&k);
        m.validate();
    }
    assert_eq!(m.len(), n as usize - (n / 2) as usize);

    // Everything removed should be gone; everything remaining should be
    // exactly right.
    let removed: HashSet<i64> = order.iter().take((n / 2) as usize).copied().collect();
    for i in 0..n {
        if removed.contains(&i) {
            assert_eq!(m.get(&i), None);
        } else {
            assert_eq!(m.get(&i), Some(&(i * 2)));
        }
    }
    m.validate();

    // Remove the rest, down to empty.
    for &k in order.iter().skip((n / 2) as usize) {
        m = m.dissoc_owned(&k);
        m.validate();
    }
    assert!(m.is_empty());
    m.validate();
}

// Miri is orders of magnitude slower than native execution; shrink the
// dense scripted sizes under it so the suite stays tractable while still
// exercising deep trees, canonical inlining cascades, and collision nodes.
#[cfg(miri)]
const DENSE_N: i64 = 200;
#[cfg(not(miri))]
const DENSE_N: i64 = 10_000;

#[cfg(miri)]
const DENSE_N_COLLISION: i64 = 80;
#[cfg(not(miri))]
const DENSE_N_COLLISION: i64 = 800;

#[cfg(miri)]
const PERSISTENT_N: i32 = 40;
#[cfg(not(miri))]
const PERSISTENT_N: i32 = 300;

#[test]
fn dense_scripted_random_hasher() {
    dense_scripted_test::<RandomState>(RandomState::new(), DENSE_N, 1);
}

#[test]
fn dense_scripted_mod8_hasher() {
    // Smaller n: mod-8 hashing means huge collision nodes, O(n) per op.
    dense_scripted_test(mod8_hasher(), DENSE_N_COLLISION, 2);
}

#[test]
fn dense_scripted_and3_hasher() {
    dense_scripted_test(and3_hasher(), DENSE_N_COLLISION, 3);
}

// ---------------------------------------------------------------------
// M3: same dense scripted coverage, run through `PersistentHashSet`
// (`validate` walks the same underlying tree — see set.rs's `validate`).
// ---------------------------------------------------------------------

fn dense_scripted_set_test<S: BuildHasher + Clone + Default>(hasher: S, n: i64, seed: u64) {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut s: PersistentHashSet<i64, S> = PersistentHashSet::with_hasher(hasher);
    for i in 0..n {
        s = s.insert_owned(i);
    }
    s.validate();
    assert_eq!(s.len(), n as usize);

    let mut order: Vec<i64> = (0..n).collect();
    for i in (1..order.len()).rev() {
        let j = (next() as usize) % (i + 1);
        order.swap(i, j);
    }
    for &k in order.iter().take((n / 2) as usize) {
        s = s.remove_owned(&k);
        s.validate();
    }
    assert_eq!(s.len(), n as usize - (n / 2) as usize);

    let removed: HashSet<i64> = order.iter().take((n / 2) as usize).copied().collect();
    for i in 0..n {
        assert_eq!(s.contains(&i), !removed.contains(&i));
    }
    s.validate();

    for &k in order.iter().skip((n / 2) as usize) {
        s = s.remove_owned(&k);
        s.validate();
    }
    assert!(s.is_empty());
    s.validate();
}

#[test]
fn set_dense_scripted_random_hasher() {
    dense_scripted_set_test::<RandomState>(RandomState::new(), DENSE_N, 11);
}

#[test]
fn set_dense_scripted_mod8_hasher() {
    dense_scripted_set_test(mod8_hasher(), DENSE_N_COLLISION, 12);
}

#[test]
fn set_dense_scripted_and3_hasher() {
    dense_scripted_set_test(and3_hasher(), DENSE_N_COLLISION, 13);
}

/// A key wrapping an exact, caller-chosen 32-bit hash plus an `id` that
/// only affects equality, never hashing — this lets tests construct
/// precise chunk collisions/divergences at any depth (including the
/// depth-6 edge case, where only 2 of the 32 hash bits remain) without
/// relying on chance. Two `ExactHash`es with equal `hash` but different
/// `id` are unequal keys with an equal hash (a real collision), exactly
/// like real `K`s that happen to collide.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ExactHash {
    hash: u32,
    id: u32,
}

impl std::hash::Hash for ExactHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.hash);
    }
}

#[derive(Clone, Default)]
struct IdentityBuildHasher;

struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("ExactHash::hash only ever calls write_u32")
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = i as u64;
    }
}

impl BuildHasher for IdentityBuildHasher {
    type Hasher = IdentityHasher;
    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher(0)
    }
}

fn chunked_hash(chunks: [u32; 7]) -> u32 {
    let mut h = 0u32;
    for (i, c) in chunks.iter().enumerate() {
        assert!(*c < 32);
        h |= c << (5 * i as u32);
    }
    h
}

/// Deliberately construct a full single-child chain from depth 1 through
/// depth 5, diverging only at depth 6 (the 2-bit final level) — the part of
/// the split logic least likely to be exercised by chance under a
/// well-distributed hasher with realistic key-set sizes. Then dissoc one of
/// the two keys back out and confirm the resulting cascade of canonical
/// inlining (chain link after chain link collapsing all the way back to a
/// single root leaf) lands on exactly the right key/value.
#[test]
fn deep_chain_diverging_at_depth_six() {
    let shared = [5u32, 7, 3, 11, 19, 23]; // chunks 0..=5, identical for both keys
    let hash_a = chunked_hash([shared[0], shared[1], shared[2], shared[3], shared[4], shared[5], 0]);
    let hash_b = chunked_hash([shared[0], shared[1], shared[2], shared[3], shared[4], shared[5], 1]);
    assert_ne!(hash_a, hash_b);
    // Sanity: everything but the depth-6 chunk truly matches.
    for d in 0..6 {
        assert_eq!(node_chunk(hash_a, d), node_chunk(hash_b, d));
    }
    assert_ne!(node_chunk(hash_a, 6), node_chunk(hash_b, 6));

    let key_a = ExactHash { hash: hash_a, id: 1 };
    let key_b = ExactHash { hash: hash_b, id: 2 };

    let m: PersistentHashMap<ExactHash, &'static str, IdentityBuildHasher> =
        PersistentHashMap::with_hasher(IdentityBuildHasher)
            .assoc(key_a, "a")
            .assoc(key_b, "b");
    assert_eq!(m.len(), 2);
    assert_eq!(m.get(&key_a), Some(&"a"));
    assert_eq!(m.get(&key_b), Some(&"b"));
    m.validate();

    // Removing one must cascade the canonical inline all the way back up
    // through every chain link to a single root leaf holding the other.
    let m2 = m.dissoc(&key_a);
    assert_eq!(m2.len(), 1);
    assert_eq!(m2.get(&key_a), None);
    assert_eq!(m2.get(&key_b), Some(&"b"));
    m2.validate();
    // Original untouched (structural sharing / persistence).
    assert_eq!(m.get(&key_a), Some(&"a"));
    m.validate();

    let m3 = m.dissoc_owned(&key_b);
    assert_eq!(m3.len(), 1);
    assert_eq!(m3.get(&key_b), None);
    assert_eq!(m3.get(&key_a), Some(&"a"));
    m3.validate();
}

/// A middle-depth divergence (not depth 0, not depth 6) plus a genuine
/// full-hash collision (collision node) coexisting with chain-building —
/// two keys collide immediately, a third shares only their first three
/// chunks then diverges.
#[test]
fn mixed_chain_and_collision_node() {
    let hash_ab = chunked_hash([2, 4, 6, 8, 10, 12, 1]);
    let key_a = ExactHash { hash: hash_ab, id: 1 };
    let key_b = ExactHash { hash: hash_ab, id: 2 }; // same hash as a, different key: collision node

    let hash_c = chunked_hash([2, 4, 6, 20, 1, 1, 1]); // shares chunks 0..=2 with a/b, diverges at 3
    let key_c = ExactHash { hash: hash_c, id: 3 };

    let m: PersistentHashMap<ExactHash, i32, IdentityBuildHasher> =
        PersistentHashMap::with_hasher(IdentityBuildHasher)
            .assoc(key_a, 1)
            .assoc(key_b, 2)
            .assoc(key_c, 3);
    assert_eq!(m.len(), 3);
    assert_eq!(m.get(&key_a), Some(&1));
    assert_eq!(m.get(&key_b), Some(&2));
    assert_eq!(m.get(&key_c), Some(&3));
    m.validate();

    let m2 = m.dissoc(&key_c);
    assert_eq!(m2.len(), 2);
    assert_eq!(m2.get(&key_a), Some(&1));
    assert_eq!(m2.get(&key_b), Some(&2));
    assert_eq!(m2.get(&key_c), None);
    m2.validate();

    let m3 = m2.dissoc_owned(&key_a);
    assert_eq!(m3.len(), 1);
    assert_eq!(m3.get(&key_b), Some(&2));
    m3.validate();
}

fn node_chunk(hash: u32, depth: u32) -> u32 {
    (hash >> (5 * depth)) & 0x1f
}

#[test]
fn persistent_variant_also_validates() {
    let mut m: PersistentHashMap<i32, i32> = PersistentHashMap::new();
    let mut history = Vec::new();
    for i in 0..PERSISTENT_N {
        m = m.assoc(i, i);
        history.push(m.clone());
    }
    for h in &history {
        h.validate();
    }
    for i in (0..PERSISTENT_N).step_by(3) {
        m = m.dissoc(&i);
        m.validate();
    }
    // Older snapshots must remain valid and untouched by later ops.
    for h in &history {
        h.validate();
    }
}
