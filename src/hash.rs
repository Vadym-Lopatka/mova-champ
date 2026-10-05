//! Deterministic default hasher, plus the order-independent `hash_unordered`
//! helper used to hash whole maps/sets.
//!
//! ## Why not `RandomState`
//!
//! `std::collections::hash_map::RandomState` seeds each instance randomly.
//! Two `PersistentHashMap`s built independently with `RandomState::new()`
//! would then hash equal keys to *different* 32-bit values, which — because
//! CHAMP's tree shape is a deterministic function of those hash values —
//! would give the two maps different (both perfectly canonical) tree shapes
//! even when they hold identical contents. [`crate::map::nodes_eq`]'s
//! bitmap-mismatch fast-fail would then report two `==` maps as unequal,
//! breaking the `=` contract entirely. [`DefaultBuildHasher`] is stateless
//! and deterministic (same algorithm, same output, every time, in every
//! process) precisely so that equality — see the contract spelled out on
//! [`crate::PersistentHashMap`] and its `PartialEq` impl — is meaningful for
//! the crate's default configuration.
//!
//! ## Algorithm
//!
//! FxHash-style word-at-a-time accumulation (`state = (state.rotate_left(5)
//! ^ word).wrapping_mul(GOLDEN)`, processing the input 8 bytes at a time
//! with a padded tail chunk), then a Murmur3 `fmix64` finalizer in
//! `finish()`. The finalizer is required, not decorative: CHAMP consumes a
//! hash's *lowest* 5 bits first (at the tree root), and FxHash-style
//! multiply-rotate accumulation alone leaves the low bits weak (poor
//! avalanche), which would skew the root-level bucket distribution and
//! defeat the whole point of a wide branching factor. `fmix64` gives every
//! output bit, including the low ones, full avalanche from the accumulated
//! state.

use std::hash::{BuildHasher, Hash, Hasher};

/// Multiplicative constant used by the accumulation step (odd, large,
/// good bit-mixing properties — the same constant FxHash uses).
const GOLDEN: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Deterministic, zero-dependency default [`BuildHasher`] for champ
/// collections.
///
/// Stateless (`Default + Clone + Copy`, all instances are identical) and
/// deterministic: hashing the same key with any two `DefaultBuildHasher`
/// instances (in the same process or a different one, same or different
/// build) always produces the same hash. See the module docs for why that
/// property, not raw hash quality, is the reason this exists instead of
/// `RandomState`.
///
/// **This is not a DoS-resistant hasher.** Like any deterministic hash, an
/// adversary who can choose input keys can construct pathological
/// collisions. Don't use `PersistentHashMap`'s default hasher for
/// attacker-controlled key sets in a context where hash-flooding is a
/// concern; plug in a keyed hasher via [`crate::PersistentHashMap::with_hasher`]
/// instead.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefaultBuildHasher;

impl BuildHasher for DefaultBuildHasher {
    type Hasher = DefaultHasher;

    #[inline]
    fn build_hasher(&self) -> DefaultHasher {
        DefaultHasher { state: 0 }
    }
}

/// [`Hasher`] implementation backing [`DefaultBuildHasher`]. See the module
/// docs for the algorithm.
#[derive(Debug)]
pub struct DefaultHasher {
    state: u64,
}

impl DefaultHasher {
    #[inline]
    fn accumulate(&mut self, word: u64) {
        self.state = (self.state.rotate_left(5) ^ word).wrapping_mul(GOLDEN);
    }
}

impl Hasher for DefaultHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let (head, tail) = bytes.split_at(8);
            // Unwrap is infallible: `head` is exactly 8 bytes.
            self.accumulate(u64::from_ne_bytes(head.try_into().unwrap()));
            bytes = tail;
        }
        if !bytes.is_empty() {
            let mut buf = [0u8; 8];
            buf[..bytes.len()].copy_from_slice(bytes);
            self.accumulate(u64::from_ne_bytes(buf));
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Murmur3 fmix64: strong avalanche finalizer, required so the low
        // bits (which CHAMP consumes first) are fully mixed — see module
        // docs.
        let mut h = self.state;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }
}

/// Order-independent hash of a collection of items: XOR-combines
/// `s.hash_one(item)` over every item, so hashing the same multiset of items
/// in any order (any iteration order over an unordered collection) produces
/// the same result.
///
/// Clients hashing a whole map (rather than a set) should feed this
/// `(k, v)` pairs, e.g. `hash_unordered(map.iter(), &hasher)` — hashing a
/// pair rather than key and value separately keeps a map's hash distinct
/// from a set-of-keys' hash even when the two share the same key multiset.
///
/// Like [`crate::PersistentHashMap`]'s equality (see its type docs), this is
/// only meaningful as an `=`-consistent hash (equal collections hash equal)
/// when every `s.hash_one` call is deterministic for equal items — true for
/// stateless/deterministic `BuildHasher`s like [`DefaultBuildHasher`], false
/// for randomly seeded ones like `RandomState`.
pub fn hash_unordered<T: Hash, S: BuildHasher>(items: impl IntoIterator<Item = T>, s: &S) -> u64 {
    items.into_iter().fold(0u64, |acc, item| acc ^ s.hash_one(item))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_instances() {
        let a = DefaultBuildHasher;
        let b = DefaultBuildHasher;
        assert_eq!(a.hash_one("hello"), b.hash_one("hello"));
        assert_eq!(a.hash_one(12345u64), b.hash_one(12345u64));
        assert_eq!(a.hash_one((1u32, "x")), b.hash_one((1u32, "x")));
    }

    #[test]
    fn distinguishes_different_inputs() {
        let s = DefaultBuildHasher;
        assert_ne!(s.hash_one("hello"), s.hash_one("world"));
        assert_ne!(s.hash_one(1u64), s.hash_one(2u64));
    }

    /// Loose statistical sanity check on the root-level chunk (low 5 bits
    /// of the folded 32-bit hash): hashing a large sequential range should
    /// hit all 32 buckets, none wildly over-represented. This is exactly
    /// the property the `fmix64` finalizer exists to guarantee (see module
    /// docs) — without it, sequential-integer low bits from the raw
    /// accumulator would be badly skewed.
    #[test]
    fn root_chunk_distribution_is_sane() {
        let s = DefaultBuildHasher;
        let n = 100_000u64;
        let mut buckets = [0u64; 32];
        for i in 0..n {
            let h = s.hash_one(i);
            let folded = (h ^ (h >> 32)) as u32;
            let chunk = folded & 0x1f;
            buckets[chunk as usize] += 1;
        }
        let mean = n as f64 / 32.0;
        for (chunk, &count) in buckets.iter().enumerate() {
            assert!(count > 0, "bucket {chunk} never hit");
            assert!(
                (count as f64) < mean * 2.0,
                "bucket {chunk} has {count} entries, > 2x mean {mean}"
            );
        }
    }

    #[test]
    fn hash_unordered_is_order_independent() {
        let s = DefaultBuildHasher;
        let a = [1, 2, 3, 4, 5];
        let mut b = a;
        b.reverse();
        assert_eq!(hash_unordered(a, &s), hash_unordered(b, &s));

        let c = [1, 2, 3, 4, 6];
        assert_ne!(hash_unordered(a, &s), hash_unordered(c, &s));

        let empty: [i32; 0] = [];
        assert_eq!(hash_unordered(empty, &s), 0);
    }

    #[test]
    fn hash_unordered_pairs_differ_from_keys_only() {
        let s = DefaultBuildHasher;
        let pairs = [(1, "a"), (2, "b")];
        let keys = [1, 2];
        // Not a hard guarantee in general, but for these concrete inputs
        // this documents/asserts the intended usage distinction.
        assert_ne!(hash_unordered(pairs, &s), hash_unordered(keys, &s));
    }
}
