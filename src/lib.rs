//! champ — a CHAMP (Compressed Hash-Array Mapped Prefix-tree)
//! persistent hash map, built for Rust: single-allocation nodes, atomic
//! refcounts, and opportunistic in-place mutation when an update path holds
//! the only reference to a node. See `DESIGN.md` in the repository root for
//! the full rationale and `SPEC-CORE.md` for the exact node layout and
//! operation contracts this module implements.
//!
//! The public surface for this milestone is [`PersistentHashMap`] with
//! `get`/`assoc`/`dissoc` (both `&self`, persistent, and `_owned`, consuming
//! with in-place mutation on uniquely-owned nodes), iteration, and equality.

mod hash;
mod iter;
mod map;
mod node;
#[cfg(feature = "pool-alloc")]
mod pool;
mod set;
mod text;
#[cfg(feature = "spike-ropey")]
mod text_ropey;
mod transient;
mod vec;

pub use hash::{DefaultBuildHasher, DefaultHasher, hash_unordered};
pub use iter::{IntoIter, IntoKeys, Iter, Keys, Values};
pub use map::PersistentHashMap;
pub use set::PersistentHashSet;
pub use text::{Chunks, PText};
#[cfg(feature = "spike-ropey")]
pub use text_ropey::RopeyText;
pub use transient::{TransientMap, TransientSet};
pub use vec::{PVecChunks, PVecIter, PVector};

/// Total bytes currently parked in the *calling thread's* node pool (see
/// `src/pool.rs`), across all size classes. A measurement/introspection
/// hook for benchmarks and memory-accounting tools (e.g.
/// `examples/memstats.rs`) that need to separate "pool-retained bytes" from
/// "live map bytes" — a pooled block still reads as globally-allocated
/// memory to any allocator-level accounting, since it hasn't been freed,
/// only parked for reuse. Not needed for, and has no effect on, normal
/// map/set usage. Only present when the (default-on) `pool-alloc` feature
/// is enabled.
#[cfg(feature = "pool-alloc")]
pub fn pool_retained_bytes() -> usize {
    pool::retained_bytes()
}

/// Free every block currently parked in the *calling thread's* node pool
/// back to the system allocator. A measurement/testing hook, not needed by
/// normal map/set usage (the pool's own per-class cap already bounds
/// retention) — exists so callers that want an apples-to-apples comparison
/// against a pool-off build (or that are about to go idle and would rather
/// return memory to the OS) can force the pool back to empty on demand.
/// Only present when the (default-on) `pool-alloc` feature is enabled.
#[cfg(feature = "pool-alloc")]
pub fn pool_drain() {
    pool::drain()
}
