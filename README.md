# mova-champ

The `champ` crate. "mova-champ" is the repository name; the crate and its
import path are `champ`. It is the collections library of
[Mova](https://github.com/Vadym-Lopatka/mova), which keeps its own copy in
`vendor/champ`.

**Persistent hash maps, sets, vectors, and text with Clojure semantics — built to be the memory-lightest way to keep every version of your data.**

CHAMP (*Compressed Hash-Array Mapped Prefix-tree*, Steindorfer & Vinju, OOPSLA 2015) is the successor to the HAMT that powers Clojure's collections. This crate is a from-scratch Rust implementation of it — plus a persistent vector and a persistent rope — with zero required dependencies.

## Why this exists

This crate was born from a measured pain point, not a hypothesis.

A Clojure dialect written in Rust originally backed its large maps with a vendored HAMT crate. Profiling a real workload — many small, short-lived maps — showed **~8.7 KB of RSS per 7-key map**. That number was the assignment: build persistent collections with unchanged Clojure semantics that demolish it.

CHAMP's design is what makes that possible, and this crate leans into Rust to push it further:

- **Two disjoint bitmaps** (`datamap`/`nodemap`) instead of one bitmap plus boxed-node sniffing — entries and children live in separate, contiguous regions of a **single exact-fit allocation** per node. No `Box<[Entry]>` double indirection, no slack.
- **Canonical form** — deletion always restores the exact tree shape a fresh build would produce. Two equal maps are *structurally identical*, so `==` can short-circuit on pointer equality and fail fast on bitmap mismatch instead of walking `O(n)` entries.
- **Ownership-aware mutation** — when a node's refcount is 1, "persistent" operations mutate in place instead of copying. Rust's ownership system replaces Clojure's transient edit-token machinery entirely, and real transients are there too when you want them.

The result, as measured by the author:

| Scenario | baseline | champ |
|---|---|---|
| Clojure-dialect interpreter, 100k live 64-key maps (RSS) | ~11.1 GB | **~1.03 GB** (10.8× smaller) |
| 100k independent 7-key maps | ~8.7 KB/map (RSS) | **~0.2 KB/map** (live heap) |
| u64→u64 map, 100k entries | `im`: 255 B/entry | **23.8 B/entry** — near-parity with *mutable* `std::HashMap` (22.3) |
| Persistent build, 100k inserts | `im`: 102 ms | **9.0 ms** |
| Equality of a clone with 1 change | `im`: 4.4 ms | **37 ns** |

The integration that produced the first row swapped collections under a full language runtime with **zero behavioral changes** — 664/664 tests passing.

## Using it

Not on crates.io. Depend on it from git (edition 2024, minimum Rust 1.94, zero required dependencies):

```toml
[dependencies]
champ = { git = "https://github.com/Vadym-Lopatka/mova-champ" }
```

See the [Examples](#examples) section below for five runnable, self-contained programs covering the use cases champ is built for.

## The collections

| Type | What it is |
|---|---|
| `PersistentHashMap<K, V>` | CHAMP hash map, Clojure `assoc`/`dissoc` semantics |
| `PersistentHashSet<K>` | CHAMP hash set (a map with `()` values, same guarantees) |
| `TransientMap` / `TransientSet` | owned batch builders — Clojure's `transient` … `persistent!` |
| `PVector<T>` | persistent vector: `push`/`pop` at both ends, `set`, `slice` |
| `PText` | persistent rope for text: char-indexed `splice`/`slice`, line queries |

Every persistent operation comes in two flavors: a `&self` method that always leaves the original untouched, and an `_owned` method that consumes `self` and mutates in place wherever it holds the only reference — same semantics, no copying tax when you don't share.

### Maps and sets

```rust
use champ::PersistentHashMap;

let empty: PersistentHashMap<&str, i32> = PersistentHashMap::new();

// Persistent: every version stays alive and shares structure.
let a = empty.assoc("x", 1).assoc("y", 2);
let b = a.assoc("x", 10);
assert_eq!(a.get(&"x"), Some(&1));   // untouched
assert_eq!(b.get(&"x"), Some(&10));

// Owned: same semantics, but mutates in place while the map is unshared.
let big = (0..100_000).fold(PersistentHashMap::new(), |m, i| m.assoc_owned(i, i * 2));

// Transient: batch edits, then an O(1) move back to persistent.
let mut t = big.transient();
t.assoc(100_000, 0);
t.dissoc(&0);
let done = t.persistent();
```

Clojure's contract is kept precisely: `dissoc` of an absent key returns the *pointer-identical* map, `assoc` of an unchanged key/value pair allocates nothing, replacing a value keeps the original key instance. Equality between independently built maps is cheap because the default hasher is deterministic — equal contents produce byte-identical trees.

### Vectors

```rust
use champ::PVector;

let v = PVector::from_slice(&[1, 2, 3]);
let w = v.push_back(4);              // persistent
let x = w.clone().set_owned(0, 99);  // owned, in-place when unique

assert_eq!(v.len(), 3);              // still [1, 2, 3]
assert_eq!(x.get(0), Some(&99));

let front = v.slice(0..2);           // structural slice, no copying
let sum: i32 = w.iter().sum();
```

`eq_by` / `try_eq_by` compare two vectors with a custom (even fallible) predicate and skip pointer-shared subtrees entirely — comparing a vector with its own lightly-edited descendant touches only what changed.

### Text

```rust
use champ::PText;

let t = PText::from("hello world");
let u = t.splice(5..6, ", ");                 // replace the space with ", "
assert_eq!(u.to_string(), "hello, world");
assert_eq!(t.to_string(), "hello world");     // the original is a kept snapshot

let head = u.slice(0..5);                     // zero-copy structural slice
let line = u.char_to_line(3);                 // line/char index queries
for chunk in u.chunks() { /* zero-copy &str leaves */ }
```

`PText` hashes identically to the equivalent `str`, so it drops into `Hash`/`Eq`-keyed collections next to ordinary strings.

## Examples

Every use case champ is built for has a runnable example — `cargo run --example <name>`:

| Example | Use case | The one-line win |
|---|---|---|
| `undo_history` | Editor undo/redo as `Vec<PText>` snapshots | No command pattern, no inverse operations — a thousand retained snapshots share structure instead of copying |
| `change_detection` | React-style "did anything change?" gating with plain `==` | Equality between related versions skips pointer-shared subtrees — no dirty flags, no change listeners |
| `config_snapshots` | Atomic hot-reloadable config shared across threads | Snapshots are `Send + Sync` pointer clones — every request sees a perfectly consistent view, concurrently |
| `interpreter_env` | Lexical scopes in a toy expression interpreter | Child scopes share all parent structure — closures capture environments by pointer clone, not map copy |
| `speculative_world` | Try-it-keep-it-only-if-valid game transactions | "Discard = drop, commit = assign" — no transaction journal, no rollback logic |
| `incremental_cache` | Incremental recomputation validated by version identity | `ptr_eq` is an unforgeable "derived from exactly that version" certificate — patch when it proves lineage, full rescan when it can't |
| `memstats`, `text_memstats`, `vec_memstats` | — | Exact-byte measurement harnesses for the memory numbers above |

## Guarantees

- **Thread-safe sharing**: all types are `Send + Sync` (given `Send + Sync` contents); internal refcounts follow the `Arc` protocol.
- **Deterministic, canonical trees**: iteration order is stable for equal contents (though not insertion order — the Clojure contract), and `validate()` (behind the `validate` feature) checks every structural invariant.
- **Tested like it matters**: property-tested against `std` collections as models — including adversarial colliding hashers — and run under Miri across the whole suite, zero errors, zero leaks.

## Feature flags

| Flag | Default | Effect |
|---|---|---|
| `pool-alloc` | on | thread-local node pool that recycles node allocations (turn off under Miri) |
| `validate` | off | expose `validate()` invariant checkers for integration tests |
| `spike-ropey` | off | `RopeyText`, a `ropey`-backed twin of `PText`'s API, for benchmarking only |

## When to reach for champ — and when not to

**A good fit** when versions of your data need to coexist: an interpreter's environments, undo histories, snapshots handed across threads, speculative edits you might discard. That's the workload champ was built for, and there it is simultaneously ~7–11× lighter than comparable persistent crates, close to *mutable* `std::HashMap` in bytes per entry, and equipped with near-free equality between related versions. Structural sharing means a thousand snapshots cost little more than one.

**A poor fit** when nothing is ever shared. If you hold exactly one current version and mutate it in a tight loop, `std::collections` and `String` will beat any persistent structure — persistence is a property you should be *using*, not just paying for.

Two honest edges even inside the good cases, measured rather than tuned away:

- Assoc-heavy update loops measured up to ~26% slower than the previous HAMT at the whole-interpreter level in that Clojure dialect — the throughput price of the 10.8× memory win.
- `PText`'s *persistent* splice path (keeping every intermediate revision of clustered edits) is roughly an order of magnitude slower than a dedicated mutable rope like `ropey` — even though `PText` wins the owned sequential-typing and bulk-replace workloads it targets.

In one line: **champ turns "keep every version" from a memory catastrophe into the cheap default — as long as keeping versions is actually what you're doing.**

## Tests

`cargo test`

## License

EPL-1.0, see LICENSE.
