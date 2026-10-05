//! Hot-reloadable config, shared across threads, with every in-flight
//! request seeing a perfectly consistent snapshot.
//!
//! **Use case:** a server keeps its config (feature flags, routing rules,
//! whatever) behind a lock so it can be hot-reloaded without a restart.
//! Naively, a request that reads the config twice mid-reload can observe
//! two different values for two different keys — a config that was never
//! valid at any single instant. The usual fix is either holding the read
//! lock for the whole request (kills reload latency and concurrency) or a
//! generation-counter/copy-on-read scheme built by hand.
//!
//! **Why champ wins:** the shared config is a
//! `RwLock<PersistentHashMap<String, String>>`. A request grabs a *read*
//! lock just long enough to `clone()` the map out — an `Arc`-style
//! refcount bump, not a copy of the entries — then releases the lock and
//! reads its own private, `Send`-able snapshot for the rest of the
//! request, however long that takes. A concurrent hot-reload swaps in a
//! brand new map under a brief *write* lock; every snapshot already handed
//! out is completely unaffected, because it's an independent immutable
//! value, not a view into mutable shared state.
//!
//! **Where the win comes from, mechanically:** champ collections are
//! `Send + Sync` (given `Send + Sync` contents) and persistent — no
//! collection is ever mutated after another handle might be reading it, so
//! handing a full snapshot across a thread boundary is exactly as cheap as
//! handing over a pointer. Keeping the last N configs around for debugging
//! (as this example does) costs a `Vec` of those same cheap clones, not N
//! copies of the config.
//!
//! Run with `cargo run --example config_snapshots`.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use champ::PersistentHashMap;

type Config = PersistentHashMap<String, String>;

fn build_config(generation: u32) -> Config {
    PersistentHashMap::new()
        .assoc("generation".to_string(), generation.to_string())
        .assoc("timeout_ms".to_string(), (100 + generation * 10).to_string())
        .assoc("feature_x".to_string(), if generation % 2 == 0 { "off".to_string() } else { "on".to_string() })
}

fn main() {
    let current: Arc<RwLock<Config>> = Arc::new(RwLock::new(build_config(0)));

    // Debug history of past configs — each entry is a cheap clone, not a
    // copy, so retaining several generations is nearly free.
    let history: Arc<RwLock<Vec<Config>>> = Arc::new(RwLock::new(Vec::new()));

    let mut request_threads = Vec::new();
    for id in 0..4u32 {
        let current = Arc::clone(&current);
        request_threads.push(thread::spawn(move || {
            // Grab a snapshot under a brief read lock, then let go of the
            // lock entirely for the rest of the "request".
            let snapshot: Config = {
                let guard = current.read().unwrap();
                guard.clone() // O(1): pointer/refcount copy, not a deep copy
            };

            let generation_at_start = snapshot.get(&"generation".to_string()).cloned();
            let timeout_at_start = snapshot.get(&"timeout_ms".to_string()).cloned();

            // Simulate a request doing several reads over some wall-clock
            // time, while the config keeps getting hot-reloaded elsewhere.
            for _ in 0..20 {
                thread::sleep(Duration::from_micros(50));
                let generation_now = snapshot.get(&"generation".to_string()).cloned();
                let timeout_now = snapshot.get(&"timeout_ms".to_string()).cloned();
                // The whole point: this thread's view never drifts mid-
                // request, no matter how many reloads happen concurrently.
                assert_eq!(generation_now, generation_at_start, "thread {id} saw a torn config read");
                assert_eq!(timeout_now, timeout_at_start, "thread {id} saw a torn config read");
            }

            println!(
                "  [reader {id}] consistent snapshot: generation={:?} timeout_ms={:?}",
                generation_at_start, timeout_at_start
            );
            generation_at_start
        }));
    }

    // Hot-reload the config a few times while readers are mid-flight.
    let mut reloader_generations = Vec::new();
    for generation in 1..=5u32 {
        thread::sleep(Duration::from_micros(80));
        let next = build_config(generation);
        {
            let mut guard = current.write().unwrap();
            *guard = next.clone();
        }
        history.write().unwrap().push(next);
        reloader_generations.push(generation);
    }

    let seen_generations: Vec<_> = request_threads
        .into_iter()
        .map(|h| h.join().expect("reader thread panicked"))
        .collect();

    println!("reloads applied:        {reloader_generations:?}");
    println!("generations readers saw at snapshot time: {seen_generations:?}");

    // The debug history retained one full config per reload, at the cost
    // of pointer clones — not five deep copies.
    let hist = history.read().unwrap();
    assert_eq!(hist.len(), 5);
    for (i, cfg) in hist.iter().enumerate() {
        let generation = (i as u32) + 1;
        assert_eq!(cfg.get(&"generation".to_string()), Some(&generation.to_string()));
    }
    println!("retained {} historical config snapshots for debugging, cheaply.", hist.len());

    // Final live config is the last reload.
    let final_cfg = current.read().unwrap().clone();
    assert_eq!(final_cfg.get(&"generation".to_string()), Some(&"5".to_string()));
    println!("final live config generation: {:?}", final_cfg.get(&"generation".to_string()));
}
