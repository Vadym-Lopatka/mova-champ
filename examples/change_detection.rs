//! React-style "did anything change?" gating, done with plain `==` — no
//! dirty flags, no change listeners.
//!
//! **Use case:** a UI (or a cache, or a downstream subscriber) needs to
//! know whether a re-render/re-computation is worth doing after an event
//! is applied to some state. The usual answers are a hand-maintained dirty
//! flag (easy to forget to set, easy to set too eagerly) or a deep
//! structural diff of old vs. new state (correct, but `O(n)` on every
//! event even when the event touched one field).
//!
//! **Why champ wins:** state lives in a `PersistentHashMap`. A `reduce`
//! function takes the *old* state by value and returns a *new* state;
//! nothing is mutated. Deciding whether to re-render is exactly
//! `new_state != old_state` — ordinary `PartialEq`, no bespoke diffing
//! code, no flags to thread through every mutation site.
//!
//! **Where the win comes from, mechanically:** champ's canonical tree form
//! means two maps with the same contents are *structurally* identical, so
//! equality checks pointer identity node-by-node and only descends where
//! pointers differ (see `nodes_eq` in `src/map.rs`). A no-op event's
//! `assoc` of an already-present key/value pair is documented to return a
//! *pointer-identical* map — so comparing a 100k-entry map against a
//! descendant with one real change touches only the edited path, and this
//! crate's benches measure that at ~37ns (vs 4.4ms for `im`'s crate — see
//! the README's numbers table) — while a no-op event's equality check is
//! effectively a single pointer compare at the root.
//!
//! Run with `cargo run --example change_detection`.

use champ::PersistentHashMap;

type State = PersistentHashMap<String, i64>;

#[derive(Debug)]
enum Event {
    SetScore(&'static str, i64),
    /// Sets a value that's already there — a genuine no-op.
    Noop(&'static str, i64),
}

fn reduce(state: State, event: &Event) -> State {
    match *event {
        Event::SetScore(key, val) => state.assoc(key.to_string(), val),
        Event::Noop(key, val) => state.assoc(key.to_string(), val),
    }
}

/// Stand-in for an expensive UI re-render. Only called when state actually
/// changed, and we count calls to prove the gate is doing its job.
fn rerender(render_count: &mut u32, state: &State) {
    *render_count += 1;
    println!("  [rerender #{render_count}] alice={:?}", state.get(&"alice".to_string()));
}

fn main() {
    let mut state: State = PersistentHashMap::new();
    state = state.assoc("alice".to_string(), 0);
    state = state.assoc("bob".to_string(), 0);

    let mut render_count = 0u32;
    rerender(&mut render_count, &state); // initial paint

    let events = [
        Event::SetScore("alice", 10), // real change
        Event::Noop("alice", 10),     // same key, same value: no-op
        Event::SetScore("bob", 5),    // real change
        Event::Noop("bob", 5),        // no-op again
        Event::SetScore("alice", 20), // a third genuine change
    ];

    for event in &events {
        let new_state = reduce(state.clone(), event);

        let changed = new_state != state;
        println!("event {event:?}: changed = {changed}");

        if changed {
            rerender(&mut render_count, &new_state);
        }

        // The no-op events must produce a map that's not just *equal* but
        // pointer-identical to the input — champ's assoc contract for an
        // unchanged key/value pair. That's what makes the `!=` check cheap
        // instead of a full O(n) structural walk.
        if matches!(event, Event::Noop(..)) {
            assert!(new_state.ptr_eq(&state), "no-op assoc should be pointer-identical");
            assert!(!changed, "no-op event must not report a change");
        }

        state = new_state;
    }

    // Real changes did NOT produce pointer-identical maps.
    assert_eq!(state.get(&"alice".to_string()), Some(&20));
    assert_eq!(state.get(&"bob".to_string()), Some(&5));

    println!("total renders: {render_count} (for {} events, 2 of which were no-ops)", events.len());
    assert_eq!(render_count, 1 + 3); // initial paint + 3 real changes
}
