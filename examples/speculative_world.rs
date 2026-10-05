//! Speculative "try it, keep it only if valid" transactions — with no
//! rollback logic at all.
//!
//! **Use case:** a game (or any simulation) wants to try an action, check
//! whether the resulting state is legal, and either commit it or throw it
//! away. The usual machinery for that is a transaction journal: record
//! every mutation you're about to make, and if validation fails, replay
//! the journal backward to undo them.
//!
//! **Why champ wins:** `try_apply` builds a *candidate* world from
//! `world.clone()` plus whatever the action does, validates the candidate,
//! and returns `Some(candidate)` or `None`. "Rollback" on failure is
//! simply never assigning the candidate anywhere — it just gets dropped.
//! "Commit" on success is a variable assignment. There is no undo log,
//! because the base `world` was never mutated in the first place; the
//! candidate is a wholly separate value from the moment it's born.
//!
//! **Where the win comes from, mechanically:** `world.clone()` is an
//! O(1) pointer/refcount copy, so building a candidate to validate is
//! cheap even for a large world, and the base stays byte-for-byte
//! unchanged whether the candidate is accepted or discarded. This also
//! means exploring several "what-if" branches from the same base state in
//! parallel (e.g. AI move search) is just N cheap clones of one shared
//! base, not N copies of the world.
//!
//! Run with `cargo run --example speculative_world`.

use champ::PersistentHashMap;

/// entity id -> hp
type World = PersistentHashMap<u32, i64>;

#[derive(Debug, Clone, Copy)]
enum Action {
    Damage { target: u32, amount: i64 },
    Heal { target: u32, amount: i64 },
}

/// Build a candidate world from a clone of `world`, apply the action, and
/// hand back `Some(candidate)` only if it's still legal (no entity's hp
/// drops below zero). On rejection, the candidate is simply dropped —
/// `world` was never touched, so there's nothing to roll back.
fn try_apply(world: &World, action: Action) -> Option<World> {
    let candidate = match action {
        Action::Damage { target, amount } => {
            let hp = *world.get(&target)?;
            world.assoc(target, hp - amount)
        }
        Action::Heal { target, amount } => {
            let hp = *world.get(&target)?;
            world.assoc(target, hp + amount)
        }
    };

    let valid = candidate.iter().all(|(_, &hp)| hp >= 0);
    if valid { Some(candidate) } else { None }
}

fn main() {
    let mut world: World = PersistentHashMap::new().assoc(1, 30).assoc(2, 10).assoc(3, 5);

    println!("base world: entity1={:?} entity2={:?} entity3={:?}", world.get(&1), world.get(&2), world.get(&3));

    // An action that keeps everyone's hp legal: accepted.
    let action_ok = Action::Damage { target: 1, amount: 20 };
    let base_before = world.clone();
    match try_apply(&world, action_ok) {
        Some(next) => {
            assert_eq!(next.get(&1), Some(&10));
            // Base world (and our separately held clone of it) is
            // completely untouched by building/validating the candidate.
            assert_eq!(world.get(&1), Some(&30));
            assert!(world.ptr_eq(&base_before));
            world = next; // commit: just an assignment
            println!("accepted {action_ok:?} -> entity1={:?}", world.get(&1));
        }
        None => panic!("expected this action to be accepted"),
    }

    // An action that would push an entity's hp negative: rejected, and the
    // world is exactly as it was before we even tried.
    let action_bad = Action::Damage { target: 3, amount: 999 };
    let world_before_attempt = world.clone();
    match try_apply(&world, action_bad) {
        Some(_) => panic!("expected this action to be rejected"),
        None => {
            // No commit happened; `world` is untouched — there was never
            // anything to undo, because nothing was ever mutated.
            assert!(world.ptr_eq(&world_before_attempt));
            assert_eq!(world.get(&3), Some(&5));
            println!("rejected {action_bad:?} -> world unchanged, entity3={:?}", world.get(&3));
        }
    }

    // Parallel what-if exploration: try several candidate futures from the
    // *same* base world, each a cheap independent clone-and-branch, and
    // pick the one we like. The base is untouched by all of them.
    let branch_heal = try_apply(&world, Action::Heal { target: 2, amount: 5 });
    let branch_damage = try_apply(&world, Action::Damage { target: 2, amount: 5 });
    let branch_fatal = try_apply(&world, Action::Damage { target: 2, amount: 999 });

    assert_eq!(branch_heal.as_ref().and_then(|w| w.get(&2)), Some(&15));
    assert_eq!(branch_damage.as_ref().and_then(|w| w.get(&2)), Some(&5));
    assert!(branch_fatal.is_none());
    assert_eq!(world.get(&2), Some(&10)); // base never moved through any of this

    println!(
        "explored 3 what-if branches from one shared base: heal={:?} damage={:?} fatal={:?}",
        branch_heal.as_ref().and_then(|w| w.get(&2)),
        branch_damage.as_ref().and_then(|w| w.get(&2)),
        branch_fatal.is_some()
    );

    println!("speculative_world: commit-or-drop, zero transaction-journal code.");
}
