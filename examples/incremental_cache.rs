//! Incremental recomputation validated by version identity — an
//! `Arc`-style pointer compare doubling as an unforgeable "was this derived
//! from exactly that version?" certificate.
//!
//! **Use case:** a derived structure (a memoized index, a layout skeleton,
//! a computed view) needs to stay in sync with a source document as it's
//! edited. Recomputing it from scratch on every edit is correct but wastes
//! the work already done for everything the edit didn't touch. Patching it
//! incrementally is cheap — but only if the patch is applied to the exact
//! version it was derived from. Apply an incremental patch meant for
//! version N to version N-1 (an undo happened, a branch was taken, the
//! document was replaced wholesale) and you get a plausible-looking but
//! silently wrong index, the worst kind of bug because nothing crashes.
//!
//! **Why champ wins:** `PText::ptr_eq` compares tree *identity*, not
//! content — it's the same "same allocation" check `Arc::ptr_eq` gives
//! you, made available because champ's persistent ops never mutate a node
//! two callers might be holding. So an edit record's `base` field (the
//! exact `PText` an edit started from) can be compared against a cache's
//! own `text` field with a cheap pointer test: if they're identical, the
//! cache really was built from what this edit is a delta against, and the
//! patch is safe to apply. If they're not — even if the two versions have
//! byte-for-byte identical *content* — the cache's provenance doesn't
//! match, so the code falls back to a full rescan instead of guessing.
//! The identity check can never rubber-stamp a wrong patch: it either
//! proves the fast path is applicable, or it doesn't run at all, so a
//! failed check degrades to correct-but-slower, never to a wrong answer.
//!
//! **Where the win comes from, mechanically:** this is the same pattern
//! A host application (an editor built on this crate) uses for
//! its layout engine: `:editor/last-splice` records `{base, result, at,
//! deleted, inserted}` for every edit, exactly like `EditRecord` below.
//! The layout engine's memoized line skeleton patches only the
//! edited-line region — dropping stale entries in the touched span,
//! rescanning only the freshly inserted text, and shifting every later
//! offset by a constant — instead of re-laying-out the whole document,
//! *because* `ptr_eq(last_splice.base, cached_skeleton.source)` proves the
//! cache's lineage before the patch is trusted.
//!
//! Run with `cargo run --example incremental_cache`.

use champ::PText;

/// A derived index over a document: the char offset where each line
/// starts. `starts[0]` is always `0`; a document with `k` newlines has
/// `k + 1` entries, matching `PText::len_lines`.
struct LineIndex {
    text: PText,
    starts: Vec<usize>,
}

/// The oracle: rebuild the index from nothing but a linear scan. Always
/// correct, always `O(n)` — what `refresh` tries to avoid paying for on
/// every edit, and what every patched result is checked against below.
fn full_scan(text: &PText) -> Vec<usize> {
    let mut starts = vec![0];
    let mut offset = 0usize;
    for c in text.chars() {
        offset += 1;
        if c == '\n' {
            starts.push(offset);
        }
    }
    starts
}

/// What every edit produces — A host application's `:editor/last-splice`, generalized.
/// `base` and `result` are the actual `PText` values the edit ran between;
/// their *identity* (not their content) is what makes a downstream cache
/// able to trust a patch instead of a rescan.
struct EditRecord {
    base: PText,
    result: PText,
    at: usize,
    deleted: usize,
    inserted: usize,
}

/// A document that remembers nothing but its current text, and hands back
/// an `EditRecord` — a receipt naming exactly which version it started
/// from — for every edit it applies.
struct Document {
    text: PText,
}

impl Document {
    fn new(initial: &str) -> Self {
        Document { text: PText::from(initial) }
    }

    fn edit(&mut self, at: usize, deleted: usize, inserted: &str) -> EditRecord {
        let base = self.text.clone(); // O(1): refcount bump, not a copy
        let result = base.splice(at..at + deleted, inserted);
        self.text = result.clone();
        EditRecord { base, result, at, deleted, inserted: inserted.chars().count() }
    }
}

/// Bring `cache` up to date with `rec`. Returns the refreshed cache and
/// whether the fast (patch) path was taken.
///
/// The safety of the fast path rests entirely on one check:
/// `PText::ptr_eq(&rec.base, &cache.text)`. That's true exactly when
/// `cache` was built from the very version `rec` is a delta against — not
/// "a version with the same text", the *same tree*. Only then is patching
/// sound: drop the stale entries the edit invalidated, rescan just the
/// inserted text for new line breaks, and shift everything after the edit
/// by the length delta. If the check fails — the record is stale, e.g. an
/// undo jumped back to an earlier version, or the document was swapped
/// out from under the cache — there is no safe patch to apply, so this
/// falls back to `full_scan`. The check never certifies a bad patch: it
/// either proves the fast path applies, or the slow path runs instead.
fn refresh(cache: &LineIndex, rec: &EditRecord) -> (LineIndex, bool) {
    if !PText::ptr_eq(&rec.base, &cache.text) {
        // Stale record: cache's provenance doesn't match this edit's
        // starting point. Correct-but-slower, never wrong.
        return (LineIndex { text: rec.result.clone(), starts: full_scan(&rec.result) }, false);
    }

    let at = rec.at;
    let deleted_end = rec.at + rec.deleted;
    let delta = rec.inserted as isize - rec.deleted as isize;

    let mut starts = Vec::with_capacity(cache.starts.len());
    let mut i = 0;

    // Entries entirely before the edit are untouched — including one that
    // lands exactly at `at`, which still opens a line in the result (the
    // inserted/deleted span just becomes that line's new content).
    while i < cache.starts.len() && cache.starts[i] <= at {
        starts.push(cache.starts[i]);
        i += 1;
    }

    // Entries strictly inside the edited span are stale: the newline that
    // defined each one sat inside `[at, deleted_end)` and is gone. Drop
    // them; if the inserted text recreates an equivalent line break, the
    // scan below re-adds it at the right offset.
    while i < cache.starts.len() && cache.starts[i] <= deleted_end {
        i += 1;
    }

    // Re-scan ONLY the freshly inserted text for newlines — not the whole
    // document — and record their offsets in the new text.
    let inserted_text = rec.result.slice(at..at + rec.inserted);
    let mut offset = at;
    for c in inserted_text.chars() {
        offset += 1;
        if c == '\n' {
            starts.push(offset);
        }
    }

    // Everything after the edited span is untouched content, just shifted
    // by however much the edit changed the document's length.
    while i < cache.starts.len() {
        starts.push((cache.starts[i] as isize + delta) as usize);
        i += 1;
    }

    (LineIndex { text: rec.result.clone(), starts }, true)
}

fn main() {
    let initial = "aaa\nbbb\nccc";
    let mut doc = Document::new(initial);
    let mut cache = LineIndex { text: doc.text.clone(), starts: full_scan(&doc.text) };
    assert_eq!(cache.starts, full_scan(&doc.text));
    println!("initial doc: {:?}", initial);
    println!("initial line starts: {:?}", cache.starts);

    // Retain this early version — and its cache — for the "undo" act below.
    let early_text = doc.text.clone();
    let early_cache_starts = cache.starts.clone();

    let mut patched = 0u32;
    let mut rescanned = 0u32;

    // --- Edit 1: insert plain text mid-line (no newlines involved). -----
    // "aaa\nbbb\nccc" -> "aXaa\nbbb\nccc"
    let rec1 = doc.edit(1, 0, "X");
    let (cache1, was_patched) = refresh(&cache, &rec1);
    assert!(was_patched, "edit 1 should take the patch path: base == cache.text");
    assert_eq!(cache1.starts, full_scan(&doc.text), "patched index must match the oracle");
    println!(
        "edit 1 (mid-line insert):      patched={was_patched:<5} starts={:?}  text={:?}",
        cache1.starts,
        doc.text.to_string()
    );
    patched += 1;
    cache = cache1;

    // --- Edit 2: insert text that itself contains newlines. -------------
    // "aXaa\nbbb\nccc" -> "aXaa\nbbbYY\n\nccc" (insert "YY\n" right before
    // the second line's own newline)
    let rec2 = doc.edit(8, 0, "YY\n");
    let (cache2, was_patched) = refresh(&cache, &rec2);
    assert!(was_patched, "edit 2 should take the patch path: base == cache.text");
    assert_eq!(cache2.starts, full_scan(&doc.text), "patched index must match the oracle");
    println!(
        "edit 2 (insert with newline):  patched={was_patched:<5} starts={:?}  text={:?}",
        cache2.starts,
        doc.text.to_string()
    );
    patched += 1;
    cache = cache2;

    // --- Edit 3: delete a span that crosses newlines, merging lines. ----
    // "aXaa\nbbbYY\n\nccc" -> "aXaa\nbbbYccc" (delete "Y\n\n")
    let rec3 = doc.edit(9, 3, "");
    let (cache3, was_patched) = refresh(&cache, &rec3);
    assert!(was_patched, "edit 3 should take the patch path: base == cache.text");
    assert_eq!(cache3.starts, full_scan(&doc.text), "patched index must match the oracle");
    println!(
        "edit 3 (delete across \\n):     patched={was_patched:<5} starts={:?}  text={:?}",
        cache3.starts,
        doc.text.to_string()
    );
    patched += 1;
    cache = cache3;

    // --- The "undo" act: jump back to the retained early version, then --
    // --- refresh the *current* cache with a record whose `base` is that -
    // --- old version. `cache.text` is the latest document, not the early
    // --- one, so the identity check must fail — exactly the situation an
    // --- undo, a branch, or a wholesale document swap produces.
    let mut undone = Document { text: early_text.clone() };
    let stale_rec = undone.edit(0, 0, ">> "); // a perfectly ordinary edit...
    assert_eq!(early_cache_starts, full_scan(&early_text)); // ...against an old version.

    let (rescan_cache, was_patched) = refresh(&cache, &stale_rec);
    assert!(!was_patched, "a record based on a retired version must NOT take the patch path");
    assert_eq!(
        rescan_cache.starts,
        full_scan(&stale_rec.result),
        "rescan path must still be correct, just slower"
    );
    println!(
        "undo + stale edit:             patched={was_patched:<5} starts={:?}  text={:?}",
        rescan_cache.starts,
        stale_rec.result.to_string()
    );
    rescanned += 1;

    println!(
        "\n{} edits total: {} patched (identity check passed), {} rescanned (identity check failed) — all verified against full_scan.",
        patched + rescanned,
        patched,
        rescanned
    );
    assert_eq!(patched, 3);
    assert_eq!(rescanned, 1);
}
