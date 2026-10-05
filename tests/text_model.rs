//! Property tests: `PText` against `String` as a model, over random splice
//! sequences (positions/lengths/content including multi-byte text),
//! comparing full content, lengths (bytes/chars/lines), and spot
//! char_at/line queries after every op. Also covers clone-and-diverge (the
//! undo-stack pattern: old versions retained and re-verified) and
//! slice-then-splice aliasing (structural sharing must not let an edit to
//! one version leak into another).

use champ::PText;
use proptest::prelude::*;

// ---------------------------------------------------------------------
// Boundary suite (SPEC-PTEXT.md's mandatory boundary coverage): empty
// text, single char, exactly LEAF_MAX, splices at 0/end/chunk boundaries,
// deleting across many chunks, degenerate repeated single-char inserts at
// the same position (sequential typing), CRLF, and emoji/combining
// sequences. Plain #[test]s, not proptest — these are specific cases the
// fuzzer might not reliably hit, checked deterministically.
// ---------------------------------------------------------------------

/// `PText`'s own `LEAF_MAX` isn't public; 2048 per SPEC-PTEXT.md/NOTES-PTEXT.md.
const LEAF_MAX: usize = 2048;

#[test]
fn boundary_empty_text() {
    let t = PText::new();
    t.validate();
    assert!(t.is_empty());
    assert_eq!(t.len_bytes(), 0);
    assert_eq!(t.len_chars(), 0);
    assert_eq!(t.len_lines(), 1);
    assert_eq!(t.to_string(), "");
    // An empty text is still one (empty) leaf, so `chunks()` yields exactly
    // one (empty) chunk, not zero — `to_string()`/`chars()` still come out
    // empty either way.
    assert_eq!(t.chunks().collect::<String>(), "");
    let t2 = t.splice(0..0, "");
    t2.validate();
    assert!(t2.is_empty());
}

#[test]
fn boundary_single_char() {
    let t = PText::from("x");
    t.validate();
    assert_eq!(t.len_bytes(), 1);
    assert_eq!(t.len_chars(), 1);
    assert_eq!(t.char_at(0), 'x');
    let t2 = t.splice(0..1, "");
    t2.validate();
    assert!(t2.is_empty());
    let t3 = t.splice(1..1, "y");
    t3.validate();
    assert_eq!(t3.to_string(), "xy");
    let t4 = t.splice(0..0, "y");
    t4.validate();
    assert_eq!(t4.to_string(), "yx");
}

#[test]
fn boundary_exactly_leaf_max() {
    let s = "a".repeat(LEAF_MAX);
    let t = PText::from(s.as_str());
    t.validate();
    assert_eq!(t.len_bytes(), LEAF_MAX);
    assert_eq!(t.to_string(), s);
    // One more byte forces a split.
    let t2 = t.splice(0..0, "b");
    t2.validate();
    assert_eq!(t2.len_bytes(), LEAF_MAX + 1);
    assert_eq!(t2.char_at(0), 'b');
    // One fewer byte stays a single leaf's worth of content.
    let t3 = t.splice(0..1, "");
    t3.validate();
    assert_eq!(t3.len_bytes(), LEAF_MAX - 1);
}

#[test]
fn boundary_splice_at_leaf_and_internal_boundaries() {
    // Many leaves' worth of content; splice right at what should be a leaf
    // boundary (a multiple of LEAF_MAX-ish) as well as at 0 and at the end.
    let s = "0123456789".repeat(LEAF_MAX * 3 / 10 + 50); // several full leaves
    let t = PText::from(s.as_str());
    t.validate();
    let n = t.len_chars();

    let at_start = t.splice(0..0, "START");
    at_start.validate();
    assert_eq!(at_start.to_string(), format!("START{s}"));

    let at_end = t.splice(n..n, "END");
    at_end.validate();
    assert_eq!(at_end.to_string(), format!("{s}END"));

    for boundary in [LEAF_MAX, LEAF_MAX * 2, LEAF_MAX * 3] {
        if boundary < n {
            let edited = t.splice(boundary..boundary, "|");
            edited.validate();
            let mut expected = s.clone();
            expected.insert(boundary, '|');
            assert_eq!(edited.to_string(), expected);
        }
    }
}

#[test]
fn boundary_delete_across_many_chunks() {
    let s = "x".repeat(LEAF_MAX * 10);
    let t = PText::from(s.as_str());
    t.validate();
    let n = t.len_chars();
    // Delete a huge middle span crossing many leaf/internal boundaries.
    let deleted = t.splice(100..n - 100, "");
    deleted.validate();
    assert_eq!(deleted.len_chars(), 200);
    assert_eq!(deleted.to_string(), "x".repeat(200));
}

#[test]
fn boundary_degenerate_repeated_single_char_inserts_same_position() {
    // Sequential typing at position 0 (prepend) via the owned path.
    let mut t = PText::new();
    let mut model = String::new();
    for ch in "abcdefghij".repeat(50).chars() {
        t = t.splice_owned(0..0, &ch.to_string());
        model.insert(0, ch);
    }
    t.validate();
    assert_eq!(t.to_string(), model);
}

#[test]
fn boundary_crlf() {
    let s = "line1\r\nline2\r\nline3\r\n";
    let t = PText::from(s);
    t.validate();
    // '\n' is what's counted as a newline; each \r\n contributes exactly 1.
    assert_eq!(t.len_lines(), 4); // 3 newlines -> 4 lines (last one empty)
    assert_eq!(t.line_to_char(0), 0);
    assert_eq!(t.line_to_char(1), 7);
    assert_eq!(t.line_to_char(2), 14);
    let spliced = t.splice(5..5, "\r\n");
    spliced.validate();
    assert_eq!(spliced.to_string(), "line1\r\n\r\nline2\r\nline3\r\n");
}

#[test]
fn boundary_emoji_and_combining_sequences() {
    // Family emoji (multiple codepoints joined by ZWJ), flag (regional
    // indicator pair), and a base char + combining accent.
    let s = "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\u{1f1fa}\u{1f1f8}e\u{0301}";
    let t = PText::from(s);
    t.validate();
    assert_eq!(t.to_string(), s);
    assert_eq!(t.len_chars(), s.chars().count());
    // Splice right in the middle of the sequence (between codepoints, a
    // valid char boundary even though it looks like "inside one grapheme").
    let mid = s.chars().count() / 2;
    let byte_mid = t.byte_of_char(mid);
    assert!(s.is_char_boundary(byte_mid));
    let edited = t.splice(mid..mid, "|");
    edited.validate();
    let mut expected = s.to_string();
    expected.insert(byte_mid, '|');
    assert_eq!(edited.to_string(), expected);
}

/// Refcount-leak stress: build a tree, fan out many clones and slices
/// (bumping/holding refcounts on shared subtrees), splice some of them
/// through both the shared (copy) and — where a clone happens to be
/// unique — owned paths, then drop everything in a shuffled order.
/// Correctness here isn't observable by content assertions alone; the
/// point is to give Miri's leak/UB checker a rich refcount workout (see
/// NOTES-PTEXT.md's Miri section — this is the crate's "refcount-leak
/// test" per SPEC-PTEXT.md, same spirit as `tests/model.rs`'s Rc-keyed
/// leak tests, adapted for a structure with no generic K/V to `Rc`-wrap).
#[test]
fn refcount_leak_stress_clone_slice_splice_shuffled_drop() {
    let base = "The quick brown fox jumps over the lazy dog. ".repeat(80);
    let root = PText::from(base.as_str());
    root.validate();

    let mut versions = vec![root.clone()];
    for i in 0..30 {
        let src = &versions[i % versions.len()];
        let n = src.len_chars();
        let pos = (i * 37) % (n + 1);
        let spliced = src.splice(pos..pos, "XYZ");
        let sliced = src.slice(0..n / 2);
        versions.push(spliced);
        versions.push(sliced);
    }
    // Also exercise the owned path on fresh unique clones.
    for v in versions.iter().take(10).cloned().collect::<Vec<_>>() {
        let n = v.len_chars();
        let _ = v.splice_owned(0.min(n)..0.min(n), "Q");
    }
    for v in &versions {
        v.validate();
    }
    // Drop in reverse-ish shuffled order (rotate, not sort, to keep it
    // deterministic without pulling in a shuffle dependency).
    let mid = versions.len() / 3;
    versions.rotate_left(mid);
    drop(versions);
    drop(root);
}

/// A small pool of chars deliberately including multi-byte (Latin-1
/// supplement, an emoji outside the BMP), a combining mark (zero-width when
/// rendered but a full `char` in Rust's model), and both line-ending
/// styles, alongside plain ASCII — proptest's `any::<char>()` would spend
/// almost all its budget on obscure codepoints instead of stressing the
/// boundary-heavy cases that actually matter for a rope.
fn char_strategy() -> impl Strategy<Value = char> {
    prop_oneof![
        3 => proptest::char::range('a', 'z'),
        2 => proptest::char::range('A', 'Z'),
        1 => proptest::char::range('0', '9'),
        1 => Just(' '),
        1 => Just('\n'),
        1 => Just('\r'),
        1 => Just('\u{00e9}'),     // e-acute, 2 bytes
        1 => Just('\u{0301}'),     // combining acute accent, 2 bytes, zero-width
        1 => Just('\u{4e2d}'),     // CJK, 3 bytes
        1 => Just('\u{1f600}'),    // emoji, 4 bytes, outside the BMP
    ]
}

fn text_strategy(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(char_strategy(), 0..max_len).prop_map(|v| v.into_iter().collect())
}

/// `(position_fraction, delete_fraction, inserted_text)` — fractions are
/// resolved against the model's *current* char length inside the test loop
/// (proptest strategies can't see runtime state), then clamped, so every
/// generated op is valid regardless of how the text has grown/shrunk so
/// far.
fn op_strategy() -> impl Strategy<Value = (f64, f64, String)> {
    (0.0..1.0f64, 0.0..1.0f64, text_strategy(6))
}

fn ops_strategy() -> impl Strategy<Value = Vec<(f64, f64, String)>> {
    #[cfg(miri)]
    let max_len = 15;
    #[cfg(not(miri))]
    let max_len = 150;
    prop::collection::vec(op_strategy(), 0..max_len)
}

fn model_splice(model: &mut String, char_pos: usize, char_del: usize, insert: &str) {
    let start = model.char_indices().nth(char_pos).map(|(b, _)| b).unwrap_or(model.len());
    let end = model.char_indices().nth(char_pos + char_del).map(|(b, _)| b).unwrap_or(model.len());
    model.replace_range(start..end, insert);
}

fn assert_matches_model(text: &PText, model: &str) {
    text.validate();
    assert_eq!(text.to_string(), model, "content mismatch");
    assert_eq!(text.len_bytes(), model.len(), "len_bytes mismatch");
    assert_eq!(text.len_chars(), model.chars().count(), "len_chars mismatch");
    assert_eq!(text.len_lines(), model.split('\n').count(), "len_lines mismatch");
    if !model.is_empty() {
        let n = model.chars().count();
        for spot in [0usize, n / 3, n / 2, n - 1] {
            assert_eq!(text.char_at(spot), model.chars().nth(spot).unwrap(), "char_at({spot}) mismatch");
        }
    }
    // Every line start round-trips through char_to_line.
    let n_lines = text.len_lines();
    for line in 0..n_lines {
        let c = text.line_to_char(line);
        assert!(c <= text.len_chars());
        assert_eq!(text.char_to_line(c), line, "char_to_line(line_to_char({line})) mismatch");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 300 }))]

    /// Random splice sequences, alternating the persistent (`splice`) and
    /// owned (`splice_owned`) paths, checked against a `String` model after
    /// every op. Every 7th op also snapshots (clone) both structures; all
    /// snapshots are re-verified at the end against their own model
    /// snapshot — the undo-stack pattern: retaining an old `PText` version
    /// must not let later edits (via either path) leak into it.
    #[test]
    fn splice_matches_string_model(ops in ops_strategy()) {
        let mut text = PText::new();
        let mut model = String::new();
        let mut retained: Vec<(PText, String)> = Vec::new();

        for (i, (pos_frac, del_frac, insert)) in ops.into_iter().enumerate() {
            let n = model.chars().count();
            let pos = ((pos_frac * n as f64) as usize).min(n);
            let max_del = n - pos;
            let del = ((del_frac * max_del as f64) as usize).min(max_del);

            model_splice(&mut model, pos, del, &insert);
            text = if i % 2 == 0 {
                text.splice(pos..pos + del, &insert)
            } else {
                text.splice_owned(pos..pos + del, &insert)
            };

            assert_matches_model(&text, &model);

            if i % 7 == 0 {
                retained.push((text.clone(), model.clone()));
            }
        }

        for (rt, rm) in &retained {
            assert_matches_model(rt, rm);
        }
        // The live (final) version must still be independently correct
        // after every retained clone has been checked (clones are shallow
        // rc bumps — verifying them must not have perturbed `text`).
        assert_matches_model(&text, &model);
    }

    /// Slice-then-splice aliasing: a slice taken from a text must be
    /// unaffected by subsequent edits to the original (structural sharing
    /// must copy-on-write, never mutate shared subtrees), and vice versa.
    #[test]
    fn slice_then_splice_aliasing(
        base in text_strategy(200),
        slice_pos_frac in 0.0..1.0f64,
        slice_len_frac in 0.0..1.0f64,
        edit_pos_frac in 0.0..1.0f64,
        insert in text_strategy(6),
    ) {
        let text = PText::from(base.as_str());
        let n = text.len_chars();
        let slice_start = ((slice_pos_frac * n as f64) as usize).min(n);
        let slice_len = ((slice_len_frac * (n - slice_start) as f64) as usize).min(n - slice_start);
        let slice_end = slice_start + slice_len;

        let sl = text.slice(slice_start..slice_end);
        let expected_slice: String = base.chars().skip(slice_start).take(slice_len).collect();
        assert_eq!(sl.to_string(), expected_slice);

        let edit_pos = ((edit_pos_frac * n as f64) as usize).min(n);
        let edited = text.splice(edit_pos..edit_pos, &insert);

        // Original and slice must be untouched by the edit to `edited`.
        assert_eq!(text.to_string(), base);
        assert_eq!(sl.to_string(), expected_slice);

        let mut expected_edited = base.clone();
        let byte_pos = base.char_indices().nth(edit_pos).map(|(b, _)| b).unwrap_or(base.len());
        expected_edited.insert_str(byte_pos, &insert);
        assert_eq!(edited.to_string(), expected_edited);
    }

    /// M7.5 (SPEC-PTEXT.md follow-up): random-position splices via the
    /// **owned** path exclusively, starting from a bulk-built multi-leaf
    /// document — the exact shape of the random-splice benchmark scenario
    /// that motivated M7.5's rewrite of `splice_owned`'s fallback. Biases
    /// toward small ranges (like the benchmark's ~0-3 char deletes) but
    /// also includes occasional larger ones so both the common
    /// single-leaf-split-and-insert-into-unique-parent path and the rarer
    /// span-fallback path (padding a shrunk subtree back to
    /// `expected_height` — see `ptext_to_outcome`'s doc comment) get
    /// exercised. Every 5th op clones first (so the *next* op's descent
    /// hits a shared node partway down, exercising `owned_splice_copy`),
    /// checked against a `String` model after every op.
    #[test]
    fn random_position_owned_splices_matches_string_model(
        base in text_strategy(if cfg!(miri) { 200 } else { 3000 }),
        ops in prop::collection::vec(
            (0.0..1.0f64, prop_oneof![3 => 0.0..0.05f64, 1 => 0.0..1.0f64], text_strategy(6)),
            0..(if cfg!(miri) { 15 } else { 100 }),
        ),
    ) {
        let mut text = PText::from(base.as_str());
        let mut model = base.clone();
        let mut retained: Vec<(PText, String)> = Vec::new();

        for (i, (pos_frac, del_frac, insert)) in ops.into_iter().enumerate() {
            if i % 5 == 0 {
                retained.push((text.clone(), model.clone()));
            }
            let n = model.chars().count();
            let pos = ((pos_frac * n as f64) as usize).min(n);
            let max_del = n - pos;
            let del = ((del_frac * max_del as f64) as usize).min(max_del);

            model_splice(&mut model, pos, del, &insert);
            text = text.splice_owned(pos..pos + del, &insert);

            assert_matches_model(&text, &model);
        }
        for (rt, rm) in &retained {
            assert_matches_model(rt, rm);
        }
    }

    /// Degenerate repeated single-char inserts at the same moving caret —
    /// the sequential-typing pattern the whole milestone is motivated by —
    /// via the owned path, checked against `String::insert`.
    #[test]
    fn sequential_typing_matches_model(chars in prop::collection::vec(char_strategy(), 0..(if cfg!(miri) { 20 } else { 300 }))) {
        let mut text = PText::new();
        let mut model = String::new();
        for (i, ch) in chars.into_iter().enumerate() {
            let pos = i; // always type at the end
            text = text.splice_owned(pos..pos, &ch.to_string());
            model.push(ch);
        }
        assert_matches_model(&text, &model);
    }

    /// M14b (SPEC-M14-TEXT-PERSIST-EQ.md): the pruned dual walk's fallback
    /// path must preserve the existing chunk-boundary-independence
    /// property exactly. `t1` is one clean `PText::from` bulk build; `t2`
    /// is content-identical but assembled from independently-`from()`-built
    /// pieces joined by `concat` at proptest-chosen cut points — a
    /// deliberately different (and, since each piece is its own fresh
    /// allocation, entirely non-shared) leaf layout than `t1`'s. Equal
    /// content must compare equal regardless; a single-char difference
    /// introduced afterward must compare unequal.
    #[test]
    fn eq_ignores_chunk_boundaries(
        s in text_strategy(if cfg!(miri) { 200 } else { 6000 }),
        cuts in prop::collection::vec(0.0..1.0f64, 0..8),
    ) {
        let t1 = PText::from(s.as_str());

        let n = s.chars().count();
        let mut cut_points: Vec<usize> = cuts.iter().map(|f| ((f * n as f64) as usize).min(n)).collect();
        cut_points.push(n);
        cut_points.sort_unstable();
        cut_points.dedup();
        let mut t2 = PText::new();
        let mut prev = 0usize;
        for &cp in &cut_points {
            t2 = PText::concat(t2, PText::from(model_slice(&s, prev, cp).as_str()));
            prev = cp;
        }

        assert_eq!(t1.to_string(), t2.to_string(), "eq_ignores_chunk_boundaries: content setup mismatch");
        prop_assert!(t1 == t2, "content-equal PTexts with different (non-shared) chunk shapes must compare equal");
        prop_assert!(t2 == t1, "PartialEq must be symmetric here too");

        if n > 0 {
            let mid = n / 2;
            // '\u{2603}' (snowman) is outside `char_strategy`'s pool, so
            // this substitution is guaranteed to actually change content.
            let t3 = t2.splice(mid..mid + 1, "\u{2603}");
            prop_assert!(t1 != t3, "a single-char difference must compare unequal");
        }
    }
}

// ---------------------------------------------------------------------
// M14b (SPEC-M14-TEXT-PERSIST-EQ.md): shared-structure equality witness,
// adapted from M13's `try_eq_by` call-count test (SPEC-M13-EQWITH.md) —
// PText has no per-element predicate to count calls against, so this pins
// correctness only; `benches/text.rs`'s `text_eq/eq_shared` group is the
// quantitative witness that the pruning this exercises is actually cheap
// (BASELINE-M14-EQ.txt has the pre-M14b numbers it's measured against).
// ---------------------------------------------------------------------

#[test]
fn eq_shared_structure_witness_v1_v3() {
    let base = "0123456789".repeat(LEAF_MAX * 5); // several internal levels
    let v1 = PText::from(base.as_str());
    let start = v1.len_chars() / 2;
    // Round-trip edit: insert then remove the same span back out, so v3 is
    // content-equal to v1 but its root (and the whole edited spine) is a
    // fresh allocation — every subtree off that spine is still the exact
    // same shared allocation as v1's, via persistent `splice`'s structural
    // sharing (M14a: `clone()` + `splice_owned`'s copy-on-shared descent).
    let v2 = v1.splice(start..start, "ZZZ");
    let v3 = v2.splice(start..start + 3, "");
    v3.validate();
    assert_eq!(v3.to_string(), base, "round-trip edit must reproduce the original content");
    assert!(!PText::ptr_eq(&v1, &v3), "v3's root must differ from v1's (the edit spine was rebuilt), so the trivial root ptr_eq fast path cannot fire");
    assert!(v1 == v3, "content-equal, structure-sharing PTexts must compare equal via the pruned walk");
    assert!(v3 == v1, "PartialEq must be symmetric here too");
}

// ---------------------------------------------------------------------
// M9 (SPEC-M9-SPLITAT.md gate 2): dedicated differential fuzz test for
// `slice`/`splice` against a `String` model, run entirely against a SHARED
// tree (mirroring `examples/slice_probe.rs`'s `_shared` pattern) so every
// op is forced through `split_at`'s persistent copy path -- the exact path
// M9's `wrap_siblings`/`join_left`/`join_right` rewrite touches (on a
// uniquely-owned root, `splice`/`slice` would still be correct but
// wouldn't exercise the sharing-preserving `extract_as_ptext` code the fix
// changed). Deterministic: a fixed LCG seed, never OS/time-derived, so a
// failure always reproduces exactly.
// ---------------------------------------------------------------------

struct FuzzLcg(u64);
impl FuzzLcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes LCG constants (matches examples/slice_probe.rs).
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
}

/// Fixed, never OS/time-derived -- a failure always reproduces exactly.
const FUZZ_SEED: u64 = 0x5EED_C0DE_5EED_C0DE;

/// ASCII plus multibyte (Latin-1 supplement, a combining mark, CJK, and an
/// emoji outside the BMP) and both line-ending styles -- deliberately
/// echoes `char_strategy()` above rather than reusing it directly, since
/// this test drives its own deterministic LCG instead of proptest's RNG.
const FUZZ_CHARSET: &[char] = &['a', 'b', 'c', 'x', 'y', 'z', ' ', '\n', '\r', '\u{00e9}', '\u{0301}', '\u{4e2d}', '\u{1f600}'];

fn fuzz_random_string(lcg: &mut FuzzLcg, n_chars: usize) -> String {
    (0..n_chars).map(|_| FUZZ_CHARSET[lcg.below(FUZZ_CHARSET.len())]).collect()
}

fn model_slice(model: &str, start: usize, end: usize) -> String {
    model.chars().skip(start).take(end - start).collect()
}

/// At least 10k random `slice`/`splice` ops, checked against a `String` model,
/// across mixed doc sizes (empty, single char, exactly one leaf, just over
/// one leaf, and two comfortably multi-level sizes) and ASCII + multibyte
/// content -- ALL run against a tree kept alive by an external clone
/// (`_shared`) for that whole size's run of ops, so every `split_at` call
/// is forced through the shared/copy path (`node::is_unique` false)
/// instead of skating past it on a uniquely-owned root. `_shared` itself
/// is checked against the size's ORIGINAL content at the end of each
/// size's run -- structural sharing must copy-on-write, never let a
/// persistent edit leak into a still-live sibling handle, which is exactly
/// the property `split_at`'s node-sharing logic is responsible for.
#[test]
fn shared_tree_slice_splice_fuzz_matches_string_model() {
    let sizes: &[usize] = if cfg!(miri) {
        &[0, 1, 37] // Miri is orders of magnitude slower; keep this cheap.
    } else {
        &[0, 1, LEAF_MAX, LEAF_MAX + 1, LEAF_MAX * 3 + 17, LEAF_MAX * 12 + 5]
    };
    let ops_per_size: usize = if cfg!(miri) { 20 } else { 2000 };

    let mut lcg = FuzzLcg(FUZZ_SEED);
    let mut total_ops = 0usize;

    for &size in sizes {
        let original_model = fuzz_random_string(&mut lcg, size);
        let mut model = original_model.clone();
        let mut text = PText::from(model.as_str());
        // Held alive for this whole size's run of ops -- forces every op
        // below through the shared/copy path (see doc comment above).
        let _shared = text.clone();
        text.validate();

        // Explicit boundary coverage before the random loop: split points
        // at 0/len and full-range/empty-range slices, deterministic
        // regardless of what the LCG happens to roll.
        {
            let n = model.chars().count();
            let s_empty_start = text.slice(0..0);
            s_empty_start.validate();
            assert_eq!(s_empty_start.to_string(), "");
            let s_empty_end = text.slice(n..n);
            s_empty_end.validate();
            assert_eq!(s_empty_end.to_string(), "");
            let s_all = text.slice(0..n);
            s_all.validate();
            assert_eq!(s_all.to_string(), model);
        }

        for _ in 0..ops_per_size {
            total_ops += 1;
            let n = model.chars().count();
            if lcg.below(2) == 0 {
                // slice: check against the model, don't mutate either.
                let start = lcg.below(n + 1);
                let end = start + lcg.below(n + 1 - start);
                let got = text.slice(start..end);
                got.validate();
                assert_eq!(got.to_string(), model_slice(&model, start, end), "slice({start}..{end}) mismatch, size={size}");
            } else {
                // splice: mutate both the model and `text` (persistent
                // path -- `text` remains a live handle, `_shared` doesn't
                // move).
                let start = lcg.below(n + 1);
                let del = lcg.below(n + 1 - start);
                let end = start + del;
                let insert_len = lcg.below(6);
                let insert = fuzz_random_string(&mut lcg, insert_len);
                let bstart = model.char_indices().nth(start).map(|(b, _)| b).unwrap_or(model.len());
                let bend = model.char_indices().nth(end).map(|(b, _)| b).unwrap_or(model.len());
                model.replace_range(bstart..bend, &insert);
                text = text.splice(start..end, &insert);
                text.validate();
                assert_eq!(text.to_string(), model, "splice({start}..{end}, {insert:?}) mismatch, size={size}");
            }
        }

        // Final cross-check: `text` matches every accumulated edit, and
        // the untouched `_shared` clone still matches this size's
        // ORIGINAL content (aliasing/sharing safety -- see doc comment).
        text.validate();
        assert_eq!(text.to_string(), model);
        _shared.validate();
        assert_eq!(_shared.to_string(), original_model);
    }

    assert!(total_ops >= if cfg!(miri) { 1 } else { 10_000 }, "fuzz op count too low: {total_ops}");
}

// ---------------------------------------------------------------------
// M9.1 (SPEC-M9-SPLITAT.md addendum): read_range/range_to_string against
// the String model, under the same shared-tree discipline as the M9 fuzz
// above -- a second, external clone (`_shared`) held alive for the whole
// run so every read descends a tree reachable through more than one
// handle (read_range never mutates, so this mostly guards against the
// descent accidentally assuming/requiring uniqueness anywhere, which it
// must not: it only ever borrows).
// ---------------------------------------------------------------------

#[test]
fn shared_tree_read_range_fuzz_matches_string_model() {
    let sizes: &[usize] = if cfg!(miri) {
        &[0, 1, 37] // Miri is orders of magnitude slower; keep this cheap.
    } else {
        &[0, 1, LEAF_MAX, LEAF_MAX + 1, LEAF_MAX * 3 + 17, LEAF_MAX * 12 + 5]
    };
    let ops_per_size: usize = if cfg!(miri) { 20 } else { 2000 };

    let mut lcg = FuzzLcg(FUZZ_SEED ^ 0xA11C_E5EE_D000_0001);
    let mut total_ops = 0usize;

    for &size in sizes {
        let model = fuzz_random_string(&mut lcg, size);
        let text = PText::from(model.as_str());
        // Held alive for this whole size's run -- forces every read below
        // through a shared root (see doc comment above).
        let _shared = text.clone();
        text.validate();

        let n = model.chars().count();
        // Explicit boundary coverage, deterministic regardless of the LCG.
        for &(start, end) in &[(0, 0), (0, n), (n, n)] {
            assert_eq!(text.range_to_string(start..end), model_slice(&model, start, end), "range_to_string({start}..{end}) mismatch, size={size}");
            let mut buf = String::new();
            text.read_range(start..end, &mut buf);
            assert_eq!(buf, model_slice(&model, start, end), "read_range({start}..{end}) mismatch, size={size}");
        }

        for _ in 0..ops_per_size {
            total_ops += 1;
            let start = lcg.below(n + 1);
            let end = start + lcg.below(n + 1 - start);
            let expected = model_slice(&model, start, end);

            let got = text.range_to_string(start..end);
            assert_eq!(got, expected, "range_to_string({start}..{end}) mismatch, size={size}");

            // Also cross-check against slice(r) flattened, and the append
            // (not just fresh-String) form of read_range.
            let via_slice = text.slice(start..end).to_string();
            assert_eq!(via_slice, expected, "slice({start}..{end}) flattened mismatch, size={size}");

            let mut buf = String::from("X");
            text.read_range(start..end, &mut buf);
            assert_eq!(buf, format!("X{expected}"), "read_range({start}..{end}) append mismatch, size={size}");
        }

        // `text` and the untouched `_shared` clone must both still match
        // the original content -- read_range/range_to_string never mutate.
        text.validate();
        assert_eq!(text.to_string(), model);
        _shared.validate();
        assert_eq!(_shared.to_string(), model);
    }

    assert!(total_ops >= if cfg!(miri) { 1 } else { 10_000 }, "fuzz op count too low: {total_ops}");
}
