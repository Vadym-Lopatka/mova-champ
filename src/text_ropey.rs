//! `RopeyText`: the `ropey` crate wearing [`crate::PText`]'s exact public
//! API, so the two engines can be compared head-to-head on the shared model
//! tests (`tests/text_model.rs` / `tests/text_ropey_model.rs`) and criterion
//! benches (`benches/text.rs`) without either test suite knowing which rope
//! it's driving.
//!
//! This is a spike, gated behind the `spike-ropey` feature and never
//! compiled into a default build: `ropey::Rope` is already a mature,
//! mutable-in-place rope, so unlike `PText` there's no persistent-tree
//! algorithm to build here — `RopeyText` is a thin `Arc<Rope>` wrapper that
//! gets `PText`'s O(1)-clone/`ptr_eq`/owned-fast-path *shape* almost for
//! free:
//!
//! - `Clone` is `Arc::clone` (O(1), matching `PText`'s O(1) refcount bump).
//! - `ptr_eq` is `Arc::ptr_eq`.
//! - The `_owned` fast path is `Arc::make_mut`: when the caller holds the
//!   only reference, this mutates the `Rope` in place; when the `Rope` is
//!   shared, `Arc::make_mut` clones it first — and `Rope::clone` is itself
//!   O(1)-ish (ropey's chunks are `Rc`-shared internally), so the
//!   fallback is one cheap clone, not a deep copy.
//!
//! See each method below for how it maps onto `ropey::Rope`'s API, and a
//! note wherever `PText`'s documented semantics and `ropey`'s don't quite
//! line up (line-ending conventions, `chunks()`'s empty-text case, etc. —
//! also covered in `tests/text_ropey_model.rs`'s doc comments).

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

use ropey::Rope;

/// `ropey::Rope` behind `PText`'s public API. See the module docs.
pub struct RopeyText(Arc<Rope>);

impl Clone for RopeyText {
    fn clone(&self) -> Self {
        RopeyText(Arc::clone(&self.0))
    }
}

impl Default for RopeyText {
    fn default() -> Self {
        RopeyText::new()
    }
}

impl RopeyText {
    /// An empty text.
    pub fn new() -> RopeyText {
        RopeyText(Arc::new(Rope::new()))
    }

    pub fn len_bytes(&self) -> usize {
        self.0.len_bytes()
    }

    pub fn len_chars(&self) -> usize {
        self.0.len_chars()
    }

    /// Number of lines. `PText` counts only `'\n'` as a line separator
    /// (`k` newlines -> `k + 1` lines); `ropey`'s default build (this
    /// crate's dev-dependency doesn't disable its `unicode_lines`/
    /// `cr_lines` default features, and — because Cargo unifies feature
    /// flags for a single resolved package version across every
    /// dependency edge that pulls it in — this crate's own `spike-ropey`
    /// edge can't unilaterally turn them off either) also treats a lone
    /// `'\r'` (not part of a `"\r\n"` pair) as a line break, plus a
    /// handful of Unicode line/paragraph separators `PText` doesn't
    /// recognize at all. `"\r\n"` itself counts as exactly one break under
    /// both conventions, so plain LF and CRLF text (the overwhelming
    /// common case) agree between the two engines; only text containing a
    /// standalone `'\r'` (or one of the Unicode separators) can disagree.
    /// See `tests/text_ropey_model.rs`'s model helper for how the ported
    /// property tests account for this.
    pub fn len_lines(&self) -> usize {
        self.0.len_lines()
    }

    pub fn is_empty(&self) -> bool {
        self.0.len_bytes() == 0
    }

    pub fn ptr_eq(a: &RopeyText, b: &RopeyText) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    /// Byte offset of char index `idx` (`idx == len_chars()` is valid,
    /// returning `len_bytes()`).
    pub fn byte_of_char(&self, idx: usize) -> usize {
        debug_assert!(idx <= self.len_chars(), "byte_of_char: index out of bounds");
        self.0.char_to_byte(idx)
    }

    /// The `char` at char index `idx` (`idx < len_chars()`).
    pub fn char_at(&self, idx: usize) -> char {
        self.0.char(idx)
    }

    /// Char index of the 0-indexed line containing char index `idx`. See
    /// [`Self::len_lines`]'s doc comment for the line-convention caveat.
    pub fn char_to_line(&self, idx: usize) -> usize {
        self.0.char_to_line(idx)
    }

    /// Char index of the start of 0-indexed `line` (`line < len_lines()`).
    /// See [`Self::len_lines`]'s doc comment for the line-convention
    /// caveat.
    pub fn line_to_char(&self, line: usize) -> usize {
        debug_assert!(line < self.len_lines(), "line_to_char: line out of bounds");
        self.0.line_to_char(line)
    }

    /// Delete `char_range` and insert `text` in its place. Persistent:
    /// `self` remains valid and unchanged — clones the inner `Rope` (cheap:
    /// ropey's chunks are internally `Rc`-shared) before mutating the
    /// clone.
    pub fn splice(&self, char_range: Range<usize>, text: &str) -> RopeyText {
        let mut r = (*self.0).clone();
        if char_range.start != char_range.end {
            r.remove(char_range.start..char_range.end);
        }
        r.insert(char_range.start, text);
        RopeyText(Arc::new(r))
    }

    /// Like [`Self::splice`], but consumes `self`: `Arc::make_mut` gives
    /// in-place mutation when `self` is the sole owner of the inner
    /// `Rope`, falling back to one clone when it's shared — the same
    /// "mutate in place when unique, copy when shared" shape as `PText`'s
    /// owned path, just resolved at the whole-rope granularity instead of
    /// `PText`'s node-by-node descent.
    pub fn splice_owned(self, char_range: Range<usize>, text: &str) -> RopeyText {
        let RopeyText(mut arc) = self;
        {
            let r = Arc::make_mut(&mut arc);
            if char_range.start != char_range.end {
                r.remove(char_range.start..char_range.end);
            }
            r.insert(char_range.start, text);
        }
        RopeyText(arc)
    }

    /// Structural slice over `char_range`.
    pub fn slice(&self, char_range: Range<usize>) -> RopeyText {
        RopeyText(Arc::new(Rope::from(self.0.slice(char_range))))
    }

    /// Append `char_range`'s content directly to `out`. Same bounds
    /// contract as [`Self::slice`]: `char_range` must be valid (`start <=
    /// end <= len_chars()`), debug-checked here, not enforced in release.
    pub fn read_range(&self, char_range: Range<usize>, out: &mut String) {
        debug_assert!(char_range.start <= char_range.end && char_range.end <= self.len_chars(), "read_range: char_range out of bounds");
        if char_range.start == char_range.end {
            return;
        }
        for chunk in self.0.slice(char_range).chunks() {
            out.push_str(chunk);
        }
    }

    /// Convenience wrapper over [`Self::read_range`]: allocates a fresh
    /// `String` sized via `char_to_byte`-derived length up front, then
    /// fills it.
    pub fn range_to_string(&self, char_range: Range<usize>) -> String {
        debug_assert!(char_range.start <= char_range.end && char_range.end <= self.len_chars(), "range_to_string: char_range out of bounds");
        let cap = if char_range.start == char_range.end {
            0
        } else {
            self.0.char_to_byte(char_range.end) - self.0.char_to_byte(char_range.start)
        };
        let mut out = String::with_capacity(cap);
        self.read_range(char_range, &mut out);
        out
    }

    /// Join two ropes into one.
    pub fn concat(a: RopeyText, b: RopeyText) -> RopeyText {
        let RopeyText(mut arc) = a;
        {
            let r = Arc::make_mut(&mut arc);
            r.append((*b.0).clone());
        }
        RopeyText(arc)
    }

    /// Walk over the text's chunks as `&str`. `PText::chunks` returns its
    /// own `Chunks<'_>` type (a hand-rolled zero-copy tree walk); ropey
    /// already has an equivalent chunk iterator (`ropey::iter::Chunks`),
    /// so this spike returns that directly rather than wrapping it in a
    /// same-named local type. One behavioral difference: an empty
    /// `PText` still yields exactly one (empty) chunk (it's always at
    /// least one leaf node), while an empty `Rope` yields *zero* chunks —
    /// see `tests/text_ropey_model.rs`'s `boundary_empty_text` for where
    /// this is relaxed to a content-equality check instead.
    pub fn chunks(&self) -> ropey::iter::Chunks<'_> {
        self.0.chunks()
    }

    /// Iterator over `char`s, built on [`Self::chunks`] via ropey's own
    /// char iterator (same declared return type as `PText::chars`).
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        self.0.chars()
    }

    /// No-op: `ropey::Rope` maintains its own internal invariants
    /// privately (there's no analogous "walk the tree and assert the
    /// canonical-form invariants" hook to call into, and this spike isn't
    /// reimplementing one). Kept only so the `tests/text_model.rs` suite,
    /// copied into `tests/text_ropey_model.rs` with `PText` renamed to
    /// `RopeyText`, compiles and runs unchanged — every `.validate()` call
    /// in the copied suite becomes a harmless no-op.
    pub fn validate(&self) {}
}

impl fmt::Debug for RopeyText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors `PText`'s `Debug` impl exactly (same struct name pattern,
        // same three fields).
        f.debug_struct("RopeyText").field("len_bytes", &self.len_bytes()).field("len_chars", &self.len_chars()).field("len_lines", &self.len_lines()).finish()
    }
}

impl From<&str> for RopeyText {
    fn from(s: &str) -> RopeyText {
        RopeyText(Arc::new(Rope::from(s)))
    }
}

impl From<String> for RopeyText {
    fn from(s: String) -> RopeyText {
        RopeyText(Arc::new(Rope::from(s)))
    }
}

impl fmt::Display for RopeyText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for RopeyText {
    fn eq(&self, other: &RopeyText) -> bool {
        if RopeyText::ptr_eq(self, other) {
            return true;
        }
        // `Rope`'s own `PartialEq` is already content-based and
        // chunk-boundary-independent (it compares via `slice(..)`), the
        // same contract `PText`'s hand-rolled chunk-by-chunk comparison
        // gives.
        *self.0 == *other.0
    }
}

impl Eq for RopeyText {}

/// Hashes the exact same byte stream `<str as Hash>::hash` produces,
/// copying `PText`'s exact contract byte-for-byte (see `PText`'s `Hash`
/// impl doc comment) rather than delegating to `Rope`'s own `Hash` impl:
/// ropey's `Hash` deliberately buffers content into fixed 256-byte blocks
/// before each `Hasher::write` call (so its hash is stable across
/// different chunkings of the *same rope type*, but is not the plain
/// "bytes then one `0xff` terminator" stream `<str as Hash>` uses) — using
/// it here would break the cross-type "a `RopeyText` and the equivalent
/// `&str` hash equal under the same `Hasher`" property this mirrors from
/// `PText`.
impl Hash for RopeyText {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for chunk in self.0.chunks() {
            state.write(chunk.as_bytes());
        }
        state.write_u8(0xff);
    }
}
