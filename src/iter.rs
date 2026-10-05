//! Borrowing iterators over [`crate::PersistentHashMap`].
//!
//! CHAMP's canonical form guarantees a node's data run always precedes its
//! children in memory-address-independent, hash-determined order, so a
//! simple stack-based cursor gives a deterministic, stable (across
//! identical contents) iteration order: emit a node's data entries, then
//! descend into its children left-to-right.

use crate::node::NodePtr;

/// Max tree depth a [`Frame`] stack ever needs to hold: root at depth 0
/// through depth-6 bitmap nodes (7 levels, since a 32-bit hash in 5-bit
/// chunks bottoms out at depth 6), plus one more level for a collision node
/// (which can appear below any bitmap node, including at depth 6).
const MAX_DEPTH: usize = 8;

/// `slice::Iter` is cheap to hold in a fixed-size array (it's just a pair of
/// pointers), which is what lets [`Iter`] use a fixed
/// `[Option<Frame>; MAX_DEPTH]` array instead of a `Vec`, with zero heap
/// allocation and zero `unsafe`. The array is seeded with an inline `const`
/// repeat expression (`[const { None }; MAX_DEPTH]`) rather than
/// `[None; MAX_DEPTH]`, since the latter would require `Frame: Copy`, which
/// in turn would (incorrectly, for a type that never actually stores a `K`
/// or `V` by value) require `K: Copy, V: Copy` from `#[derive(Copy)]`'s
/// blanket per-type-parameter bounds.
struct Frame<'a, K, V> {
    data: std::slice::Iter<'a, (K, V)>,
    nodes: std::slice::Iter<'a, NodePtr<K, V>>,
}

/// Borrowing iterator over `(&K, &V)` pairs, in hash order. Allocation-free:
/// the traversal stack is a fixed-size inline array (max CHAMP tree depth is
/// bounded, see [`MAX_DEPTH`]), not a heap-allocated `Vec`.
pub struct Iter<'a, K, V> {
    stack: [Option<Frame<'a, K, V>>; MAX_DEPTH],
    len: u8,
}

impl<'a, K, V> Iter<'a, K, V> {
    pub(crate) fn new(root: Option<&'a NodePtr<K, V>>) -> Self {
        let mut stack = [const { None }; MAX_DEPTH];
        let len = if let Some(root) = root {
            stack[0] = Some(Frame {
                data: root.data_slice().iter(),
                nodes: root.node_slice().iter(),
            });
            1
        } else {
            0
        };
        Iter { stack, len }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.len > 0 {
            let top = self.len as usize - 1;
            let frame = self.stack[top].as_mut().expect("active frame slot is always Some");
            if let Some((k, v)) = frame.data.next() {
                return Some((k, v));
            }
            match frame.nodes.next() {
                Some(child) => {
                    debug_assert!(
                        (self.len as usize) < MAX_DEPTH,
                        "iterator depth exceeded MAX_DEPTH cap"
                    );
                    self.stack[self.len as usize] = Some(Frame {
                        data: child.data_slice().iter(),
                        nodes: child.node_slice().iter(),
                    });
                    self.len += 1;
                }
                None => {
                    self.stack[top] = None;
                    self.len -= 1;
                }
            }
        }
        None
    }
}

impl<K, V> std::iter::FusedIterator for Iter<'_, K, V> {}

/// Borrowing iterator over keys, in hash order.
pub struct Keys<'a, K, V>(pub(crate) Iter<'a, K, V>);

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
}

/// Borrowing iterator over values, in hash order.
pub struct Values<'a, K, V>(pub(crate) Iter<'a, K, V>);

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
}

// ---------------------------------------------------------------------
// Consuming iterator: same fixed-depth-array shape as `Iter`, but each
// frame owns a *handle* to its node (the moved-in root for frame 0, a
// `clone_shallow` for every frame pushed below it) instead of borrowing a
// `slice::Iter` tied to an external lifetime — `IntoIter` has no external
// lifetime to borrow from, it owns the whole tree. `next()` clones each
// `(K, V)` pair out through a transient borrow of the frame's own node
// (never stored across calls, so this stays a plain, non-self-referential
// struct). Every frame's owned node handle is released with exactly one
// `drop_node()` call: either when `next()` finishes that frame during
// normal traversal, or — for a frame still `Some` when the whole iterator
// is dropped early (abandoned mid-iteration) — from `IntoIter`'s `Drop`
// impl. `NodePtr` implements neither `Clone` nor `Drop` itself (see
// node.rs's module docs), so both of those release points matter: without
// the `Drop` impl below, an `IntoIter` dropped before exhaustion would
// leak every node still referenced by an active frame.
// ---------------------------------------------------------------------

struct IntoFrame<K, V> {
    node: NodePtr<K, V>,
    data_idx: usize,
    node_idx: usize,
}

/// Consuming iterator over `(K, V)` pairs, in hash order. Yields clones of
/// each entry (see [`crate::PersistentHashMap`]'s `IntoIterator` impl docs
/// for why: a move-out-when-unique optimization is possible future work,
/// not attempted here). Allocation-free in the same sense [`Iter`] is: the
/// traversal stack is a fixed-size inline array, not a `Vec`.
pub struct IntoIter<K, V> {
    stack: [Option<IntoFrame<K, V>>; MAX_DEPTH],
    len: u8,
}

impl<K, V> IntoIter<K, V> {
    pub(crate) fn new(root: Option<NodePtr<K, V>>) -> Self {
        let mut stack = [const { None }; MAX_DEPTH];
        let len = if let Some(root) = root {
            stack[0] = Some(IntoFrame {
                node: root,
                data_idx: 0,
                node_idx: 0,
            });
            1
        } else {
            0
        };
        IntoIter { stack, len }
    }
}

impl<K: Clone, V: Clone> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.len > 0 {
            let top = self.len as usize - 1;
            let frame = self.stack[top].as_mut().expect("active frame slot is always Some");
            if frame.data_idx < frame.node.n_data() {
                let (k, v) = &frame.node.data_slice()[frame.data_idx];
                let item = (k.clone(), v.clone());
                frame.data_idx += 1;
                return Some(item);
            }
            if frame.node_idx < frame.node.n_nodes() {
                let child = frame.node.node_slice()[frame.node_idx].clone_shallow();
                frame.node_idx += 1;
                debug_assert!(
                    (self.len as usize) < MAX_DEPTH,
                    "iterator depth exceeded MAX_DEPTH cap"
                );
                self.stack[self.len as usize] = Some(IntoFrame {
                    node: child,
                    data_idx: 0,
                    node_idx: 0,
                });
                self.len += 1;
            } else {
                // This frame's data and children are both exhausted: its
                // owned node handle is released here (the one and only
                // `drop_node()` call for it on the normal-completion path).
                let finished = self.stack[top].take().expect("frame just matched Some above");
                finished.node.drop_node();
                self.len -= 1;
            }
        }
        None
    }
}

impl<K, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        // Release any frames left on the stack from an abandoned (not
        // fully exhausted) iteration — on a normal run to completion,
        // `next()` above has already released every frame and this is a
        // no-op.
        for slot in &mut self.stack[..self.len as usize] {
            if let Some(frame) = slot.take() {
                frame.node.drop_node();
            }
        }
    }
}

impl<K: Clone, V: Clone> std::iter::FusedIterator for IntoIter<K, V> {}

/// Consuming iterator over keys, in hash order — the set's `IntoIterator`
/// item type. Mirrors [`Keys`], just over [`IntoIter`] instead of [`Iter`].
pub struct IntoKeys<K>(pub(crate) IntoIter<K, ()>);

impl<K: Clone> Iterator for IntoKeys<K> {
    type Item = K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
}

impl<K: Clone> std::iter::FusedIterator for IntoKeys<K> {}
