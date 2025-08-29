/*!
# Generalized Sets and Maps

This module provides abstractions over `Set` and `Map` data structures, allowing algorithms
to choose the most efficient implementation based on context.

Examples:
- Sparse sets -> `HashSet`
- Dense sets -> `BitSetImpl`
- Node-specific sets -> `NodeSet`

The module includes:
- [`Set<T>`]: trait for generic set-like operations
- [`Map<K, V>`]: trait for generic map-like operations
- Concrete implementations: `HashSet`, `BitSetImpl`, `NodeSet`, `EmptySet`, `HashMap`, `[Option<T>]`.
*/

use std::{
    collections::{HashMap, HashSet, hash_set::Iter},
    hash::{BuildHasher, Hash},
    iter::{Cloned, Copied, Empty},
};

use itertools::Itertools;
use num::ToPrimitive;
use stream_bitset::{
    PrimIndex,
    bitset::BitSetImpl,
    prelude::{BitmaskSliceStream, BitmaskStreamConsumer, BitmaskStreamToIndices, ToBitmaskStream},
};

use crate::node::*;

/// Minimalist trait for a set-like collection.
///
/// Supports insertion, removal, membership queries, iteration, and bulk operations.
pub trait Set<T> {
    /// Inserts `value` into the set.
    /// Returns `true` if the element was already present.
    fn insert(&mut self, value: T) -> bool;

    /// Inserts multiple elements from an iterator.
    fn insert_multiple<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for value in iter {
            self.insert(value);
        }
    }

    /// Removes `value` from the set.
    /// Returns `true` if the element was present.
    fn remove(&mut self, value: &T) -> bool;

    /// Removes multiple elements from the set.
    fn remove_multiple<'a, I>(&mut self, iter: I)
    where
        T: 'a,
        I: IntoIterator<Item = &'a T> + 'a,
    {
        for value in iter {
            self.remove(value);
        }
    }

    /// Iterator over elements in set.
    ///
    /// Returned by [`Set::iter`].
    type SetIter<'a>: Iterator<Item = T>
    where
        Self: 'a,
        T: Clone;

    /// Returns an iterator over all elements in the set.
    /// May clone elements depending on the underlying data structure.
    fn iter(&self) -> Self::SetIter<'_>
    where
        T: Clone;

    /// Returns `true` if the set contains `value`.
    fn contains(&self, value: &T) -> bool;

    /// Clears all elements from the set.
    fn clear(&mut self);

    /// Returns the number of elements in the set.
    fn len(&self) -> usize;

    /// Returns `true` if the set is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, S> Set<T> for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn insert(&mut self, value: T) -> bool {
        HashSet::insert(self, value)
    }

    fn remove(&mut self, value: &T) -> bool {
        HashSet::remove(self, value)
    }

    type SetIter<'a>
        = Cloned<Iter<'a, T>>
    where
        Self: 'a,
        T: Clone;

    fn iter(&self) -> Self::SetIter<'_>
    where
        T: Clone,
    {
        HashSet::iter(self).cloned()
    }

    fn contains(&self, value: &T) -> bool {
        HashSet::contains(self, value)
    }

    fn clear(&mut self) {
        HashSet::clear(self);
    }

    fn len(&self) -> usize {
        HashSet::len(self)
    }
}

impl<I> Set<I> for BitSetImpl<I>
where
    I: PrimIndex,
{
    fn insert(&mut self, value: I) -> bool {
        self.set_bit(value)
    }

    fn remove(&mut self, value: &I) -> bool {
        self.clear_bit(*value)
    }

    type SetIter<'a>
        = BitmaskStreamToIndices<BitmaskSliceStream<'a>, I, true>
    where
        Self: 'a,
        I: Clone;

    fn iter(&self) -> Self::SetIter<'_> {
        self.bitmask_stream().iter_set_bits()
    }

    fn contains(&self, value: &I) -> bool {
        self.get_bit(*value)
    }

    fn clear(&mut self) {
        self.clear_all();
    }

    fn len(&self) -> usize {
        self.cardinality().to_usize().unwrap()
    }
}

/// A set of nodes (0..n) supporting fast insertion, removal, and iteration.
pub struct NodeSet {
    data: Vec<Node>,
    positions: Vec<Option<OptionalNode>>,
}

impl NodeSet {
    /// Creates an empty node-set of size `n`
    pub fn new(n: NumNodes) -> Self {
        Self {
            data: Vec::new(),
            positions: vec![None; n as usize],
        }
    }

    /// Creates a full node-set of size `n`.
    /// Elements are stored in increasing order.
    pub fn new_with_all(n: NumNodes) -> Self {
        Self {
            data: (0..(n as Node)).collect_vec(),
            positions: (0..n).map(OptionalNode::new).collect_vec(),
        }
    }
}

impl Set<Node> for NodeSet {
    fn insert(&mut self, value: Node) -> bool {
        let index = value.to_usize().unwrap();
        if self.positions[index].is_some() {
            return true;
        }

        self.positions[index] = OptionalNode::new(self.data.len() as Node);
        self.data.push(value);

        false
    }

    fn remove(&mut self, value: &Node) -> bool {
        let index = value.to_usize().unwrap();
        let pos = match self.positions[index] {
            Some(pos) => pos.get() as usize,
            None => return false,
        };

        self.data.swap_remove(pos);
        if pos < self.data.len() {
            self.positions[self.data[pos] as usize] = self.positions[index];
        }

        self.positions[index] = None;

        true
    }

    type SetIter<'a>
        = Copied<std::slice::Iter<'a, Node>>
    where
        Self: 'a,
        Node: Clone;

    fn iter(&self) -> Self::SetIter<'_> {
        self.data.iter().copied()
    }

    fn contains(&self, value: &Node) -> bool {
        self.positions[*value as usize].is_some()
    }

    fn clear(&mut self) {
        self.data.clear();
        self.positions.iter_mut().for_each(|p| *p = None);
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// A "dummy" empty set. Cannot store any elements.
#[derive(Debug, Copy, Clone, Default)]
pub struct EmptySet;

impl<T> Set<T> for EmptySet {
    fn insert(&mut self, _value: T) -> bool {
        unimplemented!("You can't insert something into the EmptySet!");
    }

    fn remove(&mut self, _value: &T) -> bool {
        unimplemented!("You can't remove something from the EmptySet!")
    }

    type SetIter<'a>
        = Empty<T>
    where
        Self: 'a,
        T: Clone;

    fn iter(&self) -> Self::SetIter<'_>
    where
        T: Clone,
    {
        std::iter::empty()
    }

    fn contains(&self, _value: &T) -> bool {
        false
    }

    fn clear(&mut self) {}

    fn len(&self) -> usize {
        0
    }
}

/// Minimalist trait for map-like collections.
///
/// Supports insertion, removal, lookup, clearing, and size queries.
pub trait Map<K, V> {
    /// Inserts an `(key, value)` pair into the map.
    /// If the key was present before, returns the previous value, otherwise returns `None`.
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Removes a key from the map and returns the associated value if it existed.
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Returns a reference to the value corresponding to the given key, or `None` if the key is not present.
    fn get(&self, key: &K) -> Option<&V>;

    /// Clears all elements from the map.
    fn clear(&mut self);

    /// Returns the number of elements currently stored in the map.
    fn len(&self) -> usize;

    /// Returns `true` if the map is empty. Default implementation uses `len()`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V, S> Map<K, V> for HashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        HashMap::insert(self, key, value)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        HashMap::remove(self, key)
    }

    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }

    fn clear(&mut self) {
        HashMap::clear(self)
    }

    fn len(&self) -> usize {
        HashMap::len(self)
    }
}

impl<I, T> Map<I, T> for [Option<T>]
where
    I: ToPrimitive,
{
    fn insert(&mut self, key: I, value: T) -> Option<T> {
        let key = key.to_usize().unwrap();
        self[key].replace(value)
    }

    fn remove(&mut self, key: &I) -> Option<T> {
        let key = key.to_usize().unwrap();
        self[key].take()
    }

    fn get(&self, key: &I) -> Option<&T> {
        let key = key.to_usize().unwrap();
        self[key].as_ref()
    }

    fn clear(&mut self) {
        self.iter_mut().for_each(|x| *x = None);
    }

    fn len(&self) -> usize {
        self.len()
    }
}
