//! # General Sets / Maps
//!
//! In some algorithms, it might be beneficial to use a different kind of Set/Map depending on
//! additional information, the user knows (for example use-cases where the values of elements are bounded) but the compiler does not.

use std::{
    collections::{HashMap, HashSet, hash_set::Iter},
    hash::{BuildHasher, Hash},
    iter::{Cloned, Copied, Empty},
    marker::PhantomData,
};

use itertools::Itertools;
use num::ToPrimitive;
use stream_bitset::{
    PrimIndex,
    bitset::BitSetImpl,
    prelude::{BitmaskSliceStream, BitmaskStreamConsumer, BitmaskStreamToIndices, ToBitmaskStream},
};

use crate::node::*;

/// A minimalist generalization over basic Set-Functionality
pub trait Set<T> {
    /// Inserts an element into the Set
    /// Returns *true* if the element was contained in the Set before
    fn insert(&mut self, value: T) -> bool;

    /// Inserts multiple elements into the Set
    fn insert_multiple<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for value in iter {
            self.insert(value);
        }
    }

    /// Removes an element from the Set
    /// Returns *true* if the element was contained in the Set before
    fn remove(&mut self, value: &T) -> bool;

    /// Removes multiple elements from the Set
    fn remove_multiple<'a, I>(&mut self, iter: I)
    where
        T: 'a,
        I: IntoIterator<Item = &'a T> + 'a,
    {
        for value in iter {
            self.remove(value);
        }
    }

    type SetIter<'a>: Iterator<Item = T>
    where
        Self: 'a,
        T: Clone;

    /// Iterates over all elements in the Set
    /// Depending on the underlying datastructure, this might clone each value
    fn iter(&self) -> Self::SetIter<'_>
    where
        T: Clone;

    /// Returns *true* if the element is contained in the Set
    fn contains(&self, value: &T) -> bool;

    /// Clears the Set
    fn clear(&mut self);

    /// Returns the number of elements in the Set
    fn len(&self) -> usize;

    /// Returns *true* if the Set is empty
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

/// Custom Set-Datastructure that allows for constant additions/removals/queries as well as an
/// output-sensitive iterator over its elements. Similar to `BitSetImpl<I>`, values are confined to
/// a predefined range `0..n`
pub struct NodeSet {
    data: Vec<Node>,
    positions: Vec<Option<OptionalNode>>,
}

impl NodeSet {
    pub fn new(n: NumNodes) -> Self {
        Self {
            data: Vec::new(),
            positions: vec![None; n as usize],
        }
    }

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

#[derive(Debug, Copy, Clone, Default)]
pub struct EmptySet<T>(PhantomData<T>);

impl<T> Set<T> for EmptySet<T> {
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

/// A minimalist generalization over basic Map-Functionality
pub trait Map<K, V> {
    /// Inserts an (key,value)-pair into the map
    /// If the key was present before, return the previous element, otherwise `None`
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Removes a key from the map and returns the value if it existed.
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Returns a reference to the value corresponding to the key.
    fn get(&self, key: &K) -> Option<&V>;

    /// Clears all elements in the Map
    fn clear(&mut self);

    /// Returns the number of elements in the Set
    fn len(&self) -> usize;

    /// Returns *true* if the Map is empty
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
