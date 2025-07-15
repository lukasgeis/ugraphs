//! # General Sets / Maps
//!
//! In some algorithms, it might be beneficial to use a different kind of Set/Map depending on
//! additional information, the user knows (for example use-cases where the values of elements are bounded) but the compiler does not.

use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasher, Hash},
};

use num::ToPrimitive;
use stream_bitset::{PrimIndex, bitset::BitSetImpl};

/// A minimalist generalization over basic Set-Functionality
pub trait Set<T> {
    /// Inserts an element into the Set
    /// Returns *true* if the element was contained in the Set before
    fn insert(&mut self, value: T) -> bool;

    /// Inserts multiple elements into the Set
    fn insert_multiple<I: Iterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }

    /// Removes an element from the Set
    /// Returns *true* if the element was contained in the Set before
    fn remove(&mut self, value: &T) -> bool;

    /// Removes multiple elements from the Set
    fn remove_multiple<'a, I: Iterator<Item = &'a T> + 'a>(&mut self, iter: I)
    where
        T: 'a,
    {
        for value in iter {
            self.remove(value);
        }
    }

    /// Returns *true* if the element is contained in the Set
    fn contains(&self, value: &T) -> bool;

    /// Clears the Set
    fn clear(&mut self);
}

impl<T: Eq + Hash, S: BuildHasher> Set<T> for HashSet<T, S> {
    fn insert(&mut self, value: T) -> bool {
        HashSet::insert(self, value)
    }

    fn remove(&mut self, value: &T) -> bool {
        HashSet::remove(self, value)
    }

    fn contains(&self, value: &T) -> bool {
        HashSet::contains(self, value)
    }

    fn clear(&mut self) {
        HashSet::clear(self);
    }
}

impl<I: PrimIndex> Set<I> for BitSetImpl<I> {
    fn insert(&mut self, value: I) -> bool {
        self.set_bit(value)
    }

    fn remove(&mut self, value: &I) -> bool {
        self.clear_bit(*value)
    }

    fn contains(&self, value: &I) -> bool {
        self.get_bit(*value)
    }

    fn clear(&mut self) {
        self.clear_all();
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
}

impl<K: Eq + Hash, V, S: BuildHasher> Map<K, V> for HashMap<K, V, S> {
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
}

impl<I: ToPrimitive, T> Map<I, T> for [Option<T>] {
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
}
