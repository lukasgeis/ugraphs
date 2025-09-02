/*!
# Generalized Maps

This module provides abstractions over `Map` data structures, allowing algorithms
to choose the most efficient implementation based on context.

Examples:
- Sparse maps -> `HashMap`
- Dense indexed maps -> `[Option<T>]`

The module includes:
- [`Map<K, V>`]: trait for generic map-like operations
- Concrete implementations: `HashMap`, `[Option<T>]`.
*/

use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
};

use num::ToPrimitive;

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

/// `Vec<Option<T>>` usable as `Map`
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
