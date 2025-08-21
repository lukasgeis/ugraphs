use std::{
    collections::{HashMap, HashSet},
    hash::RandomState,
};

use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use num::{One, Zero};

mod geometric;
mod multi_traits;
mod node_mapper;
mod set_map;
mod sliced_buffer;

pub use geometric::*;
pub use multi_traits::*;
pub use node_mapper::*;
pub use set_map::*;
pub use sliced_buffer::*;

use stream_bitset::{PrimIndex, bitset::BitSetImpl};

/// Helper trait for probalities
pub trait Probability {
    /// Returns *true* if the probality is valid (ie. between `0` and `1`)
    fn is_valid_probility(&self) -> bool;
}

impl<P: Zero + One + PartialOrd> Probability for P {
    fn is_valid_probility(&self) -> bool {
        Self::zero().le(self) && Self::one().ge(self)
    }
}

/// Helper trait for datastructure that can be initialized with capacity.
/// Can be interpreted as reserved space or guaranteed used space
pub trait FromCapacity: Sized {
    /// Create a new instance with a given capacity
    fn from_capacity(capacity: usize) -> Self {
        Self::from_total_used_capacity(capacity, capacity)
    }

    /// Creates a new instance from the total capacity (ie. max-value for example) and the actual
    /// capacity that will be used (space-wise)
    fn from_total_used_capacity(total: usize, used: usize) -> Self;
}

impl<T> FromCapacity for Vec<T> {
    fn from_total_used_capacity(total: usize, _used: usize) -> Self {
        // Using `Vec<T>` as a Map/Set requires intializing to the maximum element
        Self::with_capacity(total)
    }
}

impl<I: PrimIndex> FromCapacity for BitSetImpl<I> {
    fn from_total_used_capacity(total: usize, _used: usize) -> Self {
        // Using `BitSetImpl<I>` as a Map/Set requires intializing to the maximum element
        Self::new(I::from_usize(total).unwrap())
    }
}

impl<T> FromCapacity for HashSet<T, RandomState> {
    fn from_total_used_capacity(_total: usize, used: usize) -> Self {
        // Using `HashSet<T>` as a Set only requires intializing to the number of elements
        Self::with_capacity(used)
    }
}

impl<T> FromCapacity for FxHashSet<T> {
    fn from_total_used_capacity(_total: usize, used: usize) -> Self {
        // Using `FxHashSet<T>` as a Set only requires intializing to the number of elements
        Self::with_capacity_and_hasher(used, FxBuildHasher::default())
    }
}

impl<K, V> FromCapacity for HashMap<K, V, RandomState> {
    fn from_total_used_capacity(_total: usize, used: usize) -> Self {
        // Using `HashMap<K, V>` as a Map only requires intializing to the number of elements
        Self::with_capacity(used)
    }
}

impl<K, V> FromCapacity for FxHashMap<K, V> {
    fn from_total_used_capacity(_total: usize, used: usize) -> Self {
        // Using `FxHashMap<K, V>` as a Map only requires intializing to the number of elements
        Self::with_capacity_and_hasher(used, FxBuildHasher::default())
    }
}
