use std::{
    collections::{HashMap, HashSet},
    hash::RandomState,
};

use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use num::{One, Zero};

mod geometric;
mod multi_traits;
mod set_map;
mod sliced_buffer;

pub use geometric::*;
pub use multi_traits::*;
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
pub trait FromCapacity {
    /// Create a new instance with a given capacity
    fn from_capacity(capacity: usize) -> Self;
}

impl<T> FromCapacity for Vec<T> {
    fn from_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }
}

impl<I: PrimIndex> FromCapacity for BitSetImpl<I> {
    fn from_capacity(capacity: usize) -> Self {
        Self::new(I::from_usize(capacity).unwrap())
    }
}

impl<T> FromCapacity for HashSet<T, RandomState> {
    fn from_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }
}

impl<T> FromCapacity for FxHashSet<T> {
    fn from_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, FxBuildHasher::default())
    }
}

impl<K, V> FromCapacity for HashMap<K, V, RandomState> {
    fn from_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }
}

impl<K, V> FromCapacity for FxHashMap<K, V> {
    fn from_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, FxBuildHasher::default())
    }
}
