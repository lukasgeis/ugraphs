/*!
# Utilities

Provides a variety of utility traits/structs such as
- [`SlicedBuffer`](self::sliced_buffer::SlicedBuffer): the internal representation for [`CsrGraph`],
- [`GeometricJumper`](self::geometric::GeometricJumper): the generator for [`G(n,p)`](crate::gens::gnp::Gnp) graphs,
- abstractions over [`Set`] and [`Map`] for more flexibility in certain algorithms,
- the `NodeMapper`-framework for mapping one graph to another (see [`NodeMapSetter`] / [`NodeMapGetter`]),
- utility traits for combining multiple objects with less overhead.

Apart from `Set, Map, NodeMapSetter, NodeMapGetter`, you probably do not need to interact with this module directly.
*/

use std::{
    collections::{HashMap, HashSet},
    hash::RandomState,
};

use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use num::{One, Zero};

use crate::prelude::*;

pub mod geometric;
pub mod map;
pub mod multi_traits;
pub mod node_mapper;
pub mod partition;
pub mod set;
pub mod sliced_buffer;

// Only export most important traits / structs

pub use map::Map;
pub use node_mapper::{NodeMapCompose, NodeMapGetter, NodeMapInverse, NodeMapSetter, NodeMapper};
pub use partition::{IntoPartition, Partition};
pub use set::Set;

use stream_bitset::{PrimIndex, bitset::BitSetImpl};

/// Helper trait for probalities
pub trait Probability {
    /// Returns *true* if the probality is valid (ie. between `0` and `1`)
    fn is_valid_probility(&self) -> bool;
}

impl<P> Probability for P
where
    P: Zero + One + PartialOrd,
{
    fn is_valid_probility(&self) -> bool {
        Self::zero().le(self) && Self::one().ge(self)
    }
}

/// Helper trait for datastructure that can be initialized with capacity.
/// Can be interpreted as reserved space or guaranteed used space.
///
/// Note that this should mainly be used in conjunction with either [`Set`] or [`Map`]
/// datastructures and not synonym to common implementations (`Vec<T>` for example) treats
/// total capacity differently normally than intended here when `Vec<T>` is used as `Map`.
pub trait FromCapacity: Sized {
    /// Create a new instance with a given capacity
    fn from_capacity(capacity: usize) -> Self {
        Self::from_total_used_capacity(capacity, capacity)
    }

    /// Creates a new instance from the total capacity (ie. max-value for example) and the actual
    /// capacity that will be used (space-wise).
    ///
    /// While seeming complex, this method often defaults to using [`FromCapacity::from_capacity`]
    /// with either `total` or `used`. If you only have one value as an upper bound, provide it as
    /// both arguments if possible.
    fn from_total_used_capacity(total: usize, used: usize) -> Self;
}

impl<T> FromCapacity for Vec<T> {
    fn from_total_used_capacity(total: usize, _used: usize) -> Self {
        // Using `Vec<T>` as a Map/Set requires intializing to the maximum element
        Self::with_capacity(total)
    }
}

impl<I> FromCapacity for BitSetImpl<I>
where
    I: PrimIndex,
{
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
