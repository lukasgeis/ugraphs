/*!
# Node Representation

This module defines the internal representation of nodes in a graph.

We use `Node = u32` because most graphs have fewer than `2^32` nodes.
This allows us to:
1. Save memory by avoiding `usize` or `u64`.
2. Manipulate node values directly without additional abstractions.

The module also provides:
- `NumNodes` as a type alias for counting nodes.
- `NodeBitSet` for efficient node set representations.
- `OptionalNode` and `OptionalU64` types that efficiently wrap optional values
  without the extra memory overhead of `Option<Node>` or `Option<u64>`.
*/

use std::num::NonZero;
use stream_bitset::bitset::BitSetImpl;

/// Type alias for a node identifier.
///
/// A `Node` can be any unsigned integer from 0 up to `Node::MAX - 1`.
/// Most algorithms and data structures in this library use `Node` values directly.
pub type Node = u32;

/// A special `Node` value representing an invalid or missing node.
///
/// Typically used for sentinel purposes in optional node containers.
pub const INVALID_NODE: Node = Node::MAX;

/// Type alias representing a number of nodes.
///
/// Same as `Node`. Maximum value is `2^32 - 1`.
pub type NumNodes = Node;

/// Bitset specialized for nodes.
pub type NodeBitSet = BitSetImpl<Node>;

/// Efficient optional node wrapper using `NonZero<Node>`.
///
/// Avoids the memory overhead of `Option<Node>` in large arrays like `Vec<Option<Node>>`.
///
/// The `const N` parameter specifies the sentinel value used as `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OptionalNodeImpl<const N: Node>(NonZero<Node>);

/// Default optional node using `INVALID_NODE` as the `None` value.
pub type OptionalNode = OptionalNodeImpl<INVALID_NODE>;

impl<const N: Node> OptionalNodeImpl<N> {
    /// Creates a new `OptionalNodeImpl` from a node value.
    ///
    /// Returns `Some(OptionalNodeImpl)` if `n != N`, otherwise returns `None`.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::node::{OptionalNode, INVALID_NODE};
    /// let some_node = OptionalNode::new(5);
    /// let none_node = OptionalNode::new(INVALID_NODE);
    /// assert!(some_node.is_some());
    /// assert!(none_node.is_none());
    /// ```
    pub const fn new(n: Node) -> Option<Self> {
        match NonZero::new(n ^ N) {
            Some(inner) => Some(OptionalNodeImpl(inner)),
            None => None,
        }
    }

    /// Returns the underlying `Node` value stored in this optional wrapper.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::node::{OptionalNode, INVALID_NODE};
    /// let opt = OptionalNode::new(5).unwrap();
    /// assert_eq!(opt.get(), 5);
    /// ```
    pub const fn get(&self) -> Node {
        self.0.get() ^ N
    }
}

/// Efficient optional `u64` wrapper using `NonZero<u64>`.
///
/// Avoids the memory overhead of `Option<u64>` in large arrays or vectors.
/// The `const N` parameter specifies the sentinel value used as `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OptionalU64Impl<const N: u64>(NonZero<u64>);

/// Default optional u64 using `u64::MAX` as the `None` value.
pub type OptionalU64 = OptionalU64Impl<{ u64::MAX }>;

impl<const N: u64> OptionalU64Impl<N> {
    /// Creates a new `OptionalU64Impl` from a u64 value.
    ///
    /// Returns `Some(OptionalU64Impl)` if `n != N`, otherwise returns `None`.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::node::OptionalU64;
    /// let some_val = OptionalU64::new(42);
    /// let none_val = OptionalU64::new(u64::MAX);
    /// assert!(some_val.is_some());
    /// assert!(none_val.is_none());
    /// ```
    pub const fn new(n: u64) -> Option<Self> {
        match NonZero::new(n ^ N) {
            Some(inner) => Some(OptionalU64Impl(inner)),
            None => None,
        }
    }

    /// Returns the underlying `u64` value stored in this optional wrapper.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::node::OptionalU64;
    /// let opt = OptionalU64::new(42).unwrap();
    /// assert_eq!(opt.get(), 42);
    /// ```
    pub const fn get(&self) -> u64 {
        self.0.get() ^ N
    }
}
