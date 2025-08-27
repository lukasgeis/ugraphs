/*!
# Node Representation

We choose `Node = u32` as almost all use-cases often involve less than `2^32` nodes.
This allows as to (1) save space by not using `usize` or `u64` and (2) allows directly manipulating node values without abstracting over them.
*/

use std::num::NonZero;
use stream_bitset::bitset::BitSetImpl;

/// Nodes can be any unsigned integer from `0` to `Node::MAX - 1`
pub type Node = u32;

/// Node-Value that is considered invalid
pub const INVALID_NODE: Node = Node::MAX;

/// There can be at most `2^32 - 1` nodes in a graph!
pub type NumNodes = Node;

/// BitSet for Nodes
pub type NodeBitSet = BitSetImpl<Node>;

/// As `Option<Node>` uses additional bytes for padding, it can be inefficient
/// since we often need to use `Vec<Option<Node>>`. This instead uses the
/// `NonZero`-Wrapper to assign a constant value (often)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OptionalNodeImpl<const N: Node>(NonZero<Node>);

/// Often, `INVALID_NODE` is safe to pick as the `None`-Value
pub type OptionalNode = OptionalNodeImpl<INVALID_NODE>;

impl<const N: Node> OptionalNodeImpl<N> {
    /// Returns `Some(OptionalNodeImpl)` if `n != N` and `None` otherwise
    pub const fn new(n: Node) -> Option<Self> {
        match NonZero::new(n ^ N) {
            Some(inner) => Some(OptionalNodeImpl(inner)),
            None => None,
        }
    }

    /// Gets the underlying Node-Value
    pub const fn get(&self) -> Node {
        self.0.get() ^ N
    }
}

/// As `Option<u64>` uses additional bytes for padding, it can be inefficient
/// since we often need to use `Vec<Option<u64>>`. This instead uses the
/// `NonZero`-Wrapper to assign a constant value (often)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OptionalU64Impl<const N: u64>(NonZero<u64>);

/// Often, `u64::MAX` is safe to pick as the `None`-Value
pub type OptionalU64 = OptionalU64Impl<{ u64::MAX }>;

impl<const N: u64> OptionalU64Impl<N> {
    /// Returns `Some(OptionalU64Impl)` if `n != N` and `None` otherwise
    pub const fn new(n: u64) -> Option<Self> {
        match NonZero::new(n ^ N) {
            Some(inner) => Some(OptionalU64Impl(inner)),
            None => None,
        }
    }

    /// Gets the underlying u64-Value
    pub const fn get(&self) -> u64 {
        self.0.get() ^ N
    }
}
