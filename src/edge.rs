/*!
# Edge Representation

This module defines the representation of edges in graphs.

- An `Edge(u, v)` consists of a preceding node `u` and a succeeding node `v`.
- For **undirected graphs**, `Edge(u, v)` is equivalent to `Edge(v, u)`.
- An edge is **normalized** if `u <= v`.
- Provides utilities for converting integers into edges (`from_u64` and `from_u64_undir`) for enumeration of all possible edges.
*/

use std::fmt::{Debug, Display};

use stream_bitset::bitset::BitSetImpl;

use crate::node::Node;

/// Represents an edge between two nodes `u` and `v`.
///
/// The user decides whether the graph is directed or undirected.
/// For undirected graphs, consider `Edge(u, v)` equivalent to `Edge(v, u)`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge(pub Node, pub Node);

/// Type alias representing the number of edges.
///
/// Limited to `u32` (maximum 2^32 - 1 edges).
/// If not big enough, change manually to `u64` for very large graphs.
pub type NumEdges = u32;

/// Bitset specialized for edges.
pub type EdgeBitSet = BitSetImpl<NumEdges>;

impl Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.0, self.1)
    }
}

impl Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Edge {
    /// Returns a normalized edge where the smaller node comes first.
    ///
    /// # Example
    /// ```
    /// # use ugraphs::edge::Edge;
    /// let e = Edge(3, 1);
    /// assert_eq!(e.normalized(), Edge(1, 3));
    /// ```
    #[inline(always)]
    pub fn normalized(&self) -> Self {
        Edge(self.0.min(self.1), self.0.max(self.1))
    }

    /// Returns `true` if the smaller endpoint comes first (i.e., `u <= v`).
    #[inline(always)]
    pub fn is_normalized(&self) -> bool {
        self.0 <= self.1
    }

    /// Returns `true` if the edge is a self-loop (`u == v`).
    #[inline(always)]
    pub fn is_loop(&self) -> bool {
        self.0 == self.1
    }

    /// Returns the edge with endpoints swapped (`Edge(v, u)`).
    #[inline(always)]
    pub fn reverse(&self) -> Self {
        Edge(self.1, self.0)
    }

    /// Maps a number `x` in `0..n^2` to a directed edge `(u, v)` of `n` nodes.
    ///
    /// # Panics
    /// Debug-asserts if `x >= n * n`.
    ///
    /// # Example
    /// ```
    /// # use ugraphs::edge::Edge;
    ///
    /// let n = 10u64;
    /// let ub = n * n;
    ///
    /// let mut edges: Vec<Edge> = (0..ub).map(|x| Edge::from_u64(x, n)).collect();
    /// edges.sort_unstable();
    /// edges.dedup();
    ///
    /// assert_eq!(edges.len(), ub as usize);
    /// ```
    #[inline(always)]
    pub fn from_u64(x: u64, n: u64) -> Self {
        debug_assert!(x < n * n);

        let u = x / n;
        let v = x % n;
        Edge(u as Node, v as Node)
    }

    /// Maps a number `x` in `0..(n choose 2)` to a normalized undirected edge `(u, v)`.
    ///
    /// # Panics
    /// Debug-asserts if `x >= n * (n - 1) / 2`.
    ///
    /// # Example
    /// ```
    /// # use ugraphs::edge::Edge;
    ///
    /// let n = 10u64;
    /// let ub = n * (n - 1) / 2;
    ///
    /// let mut edges: Vec<Edge> = (0..ub).map(|x| Edge::from_u64_undir(x, n)).collect();
    /// edges.sort_unstable();
    /// edges.dedup();
    ///
    /// assert_eq!(edges.len(), ub as usize);
    /// ```
    pub fn from_u64_undir(mut x: u64, n: u64) -> Self {
        debug_assert!(x < n * (n - 1) / 2, "{x} >= {n}");

        let mut num_neighbors = (n - 1) / 2;
        // Easy case where `n - 1` is even and no corner cases exist
        if n & 1 == 1 {
            let u = x / num_neighbors;
            let v = (u + 1 + (x % num_neighbors)) % n;

            Edge(u as Node, v as Node).normalized()
        // Harder case where `n - 1` is odd and the number of checked neighbors alternates
        } else {
            let half_n = n / 2;
            let lower_half = num_neighbors * half_n;

            // x is in the half where we only enumerate `floor((n - 1) / 2)` neighbors
            if x < lower_half {
                let u = x / num_neighbors;
                let v = (u + 1 + (x % num_neighbors)) % n;

                // Edges are guaranteed to be normalized in the lower half
                return Edge(u as Node, v as Node);
            }

            // x is the upper half where we enumerate `ceil((n - 1) / 2)` neighbors
            x -= lower_half;
            num_neighbors += 1;

            let u = (x / num_neighbors) + half_n;
            let v = (u + 1 + (x % num_neighbors)) % n;

            Edge(u as Node, v as Node).normalized()
        }
    }
}

impl From<(Node, Node)> for Edge {
    fn from(value: (Node, Node)) -> Self {
        Edge(value.0, value.1)
    }
}

impl From<&(Node, Node)> for Edge {
    fn from(value: &(Node, Node)) -> Self {
        Edge(value.0, value.1)
    }
}

impl From<(&Node, &Node)> for Edge {
    fn from(value: (&Node, &Node)) -> Self {
        Edge(*value.0, *value.1)
    }
}

impl From<&Edge> for Edge {
    fn from(value: &Edge) -> Self {
        *value
    }
}
