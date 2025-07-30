use std::fmt::{Debug, Display};

use stream_bitset::bitset::BitSetImpl;

use crate::Node;

/// An edge is defined by two nodes/endpoints.
/// Is is up to the user whether an Edge is directed or not.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge(pub Node, pub Node);

/// We limit the number of edges to `2^32 - 1`.
/// CHANGE it to `u64` if this does not suffice (which it usually should).
pub type NumEdges = u32;

/// A BitSet over NumEdges
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
    /// Normalizes the edge such that the endpoint with smaller value comes first
    pub fn normalized(&self) -> Self {
        Edge(self.0.min(self.1), self.0.max(self.1))
    }

    /// Returns true if the endpoint with smaller index comes first
    pub fn is_normalized(&self) -> bool {
        self.0 <= self.1
    }

    /// Returns true if both endpoints are equal
    pub fn is_loop(&self) -> bool {
        self.0 == self.1
    }

    /// Reverses the edge by switching the endpoints
    pub fn reverse(&self) -> Self {
        Edge(self.1, self.0)
    }

    /// Simple bidirection from `0..n^2` to all possible (directed) edges of `n` nodes
    pub fn from_u64(x: u64, n: u64) -> Self {
        debug_assert!(x < n * n);

        let u = x / n;
        let v = x % n;
        Edge(u as Node, v as Node)
    }

    /// Simple bidirection from `0..(n choose 2)` to all possible normalized edges of `n` nodes.
    ///
    /// The bidirecton works by assigning each node the next `(n - 1)/2` neighbors modulo `n`
    /// (up to rounding) and normalizing the resulting edge
    pub fn from_u64_undir(mut x: u64, n: u64) -> Self {
        debug_assert!(x < n * (n - 1) / 2);

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
