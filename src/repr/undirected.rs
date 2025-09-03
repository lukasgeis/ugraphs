/*!
# Undirected Graph Representations

This module defines generic and concrete **undirected graph** representations.

An undirected graph is represented by parameterizing [`UndirectedGraph`] with a
[`Neighborhood`] implementation, which controls how adjacency information is stored.

## Provided Representations

- [`AdjArrayUndir`] — adjacency lists stored in `Vec<Node>`.
- [`SparseAdjArrayUndir`] — adjacency lists stored in `SmallVec<[Node; N]>`, optimized for sparse graphs.
- [`AdjMatrixUndir`] — adjacency stored as a `NodeBitSet` (boolean adjacency matrix).

All representations share the same API and differ only in memory usage
and performance characteristics.
*/

use crate::{
    repr::neighborhood::macros::{impl_common_graph_ops, impl_try_add_edge},
    testing::test_graph_ops,
};

use std::ops::Range;

use super::*;

/// Generic undirected graph representation parameterized by a [`Neighborhood`] type.
///
/// - Adjacency is stored in a `Vec<Nbs>`, where each entry corresponds
///   to the neighborhood of a vertex.
/// - Edges are always undirected, i.e., adding/removing `(u, v)` will
///   also affect `(v, u)` unless `u == v`.
///
/// # Type parameters
/// - `Nbs`: A [`Neighborhood`] implementation, e.g. [`ArrNeighborhood`],
///   [`SparseNeighborhood`], or [`BitNeighborhood`].
///
/// # Fields
/// - `nbs`: Vector of neighborhoods, one per vertex.
/// - `num_edges`: Total number of edges in the graph.
#[derive(Clone)]
pub struct UndirectedGraph<Nbs>
where
    Nbs: Neighborhood,
{
    nbs: Vec<Nbs>,
    num_edges: NumEdges,
}

/// Undirected graph using adjacency arrays (`Vec<Node>`).
///
/// - Flexible, simple representation.
/// - Best for sparse to moderately dense graphs.
pub type AdjArrayUndir = UndirectedGraph<ArrNeighborhood>;

/// Undirected graph using sparse adjacency arrays (`SmallVec<[Node; N]>`).
///
/// - Optimized for sparse graphs where most nodes have few neighbors.
/// - Reduces cache misses by storing small neighborhoods inline.
pub type SparseAdjArrayUndir = UndirectedGraph<SparseNeighborhood>;

/// Undirected graph using a bitset-based adjacency matrix (`NodeBitSet`).
///
/// - Best for dense graphs where fast membership queries (`has_edge`) dominate.
/// - Memory usage: `O(n^2)`.
pub type AdjMatrixUndir = UndirectedGraph<BitNeighborhood>;

impl_common_graph_ops!(UndirectedGraph<nbs : Nbs> => nbs, Undirected);

impl<Nbs> AdjacencyTest for UndirectedGraph<Nbs>
where
    Nbs: Neighborhood,
{
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.nbs[u as usize].has_neighbor(v)
    }

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        self.nbs[u as usize].has_neighbors(neighbors)
    }
}

impl<Nbs> GraphEdgeEditing for UndirectedGraph<Nbs>
where
    Nbs: Neighborhood,
{
    fn add_edge(&mut self, u: Node, v: Node) {
        self.nbs[u as usize].add_neighbor(v);
        if u != v {
            self.nbs[v as usize].add_neighbor(u);
        }
        self.num_edges += 1;
    }

    impl_try_add_edge!(self);

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if self.nbs[u as usize].try_remove_neighbor(v) {
            if u != v {
                assert!(self.nbs[v as usize].try_remove_neighbor(u));
            }
            self.num_edges -= 1;
            true
        } else {
            false
        }
    }
}

impl<Nbs> GraphLocalEdgeEditing for UndirectedGraph<Nbs>
where
    Nbs: Neighborhood,
{
    fn remove_edges_at_node(&mut self, u: Node) {
        let (beg, end) = self.nbs.split_at_mut(u as usize);
        let (nbs, end) = end.split_at_mut(1);
        let nbs = &mut nbs[0];

        for nb in nbs.neighbors() {
            if nb < u {
                beg[nb as usize].try_remove_neighbor(u);
            } else if nb > u {
                end[(nb - u - 1) as usize].try_remove_neighbor(u);
            }
        }

        self.num_edges -= nbs.num_of_neighbors() as NumEdges;
        nbs.clear();
    }
}

// ---------- Testing ----------

test_graph_ops!(
    test_adj_array_undir,
    AdjArrayUndir,
    true,
    (GraphNew, AdjacencyList, GraphEdgeEditing)
);

test_graph_ops!(
    test_sparse_adj_array_undir,
    SparseAdjArrayUndir,
    true,
    (GraphNew, AdjacencyList, GraphEdgeEditing)
);

test_graph_ops!(
    test_adj_matrix_undir,
    AdjMatrixUndir,
    true,
    (GraphNew, AdjacencyList, GraphEdgeEditing)
);
