/*!
# Directed Graph Representations

This module defines generic and concrete **directed graph** representations.

A directed graph is represented by parameterizing [`DirectedGraph`] or
[`DirectedGraphIn`] with one or two [`Neighborhood`] types, which
control how adjacency information is stored.

## Provided Representations

- [`AdjArray`] / [`AdjArrayIn`] — adjacency arrays for outgoing and (optionally) incoming neighbors.
- [`SparseAdjArray`] / [`SparseAdjArrayIn`] — sparse adjacency arrays using inline small vectors.
- [`AdjMatrix`] / [`AdjMatrixIn`] — bitset-based adjacency matrices.
- [`AdjArrayMatrix`] — outgoing adjacency arrays, incoming adjacency matrix.

## Design
- [`DirectedGraph`] stores **only outgoing neighborhoods** and derives
  incoming neighborhoods by scanning all vertices (costly).
- [`DirectedGraphIn`] stores **both outgoing and incoming neighborhoods**,
  enabling efficient access to in-neighbors.
*/

use stream_bitset::{bitset::BitsetStream, prelude::IntoBitmaskStream};

use std::ops::Range;

use crate::{
    repr::neighborhood::macros::{impl_common_graph_ops, impl_try_add_edge},
    testing::test_graph_ops,
};

use super::*;

/// A directed graph storing only **outgoing neighborhoods**.
///
/// - Outgoing adjacency is stored directly.
/// - Incoming adjacency is derived on demand (expensive).
///
/// # Type parameters
/// - `OutNbs`: [`Neighborhood`] implementation used for outgoing adjacency.
#[derive(Clone)]
pub struct DirectedGraph<OutNbs>
where
    OutNbs: Neighborhood,
{
    out_nbs: Vec<OutNbs>,
    num_edges: NumEdges,
}

/// A directed graph storing **both outgoing and incoming neighborhoods**.
///
/// - Outgoing adjacency is stored in `out_nbs`.
/// - Incoming adjacency is stored in `in_nbs`.
/// - Enables efficient `in_neighbors_of` and `in_degree_of`.
///
/// # Type parameters
/// - `OutNbs`: [`Neighborhood`] implementation used for outgoing adjacency.
/// - `InNbs`: [`Neighborhood`] implementation used for incoming adjacency.
#[derive(Clone)]
pub struct DirectedGraphIn<OutNbs, InNbs>
where
    OutNbs: Neighborhood,
    InNbs: Neighborhood,
{
    out_nbs: Vec<OutNbs>,
    in_nbs: Vec<InNbs>,
    num_edges: NumEdges,
}

/// Directed graph using adjacency arrays (`Vec<Node>`).
pub type AdjArray = DirectedGraph<ArrNeighborhood>;

/// Directed graph using adjacency arrays for both outgoing and incoming neighborhoods.
pub type AdjArrayIn = DirectedGraphIn<ArrNeighborhood, ArrNeighborhood>;

/// Directed graph using sparse adjacency arrays (`SmallVec<[Node; N]>`).
pub type SparseAdjArray = DirectedGraph<SparseNeighborhood>;

/// Directed graph using sparse adjacency arrays for both outgoing and incoming neighborhoods.
pub type SparseAdjArrayIn = DirectedGraphIn<SparseNeighborhood, SparseNeighborhood>;

/// Directed graph using a bitset-based adjacency matrix (`NodeBitSet`).
pub type AdjMatrix = DirectedGraph<BitNeighborhood>;

/// Directed graph using a bitset-based adjacency matrix for both outgoing and incoming neighborhoods.
pub type AdjMatrixIn = DirectedGraphIn<BitNeighborhood, BitNeighborhood>;

/// Directed graph using:
/// - adjacency arrays for outgoing neighborhoods,
/// - a bitset-based adjacency matrix for incoming neighborhoods.
pub type AdjArrayMatrix = DirectedGraphIn<ArrNeighborhood, BitNeighborhood>;

impl_common_graph_ops!(DirectedGraph<out_nbs : OutNbs> => out_nbs, Directed);
impl_common_graph_ops!(DirectedGraphIn<out_nbs : OutNbs, in_nbs: InNbs> => out_nbs, Directed);

impl<OutNbs> DirectedAdjacencyList for DirectedGraph<OutNbs>
where
    OutNbs: Neighborhood,
{
    type InNeighborIter<'a>
        = DirectedInNeighborIter<'a, OutNbs>
    where
        Self: 'a;

    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_> {
        DirectedInNeighborIter {
            graph: self,
            node: u,
            lb: 0,
        }
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        // Should be avoided as this is very costly
        self.in_neighbors_of(u).count() as NumNodes
    }

    type InNeighborsStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;

    fn in_neighbors_of_as_stream(&self, u: Node) -> Self::InNeighborsStream<'_> {
        NodeBitSet::new_with_bits_set(self.number_of_nodes(), self.in_neighbors_of(u))
            .into_bitmask_stream()
    }
}

impl<OutNbs: Neighborhood> Singletons for DirectedGraph<OutNbs> {
    /// Very inefficient as this has a runtime of `O(n^2)` for some implementations
    fn is_singleton(&self, u: Node) -> bool {
        self.total_degree_of(u) == 0
    }
}

/// Iterator over the in-neighbors of a node in a [`DirectedGraph`].
///
/// Scans all vertices, checking if they have an edge into the target node.
/// This is inherently costly compared to [`DirectedGraphIn`].
pub struct DirectedInNeighborIter<'a, OutNbs>
where
    OutNbs: Neighborhood,
{
    graph: &'a DirectedGraph<OutNbs>,
    node: Node,
    lb: Node,
}

impl<'a, OutNbs> Iterator for DirectedInNeighborIter<'a, OutNbs>
where
    OutNbs: Neighborhood,
{
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        while self.lb < self.graph.number_of_nodes() {
            self.lb += 1;

            if self.graph.has_edge(self.lb - 1, self.node) {
                return Some(self.lb - 1);
            }
        }

        None
    }
}

impl<OutNbs> AdjacencyTest for DirectedGraph<OutNbs>
where
    OutNbs: Neighborhood,
{
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.out_nbs[u as usize].has_neighbor(v)
    }

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        self.out_nbs[u as usize].has_neighbors(neighbors)
    }
}

impl<OutNbs> GraphEdgeEditing for DirectedGraph<OutNbs>
where
    OutNbs: Neighborhood,
{
    fn add_edge(&mut self, u: Node, v: Node) {
        self.out_nbs[u as usize].add_neighbor(v);
        self.num_edges += 1;
    }

    impl_try_add_edge!(self);

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if self.out_nbs[u as usize].try_remove_neighbor(v) {
            self.num_edges -= 1;
            true
        } else {
            false
        }
    }
}

impl<OutNbs> GraphDirectedEdgeEditing for DirectedGraph<OutNbs>
where
    OutNbs: Neighborhood,
{
    fn remove_edges_into_node(&mut self, u: Node) {
        // Should be avoided as this is very costly
        self.num_edges -= self
            .vertices_range()
            .map(|v| self.out_nbs[v as usize].try_remove_neighbor(u) as NumEdges)
            .sum::<NumEdges>();
    }

    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.num_edges -= self.out_nbs[u as usize].num_of_neighbors() as NumEdges;
        self.out_nbs[u as usize].clear();
    }
}

impl<OutNbs> GraphLocalEdgeEditing for DirectedGraph<OutNbs>
where
    OutNbs: Neighborhood,
{
    fn remove_edges_at_node(&mut self, u: Node) {
        self.remove_edges_into_node(u);
        self.remove_edges_out_of_node(u);
    }
}

impl<OutNbs, InNbs> DirectedAdjacencyList for DirectedGraphIn<OutNbs, InNbs>
where
    OutNbs: Neighborhood,
    InNbs: Neighborhood,
{
    type InNeighborIter<'a>
        = <InNbs as Neighborhood>::NeighborhoodIter<'a>
    where
        Self: 'a;

    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_> {
        self.in_nbs[u as usize].neighbors()
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        self.in_nbs[u as usize].num_of_neighbors()
    }

    type InNeighborsStream<'a>
        = <InNbs as Neighborhood>::NeighborhoodStream<'a>
    where
        Self: 'a;

    fn in_neighbors_of_as_stream(&self, u: Node) -> Self::InNeighborsStream<'_> {
        self.in_nbs[u as usize].neighbors_as_stream(self.number_of_nodes())
    }
}

impl<OutNbs: Neighborhood, InNbs: Neighborhood> Singletons for DirectedGraphIn<OutNbs, InNbs> {
    #[inline]
    fn is_singleton(&self, u: Node) -> bool {
        self.total_degree_of(u) == 0
    }
}

impl<OutNbs, InNbs> AdjacencyTest for DirectedGraphIn<OutNbs, InNbs>
where
    OutNbs: Neighborhood,
    InNbs: Neighborhood,
{
    fn has_edge(&self, u: Node, v: Node) -> bool {
        // Without additional knowledge, checking `out_nbs` or `in_nbs` does not make a difference:
        // since we define `AdjArrayMatrix` which uses a `BitNeighborhood` for `in_nbs`, we default
        // to `in_nbs` here
        self.in_nbs[v as usize].has_neighbor(u)
    }

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        // Without additional knowledge, checking `out_nbs` or `in_nbs` does not make a difference:
        // since we define `AdjArrayMatrix` which uses a `BitNeighborhood` for `in_nbs`, we default
        // to `in_nbs` here.
        //
        // For larger `N`, it might make sense to implement this on `out_nbs` in case we want to
        // leverage locality
        neighbors.map(|v| self.has_edge(u, v))
    }
}

impl<OutNbs, InNbs> GraphEdgeEditing for DirectedGraphIn<OutNbs, InNbs>
where
    OutNbs: Neighborhood,
    InNbs: Neighborhood,
{
    fn add_edge(&mut self, u: Node, v: Node) {
        self.out_nbs[u as usize].add_neighbor(v);
        self.in_nbs[v as usize].add_neighbor(u);
        self.num_edges += 1;
    }

    impl_try_add_edge!(self);

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if self.out_nbs[u as usize].try_remove_neighbor(v) {
            assert!(self.in_nbs[v as usize].try_remove_neighbor(u));
            self.num_edges -= 1;
            true
        } else {
            false
        }
    }
}

impl<OutNbs, InNbs> GraphDirectedEdgeEditing for DirectedGraphIn<OutNbs, InNbs>
where
    OutNbs: Neighborhood,
    InNbs: Neighborhood,
{
    fn remove_edges_into_node(&mut self, u: Node) {
        for v in self.vertices_range() {
            self.out_nbs[v as usize].try_remove_neighbor(u);
        }
        self.num_edges -= self.in_nbs[u as usize].num_of_neighbors() as NumEdges;
        self.in_nbs[u as usize].clear();
    }

    fn remove_edges_out_of_node(&mut self, u: Node) {
        for v in self.vertices_range() {
            self.in_nbs[v as usize].try_remove_neighbor(u);
        }
        self.num_edges -= self.out_nbs[u as usize].num_of_neighbors() as NumEdges;
        self.out_nbs[u as usize].clear();
    }
}

impl<OutNbs, InNbs> GraphLocalEdgeEditing for DirectedGraphIn<OutNbs, InNbs>
where
    OutNbs: Neighborhood,
    InNbs: Neighborhood,
{
    fn remove_edges_at_node(&mut self, u: Node) {
        self.remove_edges_into_node(u);
        self.remove_edges_out_of_node(u);
    }
}

// ---------- Testing ----------

test_graph_ops!(
    test_adj_array,
    AdjArray,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);

test_graph_ops!(
    test_adj_array_in,
    AdjArrayIn,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);

test_graph_ops!(
    test_sparse_adj_array,
    SparseAdjArray,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);

test_graph_ops!(
    test_sparse_adj_array_in,
    SparseAdjArrayIn,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);

test_graph_ops!(
    test_adj_matrix,
    AdjMatrix,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);

test_graph_ops!(
    test_adj_matrix_in,
    AdjMatrixIn,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);

test_graph_ops!(
    test_adj_array_matrix,
    AdjArrayMatrix,
    false,
    (
        GraphNew,
        AdjacencyList,
        DirectedAdjacencyList,
        GraphEdgeEditing,
        GraphDirectedEdgeEditing
    )
);
