use crate::{
    repr::macros::{impl_common_graph_ops, impl_try_add_edge},
    testing::test_graph_ops,
};

use super::*;

/// A directed graph representation storing only outgoing Neighborhoods
#[derive(Clone)]
pub struct DirectedGraph<OutNbs: Neighborhood> {
    out_nbs: Vec<OutNbs>,
    num_edges: NumEdges,
}

/// A directed graph representation storing both outgoing & incoming Neighborhoods
#[derive(Clone)]
pub struct DirectedGraphIn<OutNbs: Neighborhood, InNbs: Neighborhood> {
    out_nbs: Vec<OutNbs>,
    in_nbs: Vec<InNbs>,
    num_edges: NumEdges,
}

/// Representation using an Adjacency-Array
pub type AdjArray = DirectedGraph<ArrNeighborhood>;

/// Representation using an Adjacency-Array for both outgoing & incoming Neighborhoods
pub type AdjArrayIn = DirectedGraphIn<ArrNeighborhood, ArrNeighborhood>;

/// Representation using a sparse Adjacency-Array
pub type SparseAdjArray = DirectedGraph<SparseNeighborhood>;

/// Representation using a sparse Adjacency-Array for both outgoing & incoming Neighborhoods
pub type SparseAdjArrayIn = DirectedGraphIn<SparseNeighborhood, SparseNeighborhood>;

/// Representation using an Adjacency-Matrix
pub type AdjMatrix = DirectedGraph<BitNeighborhood>;

/// Representation using an Adjacency-Matrix for both outgoing & incoming Neighborhoods
pub type AdjMatrixIn = DirectedGraphIn<BitNeighborhood, BitNeighborhood>;

/// Representation using an
/// - Adjacency-Array for outgoing Neighborhoods
/// - Adjacency-Matrix for incoming Neighborhoods
pub type AdjArrayMatrix = DirectedGraphIn<ArrNeighborhood, BitNeighborhood>;

impl_common_graph_ops!(DirectedGraph<out_nbs : OutNbs> => out_nbs, true);
impl_common_graph_ops!(DirectedGraphIn<out_nbs : OutNbs, in_nbs: InNbs> => out_nbs, true);

impl<OutNbs: Neighborhood> DirectedAdjacencyList for DirectedGraph<OutNbs> {
    fn in_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
        // Should be avoided as this is very costly
        self.vertices()
            .filter(move |&v| self.out_nbs[v as usize].has_neighbor(u))
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        // Should be avoided as this is very costly
        self.in_neighbors_of(u).count() as NumNodes
    }
}

impl<OutNbs: Neighborhood> AdjacencyTest for DirectedGraph<OutNbs> {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.out_nbs[u as usize].has_neighbor(v)
    }

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        self.out_nbs[u as usize].has_neighbors(neighbors)
    }
}

impl<OutNbs: Neighborhood> GraphEdgeEditing for DirectedGraph<OutNbs> {
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

impl<OutNbs: Neighborhood> GraphDirectedEdgeEditing for DirectedGraph<OutNbs> {
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

impl<OutNbs: Neighborhood> GraphLocalEdgeEditing for DirectedGraph<OutNbs> {
    fn remove_edges_at_node(&mut self, u: Node) {
        self.remove_edges_into_node(u);
        self.remove_edges_out_of_node(u);
    }
}

impl<OutNbs: Neighborhood, InNbs: Neighborhood> DirectedAdjacencyList
    for DirectedGraphIn<OutNbs, InNbs>
{
    fn in_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
        self.in_nbs[u as usize].neighbors()
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        self.in_nbs[u as usize].num_of_neighbors()
    }

    fn in_neighbors_of_as_stream(&self, u: Node) -> impl BitmaskStream + '_ {
        self.in_nbs[u as usize].neighbors_as_stream(self.number_of_nodes())
    }
}

impl<OutNbs: Neighborhood, InNbs: Neighborhood> AdjacencyTest for DirectedGraphIn<OutNbs, InNbs> {
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

impl<OutNbs: Neighborhood, InNbs: Neighborhood> GraphEdgeEditing
    for DirectedGraphIn<OutNbs, InNbs>
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

impl<OutNbs: Neighborhood, InNbs: Neighborhood> GraphDirectedEdgeEditing
    for DirectedGraphIn<OutNbs, InNbs>
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

impl<OutNbs: Neighborhood, InNbs: Neighborhood> GraphLocalEdgeEditing
    for DirectedGraphIn<OutNbs, InNbs>
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
