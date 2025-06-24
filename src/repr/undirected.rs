use crate::{
    repr::macros::{impl_common_graph_ops, impl_try_add_edge},
    testing::test_graph_ops,
};

use super::*;

/// An undirected graph representation
#[derive(Clone)]
pub struct UndirectedGraph<Nbs: Neighborhood> {
    nbs: Vec<Nbs>,
    num_edges: NumEdges,
}

/// Representation using an Adjacency-Array
pub type AdjArrayUndir = UndirectedGraph<ArrNeighborhood>;

/// Representation using a sparse Adjacency-Array
pub type SparseAdjArrayUndir = UndirectedGraph<SparseNeighborhood>;

/// Representation using an Adjacency-Matrix
pub type AdjMatrixUndir = UndirectedGraph<BitNeighborhood>;

impl_common_graph_ops!(UndirectedGraph<nbs : Nbs> => nbs);

impl<Nbs: Neighborhood> AdjacencyTest for UndirectedGraph<Nbs> {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.nbs[u as usize].has_neighbor(v)
    }

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        self.nbs[u as usize].has_neighbors(neighbors)
    }
}

impl<Nbs: Neighborhood> GraphEdgeEditing for UndirectedGraph<Nbs> {
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

impl<Nbs: Neighborhood> GraphLocalEdgeEditing for UndirectedGraph<Nbs> {
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
