use crate::repr::macros::impl_common_graph_ops;

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
        self.nbs[u].has_neighbor(v)
    }

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        self.nbs[u].has_neighbors(neighbors)
    }
}

impl<Nbs: Neighborhood> GraphEdgeEditing for UndirectedGraph<Nbs> {
    fn try_add_edge(&mut self, u: Node, v: Node) -> bool {
        if !self.nbs[u].try_add_neighbor(v) {
            if u != v {
                assert!(!self.nbs[v].try_add_neighbor(u));
            }
            self.num_edges += 1;
            false
        } else {
            true
        }
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if self.nbs[u].try_remove_neighbor(v) {
            if u != v {
                assert!(self.nbs[v].try_remove_neighbor(u));
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
        let (beg, end) = self.nbs.split_at_mut(u.raw() as usize);
        let (nbs, end) = end.split_at_mut(1);
        let nbs = &mut nbs[0];

        for nb in nbs.neighbors() {
            if nb < u {
                beg[nb].try_remove_neighbor(u);
            } else if nb > u {
                end[nb - u - Node::ONE].try_remove_neighbor(u);
            }
        }

        self.num_edges -= nbs.num_of_neighbors() as NumEdges;
        nbs.clear();
    }
}
