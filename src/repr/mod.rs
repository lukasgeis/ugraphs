use crate::{ops::*, *};
use std::ops::Range;
use stream_bitset::prelude::BitmaskStream;

mod csr;
mod directed;
mod neighborhood;
mod undirected;

pub mod digest;

pub use csr::*;
pub use directed::*;
pub use neighborhood::*;
pub use undirected::*;

/// Trait for methods on the Neighborhood of a specified Node
pub trait Neighborhood: Clone {
    fn new(n: NumNodes) -> Self;

    /// Returns the number of neighbors in the Neighborhood
    fn num_of_neighbors(&self) -> NumNodes;

    type NeighborhoodIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns an iterator over all neighbors in the Neighborhood
    fn neighbors(&self) -> Self::NeighborhoodIter<'_>;

    type NeighborhoodStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;

    /// Returns a BitmaskStream over the Neighborhood for a given maximum number of nodes
    fn neighbors_as_stream(&self, n: NumNodes) -> Self::NeighborhoodStream<'_>;

    /// Returns *true* if `u` is in the Neighborhood
    /// ** Might panic if `u >= n` **
    fn has_neighbor(&self, v: Node) -> bool {
        self.neighbors().any(|u| u == v)
    }

    /// Performs `self.has_neighbor` for a constand number of nodes
    /// ** Might panic if `u >= n` for any `u` in `neighbors` **
    fn has_neighbors<const N: usize>(&self, neighbors: [Node; N]) -> [bool; N] {
        let mut res = [false; N];
        for node in self.neighbors() {
            for i in 0..N {
                if neighbors[i] == node {
                    res[i] = true;
                }
            }
        }
        res
    }

    /// Tries to add a neighbor to the Neighborhood.
    /// Returns *true* if the node was in the Neighborhood before.
    /// ** Might panic if `u >= n` **
    fn try_add_neighbor(&mut self, u: Node) -> bool {
        if self.has_neighbor(u) {
            true
        } else {
            self.add_neighbor(u);
            false
        }
    }

    /// Adds a neighbor to the Neighborhood without checking if this neighbor exists beforehand.
    /// For some implementations, this might lead to Multi-Edges
    fn add_neighbor(&mut self, u: Node);

    /// Tries to remove a neighbor to the Neighborhood.
    /// Returns *true* if the node was in the Neighborhood before.
    /// ** Might panic if `u >= n` **
    fn try_remove_neighbor(&mut self, u: Node) -> bool;

    /// Removes all neighbors that fit a given predicate and returns the number of removed neighbors
    fn remove_neighbors_if<F: FnMut(Node) -> bool>(&mut self, predicate: F) -> NumNodes;

    /// Removes all neighbors in the Neighborhood
    fn clear(&mut self);
}

/// Trait for indexing the Neighborhood
pub trait IndexedNeighborhood: Neighborhood {
    /// Returns the ith neighbor (0-indexed) of a given vertex
    fn ith_neighbor(&self, i: NumNodes) -> Node;
}

/// Trait for accessing the neighborhood of nodes as slices
pub trait NeighborhoodSlice: Neighborhood {
    /// Returns a slice-reference of the neighborhood of a given vertex
    fn as_slice(&self) -> &[Node];
}

/// Trait for mutably accessing the neighborhood of nodes as slices
pub trait NeighborhoodSliceMut: NeighborhoodSlice {
    /// Returns a mutable slice-reference of the neighborhood of a given vertex
    fn as_slice_mut(&mut self) -> &mut [Node];
}

impl<N: NeighborhoodSlice> IndexedNeighborhood for N {
    fn ith_neighbor(&self, i: NumNodes) -> Node {
        self.as_slice()[i as usize]
    }
}

pub(crate) mod macros {
    macro_rules! impl_common_graph_ops {
        ($struct:ident<$first_field:ident : $first_generic:ident $(, $field:ident : $generic:ident)*> => $nbs:ident, $directed:ident) => {
            impl<$first_generic: Neighborhood, $($generic: Neighborhood),*> GraphType for $struct<$first_generic, $($generic),*> {
                type Dir = $directed;
            }

            impl<$first_generic: Neighborhood,$($generic: Neighborhood),*> GraphNodeOrder for $struct<$first_generic, $($generic),*> {
                type VertexIter<'a> = Range<Node>
                where
                    Self: 'a;

                fn vertices(&self) -> Self::VertexIter<'_> {
                    self.vertices_range()
                }

                fn number_of_nodes(&self) -> NumNodes {
                    self.$nbs.len() as NumNodes
                }
            }

            impl<$first_generic: Neighborhood,$($generic: Neighborhood),*> GraphEdgeOrder for $struct<$first_generic, $($generic),*> {
                fn number_of_edges(&self) -> NumEdges {
                    self.num_edges
                }
            }

            impl<$first_generic: Neighborhood,$($generic: Neighborhood),*> AdjacencyList for $struct<$first_generic, $($generic),*> {
                type NeighborIter<'a> = <$first_generic as Neighborhood>::NeighborhoodIter<'a>
                where
                    Self: 'a;

                type ClosedNeighborIter<'a> = std::iter::Chain<std::iter::Once<Node>, Self::NeighborIter<'a>>
                where
                    Self: 'a;

                fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
                    self.$nbs[u as usize].neighbors()
                }

                fn closed_neighbors_of(&self, u: Node) -> Self::ClosedNeighborIter<'_> {
                    std::iter::once(u).chain(self.neighbors_of(u))
                }

                fn degree_of(&self, u: Node) -> NumNodes {
                    self.$nbs[u as usize].num_of_neighbors()
                }

                type NeighborsStream<'a> = <$first_generic as Neighborhood>::NeighborhoodStream<'a>
                where
                    Self: 'a;

                // Redefine to make use of `AdjMatrix` BitSet-Representation
                fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
                    self.$nbs[u as usize].neighbors_as_stream(self.number_of_nodes())
                }
            }

            impl<$first_generic: Neighborhood,$($generic: Neighborhood),*> GraphNew for $struct<$first_generic, $($generic),*> {
                fn new(n: NumNodes) -> Self {
                    assert!(n > 0);
                    Self {
                        num_edges: 0,
                        $first_field: vec![$first_generic::new(n); n as usize],
                        $(
                            $field: vec![$generic::new(n); n as usize],
                        )*
                    }
                }
            }

            impl<$first_generic: NeighborhoodSlice, $($generic: NeighborhoodSlice),*> NeighborsSlice for $struct<$first_generic, $($generic),*> {
                fn as_neighbors_slice(&self, u: Node) -> &[Node] {
                    self.$nbs[u as usize].as_slice()
                }
            }

            impl<$first_generic: NeighborhoodSliceMut,$($generic: NeighborhoodSliceMut),*> NeighborsSliceMut for $struct<$first_generic, $($generic),*> {
                fn as_neighbors_slice_mut(&mut self, u: Node) -> &mut [Node] {
                    self.$nbs[u as usize].as_slice_mut()
                }
            }
        };
    }

    pub(super) use impl_common_graph_ops;

    macro_rules! impl_try_add_edge {
        ($self:ident) => {
            fn try_add_edge(&mut $self, u: Node, v: Node) -> bool {
                if $self.has_edge(u, v) {
                    true
                } else {
                    $self.add_edge(u, v);
                    false
                }
            }
        };
    }

    pub(super) use impl_try_add_edge;
}
