use std::{iter::Copied, slice::Iter};

use itertools::Itertools;
use smallvec::{Array, SmallVec};
use stream_bitset::prelude::{
    BitmaskSliceStream, BitmaskStreamConsumer, BitmaskStreamToIndices, BitsetStream,
    IntoBitmaskStream, ToBitmaskStream,
};

use super::*;

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

    pub(crate) use impl_common_graph_ops;

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

    pub(crate) use impl_try_add_edge;
}

/// Basic Neighborhood-Impl. using `Vec<Node>`
#[derive(Default, Clone)]
pub struct ArrNeighborhood(pub Vec<Node>);

impl Neighborhood for ArrNeighborhood {
    fn new(_n: NumNodes) -> Self {
        Self(Default::default())
    }

    fn num_of_neighbors(&self) -> NumNodes {
        self.0.len() as NumNodes
    }

    type NeighborhoodIter<'a>
        = Copied<Iter<'a, Node>>
    where
        Self: 'a;

    fn neighbors(&self) -> Self::NeighborhoodIter<'_> {
        self.0.iter().copied()
    }

    type NeighborhoodStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;

    fn neighbors_as_stream(&self, n: NumNodes) -> Self::NeighborhoodStream<'_> {
        NodeBitSet::new_with_bits_set(n, self.neighbors()).into_bitmask_stream()
    }

    fn add_neighbor(&mut self, u: Node) {
        self.0.push(u);
    }

    fn try_remove_neighbor(&mut self, u: Node) -> bool {
        if let Some((pos, _)) = self.0.iter().find_position(|&&x| x == u) {
            self.0.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn remove_neighbors_if<F>(&mut self, mut predicate: F) -> NumNodes
    where
        F: FnMut(Node) -> bool,
    {
        let size_before = self.0.len();
        self.0.retain(|x| !predicate(*x));
        (size_before - self.0.len()) as NumNodes
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

impl NeighborhoodSlice for ArrNeighborhood {
    fn as_slice(&self) -> &[Node] {
        &self.0
    }
}

impl NeighborhoodSliceMut for ArrNeighborhood {
    fn as_slice_mut(&mut self) -> &mut [Node] {
        &mut self.0
    }
}

/// Like `NeighborhoodArray` but uses `SmallVec<[Node; N]>` instead.
/// Prefer this if the graph is known to be sparse.
#[derive(Default, Clone)]
pub struct SparseNeighborhood<const N: usize = 8>(pub SmallVec<[Node; N]>)
where
    [Node; N]: Array<Item = Node>;

impl<const N: usize> Neighborhood for SparseNeighborhood<N>
where
    [Node; N]: Array<Item = Node>,
{
    fn new(_n: NumNodes) -> Self {
        Self(Default::default())
    }

    fn num_of_neighbors(&self) -> NumNodes {
        self.0.len() as NumNodes
    }

    type NeighborhoodIter<'a>
        = Copied<Iter<'a, Node>>
    where
        Self: 'a;

    fn neighbors(&self) -> Self::NeighborhoodIter<'_> {
        self.0.iter().copied()
    }

    type NeighborhoodStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;

    fn neighbors_as_stream(&self, n: NumNodes) -> Self::NeighborhoodStream<'_> {
        NodeBitSet::new_with_bits_set(n, self.neighbors()).into_bitmask_stream()
    }

    fn add_neighbor(&mut self, u: Node) {
        self.0.push(u);
    }

    fn try_remove_neighbor(&mut self, u: Node) -> bool {
        if let Some((pos, _)) = self.0.iter().find_position(|&&x| x == u) {
            self.0.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn remove_neighbors_if<F>(&mut self, mut predicate: F) -> NumNodes
    where
        F: FnMut(Node) -> bool,
    {
        let size_before = self.0.len();
        self.0.retain(|x| !predicate(*x));
        (size_before - self.0.len()) as NumNodes
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

impl<const N: usize> NeighborhoodSlice for SparseNeighborhood<N>
where
    [Node; N]: Array<Item = Node>,
{
    fn as_slice(&self) -> &[Node] {
        &self.0
    }
}

impl<const N: usize> NeighborhoodSliceMut for SparseNeighborhood<N>
where
    [Node; N]: Array<Item = Node>,
{
    fn as_slice_mut(&mut self) -> &mut [Node] {
        &mut self.0
    }
}

/// A Neighborhood represented by a NodeBitSet
#[derive(Default, Clone)]
pub struct BitNeighborhood(pub NodeBitSet);

impl Neighborhood for BitNeighborhood {
    fn new(n: NumNodes) -> Self {
        Self(NodeBitSet::new(n))
    }

    fn num_of_neighbors(&self) -> NumNodes {
        self.0.cardinality()
    }

    type NeighborhoodIter<'a>
        = BitmaskStreamToIndices<BitmaskSliceStream<'a>, Node, true>
    where
        Self: 'a;

    fn neighbors(&self) -> Self::NeighborhoodIter<'_> {
        // `self.0.iter_set_bits()` is a wrapper with an opaque type so it does not work here
        self.0.bitmask_stream().iter_set_bits()
    }

    type NeighborhoodStream<'a>
        = BitmaskSliceStream<'a>
    where
        Self: 'a;

    fn neighbors_as_stream(&self, _: NumNodes) -> Self::NeighborhoodStream<'_> {
        self.0.bitmask_stream()
    }

    fn has_neighbor(&self, u: Node) -> bool {
        self.0.get_bit(u)
    }

    fn has_neighbors<const N: usize>(&self, neighbors: [Node; N]) -> [bool; N] {
        neighbors.map(|u| self.0.get_bit(u))
    }

    fn try_add_neighbor(&mut self, u: Node) -> bool {
        self.0.set_bit(u)
    }

    fn add_neighbor(&mut self, u: Node) {
        self.0.set_bit(u);
    }

    fn try_remove_neighbor(&mut self, u: Node) -> bool {
        self.0.clear_bit(u)
    }

    fn remove_neighbors_if<F>(&mut self, predicate: F) -> NumNodes
    where
        F: FnMut(Node) -> bool,
    {
        let prev_cardinality = self.num_of_neighbors();
        self.0.update_set_bits(predicate);
        prev_cardinality - self.num_of_neighbors()
    }

    fn clear(&mut self) {
        self.0.clear_all();
    }
}
