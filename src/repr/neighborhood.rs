use itertools::Itertools;
use smallvec::{Array, SmallVec};
use stream_bitset::prelude::{BitmaskStream, IntoBitmaskStream};

use super::*;

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

    fn neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.0.iter().copied()
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

    fn remove_neighbors_if<F: FnMut(Node) -> bool>(&mut self, mut predicate: F) -> NumNodes {
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

    fn neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.0.iter().copied()
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

    fn remove_neighbors_if<F: FnMut(Node) -> bool>(&mut self, mut predicate: F) -> NumNodes {
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

    fn neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.0.iter_set_bits()
    }

    fn neighbors_as_stream(&self, _n: NumNodes) -> impl BitmaskStream + '_ {
        self.0.clone().into_bitmask_stream()
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

    fn remove_neighbors_if<F: FnMut(Node) -> bool>(&mut self, predicate: F) -> NumNodes {
        let prev_cardinality = self.num_of_neighbors();
        self.0.update_set_bits(predicate);
        prev_cardinality - self.num_of_neighbors()
    }

    fn clear(&mut self) {
        self.0.clear_all();
    }
}
