/*!
# Graph Operations

*/
use std::ops::Range;

use itertools::Itertools;
use stream_bitset::prelude::*;

use super::{edge::*, node::*};

/// Possible values for graph directions
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GraphDirection {
    Directed,
    Undirected,
}

/// Marker struct for directed graphs
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Directed;

/// Marker struct for undirected graphs
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Undirected;

/// Marker trait for direction of graphs
pub trait GraphDir {
    const DIRECTION: GraphDirection;
}

impl GraphDir for Directed {
    const DIRECTION: GraphDirection = GraphDirection::Directed;
}

impl GraphDir for Undirected {
    const DIRECTION: GraphDirection = GraphDirection::Undirected;
}

/// Trait for identifying whether a graph is directed/undirected.
/// This should be implemented by *every* graph representation.
pub trait GraphType {
    /// Getter for graph direction.
    /// As `#![feature(associated_const_equality)]` is not stable yet,
    /// this allows for selective implementations of algorithms/generators
    /// that are only meant for directed/undirected graphs.
    type Dir: GraphDir;

    /// Returns *true* if the graph is directed
    #[inline(always)]
    fn is_directed() -> bool {
        Self::Dir::DIRECTION == GraphDirection::Directed
    }

    /// Returns *true* if the graph is undirected
    #[inline(always)]
    fn is_undirected() -> bool {
        Self::Dir::DIRECTION == GraphDirection::Undirected
    }
}

/// Provides getters pertaining to the node-size of a graph
pub trait GraphNodeOrder {
    type VertexIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns the number of nodes of the graph
    fn number_of_nodes(&self) -> NumNodes;

    /// Return the number of nodes as usize
    fn len(&self) -> usize {
        self.number_of_nodes() as usize
    }

    /// Returns an iterator over V.
    fn vertices(&self) -> Self::VertexIter<'_>;

    /// Returns empty bitset with one entry per node
    fn vertex_bitset_unset(&self) -> NodeBitSet {
        NodeBitSet::new(self.number_of_nodes())
    }

    /// Returns full bitset with one entry per node
    fn vertex_bitset_set(&self) -> NodeBitSet {
        NodeBitSet::new_all_set(self.number_of_nodes())
    }

    /// Returns a range of vertices possibly including deleted vertices
    /// In contrast to self.vertices(), the range returned by self.vertices_ranges() does
    /// not borrow self and hence may be used where additional mutable references of self are needed
    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    /// Returns *true* if the graph has no nodes (and thus no edges)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Provides getters pertaining to the edge-size of a graph
pub trait GraphEdgeOrder {
    /// Returns the number of edges of the graph
    fn number_of_edges(&self) -> NumEdges;

    /// Returns empty bitset with one entry per node
    fn edge_bitset_unset(&self) -> EdgeBitSet {
        EdgeBitSet::new(self.number_of_edges())
    }

    /// Returns full bitset with one entry per node
    fn edge_bitset_set(&self) -> EdgeBitSet {
        EdgeBitSet::new_all_set(self.number_of_edges())
    }

    /// Returns *true* if the graph has no edges
    fn is_singleton_graph(&self) -> bool {
        self.number_of_edges() == 0
    }
}

pub struct NodeMapIter<'a, G, T, I>
where
    I: Iterator<Item = Node>,
{
    node_iter: I,
    graph: &'a G,
    map_fn: fn(&'a G, Node) -> T,
}

impl<'a, G, T, I> Iterator for NodeMapIter<'a, G, T, I>
where
    I: Iterator<Item = Node>,
{
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some((self.map_fn)(self.graph, self.node_iter.next()?))
    }
}

macro_rules! node_iterator {
    ($iter : ident, $single : ident, $type : ty) => {
        fn $iter(&self) -> $type {
            NodeMapIter {
                node_iter: self.vertices(),
                graph: self,
                map_fn: Self::$single,
            }
        }
    };
}

macro_rules! node_bitset_of {
    ($bitset : ident, $slice : ident) => {
        fn $bitset(&self, node: Node) -> NodeBitSet {
            NodeBitSet::new_with_bits_set(self.number_of_nodes(), self.$slice(node))
        }
    };
}

pub struct EdgesOfIterImpl<I, const IN_EDGES: bool = false>
where
    I: Iterator<Item = Node>,
{
    iter: I,
    node: Node,
    only_normalized: bool,
}

impl<I, const IN_EDGES: bool> Iterator for EdgesOfIterImpl<I, IN_EDGES>
where
    I: Iterator<Item = Node>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        for u in self.iter.by_ref() {
            if IN_EDGES {
                return Some(Edge(u, self.node));
            } else {
                let edge = Edge(self.node, u);
                if edge.is_normalized() || !self.only_normalized {
                    return Some(edge);
                }
            }
        }

        None
    }
}

pub struct EdgesIterImpl<'a, G, I>
where
    I: Iterator<Item = Edge>,
{
    iter: I,
    graph: &'a G,
    edges_of_fn: fn(&'a G, Node, bool) -> I,
    node_range: Range<Node>,
    only_normalized: bool,
}

impl<'a, G: AdjacencyList, I> Iterator for EdgesIterImpl<'a, G, I>
where
    I: Iterator<Item = Edge>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(edge) = self.iter.next() {
            return Some(edge);
        }

        loop {
            let next_node = self.node_range.next()?;
            self.iter = (self.edges_of_fn)(self.graph, next_node, self.only_normalized);

            if let Some(edge) = self.iter.next() {
                return Some(edge);
            }
        }
    }
}

pub struct VerticesWithNeighborsIterImpl<'a, G>
where
    G: GraphNodeOrder + 'a,
{
    node_iter: <G as GraphNodeOrder>::VertexIter<'a>,
    graph: &'a G,
    degree_fn: fn(&'a G, Node) -> NumNodes,
}

impl<'a, G> Iterator for VerticesWithNeighborsIterImpl<'a, G>
where
    G: GraphNodeOrder + 'a,
{
    type Item = Node;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.node_iter
            .by_ref()
            .find(|&next_node| (self.degree_fn)(self.graph, next_node) > 0)
    }
}

// ---------- Iterator-Types ----------

pub type VerticesWithNeighbors<'a, G> = VerticesWithNeighborsIterImpl<'a, G>;

pub type DegreesIter<'a, G> = NodeMapIter<'a, G, NumNodes, <G as GraphNodeOrder>::VertexIter<'a>>;
pub type NeighborsIter<'a, G> = NodeMapIter<
    'a,
    G,
    <G as AdjacencyList>::NeighborIter<'a>,
    <G as GraphNodeOrder>::VertexIter<'a>,
>;
pub type NeighborsBitSetIter<'a, G> =
    NodeMapIter<'a, G, NodeBitSet, <G as GraphNodeOrder>::VertexIter<'a>>;

pub type EdgesOf<'a, G> = EdgesOfIterImpl<<G as AdjacencyList>::NeighborIter<'a>, false>;
pub type OrderedEdgesOf = std::vec::IntoIter<Edge>;

pub type Edges<'a, G> = EdgesIterImpl<'a, G, EdgesOf<'a, G>>;
pub type OrderedEdges<'a, G> = EdgesIterImpl<'a, G, OrderedEdgesOf>;

/// Traits pertaining getters for neighborhoods & edges
pub trait AdjacencyList: GraphNodeOrder + Sized {
    type NeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    type ClosedNeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns an iterator over the (open) neighborhood of a given vertex.
    /// ** Panics if `u >= n` **
    ///
    /// Note that for directed graphs, this should be equivalent to `out_neighbors_of`
    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_>;

    /// Returns an iterator over the closed neighborhood of a given vertex.
    /// ** Panics if `u >= n` **
    fn closed_neighbors_of(&self, u: Node) -> Self::ClosedNeighborIter<'_>;

    /// If v has degree two (i.e. neighbors [u, w]), this function continues
    /// the walk `u`, `v`, `w` and returns `Some(w)`. Otherwise it returns `None`.
    /// ** Panics if `u >= n || v >= n` **
    fn continue_path(&self, u: Node, v: Node) -> Option<Node> {
        (self.degree_of(v) == 2).then(|| self.neighbors_of(v).find(|&w| w != u).unwrap())
    }

    /// Returns the number of (outgoing) neighbors of `u`
    /// ** Panics if `u >= n` **
    fn degree_of(&self, u: Node) -> NumNodes;

    /// Returns an iterator to all vertices with non-zero degree
    fn vertices_with_neighbors(&self) -> VerticesWithNeighbors<'_, Self> {
        VerticesWithNeighborsIterImpl {
            node_iter: self.vertices(),
            graph: self,
            degree_fn: Self::degree_of,
        }
    }

    /// Returns the number of nodes with non-zero degree
    fn number_of_nodes_with_neighbors(&self) -> NumNodes {
        self.vertices_with_neighbors().count() as NumNodes
    }

    /// Returns a distribution sorted by degree
    fn degree_distribution(&self) -> Vec<(NumNodes, NumNodes)> {
        let mut distr = self
            .degrees()
            .counts()
            .into_iter()
            .map(|(d, n)| (d, n as NumNodes))
            .collect_vec();
        distr.sort_by_key(|(d, _)| *d);
        distr
    }

    /// Returns the maximum degree in the graph
    fn max_degree(&self) -> NumNodes {
        self.degrees().max().unwrap_or(0)
    }

    node_iterator!(degrees, degree_of, DegreesIter<'_, Self>);
    node_iterator!(neighbors, neighbors_of, NeighborsIter<'_, Self>);
    node_bitset_of!(neighbors_of_as_bitset, neighbors_of);
    node_iterator!(
        neighbors_as_bitset,
        neighbors_of_as_bitset,
        NeighborsBitSetIter<'_, Self>
    );

    type NeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;

    /// Returns a BitmaskStream over the neighbors of a given vertex.
    /// ** Panics if `u >= n` **
    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_>;

    /// Returns a NodeBitSet with bit `v` set to *true* if `u`
    /// is at most 2 hops away from a given vertex.
    /// ** Panics if `u >= n` **
    fn closed_two_neighborhood_of(&self, u: Node) -> NodeBitSet {
        let mut ns = self.vertex_bitset_unset();
        ns.set_bit(u);
        for v in self.neighbors_of(u) {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v));
        }
        ns
    }

    /// Returns a NodeBitSet with bit `v` set to *true* if `u`
    /// is at most 3 hops away from a given vertex.
    /// ** Panics if `u >= n` **
    fn closed_three_neighborhood_of(&self, u: Node) -> NodeBitSet {
        let mut ns = self.vertex_bitset_unset();
        ns.set_bit(u);
        for v in self.closed_two_neighborhood_of(u).iter_set_bits() {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v));
        }
        ns
    }

    /// Returns an iterator over outgoing edges of a given vertex.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    /// ** Panics if `u >= n` **
    fn edges_of(&self, u: Node, only_normalized: bool) -> EdgesOf<'_, Self> {
        EdgesOfIterImpl {
            iter: self.neighbors_of(u),
            node: u,
            only_normalized,
        }
    }

    /// Returns an iterator over outgoing edges of a given vertex in sorted order.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    /// ** Panics if `u >= n` **
    fn ordered_edges_of(&self, u: Node, only_normalized: bool) -> OrderedEdgesOf {
        let mut edges = self.edges_of(u, only_normalized).collect_vec();
        edges.sort();
        edges.into_iter()
    }

    /// Returns an iterator over all edges in the graph.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    fn edges(&self, only_normalized: bool) -> Edges<'_, Self> {
        EdgesIterImpl {
            iter: self.edges_of(0, only_normalized),
            graph: self,
            edges_of_fn: Self::edges_of,
            node_range: 1..self.number_of_nodes(),
            only_normalized,
        }
    }

    /// Returns an iterator over all edges in the graph in sorted order.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    fn ordered_edges(&self, only_normalized: bool) -> OrderedEdges<'_, Self> {
        EdgesIterImpl {
            iter: self.ordered_edges_of(0, only_normalized),
            graph: self,
            edges_of_fn: Self::ordered_edges_of,
            node_range: 1..self.number_of_nodes(),
            only_normalized,
        }
    }
}

macro_rules! propagate {
    ($out_fn:ident => $fn:ident($($arg:ident : $type:ty),*) -> $ret:ty) => {
        fn $out_fn(&self, $($arg: $type),*) -> $ret {
            self.$fn($($arg),*)
        }
    };
}

// ---------- Iterator-Types ----------

pub type VerticesWithOutNeighbors<'a, G> = VerticesWithNeighborsIterImpl<'a, G>;
pub type VerticesWithInNeighbors<'a, G> = VerticesWithNeighborsIterImpl<'a, G>;

pub type OutDegreesIter<'a, G> =
    NodeMapIter<'a, G, NumNodes, <G as GraphNodeOrder>::VertexIter<'a>>;
pub type OutNeighborsIter<'a, G> = NodeMapIter<
    'a,
    G,
    <G as AdjacencyList>::NeighborIter<'a>,
    <G as GraphNodeOrder>::VertexIter<'a>,
>;
pub type OutNeighborsBitSetIter<'a, G> =
    NodeMapIter<'a, G, NodeBitSet, <G as GraphNodeOrder>::VertexIter<'a>>;

pub type InDegreesIter<'a, G> = NodeMapIter<'a, G, NumNodes, <G as GraphNodeOrder>::VertexIter<'a>>;
pub type InNeighborsIter<'a, G> = NodeMapIter<
    'a,
    G,
    <G as DirectedAdjacencyList>::InNeighborIter<'a>,
    <G as GraphNodeOrder>::VertexIter<'a>,
>;
pub type InNeighborsBitSetIter<'a, G> =
    NodeMapIter<'a, G, NodeBitSet, <G as GraphNodeOrder>::VertexIter<'a>>;

pub type OutEdgesOf<'a, G> = EdgesOfIterImpl<<G as AdjacencyList>::NeighborIter<'a>, false>;
pub type OrderedOutEdgesOf = std::vec::IntoIter<Edge>;

pub type OutEdges<'a, G> = EdgesIterImpl<'a, G, EdgesOf<'a, G>>;
pub type OrderedOutEdges<'a, G> = EdgesIterImpl<'a, G, OrderedEdgesOf>;

pub type InEdgesOf<'a, G> =
    EdgesOfIterImpl<<G as DirectedAdjacencyList>::InNeighborIter<'a>, false>;
pub type OrderedInEdgesOf = std::vec::IntoIter<Edge>;

/// Extends AdjacencyList for directed graphs
pub trait DirectedAdjacencyList: AdjacencyList + GraphType<Dir = Directed> {
    propagate!(out_neighbors_of => neighbors_of(u : Node) -> Self::NeighborIter<'_>);
    propagate!(out_degree_of => degree_of(u : Node) -> NumNodes);
    propagate!(vertices_with_out_neighbors => vertices_with_neighbors() -> VerticesWithOutNeighbors<'_, Self>);
    propagate!(number_of_nodes_with_out_neighbors => number_of_nodes_with_neighbors() -> NumNodes);
    propagate!(out_degree_distribution => degree_distribution() -> Vec<(NumNodes, NumNodes)>);
    propagate!(max_out_degree => max_degree() -> NumNodes);
    propagate!(out_neighbors_of_as_stream => neighbors_of_as_stream(u: Node) -> Self::NeighborsStream<'_>);

    node_iterator!(out_degrees, out_degree_of, OutDegreesIter<'_, Self>);
    node_iterator!(out_neighbors, out_neighbors_of, OutNeighborsIter<'_, Self>);
    node_bitset_of!(out_neighbors_of_as_bitset, out_neighbors_of);
    node_iterator!(
        out_neighbors_as_bitset,
        out_neighbors_of_as_bitset,
        OutNeighborsBitSetIter<'_, Self>
    );

    fn out_edges_of(&self, u: Node) -> OutEdgesOf<'_, Self> {
        self.edges_of(u, false)
    }

    fn ordered_out_edges_of(&self, u: Node) -> OrderedOutEdgesOf {
        self.ordered_edges_of(u, false)
    }

    type InNeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns an iterator over nodes `v` with edges `(v, u)`
    /// ** Panics if `u >= n` **
    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_>;

    /// If v has degree two (i.e. neighbors [u, w]), this function backtracks
    /// the walk `u`, `v`, `w` and returns `Some(w)`. Otherwise it returns `None`.
    /// ** Panics if `u >= n || v >= n` **
    fn backtrack_path(&self, u: Node, v: Node) -> Option<Node> {
        (self.in_degree_of(v) == 2).then(|| self.in_neighbors_of(v).find(|&w| w != u).unwrap())
    }

    /// Returns the number of incoming neighbors of a given vertex
    /// ** Panics if `u >= n` **
    fn in_degree_of(&self, u: Node) -> NumNodes;

    /// Returns the out-degree and in-degree of a given vertex
    /// ** Panics if `u >= n` **
    fn total_degree_of(&self, u: Node) -> NumNodes {
        self.out_degree_of(u) + self.in_degree_of(u)
    }

    /// Returns an iterator to all vertices with non-zero in-degree
    fn vertices_with_in_neighbors(&self) -> VerticesWithInNeighbors<'_, Self> {
        VerticesWithNeighborsIterImpl {
            node_iter: self.vertices(),
            graph: self,
            degree_fn: Self::in_degree_of,
        }
    }

    /// Returns the number of nodes with non-zero in-degree
    fn number_of_nodes_with_in_neighbors(&self) -> NumNodes {
        self.vertices_with_in_neighbors().count() as NumNodes
    }

    /// Returns a distribution sorted by in-degree
    fn in_degree_distribution(&self) -> Vec<(NumNodes, NumNodes)> {
        let mut distr = self
            .in_degrees()
            .counts()
            .into_iter()
            .map(|(d, n)| (d, n as NumNodes))
            .collect_vec();
        distr.sort_by_key(|(d, _)| *d);
        distr
    }

    /// Returns the maximum in-degree in the graph
    fn max_in_degree(&self) -> NumNodes {
        self.in_degrees().max().unwrap_or(0)
    }

    node_iterator!(in_degrees, in_degree_of, InDegreesIter<'_, Self>);
    node_iterator!(in_neighbors, in_neighbors_of, InNeighborsIter<'_, Self>);
    node_bitset_of!(in_neighbors_of_as_bitset, in_neighbors_of);
    node_iterator!(
        in_neighbors_as_bitset,
        in_neighbors_of_as_bitset,
        InNeighborsBitSetIter<'_, Self>
    );

    type InNeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;

    /// Returns a BitmaskStream over the in-neighbors of a given vertex.
    /// ** Panics if `u >= n` **
    fn in_neighbors_of_as_stream(&self, u: Node) -> Self::InNeighborsStream<'_>;

    /// Returns an iterator over incoming edges of a given vertex.
    /// ** Panics if `u >= n` **
    fn in_edges_of(&self, u: Node) -> InEdgesOf<'_, Self> {
        EdgesOfIterImpl {
            iter: self.in_neighbors_of(u),
            node: u,
            only_normalized: false,
        }
    }

    /// Returns an iterator over incoming edges of a given vertex in sorted order.
    /// ** Panics if `u >= n` **
    fn ordered_in_edges_of(&self, u: Node) -> OrderedInEdgesOf {
        let mut edges = self.in_edges_of(u).collect_vec();
        edges.sort();
        edges.into_iter()
    }
}

/// Trait to test existence of certain structures in a graph.
pub trait AdjacencyTest: GraphNodeOrder {
    /// Returns *true* if the egde (u,v) exists in the graph.
    /// ** Panics if `u >= n || v >= n` **    
    fn has_edge(&self, u: Node, v: Node) -> bool;

    /// Allows multiple edge-queries for a single node
    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        neighbors.map(|v| self.has_edge(u, v))
    }

    /// Returns *true* if a self-loop (u,u) exists.
    /// ** Panics if `u >= n` **
    fn has_self_loop(&self, u: Node) -> bool {
        self.has_edge(u, u)
    }

    /// Returns *true* if a self-loop (u,u) exists at any node u.
    /// ** Panics if `u >= n` **
    fn has_self_loops(&self) -> bool {
        self.vertices().any(|u| self.has_self_loop(u))
    }

    /// Returns *true* if there exists an edge (u,v) as well as (v,u) in the graph.
    /// Note that for undirected graphs with edge {u,v} this function always returns *true*.
    /// ** Panics if `u >= n || v >= n` **
    fn has_bidirected_edge(&self, u: Node, v: Node) -> bool {
        self.has_edge(u, v) && self.has_edge(v, u)
    }
}

pub struct SingletonIter<'a, G>
where
    G: Singletons,
{
    graph: &'a G,
    nodes: <G as GraphNodeOrder>::VertexIter<'a>,
}

impl<'a, G> Iterator for SingletonIter<'a, G>
where
    G: Singletons,
{
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.by_ref().find(|u| !self.graph.is_singleton(*u))
    }
}

/// Trait for checking if nodes in the graph are singletons etc.
pub trait Singletons: GraphNodeOrder + Sized {
    /// Returns *true* if `u` is a singleton, ie.
    /// - for undirected graphs, `degree_of(u) = 0`
    /// - for directed graphs, `total_degree_of(u) = 0`
    fn is_singleton(&self, u: Node) -> bool;

    fn vertices_no_singletons(&self) -> SingletonIter<'_, Self> {
        SingletonIter {
            graph: self,
            nodes: self.vertices(),
        }
    }
}

impl<G> Singletons for G
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    #[inline]
    fn is_singleton(&self, u: Node) -> bool {
        self.degree_of(u) == 0
    }
}

/// Trait for indexed access of Neighborhoods
pub trait IndexedAdjacencyList: AdjacencyList {
    /// Returns the ith neighbor (0-indexed) of a given vertex
    /// ** Panics if `u >= n || i >= deg(u)` **
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node;
}

/// Trait for swapping the order in Neighborhoods
pub trait IndexedAdjacencySwap: IndexedAdjacencyList {
    /// Swaps the ith and the jth neighbor of a given vertex
    /// ** Panics if `u >= n || i >= deg(u) || j >= deg(u)` **
    fn swap_neighbors(&mut self, u: Node, i: NumNodes, j: NumNodes);
}

/// Trait for accessing the neighborhood of nodes as slices
pub trait NeighborsSlice {
    /// Returns a slice-reference of the neighborhood of a given vertex
    fn as_neighbors_slice(&self, u: Node) -> &[Node];
}

/// Trait for mutably accessing the neighborhood of nodes as slices
pub trait NeighborsSliceMut: NeighborsSlice {
    /// Returns a mutable slice-reference of the neighborhood of a given vertex
    fn as_neighbors_slice_mut(&mut self, u: Node) -> &mut [Node];
}

impl<G> IndexedAdjacencyList for G
where
    G: NeighborsSlice + AdjacencyList,
{
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node {
        self.as_neighbors_slice(u)[i as usize]
    }
}

impl<G> IndexedAdjacencySwap for G
where
    G: NeighborsSliceMut + AdjacencyList,
{
    fn swap_neighbors(&mut self, u: Node, i: NumNodes, j: NumNodes) {
        self.as_neighbors_slice_mut(u).swap(i as usize, j as usize);
    }
}

/// Trait for creating a new empty graph
pub trait GraphNew {
    /// Creates an empty graph with n singleton nodes.
    /// ** Panics if `n = 0` **
    fn new(n: NumNodes) -> Self;
}

/// Provides functions to insert/delete edges
pub trait GraphEdgeEditing: GraphNew {
    /// Adds the edge *(u,v)* to the graph.
    /// ** Panics if `u >= n || v >= n` or the edge was already present **
    fn add_edge(&mut self, u: Node, v: Node) {
        assert!(!self.try_add_edge(u, v))
    }

    /// Adds the edge `(u, v)` to the graph.
    /// Returns *true* exactly if the edge was present previously.
    /// ** Panics if `u >= n || v >= n` **
    fn try_add_edge(&mut self, u: Node, v: Node) -> bool;

    /// Adds all edges in the collection
    fn add_edges<I, E>(&mut self, edges: I)
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        for Edge(u, v) in edges.into_iter().map(|d| d.into()) {
            self.add_edge(u, v);
        }
    }

    /// Tries to add all edges `(u, v)` of the collection to the graph.
    /// Returns the number of successfully added edges.
    /// ** Panics if `u >= n || v >= n`  for any `(u, v)` in `edges` **
    fn try_add_edges<I, E>(&mut self, edges: I) -> NumEdges
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        edges
            .into_iter()
            .map(|e| {
                let Edge(u, v) = e.into();
                self.try_add_edge(u, v) as NumEdges
            })
            .sum()
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is not present or u, v >= n **
    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v));
    }

    /// Removes all edges in the collection
    /// ** Panics if the any edge (u, v) in `edges` is not present or u, v >= n **
    fn remove_edges<I, E>(&mut self, edges: I)
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        for Edge(u, v) in edges.into_iter().map(|d| d.into()) {
            self.remove_edge(u, v);
        }
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// Returns *true* exactly if the edge was present previously.
    /// ** Panics if u, v >= n **
    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool;
}

/// Trait extending the methods of the GraphEdgeEditing trait for directed graphs.
pub trait GraphDirectedEdgeEditing: GraphEdgeEditing + GraphType<Dir = Directed> {
    /// Removes all directed edges (v,u) where v is any predecessor of u in the graph.
    /// ** Panics if `u >= n` **
    fn remove_edges_into_node(&mut self, u: Node);

    /// Remove all directed edges (u,v) where v is any successor of u in the graph.
    /// ** Panics if `u >= n` **
    fn remove_edges_out_of_node(&mut self, u: Node);
}

/// Trait extending the methods of the GraphEdgeEditing trait for directed as well as undirected graphs.
pub trait GraphLocalEdgeEditing: GraphEdgeEditing {
    /// Removes all edges adjacent to node u in the graph.
    /// ** Panics if `u >= n` **
    fn remove_edges_at_node(&mut self, u: Node);

    /// Removes all edges adjacent to any node u in an iterator in the graph.
    /// ** Panics if any node in `nodes` is `>= n` **
    fn remove_edges_at_nodes<I>(&mut self, nodes: I)
    where
        I: IntoIterator<Item = Node>,
    {
        for node in nodes {
            self.remove_edges_at_node(node);
        }
    }
}

/// A super trait for creating a graph from scratch from a set of edges and a number of nodes
pub trait GraphFromScratch {
    /// Create a graph from a number of nodes and an iterator over Edges
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>;

    /// Create a graph from a number of nodes and an iterator over Edges, adding edges via
    /// `graph.try_add_edge` instead of `graph.add_edge`
    fn from_try_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>;
}

impl<G> GraphFromScratch for G
where
    G: GraphNew + GraphEdgeEditing,
{
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        let mut graph = Self::new(n);
        graph.add_edges(edges);
        graph
    }

    fn from_try_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        let mut graph = Self::new(n);
        graph.try_add_edges(edges);
        graph
    }
}
