#[cfg(feature = "node_range")]
use std::ops::Range;

use itertools::Itertools;
use stream_bitset::prelude::*;

use crate::*;

/// Provides getters pertaining to the node-size of a graph
pub trait GraphNodeOrder {
    /// Returns the number of nodes of the graph
    fn number_of_nodes(&self) -> NumNodes;

    /// Returns the number of nodes as Node
    fn len_as_node(&self) -> Node {
        Node::new(self.number_of_nodes())
    }

    /// Return the number of nodes as usize
    fn len(&self) -> usize {
        self.number_of_nodes() as usize
    }

    /// Returns an iterator over V.
    fn vertices(&self) -> impl Iterator<Item = Node> + '_;

    /// Returns empty bitset with one entry per node
    fn vertex_bitset_unset(&self) -> NodeBitSet {
        NodeBitSet::new(self.len_as_node())
    }

    /// Returns full bitset with one entry per node
    fn vertex_bitset_set(&self) -> NodeBitSet {
        NodeBitSet::new_all_set(self.len_as_node())
    }

    /// Returns a range of vertices possibly including deleted vertices
    /// In contrast to self.vertices(), the range returned by self.vertices_ranges() does
    /// not borrow self and hence may be used where additional mutable references of self are needed
    ///
    /// # Warning
    /// This method may iterate over deleted vertices (if supported by an implementation). It is the
    /// responsibility of the caller to identify and treat them accordingly.
    #[cfg(feature = "node_range")]
    fn vertices_range(&self) -> Range<Node> {
        Node::MIN..Node::new(self.number_of_nodes())
    }

    #[cfg(not(feature = "node_range"))]
    fn vertices_range(&self) -> NodeRange {
        NodeRange::new_to(self.len_as_node())
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
    fn is_singleton(&self) -> bool {
        self.number_of_edges() == 0
    }
}

macro_rules! node_iterator {
    ($iter : ident, $single : ident, $type : ty) => {
        fn $iter(&self) -> impl Iterator<Item = $type> + '_ {
            self.vertices().map(|u| self.$single(u))
        }
    };
}

macro_rules! node_bitset_of {
    ($bitset : ident, $slice : ident) => {
        fn $bitset(&self, node: Node) -> NodeBitSet {
            NodeBitSet::new_with_bits_set(self.len_as_node(), self.$slice(node))
        }
    };
}

/// Traits pertaining getters for neighborhoods & edges
pub trait AdjacencyList: GraphNodeOrder + Sized {
    /// Returns an iterator over the (open) neighborhood of a given vertex.
    /// ** Panics if `u >= n` **
    ///
    /// Note that for directed graphs, this should be equivalent to `out_neighbors_of`
    fn neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_;

    /// Returns an iterator over the closed neighborhood of a given vertex.
    /// ** Panics if `u >= n` **
    fn closed_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
        std::iter::once(u).chain(self.neighbors_of(u))
    }

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
    fn vertices_with_neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.degrees()
            .enumerate()
            .filter_map(|(u, d)| (d > 0).then_some(Node::new(u as RawNode)))
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

    node_iterator!(degrees, degree_of, NumNodes);
    node_iterator!(neighbors, neighbors_of, impl Iterator<Item = Node> + '_);
    node_bitset_of!(neighbors_of_as_bitset, neighbors_of);
    node_iterator!(neighbors_as_bitset, neighbors_of_as_bitset, NodeBitSet);

    /// Returns a BitmaskStream over the neighbors of a given vertex.
    /// ** Panics if `u >= n` **
    fn neighbors_of_as_stream(&self, u: Node) -> impl BitmaskStream + '_;

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
    fn edges_of(&self, u: Node, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.neighbors_of(u)
            .map(move |v| Edge(u, v))
            .filter(move |e| !only_normalized || e.is_normalized())
    }

    /// Returns an iterator over outgoing edges of a given vertex in sorted order.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    /// ** Panics if `u >= n` **
    fn ordered_edges_of(&self, u: Node, only_normalized: bool) -> impl Iterator<Item = Edge> {
        let mut edges = self.edges_of(u, only_normalized).collect_vec();
        edges.sort();
        edges.into_iter()
    }

    /// Returns an iterator over all edges in the graph.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    fn edges(&self, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.vertices_range()
            .flat_map(move |u| self.edges_of(u, only_normalized))
    }

    /// Returns an iterator over all edges in the graph in sorted order.
    /// If `only_normalized`, then only edges `(u, v)` with `u <= v` are considered.
    fn ordered_edges(&self, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.vertices_range()
            .flat_map(move |u| self.ordered_edges_of(u, only_normalized))
    }
}

macro_rules! propagate {
    ($out_fn:ident => $fn:ident($($arg:ident : $type:ty),*) -> $ret:ty) => {
        #[inline]
        fn $out_fn(&self, $($arg: $type),*) -> $ret {
            self.$fn($($arg),*)
        }
    };
}

pub trait DirectedAdjacencyList: AdjacencyList {
    propagate!(out_neighbors_of => neighbors_of(u : Node) -> impl Iterator<Item = Node> + '_);
    propagate!(out_degree_of => degree_of(u : Node) -> NumNodes);
    propagate!(vertices_with_out_neighbors => vertices_with_neighbors() -> impl Iterator<Item = Node> + '_);
    propagate!(number_of_nodes_with_out_neighbors => number_of_nodes_with_neighbors() -> NumNodes);
    propagate!(out_degree_distribution => degree_distribution() -> Vec<(NumNodes, NumNodes)>);
    propagate!(max_out_degree => max_degree() -> NumNodes);
    propagate!(out_neighbors_of_as_stream => neighbors_of_as_stream(u: Node) -> impl BitmaskStream + '_);

    node_iterator!(out_degrees, out_degree_of, NumNodes);
    node_iterator!(
        out_neighbors,
        out_neighbors_of,
        impl Iterator<Item = Node> + '_
    );
    node_bitset_of!(out_neighbors_of_as_bitset, out_neighbors_of);
    node_iterator!(
        out_neighbors_as_bitset,
        out_neighbors_of_as_bitset,
        NodeBitSet
    );

    #[inline]
    fn out_edges_of(&self, u: Node) -> impl Iterator<Item = Edge> + '_ {
        self.edges_of(u, false)
    }

    #[inline]
    fn ordered_out_edges_of(&self, u: Node) -> impl Iterator<Item = Edge> + '_ {
        self.ordered_edges_of(u, false)
    }

    /// Returns an iterator over nodes `v` with edges `(v, u)`
    /// ** Panics if `u >= n` **
    fn in_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_;

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
    #[inline]
    fn total_degree_of(&self, u: Node) -> NumNodes {
        self.out_degree_of(u) + self.in_degree_of(u)
    }

    /// Returns an iterator to all vertices with non-zero in-degree
    fn vertices_with_in_neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.in_degrees()
            .enumerate()
            .filter_map(|(u, d)| (d > 0).then_some(Node::new(u as RawNode)))
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

    node_iterator!(in_degrees, in_degree_of, NumNodes);
    node_iterator!(
        in_neighbors,
        in_neighbors_of,
        impl Iterator<Item = Node> + '_
    );
    node_bitset_of!(in_neighbors_of_as_bitset, in_neighbors_of);
    node_iterator!(
        in_neighbors_as_bitset,
        in_neighbors_of_as_bitset,
        NodeBitSet
    );

    /// Returns a BitmaskStream over the in-neighbors of a given vertex.
    /// ** Panics if `u >= n` **
    fn in_neighbors_of_as_stream(&self, u: Node) -> impl BitmaskStream + '_;

    /// Returns an iterator over incoming edges of a given vertex.
    /// ** Panics if `u >= n` **
    fn in_edges_of(&self, u: Node) -> impl Iterator<Item = Edge> + '_ {
        self.in_neighbors_of(u).map(move |v| Edge(u, v))
    }

    /// Returns an iterator over incoming edges of a given vertex in sorted order.
    /// ** Panics if `u >= n` **
    fn ordered_in_edges_of(&self, u: Node) -> impl Iterator<Item = Edge> {
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

    /// Returns *true* if there exists an edge (u,v) as well as (v,u) in the graph.
    /// Note that for undirected graphs with edge {u,v} this function always returns *true*.
    /// ** Panics if `u >= n || v >= n` **
    fn has_bidirected_edge(&self, u: Node, v: Node) -> bool {
        self.has_edge(u, v) && self.has_edge(v, u)
    }
}

pub trait IndexedAdjacencyList: AdjacencyList {
    /// Returns the ith neighbor (0-indexed) of a given vertex
    /// ** Panics if `u >= n || i >= deg(u)`
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node;
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

impl<G: NeighborsSlice + AdjacencyList> IndexedAdjacencyList for G {
    #[inline]
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node {
        self.as_neighbors_slice(u)[i as usize]
    }
}

/// Trait for creating a new empty graph
pub trait GraphNew {
    /// Creates an empty graph with n singleton nodes
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
    /// Returns *true* exactly if the edge was not present previously.
    /// ** Panics if `u >= n || v >= n` **
    fn try_add_edge(&mut self, u: Node, v: Node) -> bool;

    /// Adds all edges in the collection
    fn add_edges(&mut self, edges: impl Iterator<Item = impl Into<Edge>>) {
        for Edge(u, v) in edges.map(|d| d.into()) {
            self.add_edge(u, v);
        }
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is not present or u, v >= n **
    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v));
    }

    /// Removes all edges in the collection
    /// ** Panics if the any edge (u, v) in `edges` is not present or u, v >= n **
    fn remove_edges(&mut self, edges: impl Iterator<Item = impl Into<Edge>>) {
        for Edge(u, v) in edges.map(|d| d.into()) {
            self.remove_edge(u, v);
        }
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// If the edge was removed, returns *true* and *false* otherwise.
    /// ** Panics if u, v >= n **
    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool;
}

/// Trait extending the methods of the GraphEdgeEditing trait for directed graphs.
pub trait GraphDirectedEdgeEditing: GraphEdgeEditing {
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
    fn remove_edges_at_nodes<I: Iterator<Item = Node>>(&mut self, nodes: I) {
        for node in nodes {
            self.remove_edges_at_node(node);
        }
    }
}

/// A super trait for creating a graph from scratch from a set of edges and a number of nodes
pub trait GraphFromScratch {
    /// Create a graph from a number of nodes and an iterator over Edges
    fn from_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self;
}

impl<G: GraphNew + GraphEdgeEditing> GraphFromScratch for G {
    fn from_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        let mut graph = Self::new(n);
        graph.add_edges(edges);
        graph
    }
}
