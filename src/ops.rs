/*!
# Graph Operations

Core graph traits and operations.

This module defines the **fundamental traits** that all graph
representations in `ugraphs` should implement (if possible).
It covers:
- **Graph type metadata** ([`GraphType`], [`GraphDir`], [`GraphDirection`]).
- **Node and edge counts** ([`GraphNodeOrder`], [`GraphEdgeOrder`]).
- **Neighborhood access** ([`AdjacencyList`], [`DirectedAdjacencyList`]).
- **Edge testing and editing** ([`AdjacencyTest`], [`GraphEdgeEditing`], etc.).

These traits form the backbone for algorithms in `ugraphs` to work across
multiple graph representations (static or dynamic, directed or undirected).

# Examples
```
use ugraphs::prelude::*;

// Build a simple undirected triangle graph
let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2), (2,0)]);

assert_eq!(g.number_of_nodes(), 3);
assert_eq!(g.number_of_edges(), 3);
assert!(g.has_edge(0,1));
assert!(g.has_edge(1,0)); // undirected
```
*/

use std::ops::Range;

use itertools::Itertools;
use stream_bitset::prelude::*;

use super::{edge::*, node::*};

/// Whether a graph is `Directed` or `Undirected`.
///
/// Used by [`GraphType`] to specialize behavior.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GraphDirection {
    Directed,
    Undirected,
}

/// Marker type representing a directed graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Directed;

/// Marker type representing an undirected graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Undirected;

/// Trait implemented by [`Directed`] and [`Undirected`].
///
/// Provides a compile-time constant [`GraphDirection`].
pub trait GraphDir {
    const DIRECTION: GraphDirection;
}

impl GraphDir for Directed {
    const DIRECTION: GraphDirection = GraphDirection::Directed;
}

impl GraphDir for Undirected {
    const DIRECTION: GraphDirection = GraphDirection::Undirected;
}

/// Identifies whether a graph is directed or undirected.
///
/// Every graph representation **must implement this trait**.
///
/// Algorithms can use this to specialize for only directed/undirected graphs.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// type G = AdjArrayUndir;
/// assert!(G::is_undirected());
/// assert!(!G::is_directed());
/// ```
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

/// Provides accessors related to the number of nodes.
///
/// Implemented by all graph representations.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// assert_eq!(g.number_of_nodes(), 3);
/// assert_eq!(g.len(), 3);
/// assert!(!g.is_empty());
/// assert_eq!(g.vertices().collect::<Vec<_>>(), vec![0,1,2]);
/// ```
pub trait GraphNodeOrder {
    /// Iterator over all nodes in the graph.
    ///
    /// Returned by [`GraphNodeOrder::vertices`].
    type VertexIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns the number of nodes in the graph.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.number_of_nodes(), 3);
    /// ```
    fn number_of_nodes(&self) -> NumNodes;

    /// Returns the number of nodes as a `usize`.
    ///
    /// Equivalent to `number_of_nodes()` but as a `usize`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.len(), 3);
    /// ```
    fn len(&self) -> usize {
        self.number_of_nodes() as usize
    }

    /// Returns an iterator over all nodes in the graph.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let nodes: Vec<_> = g.vertices().collect();
    /// assert_eq!(nodes, vec![0,1,2]);
    /// ```
    fn vertices(&self) -> Self::VertexIter<'_>;

    /// Returns an empty bitset with one entry per node.
    ///
    /// Useful for marking or filtering nodes.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let bs = g.vertex_bitset_unset();
    /// assert_eq!(bs.number_of_bits(), 3);
    /// assert!(bs.iter_set_bits().next().is_none());
    /// ```
    fn vertex_bitset_unset(&self) -> NodeBitSet {
        NodeBitSet::new(self.number_of_nodes())
    }

    /// Returns a bitset with all bits set, one per node.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let bs = g.vertex_bitset_set();
    /// assert_eq!(bs.iter_set_bits().collect::<Vec<_>>(), vec![0,1,2]);
    /// ```
    fn vertex_bitset_set(&self) -> NodeBitSet {
        NodeBitSet::new_all_set(self.number_of_nodes())
    }

    /// Returns a range of all nodes, possibly including deleted nodes (not implemented yet).
    ///
    /// Unlike `vertices()`, this does not borrow `self` and can be used
    /// where additional mutable references are needed.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let range: Vec<_> = g.vertices_range().collect();
    /// assert_eq!(range, vec![0,1,2]);
    /// ```
    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    /// Returns `true` if the graph has no nodes (and therefore no edges).
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::new(0);
    /// assert!(g.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Provides accessors related to the number of edges.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// assert_eq!(g.number_of_edges(), 2);
/// assert!(!g.is_singleton_graph());
/// ```
pub trait GraphEdgeOrder {
    /// Returns the number of edges in the graph.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.number_of_edges(), 2);
    /// ```
    fn number_of_edges(&self) -> NumEdges;

    /// Returns an empty bitset with one entry per edge.
    ///
    /// Useful for marking or filtering edges.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let bs = g.edge_bitset_unset();
    /// assert_eq!(bs.number_of_bits(), 2);
    /// assert!(bs.iter_set_bits().next().is_none());
    /// ```
    fn edge_bitset_unset(&self) -> EdgeBitSet {
        EdgeBitSet::new(self.number_of_edges())
    }

    /// Returns a bitset with all bits set, one per edge.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let bs = g.edge_bitset_set();
    /// assert_eq!(bs.iter_set_bits().collect::<Vec<_>>(), vec![0,1]);
    /// ```
    fn edge_bitset_set(&self) -> EdgeBitSet {
        EdgeBitSet::new_all_set(self.number_of_edges())
    }

    /// Returns `true` if the graph has no edges.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::new(3);
    /// assert!(g.is_singleton_graph());
    /// ```
    fn is_singleton_graph(&self) -> bool {
        self.number_of_edges() == 0
    }
}

/// Generic iterator that maps each node to some value `T`.
///
/// This is a helper iterator used internally by node-based traversal
/// methods (e.g. [`AdjacencyList::degrees`], [`AdjacencyList::neighbors`],
/// [`AdjacencyList::neighbors_as_bitset`]).
///
/// Users will typically encounter it via its type aliases:
/// [`DegreesIter`], [`NeighborsIter`], [`NeighborsBitSetIter`].
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
    ($iter : ident, $single : ident, $type : ty, $($doc:tt)*) => {
        $($doc)*
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
    ($bitset : ident, $slice : ident, $($doc:tt)*) => {
        $($doc)*
        fn $bitset(&self, node: Node) -> NodeBitSet {
            NodeBitSet::new_with_bits_set(self.number_of_nodes(), self.$slice(node))
        }
    };
}

/// Iterator over the edges adjacent to a single node.
///
/// Parameterized by whether it produces out-edges or in-edges (via `IN_EDGES`).
/// Used internally by [`AdjacencyList::edges_of`] and
/// [`DirectedAdjacencyList::in_edges_of`].
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

/// Iterator over all edges in a graph.
///
/// This drives [`AdjacencyList::edges`] and [`AdjacencyList::ordered_edges`],
/// traversing the adjacency lists of all nodes in sequence.
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

/// Iterator over nodes that have at least one neighbor (i.e. non-isolated nodes).
///
/// Used internally by [`AdjacencyList::vertices_with_neighbors`].
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

/// Iterator over nodes with neighbors (non-isolated).
///
/// Returned by [`AdjacencyList::vertices_with_neighbors`].
pub type VerticesWithNeighbors<'a, G> = VerticesWithNeighborsIterImpl<'a, G>;

/// Iterator over the degrees of all nodes in a graph.
///
/// Returned by [`AdjacencyList::degrees`].
pub type DegreesIter<'a, G> = NodeMapIter<'a, G, NumNodes, <G as GraphNodeOrder>::VertexIter<'a>>;

/// Iterator over the neighbor lists of all nodes in a graph.
///
/// Returned by [`AdjacencyList::neighbors`].
pub type NeighborsIter<'a, G> = NodeMapIter<
    'a,
    G,
    <G as AdjacencyList>::NeighborIter<'a>,
    <G as GraphNodeOrder>::VertexIter<'a>,
>;

/// Iterator over the neighbor sets of all nodes as bitsets.
///
/// Returned by [`AdjacencyList::neighbors_as_bitset`].
pub type NeighborsBitSetIter<'a, G> =
    NodeMapIter<'a, G, NodeBitSet, <G as GraphNodeOrder>::VertexIter<'a>>;

/// Iterator over the edges adjacent to a given node.
///
/// Returned by [`AdjacencyList::edges_of`].
pub type EdgesOf<'a, G> = EdgesOfIterImpl<<G as AdjacencyList>::NeighborIter<'a>, false>;

/// Iterator over edges of a node, in deterministic order.
///
/// Returned by [`AdjacencyList::ordered_edges_of`].
pub type OrderedEdgesOf = std::vec::IntoIter<Edge>;

/// Iterator over all edges in a graph.
///
/// Returned by [`AdjacencyList::edges`].
pub type Edges<'a, G> = EdgesIterImpl<'a, G, EdgesOf<'a, G>>;

/// Iterator over all edges in a graph, in deterministic order.
///
/// Returned by [`AdjacencyList::ordered_edges`].
pub type OrderedEdges<'a, G> = EdgesIterImpl<'a, G, OrderedEdgesOf>;

/// Trait providing access to neighborhoods and edges.
///
/// Implemented by both directed and undirected graphs.
/// For directed graphs, `neighbors_of(u)` should correspond to outgoing neighbors.
///
/// Many algorithms rely on this trait for traversals.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
///
/// assert_eq!(g.degree_of(1), 2);
/// assert_eq!(g.neighbors_of(1).collect::<Vec<_>>(), vec![0,2]);
///
/// let edges: Vec<_> = g.edges(true).collect();
/// assert_eq!(edges.len(), 2); // normalized edges only
/// ```
pub trait AdjacencyList: GraphNodeOrder + Sized {
    /// Iterator over all neighbors in the open neighborhood of a vertex in the graph.
    ///
    /// Returned by [`AdjacencyList::neighbors_of`].
    type NeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Iterator over all neighbors in the closed neighborhood of a vertex in the graph.
    ///
    /// Returned by [`AdjacencyList::closed_neighbors_of`].
    type ClosedNeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns an iterator over the (open) neighborhood of a given vertex.
    ///
    /// **Panics if `u >= n`.**
    ///
    /// Note: For directed graphs, this is equivalent to `out_neighbors_of`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let neighbors: Vec<_> = g.neighbors_of(1).collect();
    /// assert_eq!(neighbors, vec![0,2]);
    /// ```
    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_>;

    /// Returns an iterator over the closed neighborhood of a given vertex,
    /// including the vertex itself.
    ///
    /// **Panics if `u >= n`.**
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let mut closed_neighbors: Vec<_> = g.closed_neighbors_of(1).collect();
    /// closed_neighbors.sort_unstable();
    /// assert_eq!(closed_neighbors, vec![0,1,2]);
    /// ```
    fn closed_neighbors_of(&self, u: Node) -> Self::ClosedNeighborIter<'_>;

    /// Returns the number of neighbors (degree) of a vertex.
    ///
    ///  **Panics if `u >= n`.**
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.degree_of(1), 2);
    /// ```
    fn degree_of(&self, u: Node) -> NumNodes;

    /// Returns an iterator over vertices with non-zero degree.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1)]);
    /// let nodes: Vec<_> = g.vertices_with_neighbors().collect();
    /// assert_eq!(nodes, vec![0,1]);
    /// ```
    fn vertices_with_neighbors(&self) -> VerticesWithNeighbors<'_, Self> {
        VerticesWithNeighborsIterImpl {
            node_iter: self.vertices(),
            graph: self,
            degree_fn: Self::degree_of,
        }
    }

    /// Returns the number of vertices with at least one neighbor.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1)]);
    /// assert_eq!(g.number_of_nodes_with_neighbors(), 2);
    /// ```
    fn number_of_nodes_with_neighbors(&self) -> NumNodes {
        self.vertices_with_neighbors().count() as NumNodes
    }

    /// Returns a sorted vector of `(degree, count)` pairs.
    ///
    /// Each pair indicates how many nodes have that degree.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let distr = g.degree_distribution();
    /// assert_eq!(distr, vec![(1,2),(2,1)]);
    /// ```
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

    /// Returns the maximum degree among all vertices.
    ///
    /// Returns `0` if the graph has no vertices.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::prelude::*;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.max_degree(), 2);
    /// ```
    fn max_degree(&self) -> NumNodes {
        self.degrees().max().unwrap_or(0)
    }

    node_iterator!(
        degrees,
        degree_of,
        DegreesIter<'_, Self>,
        /// Returns an iterator over the degree of each vertex in the graph.
        ///
        /// Equivalent to mapping `degree_of` over all vertices.
        ///
        /// # Examples
        /// ```
        /// use ugraphs::prelude::*;
        ///
        /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
        /// let degs: Vec<_> = g.degrees().collect();
        /// assert_eq!(degs, vec![1,2,1]);
        /// ```
    );
    node_iterator!(
        neighbors,
        neighbors_of,
        NeighborsIter<'_, Self>,
        /// Returns an iterator over the neighbors of each vertex in the graph.
        ///
        /// Equivalent to mapping `neighbors_of` over all vertices.
        ///
        /// # Examples
        /// ```
        /// use ugraphs::prelude::*;
        ///
        /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
        /// let all_neighbors: Vec<Vec<_>> = g.neighbors().map(|iter| iter.collect()).collect();
        /// assert_eq!(all_neighbors, vec![vec![1], vec![0,2], vec![1]]);
        /// ```
    );
    node_bitset_of!(
        neighbors_of_as_bitset,
        neighbors_of,
        /// Returns a `NodeBitSet` representing the neighbors of a given vertex.
        ///
        /// The bitset has `true` for positions corresponding to neighbors of `u`.
        ///
        /// **Panics if `u >= n`.** 
        ///
        /// # Examples
        /// ```
        /// use ugraphs::prelude::*;
        ///
        /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
        /// let bitset = g.neighbors_of_as_bitset(1);
        /// assert!(bitset.get_bit(0));
        /// assert!(bitset.get_bit(2));
        /// assert!(!bitset.get_bit(1));
        /// ```
    );
    node_iterator!(
        neighbors_as_bitset,
        neighbors_of_as_bitset,
        NeighborsBitSetIter<'_, Self>,
        /// Returns an iterator over `NodeBitSet`s for the neighbors of each vertex.
        ///
        /// Equivalent to mapping `neighbors_of_as_bitset` over all vertices.
        ///
        /// # Examples
        /// ```
        /// use ugraphs::prelude::*;
        ///
        /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
        /// let bitsets: Vec<_> = g.neighbors_as_bitset().collect();
        /// assert_eq!(bitsets[0].iter_set_bits().collect::<Vec<_>>(), vec![1]);
        /// assert_eq!(bitsets[1].iter_set_bits().collect::<Vec<_>>(), vec![0,2]);
        /// assert_eq!(bitsets[2].iter_set_bits().collect::<Vec<_>>(), vec![1]);
        /// ```
    );

    /// BitmaskStream over all neighbors in the open neighborhood of a vertex in the graph.
    ///
    /// Returned by [`AdjacencyList::neighbors_of_as_stream`].
    type NeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;

    /// Returns a BitmaskStream over the neighbors of a vertex.
    ///
    /// **Panics if `u >= n`.**
    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_>;

    /// Returns a NodeBitSet with bits set for nodes at most 2 hops away from `u`.
    ///
    ///  **Panics if `u >= n`.**
    fn closed_two_neighborhood_of(&self, u: Node) -> NodeBitSet {
        let mut ns = self.vertex_bitset_unset();
        ns.set_bit(u);
        for v in self.neighbors_of(u) {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v));
        }
        ns
    }

    /// Returns a NodeBitSet with bits set for nodes at most 3 hops away from `u`.
    ///
    ///  **Panics if `u >= n`.**
    fn closed_three_neighborhood_of(&self, u: Node) -> NodeBitSet {
        let mut ns = self.vertex_bitset_unset();
        ns.set_bit(u);
        for v in self.closed_two_neighborhood_of(u).iter_set_bits() {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v));
        }
        ns
    }

    /// Returns an iterator over outgoing edges of a vertex.
    ///
    /// If `only_normalized` is `true`, only edges `(u,v)` with `u <= v` are returned.
    ///
    ///  **Panics if `u >= n`.**
    fn edges_of(&self, u: Node, only_normalized: bool) -> EdgesOf<'_, Self> {
        EdgesOfIterImpl {
            iter: self.neighbors_of(u),
            node: u,
            only_normalized,
        }
    }

    /// Returns an iterator over outgoing edges of a vertex in sorted order.
    ///
    /// If `only_normalized` is `true`, only edges `(u,v)` with `u <= v` are returned.
    ///
    ///  **Panics if `u >= n`.**
    fn ordered_edges_of(&self, u: Node, only_normalized: bool) -> OrderedEdgesOf {
        let mut edges = self.edges_of(u, only_normalized).collect_vec();
        edges.sort();
        edges.into_iter()
    }

    /// Returns an iterator over all edges in the graph.
    ///
    /// If `only_normalized` is `true`, only edges `(u,v)` with `u <= v` are returned.
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
    ///
    /// If `only_normalized` is `true`, only edges `(u,v)` with `u <= v` are returned.
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
    ($out_fn:ident => $fn:ident($($arg:ident : $type:ty),*) -> $ret:ty, $($doc:tt)*) => {
        $($doc)*
        fn $out_fn(&self, $($arg: $type),*) -> $ret {
            self.$fn($($arg),*)
        }
    };
}

// ---------- Iterator-Types ----------

/// Iterator over nodes that have at least one outgoing neighbor (out-degree > 0).
///
/// Returned by [`DirectedAdjacencyList::vertices_with_out_neighbors`].
pub type VerticesWithOutNeighbors<'a, G> = VerticesWithNeighborsIterImpl<'a, G>;

/// Iterator over nodes that have at least one incoming neighbor (in-degree > 0).
///
/// Returned by [`DirectedAdjacencyList::vertices_with_in_neighbors`].
pub type VerticesWithInNeighbors<'a, G> = VerticesWithNeighborsIterImpl<'a, G>;

/// Iterator over out-degrees of all nodes in a directed graph.
///
/// Returned by [`DirectedAdjacencyList::out_degrees`].
pub type OutDegreesIter<'a, G> =
    NodeMapIter<'a, G, NumNodes, <G as GraphNodeOrder>::VertexIter<'a>>;

/// Iterator over the outgoing neighbor lists of all nodes in a graph.
///
/// Returned by [`DirectedAdjacencyList::out_neighbors`].
pub type OutNeighborsIter<'a, G> = NodeMapIter<
    'a,
    G,
    <G as AdjacencyList>::NeighborIter<'a>,
    <G as GraphNodeOrder>::VertexIter<'a>,
>;

/// Iterator over the outgoing neighbor sets of all nodes as bitsets.
///
/// Returned by [`DirectedAdjacencyList::out_neighbors_as_bitset`].
pub type OutNeighborsBitSetIter<'a, G> =
    NodeMapIter<'a, G, NodeBitSet, <G as GraphNodeOrder>::VertexIter<'a>>;

/// Iterator over in-degrees of all nodes in a directed graph.
///
/// Returned by [`DirectedAdjacencyList::in_degrees`].
pub type InDegreesIter<'a, G> = NodeMapIter<'a, G, NumNodes, <G as GraphNodeOrder>::VertexIter<'a>>;

/// Iterator over the incoming neighbor lists of all nodes in a graph.
///
/// Returned by [`DirectedAdjacencyList::in_neighbors`].
pub type InNeighborsIter<'a, G> = NodeMapIter<
    'a,
    G,
    <G as DirectedAdjacencyList>::InNeighborIter<'a>,
    <G as GraphNodeOrder>::VertexIter<'a>,
>;

/// Iterator over the incoming neighbor sets of all nodes as bitsets.
///
/// Returned by [`DirectedAdjacencyList::in_neighbors_as_bitset`].
pub type InNeighborsBitSetIter<'a, G> =
    NodeMapIter<'a, G, NodeBitSet, <G as GraphNodeOrder>::VertexIter<'a>>;

/// Iterator over the outgoing edges of a given node.
///
/// Returned by [`DirectedAdjacencyList::out_edges_of`].
pub type OutEdgesOf<'a, G> = EdgesOfIterImpl<<G as AdjacencyList>::NeighborIter<'a>, false>;

/// Iterator over outgoing edges of a node, in deterministic order.
///
/// Returned by [`DirectedAdjacencyList::ordered_out_edges_of`].
pub type OrderedOutEdgesOf = std::vec::IntoIter<Edge>;

/// Iterator over the incoming edges of a given node.
///
/// Returned by [`DirectedAdjacencyList::in_edges_of`].
pub type InEdgesOf<'a, G> =
    EdgesOfIterImpl<<G as DirectedAdjacencyList>::InNeighborIter<'a>, false>;

/// Iterator over incoming edges of a node, in deterministic order.
///
/// Returned by [`DirectedAdjacencyList::ordered_in_edges_of`].
pub type OrderedInEdgesOf = std::vec::IntoIter<Edge>;

/// Extends [`AdjacencyList`] with in-neighbor access for directed graphs.
/// 
/// Also aliases functions of [`AdjacencyList`] with `out`-versions
/// (aka. [`AdjacencyList::neighbors_of`] => [`DirectedAdjacencyList::out_neighbors_of`])
///
/// Only implemented by directed graph types (`AdjArray`, `AdjArrayIn`, etc.).
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArray::from_edges(3, [(0,1), (2,1)]);
///
/// assert_eq!(g.out_degree_of(0), 1);
/// assert_eq!(g.in_degree_of(1), 2);
/// assert_eq!(g.in_neighbors_of(1).collect::<Vec<_>>(), vec![0,2]);
/// ```
pub trait DirectedAdjacencyList: AdjacencyList + GraphType<Dir = Directed> {
    propagate!(
        out_neighbors_of => neighbors_of(u : Node) -> Self::NeighborIter<'_>,
        /// Returns an iterator over outgoing neighbors of a given vertex.
        /// Delegates to [`AdjacencyList::neighbors_of`].
        /// **Panics if `u >= n`**
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1), (0,2)]);
        /// let out: Vec<_> = g.out_neighbors_of(0).collect();
        /// assert_eq!(out, vec![1,2]);
        /// ```
    );
    propagate!(
        out_degree_of => degree_of(u : Node) -> NumNodes,
        /// Returns the out-degree of a given vertex.
        /// Delegates to [`AdjacencyList::degree_of`].
        /// **Panics if `u >= n`**
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1), (0,2)]);
        /// assert_eq!(g.out_degree_of(0), 2);
        /// ```
    );
    propagate!(
        vertices_with_out_neighbors => vertices_with_neighbors() -> VerticesWithOutNeighbors<'_, Self>,
        /// Returns an iterator over vertices with non-zero out-degree.
        /// Delegates to [`AdjacencyList::vertices_with_neighbors`].
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1)]);
        /// let verts: Vec<_> = g.vertices_with_out_neighbors().collect();
        /// assert_eq!(verts, vec![0]);
        /// ```
    );
    propagate!(
        number_of_nodes_with_out_neighbors => number_of_nodes_with_neighbors() -> NumNodes,
        /// Returns the number of vertices with non-zero out-degree.
        /// Delegates to [`AdjacencyList::number_of_nodes_with_neighbors`].
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1)]);
        /// assert_eq!(g.number_of_nodes_with_out_neighbors(), 1);
        /// ```
    );
    propagate!(
        out_degree_distribution => degree_distribution() -> Vec<(NumNodes, NumNodes)>,
        /// Returns a distribution of out-degrees as `(degree, count)`.
        /// Delegates to [`AdjacencyList::degree_distribution`].
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1), (0,2)]);
        /// let dist = g.out_degree_distribution();
        /// assert_eq!(dist, vec![(0,2),(2,1)]); // depending on implementation
        /// ```
    );
    propagate!(
        max_out_degree => max_degree() -> NumNodes,
        /// Returns the maximum out-degree in the graph.
        /// Delegates to [`AdjacencyList::max_degree`].
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1), (0,2)]);
        /// assert_eq!(g.max_out_degree(), 2);
        /// ```
    );
    propagate!(
        out_neighbors_of_as_stream => neighbors_of_as_stream(u: Node) -> Self::NeighborsStream<'_>,
        /// Returns a [`BitmaskStream`] over outgoing neighbors of a vertex.
        /// Delegates to [`AdjacencyList::neighbors_of_as_stream`].
        /// **Panics if `u >= n`**
    );

    node_iterator!(
        out_degrees,
        out_degree_of,
        OutDegreesIter<'_, Self>,
        /// Returns an iterator over all vertex out-degrees.
        /// Equivalent to calling [`DirectedAdjacencyList::out_degree_of`] for each vertex.
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1),(0,2)]);
        /// let degs: Vec<_> = g.out_degrees().collect();
        /// assert_eq!(degs, vec![2,0,0]);
        /// ```
    );
    node_iterator!(
        out_neighbors,
        out_neighbors_of,
        OutNeighborsIter<'_, Self>,
        /// Returns an iterator over outgoing neighbors for each vertex.
        /// Yields [`NeighborsIter`] for each vertex.
        ///
        /// # Examples
        /// ```
        /// # use ugraphs::prelude::*;
        /// let g = AdjArray::from_edges(3, [(0,1)]);
        /// let neighbors: Vec<Vec<_>> = g.out_neighbors().map(|it| it.collect()).collect();
        /// assert_eq!(neighbors, vec![vec![1], vec![], vec![]]);
        /// ```
    );
    node_bitset_of!(
        out_neighbors_of_as_bitset,
        out_neighbors_of,
        /// Returns a [`NodeBitSet`] where bits corresponding to outgoing neighbors are set.
        /// **Panics if `u >= n`**
    );
    node_iterator!(
        out_neighbors_as_bitset,
        out_neighbors_of_as_bitset,
        OutNeighborsBitSetIter<'_, Self>,
        /// Returns an iterator over [`NodeBitSet`] for all vertices representing their outgoing neighbors.
    );

    /// Returns an iterator over outgoing edges of a given vertex `(u, v)`.
    /// Equivalent to [`edges_of(u, false)`].
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArray::from_edges(3, [(0,1)]);
    /// let edges: Vec<_> = g.out_edges_of(0).collect();
    /// assert_eq!(edges, vec![Edge(0,1)]);
    /// ```
    fn out_edges_of(&self, u: Node) -> OutEdgesOf<'_, Self> {
        self.edges_of(u, false)
    }

    /// Returns an iterator over outgoing edges sorted by `(u,v)`.
    /// Equivalent to [`ordered_edges_of(u, false)`].
    /// **Panics if `u >= n`**
    fn ordered_out_edges_of(&self, u: Node) -> OrderedOutEdgesOf {
        self.ordered_edges_of(u, false)
    }

    /// Iterator over all incoming neighbors of a vertex in the graph.
    ///
    /// Returned by [`DirectedAdjacencyList::in_neighbors_of`].
    type InNeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns an iterator over incoming neighbors of a vertex (`v` such that `(v, u)` exists).
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayIn::from_edges(3, [(0,1),(2,1)]);
    /// let incoming: Vec<_> = g.in_neighbors_of(1).collect();
    /// assert_eq!(incoming, vec![0,2]);
    /// ```
    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_>;

    /// Returns the number of incoming edges for vertex `u`.
    /// **Panics if `u >= n`**
    fn in_degree_of(&self, u: Node) -> NumNodes;

    /// Returns the sum of in-degree and out-degree for vertex `u`.
    /// **Panics if `u >= n`**
    fn total_degree_of(&self, u: Node) -> NumNodes {
        self.out_degree_of(u) + self.in_degree_of(u)
    }

    /// Returns an iterator to all vertices with non-zero in-degre/// Returns an iterator over vertices with non-zero in-degree.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayIn::from_edges(3, [(0,1),(2,1)]);
    /// let verts: Vec<_> = g.vertices_with_in_neighbors().collect();
    /// assert_eq!(verts, vec![1]);
    /// ```
    fn vertices_with_in_neighbors(&self) -> VerticesWithInNeighbors<'_, Self> {
        VerticesWithNeighborsIterImpl {
            node_iter: self.vertices(),
            graph: self,
            degree_fn: Self::in_degree_of,
        }
    }

    /// Returns the number of vertices with non-zero in-degree.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayIn::from_edges(3, [(0,1),(2,1)]);
    /// assert_eq!(g.number_of_nodes_with_in_neighbors(), 1);
    /// ```
    fn number_of_nodes_with_in_neighbors(&self) -> NumNodes {
        self.vertices_with_in_neighbors().count() as NumNodes
    }

    /// Returns a distribution of in-degrees as `(degree, count)`.
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

    /// Returns the maximum in-degree in the graph.
    fn max_in_degree(&self) -> NumNodes {
        self.in_degrees().max().unwrap_or(0)
    }

    node_iterator!(
        in_degrees,
        in_degree_of,
        InDegreesIter<'_, Self>,
        /// Returns an iterator over all vertex in-degrees.
    );
    node_iterator!(
        in_neighbors,
        in_neighbors_of,
        InNeighborsIter<'_, Self>,
        /// Returns an iterator over incoming neighbors for all vertices.
    );
    node_bitset_of!(
        in_neighbors_of_as_bitset,
        in_neighbors_of,
        /// Returns a [`NodeBitSet`] where bits corresponding to incoming neighbors are set.
        /// **Panics if `u >= n`**
    );
    node_iterator!(
        in_neighbors_as_bitset,
        in_neighbors_of_as_bitset,
        InNeighborsBitSetIter<'_, Self>,
        /// Returns a [`NodeBitSet`] where bits corresponding to incoming neighbors are set.
        /// **Panics if `u >= n`**
    );

    /// BitmaskStream over all incoming neighbors in the open neighborhood of a vertex in the graph.
    ///
    /// Returned by [`DirectedAdjacencyList::in_neighbors_of_as_stream`].
    type InNeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;

    /// Returns a [`BitmaskStream`] over incoming neighbors of a vertex.
    /// **Panics if `u >= n`**
    fn in_neighbors_of_as_stream(&self, u: Node) -> Self::InNeighborsStream<'_>;

    /// Returns an iterator over incoming edges of a vertex `(v, u)`.
    /// **Panics if `u >= n`**
    fn in_edges_of(&self, u: Node) -> InEdgesOf<'_, Self> {
        EdgesOfIterImpl {
            iter: self.in_neighbors_of(u),
            node: u,
            only_normalized: false,
        }
    }

    /// Returns an iterator over incoming edges of a vertex in sorted order.
    /// **Panics if `u >= n`**
    fn ordered_in_edges_of(&self, u: Node) -> OrderedInEdgesOf {
        let mut edges = self.in_edges_of(u).collect_vec();
        edges.sort();
        edges.into_iter()
    }
}

/// Trait for testing the existence of edges.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
///
/// assert!(g.has_edge(0,1));
/// assert!(g.has_bidirected_edge(1,2));
/// assert!(!g.has_edge(0,2));
/// ```
pub trait AdjacencyTest: GraphNodeOrder {
    /// Returns `true` if the edge `(u, v)` exists in the graph.
    /// **Panics if `u >= n || v >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(0,1)]);
    /// assert!(g.has_edge(0,1));
    /// assert!(!g.has_edge(1,2));
    /// ```
    fn has_edge(&self, u: Node, v: Node) -> bool;

    /// Checks multiple neighbors for a single node at once, returning an array of booleans
    /// indicating for each neighbor whether the edge exists.
    ///
    /// **Panics if `u >= n` or any neighbor `v >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (0,2)]);
    /// let res = g.has_neighbors(0, [1,2,0]);
    /// assert_eq!(res, [true,true,false]);
    /// ```
    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        neighbors.map(|v| self.has_edge(u, v))
    }

    /// Returns `true` if a self-loop `(u, u)` exists at the given vertex.
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(1,1)]);
    /// assert!(g.has_self_loop(1));
    /// assert!(!g.has_self_loop(0));
    /// ```
    fn has_self_loop(&self, u: Node) -> bool {
        self.has_edge(u, u)
    }

    /// Returns `true` if any vertex in the graph has a self-loop.
    /// Iterates over all vertices and checks for self-loops.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(0,0),(1,2)]);
    /// assert!(g.has_self_loops());
    /// ```
    fn has_self_loops(&self) -> bool {
        self.vertices().any(|u| self.has_self_loop(u))
    }

    /// Returns `true` if both `(u, v)` and `(v, u)` exist in the graph.
    /// For undirected graphs, any edge `{u,v}` always returns `true`.
    /// **Panics if `u >= n || v >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArray::from_edges(3, [(0,1)]);
    /// assert!(!g.has_bidirected_edge(0,1)); // directed graph
    ///
    /// let g2 = AdjArrayUndir::from_edges(3, [(0,1)]);
    /// assert!(g2.has_bidirected_edge(0,1)); // undirected graph
    /// ```
    fn has_bidirected_edge(&self, u: Node, v: Node) -> bool {
        self.has_edge(u, v) && self.has_edge(v, u)
    }
}

/// Iterator over all **non-singleton** vertices of a graph.
///
/// A singleton is a node without any edges:
/// - in undirected graphs: `degree_of(u) == 0`
/// - in directed graphs: `total_degree_of(u) == 0`
///
/// Returned by [`Singletons::vertices_no_singletons`].
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

/// Provides methods for detecting singleton (isolated) nodes.
///
/// A singleton is defined
/// - for undirected graphs, by `degree_of(u) == 0`.
/// - for directed graphs, by `total_degree_of(u) == 0`.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(4, [(0,1), (1,2)]);
///
/// assert!(g.is_singleton(3));
/// let non_singletons: Vec<_> = g.vertices_no_singletons().collect();
/// assert_eq!(non_singletons, vec![0,1,2]);
/// ```
pub trait Singletons: GraphNodeOrder + Sized {
    /// Returns `true` if the vertex `u` is a singleton (isolated node).
    ///
    /// - For undirected graphs: `degree_of(u) == 0`
    /// - For directed graphs: `total_degree_of(u) == 0`
    ///
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(4, [(0,1), (1,2)]);
    /// assert!(g.is_singleton(3));
    /// assert!(!g.is_singleton(1));
    /// ```
    fn is_singleton(&self, u: Node) -> bool;

    /// Returns an iterator over all vertices that are **not** singletons (i.e., have at least one edge).
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(4, [(0,1), (1,2)]);
    /// let non_singletons: Vec<_> = g.vertices_no_singletons().collect();
    /// assert_eq!(non_singletons, vec![0,1,2]);
    /// ```
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

/// Provides indexed access to neighbors.
///
/// Useful for algorithms that need deterministic order.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// let n0 = g.ith_neighbor(1, 0);
/// assert!(n0 == 0 || n0 == 2);
/// ```
pub trait IndexedAdjacencyList: AdjacencyList {
    /// Returns the `i`-th neighbor (0-indexed) of vertex `u`.
    ///
    /// **Panics if `u >= n` or `i >= degree_of(u)`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.ith_neighbor(1, 0), 0);
    /// assert_eq!(g.ith_neighbor(1, 1), 2);
    /// ```
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node;
}

/// Extends [`IndexedAdjacencyList`] with the ability to swap neighbors.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// let before = g.as_neighbors_slice(1).to_vec();
/// g.swap_neighbors(1, 0, 1);
/// let after = g.as_neighbors_slice(1).to_vec();
/// assert_eq!(before.len(), after.len());
/// ```
pub trait IndexedAdjacencySwap: IndexedAdjacencyList {
    /// Swaps the `i`-th and `j`-th neighbors of vertex `u`.
    ///
    /// **Panics if `u >= n`, `i >= degree_of(u)`, or `j >= degree_of(u)`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let before = g.as_neighbors_slice(1).to_vec();
    /// g.swap_neighbors(1, 0, 1);
    /// let after = g.as_neighbors_slice(1).to_vec();
    /// assert_eq!(before.len(), after.len());
    /// ```
    fn swap_neighbors(&mut self, u: Node, i: NumNodes, j: NumNodes);
}

/// Provides **read-only access** to the neighbors of a node as a slice.
///
/// This trait is mainly intended for performance-sensitive algorithms that
/// want direct slice access instead of iterators.  
/// It is implemented by adjacency-array based representations such as
/// [`AdjArrayUndir`](crate::repr::AdjArrayUndir).
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// let slice = g.as_neighbors_slice(1);
/// assert!(slice.contains(&0));
/// assert!(slice.contains(&2));
/// ```
pub trait NeighborsSlice {
    /// Returns a read-only slice of the neighbors of vertex `u`.
    ///
    /// This is intended for performance-sensitive algorithms that require
    /// direct slice access instead of iterators.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let slice = g.as_neighbors_slice(1);
    /// assert!(slice.contains(&0));
    /// assert!(slice.contains(&2));
    /// ```
    fn as_neighbors_slice(&self, u: Node) -> &[Node];
}

/// Provides **mutable access** to the neighbors of a node as a slice.
///
/// This is a low-level API, mainly intended for specialized algorithms that
/// need to reorder or modify adjacency arrays in place.  
/// Normal users will rarely need this; prefer higher-level methods unless you
/// are writing optimized routines.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// let slice = g.as_neighbors_slice_mut(1);
///
/// // Swap the two neighbors of node 1
/// slice.swap(0, 1);
///
/// // The set of neighbors is unchanged
/// assert!(slice.contains(&0));
/// assert!(slice.contains(&2));
/// ```
pub trait NeighborsSliceMut: NeighborsSlice {
    /// Returns a mutable slice of the neighbors of vertex `u`.
    ///
    /// This is a low-level API for algorithms that need to reorder or
    /// modify adjacency arrays in place. Prefer higher-level methods
    /// unless writing optimized routines.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// let slice = g.as_neighbors_slice_mut(1);
    /// slice.swap(0, 1); // Swap neighbors of node 1
    /// assert!(slice.contains(&0));
    /// assert!(slice.contains(&2));
    /// ```
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

/// Creates a new empty graph with a given number of nodes.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::new(5);
/// assert_eq!(g.number_of_nodes(), 5);
/// assert_eq!(g.number_of_edges(), 0);
/// ```
pub trait GraphNew {
    /// Creates a new graph with `n` singleton nodes (nodes with no edges).
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::new(3);
    /// assert_eq!(g.number_of_nodes(), 3);
    /// assert!(g.is_singleton_graph());
    /// ```
    fn new(n: NumNodes) -> Self;
}

/// Provides edge insertion and deletion operations.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let mut g = AdjArrayUndir::new(3);
/// g.add_edge(0,1);
/// assert!(g.has_edge(0,1));
/// g.remove_edge(0,1);
/// assert!(!g.has_edge(0,1));
/// ```
pub trait GraphEdgeEditing: GraphNew {
    /// Adds the edge `(u, v)` to the graph.
    ///
    /// **Panics if `u >= n` or `v >= n`, or if the edge already exists**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::new(3);
    /// g.add_edge(0, 1);
    /// assert!(g.has_edge(0, 1));
    /// ```
    fn add_edge(&mut self, u: Node, v: Node) {
        assert!(!self.try_add_edge(u, v))
    }

    /// Adds the edge `(u, v)` to the graph.
    ///
    /// Returns `true` exactly if the edge was present previously.
    ///
    /// **Panics if `u >= n || v >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::new(3);
    /// assert!(!g.try_add_edge(0, 1));
    /// assert!(g.try_add_edge(0, 1)); // Already present
    /// ```
    fn try_add_edge(&mut self, u: Node, v: Node) -> bool;

    /// Adds all edges in the provided collection to the graph.
    ///
    /// **Panics if any edge `(u, v)` is invalid or already exists**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::new(3);
    /// g.add_edges([(0,1), (1,2)]);
    /// assert!(g.has_edge(0,1));
    /// assert!(g.has_edge(1,2));
    /// ```
    fn add_edges<I, E>(&mut self, edges: I)
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        for Edge(u, v) in edges.into_iter().map(|d| d.into()) {
            self.add_edge(u, v);
        }
    }

    /// Tries to add all edges in the provided collection to the graph.
    ///
    /// Returns the number of edges successfully added.
    ///
    /// **Panics if any edge `(u, v)` is invalid**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::new(3);
    /// let added = g.try_add_edges([(0,1), (1,2)]);
    /// assert_eq!(added, 2);
    /// ```
    fn try_add_edges<I, E>(&mut self, edges: I) -> NumEdges
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        edges
            .into_iter()
            .map(|e| {
                let Edge(u, v) = e.into();
                !self.try_add_edge(u, v) as NumEdges
            })
            .sum()
    }

    /// Removes the edge `(u, v)` from the graph.
    ///
    /// **Panics if the edge does not exist or if `u >= n || v >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1)]);
    /// g.remove_edge(0,1);
    /// assert!(!g.has_edge(0,1));
    /// ```
    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v));
    }

    /// Removes all edges in the provided collection from the graph.
    ///
    /// **Panics if any edge does not exist or is invalid**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// g.remove_edges([(0,1), (1,2)]);
    /// assert!(!g.has_edge(0,1));
    /// assert!(!g.has_edge(1,2));
    /// ```
    fn remove_edges<I, E>(&mut self, edges: I)
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        for Edge(u, v) in edges.into_iter().map(|d| d.into()) {
            self.remove_edge(u, v);
        }
    }

    /// Removes the edge `(u, v)` from the graph.
    ///
    /// Returns `true` if the edge was present previously.
    ///
    /// **Panics if `u >= n || v >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1)]);
    /// assert!(g.try_remove_edge(0,1));
    /// assert!(!g.try_remove_edge(0,1)); // Already removed
    /// ```
    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool;
}

/// Extends [`GraphEdgeEditing`] with directed edge removals.
///
/// Only implemented by directed graphs.
///
/// # Example
/// ```
/// use ugraphs::prelude::*;
///
/// let mut g = AdjArray::from_edges(3, [(0,1), (2,1)]);
/// g.remove_edges_into_node(1);
/// assert!(!g.has_edge(0,1));
/// assert!(!g.has_edge(2,1));
/// ```
pub trait GraphDirectedEdgeEditing: GraphEdgeEditing + GraphType<Dir = Directed> {
    /// Removes all incoming edges `(v, u)` to the given node `u`.
    ///
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayIn::from_edges(3, [(0,1), (2,1)]);
    /// g.remove_edges_into_node(1);
    /// assert_eq!(g.in_degree_of(1), 0);
    /// ```
    fn remove_edges_into_node(&mut self, u: Node);

    /// Removes all outgoing edges `(u, v)` from the given node `u`.
    ///
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArray::from_edges(3, [(0,1), (0,2)]);
    /// g.remove_edges_out_of_node(0);
    /// assert_eq!(g.out_degree_of(0), 0);
    /// ```
    fn remove_edges_out_of_node(&mut self, u: Node);
}

/// Extends [`GraphEdgeEditing`] with local edge removals at nodes.
///
/// Implemented by both directed and undirected graphs.
///
/// # Example
/// ```
/// use ugraphs::prelude::*;
///
/// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// g.remove_edges_at_node(1);
/// assert_eq!(g.number_of_edges(), 0);
/// ```
pub trait GraphLocalEdgeEditing: GraphEdgeEditing {
    /// Removes all edges adjacent to the given node `u` (both incoming and outgoing for directed graphs, or all incident edges for undirected graphs).
    ///
    /// **Panics if `u >= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// g.remove_edges_at_node(1);
    /// assert_eq!(g.number_of_edges(), 0);
    /// ```
    fn remove_edges_at_node(&mut self, u: Node);

    /// Removes all edges adjacent to any node in the provided iterator.
    ///
    /// **Panics if any node in the iterator is `>= n`**
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let mut g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// g.remove_edges_at_nodes([0,2]);
    /// assert_eq!(g.number_of_edges(), 0);
    /// ```
    fn remove_edges_at_nodes<I>(&mut self, nodes: I)
    where
        I: IntoIterator<Item = Node>,
    {
        for node in nodes {
            self.remove_edges_at_node(node);
        }
    }
}

/// Build a graph from scratch given a number of nodes and edges.
///
/// Preferred over building with `new + add_edges` when possible,
/// since some static graph representations only support this.
///
/// # Examples
/// ```
/// use ugraphs::prelude::*;
///
/// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
/// assert_eq!(g.number_of_nodes(), 3);
/// assert!(g.has_edge(0,1));
/// ```
pub trait GraphFromScratch {
    /// Creates a new graph with `n` nodes and the given edges.
    ///
    /// This is the preferred method for building graphs from scratch instead of using `new + add_edges`.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_edges(3, [(0,1), (1,2)]);
    /// assert_eq!(g.number_of_nodes(), 3);
    /// assert!(g.has_edge(0,1));
    /// ```
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>;

    /// Creates a new graph with `n` nodes and the given edges,
    /// adding edges via `try_add_edge` instead of `add_edge` (if implemented over [`GraphNew`] +
    /// [`GraphEdgeEditing`]).
    ///
    /// This allows edges to be ignored if they already exist.
    ///
    /// # Examples
    /// ```
    /// # use ugraphs::prelude::*;
    /// let g = AdjArrayUndir::from_try_edges(3, [(0,1), (1,2), (0,1)]);
    /// assert_eq!(g.number_of_nodes(), 3);
    /// assert!(g.has_edge(0,1));
    /// assert!(g.has_edge(1,2));
    /// ```
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
