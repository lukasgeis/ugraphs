/*!
# Compressed Sparse Row (CSR) Graph Representations

This module provides adjacency representations based on the **Compressed Sparse Row (CSR)** format.
They are designed for **memory efficiency** and **fast iteration** over adjacency lists in sparse graphs.

CSR graphs store all adjacency lists in a single flattened array, with offset indices
marking the start of each vertex’s neighbor list. This structure provides:

- **Compact storage** compared to adjacency arrays (`Vec<Vec<Node>>`).
- **Fast sequential access** to neighbors due to good cache locality.
- **Efficient memory usage** for sparse graphs.
- **Higher construction cost**, but immutable and optimized for traversal.

## Types

- [`CsrGraph`]: Directed CSR graph storing outgoing edges only.
- [`CsrGraphIn`]: Directed CSR graph storing both outgoing and incoming edges.
- [`CsrGraphUndir`]: Undirected CSR graph storing adjacency symmetrically.
- [`CrossCsrGraph`]: Undirected CSR graph with *cross pointers* between reciprocal edges.

Additionally:
- [`NodeWithCrossPos`] is used internally by [`CrossCsrGraph`] to store the "cross position" of reciprocal edges.
*/

use super::*;
use crate::{testing::test_graph_ops, utils::sliced_buffer::SlicedBuffer};
use std::{iter::Copied, ops::Range, slice::Iter};
use stream_bitset::{bitmask_stream::IntoBitmaskStream, bitset::BitsetStream};

/// A neighbor node paired with its **cross position**.
///
/// Used in [`CrossCsrGraph`] to enable *O(1)* access to the reciprocal edge
/// in an undirected graph.
///
/// - `node`: ID of the neighboring vertex.
/// - `cross_pos`: Index of the reciprocal edge in the neighbor’s adjacency list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeWithCrossPos {
    pub node: Node,
    pub cross_pos: NumNodes,
}

/// Directed **CSR graph** representation.
///
/// - Stores adjacency as a single flattened array of outgoing edges.
/// - Memory-efficient and fast for sparse graphs.
/// - Incoming edges are not stored (see [`CsrGraphIn`] if you need them).
#[derive(Clone)]
pub struct CsrGraph {
    out_nbs: SlicedBuffer<Node, NumEdges>,
}

/// Directed **CSR graph** with both outgoing and incoming edges stored.
///
/// - Outgoing neighbors: `out_nbs`.
/// - Incoming neighbors: `in_nbs`.
///
/// Useful for algorithms requiring fast access to both directions.
#[derive(Clone)]
pub struct CsrGraphIn {
    out_nbs: SlicedBuffer<Node, NumEdges>,
    in_nbs: SlicedBuffer<Node, NumEdges>,
}

/// Undirected **CSR graph** representation.
///
/// - Stores symmetric adjacency lists.
/// - Each edge is stored twice (once per endpoint) (except self-loops).
#[derive(Clone)]
pub struct CsrGraphUndir {
    nbs: SlicedBuffer<Node, NumEdges>,
    self_loops: NumNodes,
}

/// Undirected **CSR graph** with cross-pointers between reciprocal edges.
///
/// - Like [`CsrGraphUndir`], but each neighbor entry knows the index of its reverse edge.
/// - Useful for algorithms that need *constant-time edge reversal*, e.g. flow algorithms.
#[derive(Clone)]
pub struct CrossCsrGraph {
    nbs: SlicedBuffer<NodeWithCrossPos, NumEdges>,
    self_loops: NumNodes,
}

macro_rules! impl_common_csr_graph_ops {
    ($struct:ident, $nbs:ident, $directed:ident) => {
        impl GraphType for $struct {
            type Dir = $directed;
        }

        impl GraphNodeOrder for $struct {
            type VertexIter<'a>
                = Range<Node>
            where
                Self: 'a;

            fn vertices(&self) -> Self::VertexIter<'_> {
                self.vertices_range()
            }

            fn number_of_nodes(&self) -> NumNodes {
                self.$nbs.number_of_slices()
            }
        }

        impl AdjacencyList for $struct {
            type NeighborIter<'a>
                = Copied<Iter<'a, Node>>
            where
                Self: 'a;

            type ClosedNeighborIter<'a>
                = std::iter::Chain<std::iter::Once<Node>, Self::NeighborIter<'a>>
            where
                Self: 'a;

            fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
                self.$nbs[u].iter().copied()
            }

            fn closed_neighbors_of(&self, u: Node) -> Self::ClosedNeighborIter<'_> {
                std::iter::once(u).chain(self.neighbors_of(u))
            }

            fn degree_of(&self, u: Node) -> NumNodes {
                self.$nbs.size_of(u) as NumNodes
            }

            type NeighborsStream<'a>
                = BitsetStream<Node>
            where
                Self: 'a;

            fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
                self.neighbors_of_as_bitset(u).into_bitmask_stream()
            }
        }

        impl AdjacencyTest for $struct {
            fn has_edge(&self, u: Node, v: Node) -> bool {
                self.$nbs[u].contains(&v)
            }
        }

        impl NeighborsSlice for $struct {
            fn as_neighbors_slice(&self, u: Node) -> &[Node] {
                &self.$nbs[u]
            }
        }

        impl NeighborsSliceMut for $struct {
            fn as_neighbors_slice_mut(&mut self, u: Node) -> &mut [Node] {
                &mut self.$nbs[u]
            }
        }
    };
}

impl_common_csr_graph_ops!(CsrGraph, out_nbs, Directed);
impl_common_csr_graph_ops!(CsrGraphIn, out_nbs, Directed);
impl_common_csr_graph_ops!(CsrGraphUndir, nbs, Undirected);

impl DirectedAdjacencyList for CsrGraph {
    type InNeighborIter<'a>
        = DirectedInCsrIter<'a>
    where
        Self: 'a;

    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_> {
        // Should be avoided as it is very inefficient
        DirectedInCsrIter {
            graph: self,
            node: u,
            lb: 0,
        }
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        // Should be avoided as it is very inefficient
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

impl DirectedAdjacencyList for CsrGraphIn {
    type InNeighborIter<'a>
        = Copied<Iter<'a, Node>>
    where
        Self: 'a;

    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_> {
        self.in_nbs[u].iter().copied()
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        self.in_nbs.size_of(u)
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

impl Singletons for CsrGraph {
    /// Very inefficient as this has a runtime of `O(n^2)` for some implementations
    fn is_singleton(&self, u: Node) -> bool {
        self.total_degree_of(u) == 0
    }
}

impl Singletons for CsrGraphIn {
    #[inline]
    fn is_singleton(&self, u: Node) -> bool {
        self.total_degree_of(u) == 0
    }
}

impl GraphEdgeOrder for CsrGraph {
    fn number_of_edges(&self) -> NumEdges {
        self.out_nbs.number_of_entries()
    }
}

impl GraphEdgeOrder for CsrGraphIn {
    fn number_of_edges(&self) -> NumEdges {
        self.out_nbs.number_of_entries()
    }
}

impl GraphEdgeOrder for CsrGraphUndir {
    fn number_of_edges(&self) -> NumEdges {
        (self.nbs.number_of_entries() + self.self_loops as NumEdges) / 2
    }
}

impl GraphType for CrossCsrGraph {
    type Dir = Undirected;
}

impl GraphNodeOrder for CrossCsrGraph {
    type VertexIter<'a>
        = Range<Node>
    where
        Self: 'a;

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }

    fn number_of_nodes(&self) -> NumNodes {
        self.nbs.number_of_slices()
    }
}

impl GraphEdgeOrder for CrossCsrGraph {
    fn number_of_edges(&self) -> NumEdges {
        (self.nbs.number_of_entries() + self.self_loops as NumEdges) / 2
    }
}

impl AdjacencyList for CrossCsrGraph {
    type NeighborIter<'a>
        = CrossPosNeighborIter<'a>
    where
        Self: 'a;

    type ClosedNeighborIter<'a>
        = std::iter::Chain<std::iter::Once<Node>, Self::NeighborIter<'a>>
    where
        Self: 'a;

    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
        CrossPosNeighborIter(self.nbs[u].iter())
    }

    fn closed_neighbors_of(&self, u: Node) -> Self::ClosedNeighborIter<'_> {
        std::iter::once(u).chain(self.neighbors_of(u))
    }

    fn degree_of(&self, u: Node) -> NumNodes {
        self.nbs.size_of(u) as NumNodes
    }

    type NeighborsStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;

    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
        self.neighbors_of_as_bitset(u).into_bitmask_stream()
    }
}

impl AdjacencyTest for CrossCsrGraph {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.nbs[u].iter().any(|cp| cp.node == v)
    }
}

impl IndexedAdjacencyList for CrossCsrGraph {
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node {
        self.nbs[u][i as usize].node
    }
}

impl IndexedAdjacencySwap for CrossCsrGraph {
    fn swap_neighbors(&mut self, u: Node, i: NumNodes, j: NumNodes) {
        let nb1 = self.nbs[u][i as usize];
        let nb2 = self.nbs[u][j as usize];

        // Update CrossPositions
        self.nbs[nb1.node][nb1.cross_pos as usize].cross_pos = j;
        self.nbs[nb2.node][nb2.cross_pos as usize].cross_pos = i;

        self.nbs[u].swap(i as usize, j as usize);
    }
}

// ---------- Custom Iterators ----------
//
// As of now `#![feature(impl_trait_in_assoc_type)]` is not stable yet which is why we rely on
// custom Wrappers around iterators if the *real* type is obfuscated by a closure.

/// Iterator over **incoming neighbors** in [`CsrGraph`].
///
/// This iterator computes in-neighbors on-the-fly by scanning all edges.
/// As a result, it is **very inefficient** and should be avoided if possible.
/// Prefer [`CsrGraphIn`] when incoming adjacency is required.
pub struct DirectedInCsrIter<'a> {
    graph: &'a CsrGraph,
    node: Node,
    lb: Node,
}

impl<'a> Iterator for DirectedInCsrIter<'a> {
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

/// Iterator over neighbors in [`CrossCsrGraph`].
///
/// Wraps a slice iterator over [`NodeWithCrossPos`] but exposes only the neighbor node.
pub struct CrossPosNeighborIter<'a>(Iter<'a, NodeWithCrossPos>);

impl<'a> Iterator for CrossPosNeighborIter<'a> {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|cp| cp.node)
    }
}

// ---------- GraphFromScratch ----------

impl GraphFromScratch for CsrGraph {
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        assert!(n > 0);
        let mut edges: Vec<Edge> = edges.into_iter().map(|e| e.into()).collect();
        edges.sort_unstable();
        edges.dedup();

        let mut offsets: Vec<NumEdges> = Vec::with_capacity(n as usize + 1);
        offsets.push(0);

        let mut curr_node = 0;
        let mut counter = 0;
        let edges: Vec<Node> = edges
            .into_iter()
            .map(|Edge(u, v)| {
                while u > curr_node {
                    offsets.push(counter);
                    curr_node += 1;
                }
                counter += 1;

                v
            })
            .collect();

        assert!(offsets.len() as NumNodes <= n + 1);
        while offsets.len() as NumNodes <= n {
            offsets.push(edges.len() as NumEdges);
        }

        Self {
            out_nbs: SlicedBuffer::new(edges, offsets),
        }
    }

    fn from_try_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        Self::from_edges(n, edges)
    }
}

impl GraphFromScratch for CsrGraphIn {
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        assert!(n > 0);
        let mut edges: Vec<Edge> = edges.into_iter().map(|e| e.into()).collect();
        edges.sort_unstable();
        edges.dedup();

        let mut in_edges: Vec<Edge> = edges.iter().map(|e| e.reverse()).collect();
        in_edges.sort_unstable();

        let mut offsets: Vec<NumEdges> = Vec::with_capacity(n as usize + 1);
        offsets.push(0);

        let mut curr_node = 0;
        let mut counter = 0;
        let edges: Vec<Node> = edges
            .into_iter()
            .map(|Edge(u, v)| {
                while u > curr_node {
                    offsets.push(counter);
                    curr_node += 1;
                }
                counter += 1;

                v
            })
            .collect();

        assert!(offsets.len() as NumNodes <= n + 1);
        while offsets.len() as NumNodes <= n {
            offsets.push(edges.len() as NumEdges);
        }

        let mut in_offsets: Vec<NumEdges> = Vec::with_capacity(n as usize + 1);
        in_offsets.push(0);

        curr_node = 0;
        counter = 0;
        let in_edges: Vec<Node> = in_edges
            .into_iter()
            .map(|Edge(u, v)| {
                while u > curr_node {
                    in_offsets.push(counter);
                    curr_node += 1;
                }
                counter += 1;

                v
            })
            .collect();

        assert!(in_offsets.len() as NumNodes <= n + 1);
        while in_offsets.len() as NumNodes <= n {
            in_offsets.push(in_edges.len() as NumEdges);
        }

        Self {
            out_nbs: SlicedBuffer::new(edges, offsets),
            in_nbs: SlicedBuffer::new(in_edges, in_offsets),
        }
    }

    fn from_try_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        Self::from_edges(n, edges)
    }
}

impl GraphFromScratch for CsrGraphUndir {
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        assert!(n > 0);
        let mut edges: Vec<Edge> = edges.into_iter().map(|e| e.into().normalized()).collect();
        edges.sort_unstable();
        edges.dedup();

        let mut edges: Vec<Edge> = edges
            .into_iter()
            .flat_map(|edge| [edge, edge.reverse()])
            .collect();
        edges.sort_unstable();
        // Remove doubled Self-Loops
        edges.dedup();

        let mut self_loops = 0;

        let mut offsets: Vec<NumEdges> = Vec::with_capacity(n as usize + 1);
        offsets.push(0);

        let mut curr_node = 0;
        let mut counter = 0;
        let edges: Vec<Node> = edges
            .into_iter()
            .map(|Edge(u, v)| {
                while u > curr_node {
                    offsets.push(counter);
                    curr_node += 1;
                }
                counter += 1;
                self_loops += (u == v) as NumNodes;

                v
            })
            .collect();

        assert!(offsets.len() as NumNodes <= n + 1);
        while offsets.len() as NumNodes <= n {
            offsets.push(edges.len() as NumEdges);
        }

        Self {
            nbs: SlicedBuffer::new(edges, offsets),
            self_loops,
        }
    }

    fn from_try_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        Self::from_edges(n, edges)
    }
}

impl GraphFromScratch for CrossCsrGraph {
    fn from_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        assert!(n > 0);
        let n = n as usize;

        let mut num_self_loops = 0usize;

        let mut num_of_neighbors: Vec<NumNodes> = vec![0; n];
        let mut temp_edges: Vec<Edge> = edges
            .into_iter()
            .map(|edge| {
                let Edge(u, v) = edge.into().normalized();

                num_of_neighbors[u as usize] += 1;
                if u != v {
                    num_of_neighbors[v as usize] += 1;
                } else {
                    num_self_loops += 1;
                }

                Edge(u, v)
            })
            .collect();
        temp_edges.sort_unstable();
        temp_edges.dedup();

        let m = temp_edges.len() * 2 - num_self_loops;

        let mut offsets = Vec::with_capacity(n + 1);
        let mut edges = vec![NodeWithCrossPos::default(); m];

        offsets.push(0);

        let mut running_offset = num_of_neighbors[0];
        for num_nb_u in num_of_neighbors.iter_mut().skip(1) {
            offsets.push(running_offset as NumEdges);
            running_offset += *num_nb_u;
            *num_nb_u = 0;
        }

        offsets.push(running_offset as NumEdges);
        num_of_neighbors[0] = 0;

        for Edge(u, v) in temp_edges {
            let addr_u = offsets[u as usize] as usize + num_of_neighbors[u as usize] as usize;
            let addr_v = offsets[v as usize] as usize + num_of_neighbors[v as usize] as usize;

            edges[addr_u] = NodeWithCrossPos {
                node: v,
                cross_pos: num_of_neighbors[v as usize],
            };
            num_of_neighbors[u as usize] += 1;

            if u != v {
                edges[addr_v] = NodeWithCrossPos {
                    node: u,
                    cross_pos: num_of_neighbors[u as usize],
                };
                num_of_neighbors[v as usize] += 1;
            }
        }

        Self {
            nbs: SlicedBuffer::new(edges, offsets),
            self_loops: num_self_loops as NumNodes,
        }
    }

    fn from_try_edges<I, E>(n: NumNodes, edges: I) -> Self
    where
        E: Into<Edge>,
        I: IntoIterator<Item = E>,
    {
        Self::from_edges(n, edges)
    }
}

// ---------- Testing ----------

test_graph_ops!(
    test_csr_graph,
    CsrGraph,
    false,
    (AdjacencyList, DirectedAdjacencyList)
);

test_graph_ops!(
    test_csr_graph_in,
    CsrGraphIn,
    false,
    (AdjacencyList, DirectedAdjacencyList)
);

test_graph_ops!(test_csr_graph_undir, CsrGraphUndir, true, (AdjacencyList));

test_graph_ops!(test_cross_csr_graph, CrossCsrGraph, true, (AdjacencyList));
