use crate::{ops::*, testing::test_graph_ops, utils::SlicedBuffer, *};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeWithCrossPos {
    pub node: Node,
    pub cross_pos: NumNodes,
}

#[derive(Clone)]
pub struct CsrGraph {
    out_nbs: SlicedBuffer<Node, NumEdges>,
}

#[derive(Clone)]
pub struct CsrGraphIn {
    out_nbs: SlicedBuffer<Node, NumEdges>,
    in_nbs: SlicedBuffer<Node, NumEdges>,
}

#[derive(Clone)]
pub struct CsrGraphUndir {
    nbs: SlicedBuffer<Node, NumEdges>,
    self_loops: NumNodes,
}

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
            fn number_of_nodes(&self) -> NumNodes {
                self.$nbs.number_of_slices()
            }
        }

        impl AdjacencyList for $struct {
            fn neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
                self.$nbs[u].iter().copied()
            }

            fn degree_of(&self, u: Node) -> NumNodes {
                self.$nbs.size_of(u) as NumNodes
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
    fn in_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
        // Should be avoided as it is very inefficient
        self.vertices_range().filter(move |&v| self.has_edge(v, u))
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        // Should be avoided as it is very inefficient
        self.in_neighbors_of(u).count() as NumNodes
    }
}

impl DirectedAdjacencyList for CsrGraphIn {
    fn in_neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
        self.in_nbs[u].iter().copied()
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        self.in_nbs.size_of(u)
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
    fn neighbors_of(&self, u: Node) -> impl Iterator<Item = Node> + '_ {
        self.nbs[u].iter().map(|cp| cp.node)
    }

    fn degree_of(&self, u: Node) -> NumNodes {
        self.nbs.size_of(u) as NumNodes
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

// ---------- GraphFromScratch ----------

impl GraphFromScratch for CsrGraph {
    fn from_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        let mut edges: Vec<Edge> = edges.map(|e| e.into()).collect();
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

    fn from_try_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        Self::from_edges(n, edges)
    }
}

impl GraphFromScratch for CsrGraphIn {
    fn from_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        let mut edges: Vec<Edge> = edges.map(|e| e.into()).collect();
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

    fn from_try_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        Self::from_edges(n, edges)
    }
}

impl GraphFromScratch for CsrGraphUndir {
    fn from_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        let mut edges: Vec<Edge> = edges.map(|e| e.into().normalized()).collect();
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

    fn from_try_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
        Self::from_edges(n, edges)
    }
}

impl GraphFromScratch for CrossCsrGraph {
    fn from_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
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

    fn from_try_edges(n: NumNodes, edges: impl Iterator<Item = impl Into<Edge>>) -> Self {
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
