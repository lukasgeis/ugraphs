/*!
# Bridge-Finding Algorithms

This module provides functionality for detecting **bridges** in undirected graphs.

- A **bridge** (or cut-edge) is an edge whose removal increases the number of connected components.
- Implements an efficient depth-first search–based algorithm for computing all bridges.
- Exposes a trait interface for computing bridges on any undirected graph type.
*/

use super::*;

/// A trait for computing **bridges** in undirected graphs.
///
/// A *bridge* is an edge that, if removed, increases the number of connected components
/// of the graph.
pub trait Bridges: GraphType<Dir = Undirected> {
    /// Computes all bridges in the graph and returns them as a vector of edges.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*, gens::*};
    ///
    /// let mut g = AdjArrayUndir::new(10);
    /// g.connect_path(0..10 as Node);
    ///
    /// let mut bridges = g.compute_bridges();
    /// bridges.sort_unstable();
    /// assert_eq!(bridges, g.ordered_edges(true).collect::<Vec<Edge>>());
    /// ```
    fn compute_bridges(&self) -> Vec<Edge>;
}

impl<G> Bridges for G
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    fn compute_bridges(&self) -> Vec<Edge> {
        BridgeSearch::new(self).compute()
    }
}

/// Helper struct that implements the depth-first search–based bridge-finding algorithm.
///
/// Maintains:
/// - A reference to the input graph
/// - A visited set of nodes
/// - Per-node discovery and low-link information
/// - The current DFS time counter
/// - The list of discovered bridges
struct BridgeSearch<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    graph: &'a G,
    visited: NodeBitSet,
    nodes_info: Vec<NodeInfo>,
    time: Node,
    bridges: Vec<Edge>,
}

impl<'a, G> BridgeSearch<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    /// Creates a new `BridgeSearch` instance for the given graph.
    fn new(graph: &'a G) -> Self {
        let n = graph.number_of_nodes();
        Self {
            graph,
            visited: NodeBitSet::new(n),
            nodes_info: vec![NodeInfo::default(); n as usize],
            time: 0,
            bridges: Vec::new(),
        }
    }

    /// Executes the bridge-finding algorithm and returns the list of bridges.
    fn compute(mut self) -> Vec<Edge> {
        for u in self.graph.vertices_no_singletons() {
            if self.visited.set_bit(u) {
                continue;
            }

            self.compute_node(u, u);
        }

        self.bridges
    }

    /// Recursive DFS routine for processing node `u` with parent `parent`.
    ///
    /// - Updates discovery and low values
    /// - Detects bridges when a child’s low value is higher than the parent’s discovery time
    fn compute_node(&mut self, parent: Node, u: Node) -> NodeInfo {
        self.time += 1;

        self.nodes_info[u as usize] = NodeInfo {
            parent,
            discovery: self.time,
            low: self.time,
        };

        for v in self.graph.neighbors_of(u) {
            if !self.visited.set_bit(v) {
                let info_v = self.compute_node(u, v);

                self.nodes_info[u as usize].update_low(info_v.low);

                if info_v.low > self.nodes_info[u as usize].discovery {
                    self.bridges.push(Edge(u, v));
                }
            } else if v != self.nodes_info[u as usize].parent {
                let v_disc = self.nodes_info[v as usize].discovery;
                self.nodes_info[u as usize].update_low(v_disc);
            }
        }

        self.nodes_info[u as usize]
    }
}

/// Stores DFS metadata for a single node during bridge search:
/// - `discovery`: the DFS discovery time of the node
/// - `low`: the lowest reachable discovery time via back edges
/// - `parent`: the parent of the node in the DFS tree
#[derive(Clone, Copy, Default)]
struct NodeInfo {
    low: Node,
    discovery: Node,
    parent: Node,
}

impl NodeInfo {
    /// Updates the `low` value with the minimum of the current `low` and the given value.
    fn update_low(&mut self, value: Node) {
        self.low = self.low.min(value);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn bridges_in_path() {
        for n in [1, 5, 10, 15] {
            let mut graph = AdjArrayUndir::new(n);
            for u in 0..(n - 1) {
                graph.add_edge(u, u + 1);
            }

            let mut bridges = graph.compute_bridges();
            bridges.sort();

            assert_eq!(bridges, graph.ordered_edges(true).collect_vec());
        }
    }

    #[test]
    fn bridge_in_example() {
        let mut graph = AdjArrayUndir::new(6);
        graph.add_edges([(0, 1), (0, 2), (2, 1), (1, 3), (3, 4), (4, 5), (5, 3)]);

        assert_eq!(graph.compute_bridges(), vec![Edge(1, 3)]);
    }
}
