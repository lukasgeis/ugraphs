/*!
# Substructure Generators

This module provides utility methods to generate additional **substructures**
inside an already existing graph.

It allows adding common motifs such as:

- **Paths**
- **Cycles**
- **Cliques**

These methods are useful when enriching a graph with specific structures for
testing algorithms, generating benchmark instances, or modeling networks with
known sub-components.

# Example

```
use ugraphs::{prelude::*, gens::*};

let mut g = AdjArray::new(5);
g.connect_path([0, 1, 2]);
g.connect_cycle([2, 3, 4]);
g.connect_clique(&NodeBitSet::new_with_bits_set(5, [0 as Node, 2, 4]), false);

assert_eq!(
    g.ordered_edges(false).collect::<Vec<Edge>>(),
    vec![Edge(0, 1), Edge(0, 2), Edge(0, 4), Edge(1, 2), Edge(2, 0), Edge(2, 3), Edge(2, 4), Edge(3, 4), Edge(4, 0), Edge(4, 2)]
);
```
*/

use itertools::Itertools;

use crate::utils::Set;

use super::*;

/// Trait for creating additional **substructures** (paths, cycles, cliques)
/// inside an already existing graph.
///
/// Implemented for all graphs that support edge editing and type queries.
pub trait GeneratorSubstructures {
    /// Connects the given nodes in order with a **simple path**.
    ///
    /// Each consecutive pair of nodes is connected by a single edge.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let mut g = AdjArray::new(4);
    /// g.connect_path([0, 1, 2, 3]);
    ///
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 2));
    /// assert!(g.has_edge(2, 3));
    /// ```
    fn connect_path<P>(&mut self, nodes_on_path: P)
    where
        P: IntoIterator<Item = Node>;

    /// Connects the given nodes with a **cycle**.
    ///
    /// - Consecutive nodes are connected by edges.
    /// - Additionally, the last node is connected back to the first.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let mut g = AdjArray::new(3);
    /// g.connect_cycle([0, 1, 2]);
    ///
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 2));
    /// assert!(g.has_edge(2, 0));
    /// ```
    fn connect_cycle<C>(&mut self, nodes_in_cycle: C)
    where
        C: IntoIterator<Item = Node>;

    /// Connects all given nodes into a **clique** (complete subgraph).
    ///
    /// - If `with_loops` is `true`, each node also gets a self-loop.
    /// - For undirected graphs, edges are normalized so duplicates are avoided.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let mut g = AdjArray::new(3);
    /// g.connect_clique(&NodeBitSet::new_with_bits_set(3, [0 as Node, 1, 2]), false);
    ///
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 2));
    /// assert!(g.has_edge(0, 2));
    /// ```
    fn connect_clique<C: Set<Node>>(&mut self, nodes: &C, with_loops: bool);
}

impl<G> GeneratorSubstructures for G
where
    G: GraphEdgeEditing + GraphType,
{
    fn connect_path<P>(&mut self, nodes_on_path: P)
    where
        P: IntoIterator<Item = Node>,
    {
        for (u, v) in nodes_on_path.into_iter().tuple_windows() {
            self.add_edge(u, v);
        }
    }

    fn connect_cycle<C>(&mut self, nodes_in_cycle: C)
    where
        C: IntoIterator<Item = Node>,
    {
        let mut iter = nodes_in_cycle.into_iter();

        // we use a rather tedious implementation to avoid needing to clone the iterator
        if let Some(first) = iter.next() {
            let mut prev = first;
            for cur in iter {
                self.add_edge(prev, cur);
                prev = cur;
            }

            self.add_edge(prev, first);
        }
    }

    fn connect_clique<C: Set<Node>>(&mut self, nodes: &C, with_loops: bool) {
        for u in nodes.iter() {
            for v in nodes.iter() {
                let e = Edge(u, v);
                if (!with_loops && e.is_loop()) || (Self::is_undirected() && !e.is_normalized()) {
                    continue;
                }

                self.try_add_edge(u as Node, v as Node);
            }
        }
    }
}

/// Trait for constructing new graphs that are initialized with
/// common **substructures**: paths, cycles, and cliques.
///
/// Implemented for all graph types that support creation and edge editing.
///
/// This is complementary to [`GeneratorSubstructures`], which modifies
/// an *existing* graph by adding substructures, whereas
/// `NewStructuredGraph` creates a fresh graph directly.
pub trait NewStructuredGraph {
    /// Creates a new graph with `n` nodes arranged in a **path**.
    ///
    /// - Nodes are numbered `0..n`.
    /// - Each consecutive pair `(i, i+1)` is connected by an edge.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let g = AdjArray::path(3);
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 2));
    /// ```
    fn path(n: NumNodes) -> Self;

    /// Creates a new graph with `n` nodes arranged in a **cycle**.
    ///
    /// - Nodes are numbered `0..n`.
    /// - Each consecutive pair `(i, i+1)` is connected by an edge.
    /// - Additionally, the last node `(n-1)` is connected back to `0`.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let g = AdjArray::cycle(3);
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 2));
    /// assert!(g.has_edge(2, 0));
    /// ```
    fn cycle(n: NumNodes) -> Self;

    /// Creates a new graph with `n` nodes arranged as a **clique** (complete graph).
    ///
    /// - Every pair of nodes is connected by an edge.
    /// - If `self_loops` is `true`, each node additionally has a loop `(u, u)`.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let g = AdjArrayUndir::clique(3, false);
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 2));
    /// assert!(g.has_edge(0, 2));
    /// ```
    fn clique(n: NumNodes, self_loops: bool) -> Self;

    /// Creates a new grid graph with `n * n` nodes arranged as a `n x n` grid.
    /// Nodes are numbered as follows:
    /// ```text
    ///   0       1     ...  n-1
    ///   n      n+1    ... 2n-1
    ///   .       .     ...   .
    ///   .       .     ...   .
    ///   .       .     ...   .
    /// n(n-1) n(n-1)+1 ... n^2-1
    /// ```
    ///
    /// If the graph is directed, edges are connected via 2-cycles.
    ///
    /// ```
    /// use ugraphs::{prelude::*, gens::*};
    ///
    /// let g = AdjArrayUndir::grid(2);
    /// assert_eq!(g.number_of_nodes(), 4);
    /// assert_eq!(g.number_of_edges(), 4);
    /// assert_eq!(
    ///     g.ordered_edges(true).collect::<Vec<Edge>>(),
    ///     vec![Edge(0,1), Edge(0,2), Edge(1,3), Edge(2,3)]
    /// );
    /// ```
    fn grid(n: NumNodes) -> Self;
}

impl<G> NewStructuredGraph for G
where
    G: GraphNew + GraphEdgeEditing + GraphType,
{
    fn path(n: NumNodes) -> Self {
        let mut g = Self::new(n);
        g.connect_path(0..n as Node);
        g
    }

    fn cycle(n: NumNodes) -> Self {
        let mut g = Self::new(n);
        g.connect_cycle(0..n as Node);
        g
    }

    fn clique(n: NumNodes, self_loops: bool) -> Self {
        let mut g = Self::new(n);
        g.connect_clique(&NodeBitSet::new_all_set(n), self_loops);
        g
    }

    fn grid(n: NumNodes) -> Self {
        let mut g = Self::new(n * n);

        for i in 1..n {
            for u in 0..n {
                // Horizontal
                g.add_edge(u * n + i - 1, u * n + i);
                // Vertical
                g.add_edge(u + n * i - n, u + n * i);

                if G::is_directed() {
                    // Horizontal
                    g.add_edge(u * n + i, u * n + i - 1);
                    // Vertical
                    g.add_edge(u + n * i, u + n * i - n);
                }
            }
        }

        g
    }
}

#[cfg(test)]
mod tests {
    use crate::{ops::*, repr::AdjArrayMatrix};

    use super::*;

    #[test]
    fn test_connect_path() {
        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_path([]);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_path([1]);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_path([2, 1]);
            assert_eq!(g.number_of_edges(), 1);
            assert!(g.has_edge(2, 1));
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_path([0, 3, 1, 4]);
            assert_eq!(
                g.edges(false).collect_vec(),
                vec![Edge(0, 3), Edge(1, 4), Edge(3, 1)]
            );
        }
    }

    #[test]
    fn test_connect_cycle() {
        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_cycle([]);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_cycle([1]);
            assert_eq!(g.number_of_edges(), 1);
            assert!(g.has_edge(1, 1));
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_cycle([0, 3, 1, 4]);
            assert_eq!(
                g.edges(false).collect_vec(),
                vec![Edge(0, 3), Edge(1, 4), Edge(3, 1), Edge(4, 0)]
            );
        }
    }

    #[test]
    fn test_connect_nodes() {
        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_clique(&NodeBitSet::new(6), true);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_clique(&NodeBitSet::new_with_bits_set(6, [1u32]), false);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_clique(&NodeBitSet::new_with_bits_set(6, [1u32]), true);
            assert_eq!(g.number_of_edges(), 1);
            assert!(g.has_edge(1, 1));
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_clique(&NodeBitSet::new_with_bits_set(6, [1u32, 2, 4]), false);
            assert_eq!(g.number_of_edges(), 6);
        }

        {
            let mut g = AdjArrayMatrix::new(6);
            g.connect_clique(&NodeBitSet::new_with_bits_set(6, [1u32, 2, 4]), true);
            assert_eq!(g.number_of_edges(), 9);
        }
    }
}
