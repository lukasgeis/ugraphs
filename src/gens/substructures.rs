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

```rust
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
    /// ```rust
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
    /// ```rust
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
    /// ```rust
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
