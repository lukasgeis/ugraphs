//! # Substructure Generators
//!
//! Provides methods to generate multiple substructures in an already existing graph.

use itertools::Itertools;

use super::*;

/// Trait for creating additional substructures in a graph.
pub trait GeneratorSubstructures {
    // connects the nodes passed with a simple path
    fn connect_path<T>(&mut self, nodes_on_path: T)
    where
        T: IntoIterator<Item = Node>;

    // connects the nodes passed with a simple path and adds a connection between the first and the last node
    fn connect_cycle<T>(&mut self, nodes_in_cycle: T)
    where
        T: IntoIterator<Item = Node>;

    // generates a clique of all nodes includes in the bit
    fn connect_clique(&mut self, nodes: &NodeBitSet, with_loops: bool);
}

impl<G> GeneratorSubstructures for G
where
    G: GraphEdgeEditing + GraphType,
{
    fn connect_path<T>(&mut self, nodes_on_path: T)
    where
        T: IntoIterator<Item = Node>,
    {
        for (u, v) in nodes_on_path.into_iter().tuple_windows() {
            self.add_edge(u, v);
        }
    }

    fn connect_cycle<T>(&mut self, nodes_in_cycle: T)
    where
        T: IntoIterator<Item = Node>,
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

    fn connect_clique(&mut self, nodes: &NodeBitSet, with_loops: bool) {
        for u in nodes.iter_set_bits() {
            for v in nodes.iter_set_bits() {
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
