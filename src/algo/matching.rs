/*!
# Matching Algorithms

This module provides algorithms for computing **matchings** in graphs.

- Supports **maximal matchings** in undirected graphs (greedy, not necessarily optimal).
- Supports **maximum bipartite matchings** using a flow-based approach.

A *matching* is a set of edges without shared endpoints.
- A **maximal matching** cannot be extended by adding another edge, but may not be optimal in size.
- A **maximum matching** is the largest possible matching.
*/

use super::*;
use itertools::Itertools;

/// A trait providing matching algorithms on undirected graphs.
///
/// Implementations provide:
/// - Greedy maximal matching
/// - Maximal matching restricted to an induced subgraph
/// - Maximum bipartite matching via flow reduction
pub trait Matching: GraphType<Dir = Undirected> {
    /// Computes a **maximal matching** in an undirected graph.
    ///
    /// Each edge `{u, v}` in the matching is returned only once as `(u, v)` with `u <= v`.  
    /// The resulting vector is sorted lexicographically.
    fn maximal_undirected_matching(&self) -> Vec<(Node, Node)> {
        self.maximal_undirected_matching_excluding(std::iter::empty())
    }

    /// Computes a **maximal matching** on an induced subgraph of an undirected graph.  
    ///
    /// - The subgraph excludes all vertices provided by the iterator `excl`.  
    /// - Each edge `{u, v}` in the matching is returned only once as `(u, v)` with `u <= v`.  
    /// - The output is sorted lexicographically.  
    ///
    /// Note: The original graph may contain directed edges, but the induced subgraph
    /// must not contain asymmetric edges `(u, v)` without `(v, u)`.
    fn maximal_undirected_matching_excluding<I>(&self, excl: I) -> Vec<(Node, Node)>
    where
        I: IntoIterator<Item = Node>;

    /// Computes a **maximum matching** on a bipartite (sub)graph.  
    ///
    /// - The bipartition is given by two disjoint sets of nodes: `class_a` and `class_b`.  
    /// - Edges within a class and self-loops are ignored.  
    ///
    /// Returns pairs `(a, b)` where `a` in `A` and `b` in `B`.  
    /// The order of the pairs is unspecified.
    fn maximum_bipartite_matching<A, B>(&self, class_a: &A, class_b: &B) -> Vec<(Node, Node)>
    where
        A: Set<Node>,
        B: Set<Node>;
}

impl<G> Matching for G
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    /// Greedy maximal matching implementation:
    /// - Iterates through vertices
    /// - Picks the first available unmatched neighbor
    /// - Marks both endpoints as matched
    fn maximal_undirected_matching_excluding<I>(&self, excl: I) -> Vec<(Node, Node)>
    where
        I: IntoIterator<Item = Node>,
    {
        let mut matching = Vec::new();
        let mut matched = NodeBitSet::new_with_bits_set(self.number_of_nodes(), excl);

        for u in self.vertices() {
            if matched.get_bit(u) {
                continue;
            }

            if let Some(v) = self.neighbors_of(u).find(|&v| !matched.get_bit(v)) {
                matched.set_bit(u);
                matched.set_bit(v);
                matching.push((u, v));
            }
        }

        matching
    }

    /// Maximum bipartite matching via flow reduction:
    /// - Constructs a bipartite flow network with source `s` and sink `t`  
    /// - Runs Edmonds–Karp style flow algorithm  
    /// - Extracts matching edges from residual network
    fn maximum_bipartite_matching<A, B>(&self, class_a: &A, class_b: &B) -> Vec<(Node, Node)>
    where
        A: Set<Node>,
        B: Set<Node>,
    {
        if class_a.is_empty() || class_b.is_empty() {
            return Vec::new();
        }

        debug_assert!(class_a.iter().all(|a| !class_b.contains(&a)));

        let n = 2 + class_a.len() + class_b.len();

        let labels = [self.number_of_nodes(), self.number_of_nodes() + 1]
            .into_iter()
            .chain(class_a.iter().chain(class_b.iter()).map(|u| u as Node))
            .collect_vec();

        let mut network = AdjArray::new(n as NumNodes);
        // edges s -> all nodes in class_a
        for i in 0..class_a.len() {
            network.add_edge(0, 2 + i as Node);
            network.add_edge(2 + i as Node, 0);
        }

        // edges class_a -> class_b
        {
            let mapping_b = {
                let mut mapping_b = vec![n; self.len()];
                for (mapped, org) in class_b.iter().enumerate() {
                    mapping_b[org as usize] = 2 + class_a.len() + mapped;
                }
                mapping_b
            };

            for (ui, u) in class_a.iter().enumerate() {
                for v in self
                    .neighbors_of(u as Node)
                    .map(|v| mapping_b[v as usize])
                    .filter(|&v| v < n)
                {
                    network.add_edge(2 + ui as Node, v as Node);
                }
            }
        }

        // edges class_b -> t
        for v in 0..class_b.len() {
            network.add_edge((2 + class_a.len() + v) as Node, 1);
            network.add_edge(1, (2 + class_a.len() + v) as Node);
        }

        for _ in network.st_flow_keep_changes(|u| labels[u as usize], 0, 1) {} // execute EK to completion

        // iterate over all nodes of class b -- each should have exactly one out neighbor; if its
        // the target t (id = 1), then the node is unmatched; otherwise it's the matching partner
        class_b
            .iter()
            .enumerate()
            .filter_map(|(bi, b)| {
                let bi = (2 + class_a.len() + bi) as Node;
                let b = b as Node;
                debug_assert_eq!(network.degree_of(bi), 1);
                let a = network.neighbors_of(bi).next().unwrap();
                if a == 1 {
                    None
                } else {
                    Some((labels[a as usize], b as Node))
                }
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximal_undirected_matching() {
        let graph = AdjArrayUndir::from_edges(4, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]); // 0 <=> 1 <=> 2 <=> 3
        let matching = graph.maximal_undirected_matching();
        assert!(matching == vec![(0, 1), (2, 3)] || matching == vec![(1, 2)]);
    }

    #[test]
    fn maximal_undirected_matching_excluding() {
        let graph = AdjArrayUndir::from_edges(4, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]); // 0 <=> 1 <=> 2 <=> 3
        let matching = graph.maximal_undirected_matching_excluding(std::iter::once(1));
        assert_eq!(matching, vec![(2, 3)]);
    }

    #[test]
    fn maximum_bipartite_matching() {
        let graph = AdjArrayUndir::from_edges(4, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]); // 0 <=> 1 <=> 2 <=> 3
        let matching = graph.maximum_bipartite_matching(
            &NodeBitSet::new_with_bits_set(4, [0 as Node, 2].into_iter()),
            &NodeBitSet::new_with_bits_set(4, [1 as Node, 3].into_iter()),
        );
        assert_eq!(matching.len(), 2);
    }
}
