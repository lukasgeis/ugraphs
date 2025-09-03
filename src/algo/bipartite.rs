/*!
# Bipartite Graph Algorithms

This module provides traits and algorithms for working with **bipartite graphs**.

Functionality includes:
- Defining and handling bipartitions of a graph
- Testing whether a graph is bipartite
- Computing a valid bipartition if one exists
- Editing a graph to remove edges that violate bipartiteness
*/

use super::{traversal::*, *};

/// A trait for representing a bipartition of the node set.
///
/// - Nodes in the set are considered to be on the **right** (1) side  
/// - Nodes not in the set are considered to be on the **left** (0) side
///
/// Provides convenience methods to check the side of a node.
pub trait Bipartition: Set<Node> {
    /// Returns `true` if the node is on the left (0) side of the partition.
    fn is_on_left_side(&self, u: Node) -> bool;

    /// Returns `true` if the node is on the right (1) side of the partition.
    fn is_on_right_side(&self, u: Node) -> bool;
}

impl<B> Bipartition for B
where
    B: Set<Node>,
{
    #[inline]
    fn is_on_left_side(&self, u: Node) -> bool {
        !self.contains(&u)
    }

    #[inline]
    fn is_on_right_side(&self, u: Node) -> bool {
        self.contains(&u)
    }
}

/// A trait for testing and computing bipartitions in graphs.
///
/// Provides methods to:
/// - Verify whether a given bipartition is valid
/// - Compute a bipartition of the graph, if one exists
/// - Test whether the graph is bipartite
pub trait BipartiteTest {
    /// Tests whether the given candidate partition is a valid bipartition.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*, gens::*};
    ///
    /// let mut g = AdjArrayUndir::new(10);
    /// g.connect_path(0..10 as Node);
    ///
    /// assert!(g.is_bipartition(&NodeBitSet::new_with_bits_set(10,  vec![0 as Node, 2, 4, 6, 8])));
    /// ```
    fn is_bipartition<B>(&self, bipartition: &B) -> bool
    where
        B: Bipartition;

    /// Computes a valid bipartition of the graph, if one exists.
    /// Returns `None` if the graph is not bipartite.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*, gens::*};
    ///
    /// let mut g = AdjArrayUndir::new(10);
    /// g.connect_path(0..10 as Node);
    ///
    /// let bip: NodeBitSet = g.compute_bipartition().unwrap();
    /// assert_eq!(bip.cardinality(), 5);
    /// ```
    fn compute_bipartition<B>(&self) -> Option<B>
    where
        B: Bipartition + FromCapacity;

    /// Tests whether the graph is bipartite.
    ///
    /// This is equivalent to checking whether `compute_bipartition` succeeds
    /// when using `NodeBitSet` as the underlying bipartition representation.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*, gens::*};
    ///
    /// let mut g = AdjArrayUndir::new(10);
    /// g.connect_path(0..10 as Node);
    ///
    /// assert!(g.is_bipartite());
    /// ```
    fn is_bipartite(&self) -> bool {
        self.compute_bipartition::<NodeBitSet>().is_some()
    }
}

impl<G> BipartiteTest for G
where
    G: AdjacencyList,
{
    fn is_bipartition<B>(&self, bipartition: &B) -> bool
    where
        B: Bipartition,
    {
        self.edges(false)
            .all(|Edge(u, v)| bipartition.is_on_left_side(u) != bipartition.is_on_left_side(v))
    }

    fn compute_bipartition<B>(&self) -> Option<B>
    where
        B: Bipartition + FromCapacity,
    {
        let bipartition = propose_possibly_illegal_bipartition(self);
        self.is_bipartition(&bipartition).then_some(bipartition)
    }
}

/// A trait for editing graphs with respect to bipartitions.
///
/// Provides methods to enforce bipartiteness by removing edges that connect nodes
/// within the same side of a bipartition.
pub trait BipartiteEdit {
    /// Removes all edges that connect nodes within the same bipartition class.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*, gens::*};
    ///
    /// let mut g = AdjArrayUndir::new(10);
    /// g.connect_path(0..10 as Node);
    ///
    /// let bip = NodeBitSet::new_with_bits_set(10, vec![0 as Node, 2, 4, 6, 8]);
    /// g.connect_clique(&bip, false);
    ///
    /// assert!(!g.is_bipartite());
    ///
    /// g.remove_edges_within_bipartition_class(&bip);
    /// assert!(g.is_bipartition(&bip));
    /// ```
    fn remove_edges_within_bipartition_class<B>(&mut self, bipartition: &B)
    where
        B: Bipartition;
}

impl<G> BipartiteEdit for G
where
    G: AdjacencyList + GraphEdgeEditing + GraphType,
{
    fn remove_edges_within_bipartition_class<B>(&mut self, bipartition: &B)
    where
        B: Bipartition,
    {
        let to_delete: Vec<_> = self
            .edges(G::is_undirected())
            .filter(|&Edge(u, v)| bipartition.is_on_left_side(u) == bipartition.is_on_left_side(v))
            .collect();
        self.remove_edges(to_delete);
    }
}

/// Computes a candidate bipartition of the graph using BFS traversal.
///
/// - If the graph is bipartite, the returned partition is valid
/// - If the graph is not bipartite, the returned partition may be invalid
///
/// Used internally as a heuristic before validation.
fn propose_possibly_illegal_bipartition<G, B>(graph: &G) -> B
where
    G: AdjacencyList,
    B: Bipartition + FromCapacity,
{
    let mut bfs = graph.bfs_with_predecessor(0);

    let mut bipartition = B::from_total_used_capacity(
        graph.number_of_nodes() as usize,
        graph.number_of_nodes() as usize,
    );

    loop {
        for (node, pred) in bfs
            .by_ref()
            .filter_map(|x| Some((x.item(), x.predecessor()?)))
        {
            if !bipartition.contains(&pred) {
                bipartition.insert(node);
            }
        }

        if !bfs.try_restart_at_unvisited() {
            break;
        }
    }

    bipartition
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn path() {
        for n in 1..10 {
            let mut graph = AdjArrayUndir::new(n);
            for u in 0..n - 1 {
                graph.add_edge(u, u + 1);
            }

            assert!(graph.is_bipartite());

            if n > 2 {
                let mut graph = graph.clone();
                graph.remove_edge(n / 2, n / 2 + 1);
                assert!(graph.is_bipartite());
            }

            if n > 2 {
                let mut graph = graph.clone();
                graph.add_edge(1 - (n % 2), n - 1);
                assert!(!graph.is_bipartite());
            }
        }
    }
}
