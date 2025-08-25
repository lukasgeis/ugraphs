use crate::{
    ops::*,
    utils::{FromCapacity, Set},
    *,
};

use super::*;

/// We define a Bipartition over a set of nodes where nodes in the Set are considered to be on the
/// 'right' side whereas nodes not in the Set are considered to be on the 'left' side.
pub trait Bipartition: Set<Node> {
    #[inline]
    fn is_on_left_side(&self, u: Node) -> bool {
        !self.contains(&u)
    }

    #[inline]
    fn is_on_right_side(&self, u: Node) -> bool {
        self.contains(&u)
    }
}

impl<B: Set<Node>> Bipartition for B {}

pub trait BipartiteTest {
    /// Tests whether the given candidate partition is a valid bipartition.
    fn is_bipartition<B: Bipartition>(&self, bipartition: &B) -> bool;

    /// Computes a valid bipartition of the graph, if one exists.
    fn compute_bipartition<B: Bipartition + FromCapacity>(&self) -> Option<B>;

    /// Tests whether the graph is bipartite.
    fn is_bipartite(&self) -> bool {
        self.compute_bipartition::<NodeBitSet>().is_some()
    }
}

impl<G> BipartiteTest for G
where
    G: AdjacencyList,
{
    fn is_bipartition<B: Bipartition>(&self, bipartition: &B) -> bool {
        self.edges(true)
            .all(|Edge(u, v)| bipartition.is_on_left_side(u) != bipartition.is_on_left_side(v))
    }

    fn compute_bipartition<B: Bipartition + FromCapacity>(&self) -> Option<B> {
        let bipartition = propose_possibly_illegal_bipartition(self);
        self.is_bipartition(&bipartition).then_some(bipartition)
    }
}

pub trait BipartiteEdit {
    /// Remove all edges that connect nodes in the same partition.
    fn remove_edges_within_bipartition_class<B: Bipartition>(&mut self, bipartition: &B);
}

impl<G> BipartiteEdit for G
where
    G: AdjacencyList + GraphEdgeEditing,
{
    fn remove_edges_within_bipartition_class<B: Bipartition>(&mut self, bipartition: &B) {
        let to_delete: Vec<_> = self
            .edges(true)
            .filter(|&Edge(u, v)| bipartition.is_on_left_side(u) == bipartition.is_on_left_side(v))
            .collect();
        self.remove_edges(to_delete);
    }
}

// Compute a bipartition of `graph` if `graph` is bipartite; otherwise an arbitrary
// partition is returned
fn propose_possibly_illegal_bipartition<G: AdjacencyList, B: Bipartition + FromCapacity>(
    graph: &G,
) -> B {
    let mut bfs = graph.bfs_with_predecessor(0);

    // propose a bipartition
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
    use crate::repr::AdjArrayUndir;

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
