/*!
# Path Iterator

This module provides functionality to iterate over *induced subpaths* in an undirected graph.

An induced subpath is defined as a maximal sequence of consecutive nodes of degree 2
(with length ≥ 1). The returned path is represented as a vector of nodes, starting and
ending with endpoints (nodes of degree ≠ 2) or identical nodes in the case of cycles.

## Key features
- Iteration over all induced paths in a graph.
- Support for filtering paths by minimum number of path-internal nodes.
- Handles both open paths and induced cycles correctly.
*/

use super::*;
use itertools::Itertools;

/// Provides iterators over induced paths in an undirected graph.
///
/// A path is represented as a vector of nodes where the endpoints are the first and
/// last entries. Endpoints have degree ≠ 2 unless the path is part of an induced cycle.
pub trait PathIterator: AdjacencyList + GraphType<Dir = Undirected> {
    /// Returns an iterator over all induced paths in the graph.
    ///
    /// Each path is represented as a `Vec<Node>` where:
    /// - The first and last elements are endpoints (degree ≠ 2), except in the case of induced cycles,
    ///   where they are equal.
    /// - Internal nodes always have degree 2.
    fn path_iter(&self) -> Paths<'_, Self>;

    /// Returns an iterator over induced paths with at least `min_length` path nodes.
    ///
    /// A *path node* is an internal node of degree 2, not counting the endpoints.
    /// Therefore, the returned vector for each path has length at least `min_length + 2`.
    fn path_iter_with_atleast_path_nodes(&self, min_path_nodes: NumNodes) -> Paths<'_, Self>;
}

impl<G> PathIterator for G
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    fn path_iter(&self) -> Paths<'_, Self> {
        Paths::new(self)
    }

    /// Same as `path_iter`, but does not return paths with less than
    /// `min_length` path nodes; i.e. excluding the end pointer.
    /// In other words: if a path is returned, the vector has length
    /// at leasat `min_length + 2`
    fn path_iter_with_atleast_path_nodes(&self, min_path_nodes: NumNodes) -> Paths<'_, Self> {
        let mut paths = Paths::new(self);
        paths.set_min_path_nodes(min_path_nodes);
        paths
    }
}

/// Iterator structure for enumerating induced paths in an undirected graph.
///
/// Holds state during traversal including visited nodes, current position, and
/// minimum path length constraints.
pub struct Paths<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    graph: &'a G,
    min_length: NumNodes,
    visited: NodeBitSet,
    search_at: Node,
}

impl<'a, G> Paths<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    fn new(graph: &'a G) -> Self {
        Self {
            graph,
            min_length: 0,
            visited: graph.vertex_bitset_unset(),
            search_at: 0,
        }
    }

    /// Sets the minimum number of internal path nodes required for yielded paths.
    ///
    /// Paths shorter than this will be skipped.
    pub fn set_min_path_nodes(&mut self, min_length: NumNodes) {
        self.min_length = min_length;
    }

    /// Builder-style variant of [`Self::set_min_path_nodes`].
    pub fn min_path_nodes(mut self, min_length: NumNodes) -> Self {
        self.set_min_path_nodes(min_length);
        self
    }

    fn complete_path(&mut self, u: Node, parent: Node, path: &mut Vec<Node>) {
        if self.graph.degree_of(u) != 2 || self.visited.set_bit(u) {
            path.push(u);
            return;
        }

        if let Some((n1, n2)) = self.graph.neighbors_of(u).collect_tuple() {
            path.push(u);
            if n1 != parent {
                self.complete_path(n1, u, path);
            }
            if n2 != parent {
                self.complete_path(n2, u, path);
            }
        }
    }
}

impl<G> Iterator for Paths<'_, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut path = Vec::with_capacity(2 + self.min_length as usize);
        while self.search_at < self.graph.number_of_nodes() {
            let start_node = self.search_at;
            self.search_at += 1;

            if self.graph.degree_of(start_node) != 2 || self.visited.set_bit(start_node) {
                continue;
            }

            let (n1, n2) = self.graph.neighbors_of(start_node).collect_tuple().unwrap();
            path.push(start_node);
            self.complete_path(n1, start_node, &mut path);
            path.reverse();

            if path.len() == 1 || path[0] != start_node {
                // we need the check to properly deal with circles
                self.complete_path(n2, start_node, &mut path);
            }

            if 2 + self.min_length as usize > path.len() {
                path.clear();
                continue;
            }

            return Some(path);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, seq::SliceRandom};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    fn match_or_reverse(a: &[Node], b: &[Node]) -> bool {
        if a == b {
            return true;
        }
        let mut c = Vec::from(b);
        c.reverse();
        a == c
    }

    #[test]
    fn single_path() {
        let graph = AdjArrayUndir::from_edges(4, [(0, 3), (3, 2), (2, 1)]);
        let paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 1);
        assert!(
            paths[0] == vec![0, 3, 2, 1] || paths[0] == vec![1, 2, 3, 0],
            "paths: {paths:?}"
        );
    }

    #[test]
    fn single_path_high_degree() {
        let graph = AdjArrayUndir::from_edges(6, [(0, 3), (3, 2), (2, 1), (0, 4), (0, 5)]);
        let paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 1);
        assert!(
            paths[0] == vec![0, 3, 2, 1] || paths[0] == vec![1, 2, 3, 0],
            "paths: {paths:?}"
        );
    }

    #[test]
    fn starlike() {
        let graph = AdjArrayUndir::from_edges(
            10,
            [
                (0, 1),
                (1, 2),
                (0, 3),
                (3, 4),
                (4, 5),
                (0, 6),
                (6, 7),
                (7, 8),
                (8, 9),
            ],
        );
        let mut paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 3);
        paths.sort_by_key(|x| x.len());

        assert!(match_or_reverse(&paths[0], &[0, 1, 2]));
        assert!(match_or_reverse(&paths[1], &[0, 3, 4, 5]));
        assert!(match_or_reverse(&paths[2], &[0, 6, 7, 8, 9]));
    }

    #[test]
    fn single_circle() {
        let graph = AdjArrayUndir::from_edges(4, [(0, 3), (3, 2), (2, 1), (1, 0)]);
        let paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 5, "{paths:?}");
        assert_eq!(paths[0][0], paths[0][4], "{paths:?}");
    }

    #[test]
    fn randomized_single_path() {
        let mut rng = Pcg64Mcg::seed_from_u64(12345);
        for _ in 0..1000 {
            let n = rng.random_range(3..100);
            let mut order = (0..n as Node).collect_vec();
            order.shuffle(&mut rng);

            let edges = (0..n - 1).map(|i| (order[i], order[i + 1])).collect_vec();
            let graph = AdjArrayUndir::from_edges(n as Node, &edges);
            let paths = graph.path_iter().collect_vec();
            assert_eq!(paths.len(), 1, "{order:?}");

            assert!(match_or_reverse(&paths[0], &order));
        }
    }

    #[test]
    fn randomized_two_path() {
        let mut rng = Pcg64Mcg::seed_from_u64(12345);
        for _ in 0..1000 {
            let n = rng.random_range(6..100);
            let n1 = rng.random_range(3..n - 2);
            if 2 * n1 == n {
                // if both subproblems have the same length, we would need addtional logic to find the correct path order; so ignore that case
                continue;
            }
            let mut order = (0..n as Node).collect_vec();
            order.shuffle(&mut rng);

            let edges = (0..n - 1)
                .filter_map(|i| (i + 1 != n1).then_some((order[i], order[i + 1])))
                .collect_vec();
            let graph = AdjArrayUndir::from_edges(n as Node, &edges);
            let mut paths = graph.path_iter().collect_vec();
            assert_eq!(paths.len(), 2, "{order:?}");

            if paths[0].len() != n1 {
                paths.reverse();
            }
            assert_eq!(paths[0].len(), n1, "{paths:?}");
            assert_eq!(paths[1].len(), n - n1, "{paths:?}");

            assert!(
                match_or_reverse(&paths[0], &order[..n1]),
                "{paths:?} {n1} {order:?}"
            );
            assert!(
                match_or_reverse(&paths[1], &order[n1..]),
                "{paths:?} {n1} {order:?}"
            );
        }
    }
}
