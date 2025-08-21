//! # Random Minimum Spanning Tree
//!
//! Provides a generator for a random minimum spanning tree.

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use crate::{
    gens::{GraphGenerator, NumNodesGen},
    *,
};

/// Generator for a random Mst where all edges (if directed) are oriented away from a given root (0 by default).
#[derive(Debug, Copy, Clone, Default)]
pub struct Mst {
    n: NumNodes,
    root: Node,
}

impl Mst {
    /// Shorthand for default
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the root of the Mst
    pub fn root(mut self, root: Node) -> Self {
        self.root = root;
        self
    }
}

impl NumNodesGen for Mst {
    fn nodes(mut self, n: NumNodes) -> Self {
        self.n = n;
        self
    }
}

impl GraphGenerator for Mst {
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge> {
        MstGenerator::new(self.n, self.root, rng)
    }
}

/// Generator for a random Mst using a naive loop-less random walk.
pub struct MstGenerator<'a, R: Rng> {
    rng: &'a mut R,
    node_gen: Uniform<Node>,
    connected: NodeBitSet,
    on_path: NodeBitSet,
    path: Vec<Node>,
    path_skip: usize,
}

impl<'a, R: Rng> MstGenerator<'a, R> {
    pub fn new(n: NumNodes, root: Node, rng: &'a mut R) -> Self {
        assert!(root < n as Node);

        Self {
            rng,
            node_gen: Uniform::new(0 as Node, n as Node).unwrap(),
            connected: NodeBitSet::new_with_bits_set(n as Node, [root]),
            on_path: NodeBitSet::new(n as Node),
            path: Vec::new(),
            path_skip: usize::MAX - 1,
        }
    }
}

impl<'a, R: Rng> Iterator for MstGenerator<'a, R> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.path_skip + 1 < self.path.len() {
            self.path_skip += 1;
            return Some(Edge(
                self.path[self.path_skip],
                self.path[self.path_skip - 1],
            ));
        }

        if self.connected.are_all_set() {
            return None;
        }

        self.path_skip = usize::MAX - 1;
        self.on_path.clear_all();
        self.path.clear();

        loop {
            let u = self.node_gen.sample(self.rng);

            if self.path.is_empty() && self.connected.get_bit(u) {
                // TBD: might be a bottleneck on large graphs -> use HashSet
                continue;
            }

            if self.on_path.set_bit(u) {
                // avoid loops
                continue;
            }

            self.path.push(u);

            if self.connected.set_bit(u) {
                self.path_skip = 1;
                return Some(Edge(self.path[1], self.path[0]));
            }
        }
    }
}
