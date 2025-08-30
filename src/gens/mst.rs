/*!
# Random Minimum Spanning Tree (MST)

This module provides generators for random **minimum spanning trees** (MSTs).
Unlike general random graph generators, MST generation ensures:

- The graph is connected.
- It contains exactly `n-1` edges for `n` nodes.
- If directed, all edges are oriented away from a designated root (default `0`).

The generator uses a naive random-walk approach without explicit loops to
incrementally grow a spanning tree.

# Examples

```
use ugraphs::gens::*;

let mut rng = rand::rng();
// Generate a random MST on 5 nodes, rooted at 0
let mst = Mst::new().nodes(5).root(0);
let edges: Vec<_> = mst.generate(&mut rng);

assert_eq!(edges.len(), 4); // Always n-1 edges
```
*/

use rand::Rng;
use rand_distr::{Distribution, Uniform};

use super::*;

/// Generator for a random **Minimum Spanning Tree (MST)**.
///
/// The MST generator creates a tree of `n` nodes where edges are chosen randomly
/// in a way that avoids cycles. If the graph is directed, all edges are oriented
/// away from the specified root node (default: `0`).
#[derive(Debug, Copy, Clone, Default)]
pub struct Mst {
    n: NumNodes,
    root: Node,
}

impl Mst {
    /// Creates a new MST generator with default settings.
    ///
    /// By default:
    /// - `n = 0`
    /// - `root = 0`
    ///
    /// # Example
    /// ```
    /// use ugraphs::gens::*;
    ///
    /// let mst = Mst::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the root node of the MST.
    ///
    /// All edges in a directed MST are oriented away from this root.
    ///
    /// # Panics
    /// Panics if `root >= n` when used in generation.
    pub fn set_root(&mut self, root: Node) {
        self.root = root;
    }

    /// Sets the root node of the MST.
    ///
    /// All edges in a directed MST are oriented away from this root.
    ///
    /// # Panics
    /// Panics if `root >= n` when used in generation.
    pub fn root(mut self, root: Node) -> Self {
        self.set_root(root);
        self
    }
}

impl NumNodesGen for Mst {
    fn set_nodes(&mut self, n: NumNodes) {
        self.n = n;
    }
}

impl GraphGenerator for Mst {
    type EdgeStream<'a, R>
        = MstGenerator<'a, R>
    where
        R: Rng + 'a,
        Self: 'a;

    fn stream<'a, R>(&'a self, rng: &'a mut R) -> Self::EdgeStream<'a, R>
    where
        R: Rng,
    {
        MstGenerator::new(self.n, self.root, rng)
    }
}

/// Streaming generator for MST edges using a random-walk method.
///
/// This generator avoids explicit loops by marking visited nodes
/// with bitsets and incrementally attaching them to the connected
/// component until a spanning tree is formed.
///
/// Yields exactly `n-1` edges, where `n` is the number of nodes.
///
/// Implements [`Iterator`] with `Item = Edge`.
///
/// # Internal Algorithm
///
/// - Starts with the root as connected.
/// - Performs a random walk over unconnected nodes.
/// - When hitting a new node, connects it to the tree and yields an edge.
/// - Ensures no cycles by rejecting already-on-path nodes.
///
/// # Example
/// ```
/// use ugraphs::gens::*;
///
/// let mut rng = rand::rng();
/// let mut mst = MstGenerator::new(5, 0, &mut rng);
///
/// let edges: Vec<_> = mst.collect();
/// assert_eq!(edges.len(), 4);
/// ```
pub struct MstGenerator<'a, R>
where
    R: Rng,
{
    rng: &'a mut R,
    node_gen: Uniform<Node>,
    connected: NodeBitSet,
    on_path: NodeBitSet,
    path: Vec<Node>,
    path_skip: usize,
}

impl<'a, R> MstGenerator<'a, R>
where
    R: Rng,
{
    /// Creates a new [`MstGenerator`] for a tree of `n` nodes rooted at `root`.
    ///
    /// # Panics
    /// Panics if `root >= n`.
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

impl<'a, R> Iterator for MstGenerator<'a, R>
where
    R: Rng,
{
    type Item = Edge;

    /// Advances the MST generation and returns the next edge.
    ///
    /// - If there are still nodes to connect, yields an edge
    ///   that attaches a new node to the existing tree.
    /// - Once all nodes are connected, returns `None`.
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
