/*!
Graph traversal algorithms and traversal-derived utilities.

This module provides:
- Generic traversal iterators (BFS, DFS, with and without predecessor tracking).
- Abstractions (`TraversalSearch`, `TraversalTree`, `RankFromOrder`) that
  turn traversals into useful structures such as parent arrays, rankings,
  or depth arrays.
- Topological ordering for directed acyclic graphs.
- A high-level `Traversal` trait that exposes traversal algorithms
  directly as methods on graph data structures.

The implementation emphasizes composability: traversal iterators can be
combined, filtered, or extended with additional state while still being
efficient and lazy.
*/

use super::*;
use std::{collections::VecDeque, marker::PhantomData};

/// Common interface for maintaining and querying visited-states
/// during a traversal.
///
/// Implementations wrap a [`Set<Node>`] that tracks which nodes
/// have already been discovered.
///
/// This allows traversal algorithms to be parameterized by different
/// set implementations (e.g., `BitSet`, `HashSet`) without changing
/// the traversal logic.
pub trait TraversalState<S>
where
    S: Set<Node>,
{
    /// Returns a reference to the set of visited nodes.
    fn visited(&self) -> &S;

    /// Checks if a given node `u` has already been visited.
    fn did_visit_node(&self, u: Node) -> bool {
        self.visited().contains(&u)
    }
}

/// Abstraction for items yielded by a traversal iterator.
///
/// A `SequencedItem` encodes both the **node currently visited**
/// and an **optional predecessor** that represents its parent
/// in the traversal tree.
///
/// Two implementations are provided:
/// - [`Node`] — stores only the node (no predecessor information).
/// - [`PredecessorOfNode`] — stores `(predecessor, node)` pairs.
pub trait SequencedItem: Clone + Copy {
    /// Constructs a new item with a predecessor.
    fn new_with_predecessor(predecessor: Node, item: Node) -> Self;

    /// Constructs a new item without predecessor information.
    fn new_without_predecessor(item: Node) -> Self;

    /// Returns the node represented by this item.
    fn item(&self) -> Node;

    /// Returns the predecessor of this node, if any.
    fn predecessor(&self) -> Option<Node>;

    /// Returns a pair `(predecessor, item)` where the predecessor
    /// may be `None` if not tracked.
    fn predecessor_with_item(&self) -> (Option<Node>, Node) {
        (self.predecessor(), self.item())
    }
}

impl SequencedItem for Node {
    fn new_with_predecessor(_: Node, item: Node) -> Self {
        item
    }
    fn new_without_predecessor(item: Node) -> Self {
        item
    }
    fn item(&self) -> Node {
        *self
    }
    fn predecessor(&self) -> Option<Node> {
        None
    }
}

/// Compact representation of `(predecessor, node)` used for
/// traversals with parent tracking.
///
/// Internally, the absence of a predecessor is encoded by
/// setting both tuple entries to the same node value.
pub type PredecessorOfNode = (Node, Node);
impl SequencedItem for PredecessorOfNode {
    fn new_with_predecessor(predecessor: Node, item: Node) -> Self {
        (predecessor, item)
    }
    fn new_without_predecessor(item: Node) -> Self {
        (item, item)
    }

    /// Returns the node represented by this item.
    fn item(&self) -> Node {
        self.1
    }

    /// Returns the predecessor of this node in the traversal tree, if any.
    fn predecessor(&self) -> Option<Node> {
        if self.0 == self.1 { None } else { Some(self.0) }
    }
}

/// Abstraction for the traversal frontier data structure.
///
/// A `NodeSequencer` is responsible for storing the "to be visited"
/// nodes during a traversal. Different implementations determine
/// the traversal order:
///
/// - [`VecDeque`] -> queue semantics -> **BFS**
/// - [`Vec`] -> stack semantics -> **DFS**
pub trait NodeSequencer<T> {
    /// Creates a new sequencer initialized with a single node.
    fn init(u: T) -> Self;

    /// Pushes a node into the frontier.
    fn push(&mut self, item: T);

    /// Removes and returns the next node from the frontier.
    fn pop(&mut self) -> Option<T>;

    /// Returns a clone of the next node without removing it.
    fn peek(&self) -> Option<T>;

    /// Returns the number of items currently in the frontier.
    fn cardinality(&self) -> usize;
}

impl<T> NodeSequencer<T> for VecDeque<T>
where
    T: Clone,
{
    fn init(u: T) -> Self {
        Self::from(vec![u])
    }
    fn push(&mut self, u: T) {
        self.push_back(u)
    }
    fn pop(&mut self) -> Option<T> {
        self.pop_front()
    }
    fn peek(&self) -> Option<T> {
        self.front().cloned()
    }
    fn cardinality(&self) -> usize {
        self.len()
    }
}

impl<T> NodeSequencer<T> for Vec<T>
where
    T: Clone,
{
    fn init(u: T) -> Self {
        vec![u]
    }
    fn push(&mut self, u: T) {
        self.push(u)
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn peek(&self) -> Option<T> {
        self.last().cloned()
    }
    fn cardinality(&self) -> usize {
        self.len()
    }
}

/// Generic traversal iterator supporting BFS and DFS variants.
///
/// Maintains an explicit "frontier" (queue or stack) of nodes to visit,
/// a set of visited nodes, and optionally records predecessor information.
/// Parameterized by the container type for the frontier and the type of
/// items yielded (either `Node` or `PredecessorOfNode`).
pub struct TraversalSearch<'a, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node>,
{
    graph: &'a G,
    visited: V,
    sequencer: S,
    stop_at: Option<Node>,
    _item: PhantomData<I>,
}

/// Type alias for a **breadth-first search** iterator using a queue (`VecDeque`).
pub type BFSWithSet<'a, G, V> = TraversalSearch<'a, G, VecDeque<Node>, Node, V>;

/// Type alias for a **depth-first search** iterator using a stack (`Vec`).
pub type DFSWithSet<'a, G, V> = TraversalSearch<'a, G, Vec<Node>, Node, V>;

/// A BFS traversal iterator over the graph, visiting nodes in
/// breadth-first order from a given starting node.
pub type BFS<'a, G> = TraversalSearch<'a, G, VecDeque<Node>, Node, NodeBitSet>;

/// A DFS traversal iterator over the graph, visiting nodes in
/// depth-first order from a given starting node.
pub type DFS<'a, G> = TraversalSearch<'a, G, Vec<Node>, Node, NodeBitSet>;

/// A BFS traversal iterator that records predecessor information,
/// producing a spanning tree of the search.
pub type BFSWithPredecessor<'a, G> =
    TraversalSearch<'a, G, VecDeque<PredecessorOfNode>, PredecessorOfNode, NodeBitSet>;

/// A DFS traversal iterator that records predecessor information,
/// producing a spanning tree of the search.
pub type DFSWithPredecessor<'a, G> =
    TraversalSearch<'a, G, Vec<PredecessorOfNode>, PredecessorOfNode, NodeBitSet>;

impl<G, S, I, V> WithGraphRef<G> for TraversalSearch<'_, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node>,
{
    /// Returns the graph being traversed.
    fn graph_ref(&self) -> &G {
        self.graph
    }
}

impl<G, S, I, V> TraversalState<V> for TraversalSearch<'_, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node>,
{
    fn visited(&self) -> &V {
        &self.visited
    }
}

impl<G, S, I, V> Iterator for TraversalSearch<'_, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node>,
{
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        let popped = self.sequencer.pop()?;
        let u = popped.item();

        if self.stop_at == Some(u) {
            while self.sequencer.pop().is_some() {} // drop all
        } else {
            for v in self.graph.neighbors_of(u) {
                if !self.visited.contains(&v) {
                    self.sequencer.push(I::new_with_predecessor(u, v));
                    self.visited.insert(v);
                }
            }
        }

        Some(popped)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.sequencer.cardinality(),
            Some(self.graph.len() - self.visited.len()),
        )
    }
}

impl<'a, G, S, I, V> TraversalSearch<'a, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node> + FromCapacity,
{
    /// Creates a new traversal iterator starting from `start`.
    ///
    /// - `graph`: The graph to traverse.
    /// - `start`: The starting node.
    pub fn new(graph: &'a G, start: Node) -> Self {
        let len = graph.len();
        let mut visited = V::from_total_used_capacity(len, len);
        visited.insert(start);
        Self {
            graph,
            visited,
            sequencer: S::init(I::new_without_predecessor(start)),
            stop_at: None,
            _item: PhantomData,
        }
    }
}

impl<'a, G, S, I, V> TraversalSearch<'a, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node>,
{
    /// Tries to restart the search at an yet unvisited node and returns
    /// true iff successful. Requires that search came to a hold earlier,
    /// i.e. self.next() returned None
    pub fn try_restart_at_unvisited(&mut self) -> bool {
        assert_eq!(self.sequencer.cardinality(), 0);
        let node = self.graph.vertices().find(|u| !self.visited.contains(u));
        match node {
            None => false,
            Some(x) => {
                self.visited.insert(x);
                self.sequencer.push(I::new_without_predecessor(x as Node));
                true
            }
        }
    }

    /// Sets a stopper node. If this node is reached, the iterator returns it and afterwards only None.
    pub fn set_stop_at(&mut self, stopper: Node) {
        self.stop_at = Some(stopper);
    }

    /// Sets a stopper node. If this node is reached, the iterator returns it and afterwards only None.
    pub fn stop_at(mut self, stopper: Node) -> Self {
        self.set_stop_at(stopper);
        self
    }

    /// Excludes a node from the search. It will be treated as if it was already visited,
    /// i.e. no edges to or from that node will be taken. If the node was already visited,
    /// this is a non-op.
    ///
    /// # Warning
    /// Calling this method has no effect if the node is already on the stack. It is therefore highly
    /// recommended to call this method directly after the constructor.
    pub fn exclude_node(&mut self, u: Node) {
        self.visited.insert(u);
    }

    /// Excludes a node from the search. It will be treated as if it was already visited,
    /// i.e. no edges to or from that node will be taken. If the node was already visited,
    /// this is a non-op.
    ///
    /// # Warning
    /// Calling this method has no effect if the node is already on the stack. It is therefore highly
    /// recommended to call this method directly after the constructor.
    pub fn with_node_excluded(mut self, u: Node) -> Self {
        self.exclude_node(u);
        self
    }

    /// Exclude multiple nodes from traversal. It is functionally equivalent to repeatedly
    /// calling [`TraversalSearch::exclude_node`].
    ///
    /// # Warning
    /// Calling this method has no effect for nodes that are already on the stack. It is
    /// therefore highly recommended to call this method directly after the constructor.
    pub fn exclude_nodes<N>(&mut self, us: N)
    where
        N: IntoIterator<Item = Node>,
    {
        for u in us {
            self.exclude_node(u);
        }
    }

    /// Exclude multiple nodes from traversal. It is functionally equivalent to repeatedly
    /// calling [`TraversalSearch::with_node_excluded`].
    ///
    /// # Warning
    /// Calling this method has no effect for nodes that are already on the stack. It is
    /// therefore highly recommended to call this method directly after the constructor.
    pub fn with_nodes_excluded<N>(mut self, us: N) -> Self
    where
        N: IntoIterator<Item = Node>,
    {
        self.exclude_nodes(us);
        self
    }

    /// Consumes the traversal search and returns true iff the requested node can be visited, i.e.
    /// if there exists a directed path from the start node to u.
    ///
    /// # Warning
    /// It is undefined behavior to call the method on a partially executed iterator.
    pub fn is_node_reachable(mut self, u: Node) -> bool {
        assert_eq!(self.sequencer.cardinality(), 1);
        self.visited.remove(&u);
        self.next();
        self.any(|v| v.item() == u)
    }
}

/// Extension trait for traversal iterators that allows computing a ranking (iteration order)
/// of the nodes in the graph.
pub trait RankFromOrder<'a, G>: WithGraphRef<G> + Iterator<Item = Node> + Sized
where
    G: 'a + AdjacencyList,
{
    /// Consumes the traversal iterator and produces a vector `ranking` where
    /// `ranking[u]` gives the position (rank, starting at 0) at which node `u`
    /// was visited.
    ///
    /// - Returns `Some(ranking)` if **all nodes of the graph** were visited.
    /// - Returns `None` if the iterator did not cover every node.
    ///
    /// # Panics
    /// Panics if the iterator yields the same node more than once.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0, 1), (1, 2)]);
    ///
    /// let ranking = g.bfs(0).ranking().unwrap();
    /// assert_eq!(ranking, vec![0, 1, 2]);
    /// ```
    fn ranking(mut self) -> Option<Vec<Node>> {
        let mut ranking = vec![INVALID_NODE; self.graph_ref().len()];
        let mut rank: Node = 0;

        for u in self.by_ref() {
            assert_eq!(ranking[u as usize], INVALID_NODE); // assert no item is repeated by iterator
            ranking[u as usize] = rank;
            rank += 1;
        }

        if rank == self.graph_ref().number_of_nodes() {
            Some(ranking)
        } else {
            None
        }
    }
}

impl<'a, G, S, V> RankFromOrder<'a, G> for TraversalSearch<'a, G, S, Node, V>
where
    G: AdjacencyList,
    S: NodeSequencer<Node>,
    V: Set<Node>,
{
}

/// Extension trait for traversal iterators that return `PredecessorOfNode`,
/// enabling extraction of the implied spanning tree structure (parents, depths).
pub trait TraversalTree<'a, G>:
    WithGraphRef<G> + Iterator<Item = PredecessorOfNode> + Sized
where
    G: 'a + AdjacencyList,
{
    /// Consumes the iterator and records the parent of each node in the implied
    /// traversal tree into the provided slice `tree`.
    ///
    /// - For each visited node `v`, `tree[v]` is set to its predecessor.
    /// - Unvisited entries remain unchanged.
    ///
    /// # Requirements
    /// - `tree.len()` must be at least `graph.len()`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0, 1), (1, 2)]);
    ///
    /// let mut parents: Vec<Node> = g.vertices_range().collect();
    /// g.bfs_with_predecessor(0).parent_array_into(&mut parents);
    /// assert_eq!(parents, vec![0, 0, 1]);
    /// ```
    fn parent_array_into(&mut self, tree: &mut [Node]) {
        for pred_with_item in self.by_ref() {
            if let Some(p) = pred_with_item.predecessor() {
                tree[pred_with_item.item() as usize] = p;
            }
        }
    }

    /// Constructs a fresh parent array of size `graph.len()` where
    /// each node is initially set to be its own parent.
    /// Then fills in the traversal tree structure using `parent_array_into`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(2, [(0, 1)]);
    ///
    /// let parents = g.bfs_with_predecessor(0).parent_array();
    /// assert_eq!(parents, vec![0, 0]);
    /// ```
    fn parent_array(&mut self) -> Vec<Node> {
        let mut tree: Vec<_> = self.graph_ref().vertices_range().collect();
        self.parent_array_into(&mut tree);
        tree
    }

    /// Consumes the iterator and computes the depth of each visited node in
    /// the traversal tree (root depth = 0).
    ///
    /// - For each visited node `v`, `depths[v]` is set accordingly.
    /// - Unvisited entries remain unchanged.
    ///
    /// # Requirements
    /// - `depths.len()` must be at least `graph.len()`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0, 1), (1, 2)]);
    ///
    /// let mut depths = vec![0; g.number_of_nodes() as usize];
    /// g.bfs_with_predecessor(0).depths_into(&mut depths);
    /// assert_eq!(depths, vec![0, 1, 2]);
    /// ```
    fn depths_into(&mut self, depths: &mut [Node]) {
        for pred_with_item in self.by_ref() {
            depths[pred_with_item.item() as usize] = pred_with_item
                .predecessor()
                .map_or(0, |p| depths[p as usize] + 1);
        }
    }

    /// Constructs a fresh depth array of size `graph.len()` initialized with 0.
    /// Then fills in the traversal tree depths using `depths_into`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0, 1), (1, 2)]);
    ///
    /// let depths = g.bfs_with_predecessor(0).depths();
    /// assert_eq!(depths, vec![0, 1, 2]);
    /// ```
    fn depths(&mut self) -> Vec<Node> {
        let mut depths: Vec<_> = vec![0; self.graph_ref().number_of_nodes() as usize];
        self.depths_into(&mut depths);
        depths
    }
}

impl<'a, G, S, V> TraversalTree<'a, G> for TraversalSearch<'a, G, S, PredecessorOfNode, V>
where
    G: AdjacencyList,
    S: NodeSequencer<PredecessorOfNode>,
    V: Set<Node>,
{
}

/// Iterator implementing topological ordering over a directed acyclic graph (DAG).
///
/// Uses a variant of Kahn's algorithm:
/// - Initializes with all nodes of in-degree 0.
/// - Repeatedly removes a node, decreasing in-degrees of its successors,
///   and enqueues new nodes of in-degree 0.
/// - Stops once all nodes are output or a cycle is detected.
pub struct TopoSearch<'a, G> {
    graph: &'a G,
    in_degs: Vec<Node>,
    stack: Vec<Node>,
}

impl<'a, G> WithGraphRef<G> for TopoSearch<'a, G>
where
    G: DirectedAdjacencyList,
{
    fn graph_ref(&self) -> &G {
        self.graph
    }
}

impl<'a, G> Iterator for TopoSearch<'a, G>
where
    G: DirectedAdjacencyList,
{
    type Item = Node;

    /// Returns the next node in topological order, if available.
    ///
    /// - Each returned node is guaranteed to appear after all its predecessors.
    /// - If the graph has a cycle, iteration will terminate early without
    ///   covering all nodes.
    fn next(&mut self) -> Option<Self::Item> {
        let u = self.stack.pop()?;

        for v in self.graph.out_neighbors_of(u) {
            self.in_degs[v as usize] -= 1;
            if self.in_degs[v as usize] == 0 {
                self.stack.push(v);
            }
        }

        Some(u)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), Some(self.graph.len()))
    }
}

impl<'a, G> TopoSearch<'a, G>
where
    G: DirectedAdjacencyList,
{
    /// Constructs a new topological search on the given directed graph,
    /// initializing in-degree counts and collecting the initial set of
    /// zero in-degree nodes.
    fn new(graph: &'a G) -> Self {
        // add an in_degree getter to each graph?
        let mut in_degs: Vec<Node> = vec![0; graph.len()];
        for u in graph.vertices() {
            for v in graph.out_neighbors_of(u) {
                // u -> v
                in_degs[v as usize] += 1;
            }
        }

        let stack: Vec<Node> = in_degs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| if *d == 0 { Some(i as Node) } else { None })
            .collect();

        Self {
            graph,
            in_degs,
            stack,
        }
    }
}

impl<'a, G> RankFromOrder<'a, G> for TopoSearch<'a, G> where G: DirectedAdjacencyList {}

/// Provides convenient traversal methods (BFS, DFS, topological order, etc.)
pub trait Traversal: AdjacencyList + Sized {
    /// Returns an iterator that traverses nodes reachable from `start`
    /// in **breadth-first search (BFS) order**.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(2, [(0, 1)]);
    ///
    /// let order: Vec<_> = g.bfs(0).collect();
    /// assert_eq!(order, vec![0, 1]);
    /// ```
    fn bfs(&self, start: Node) -> BFS<'_, Self> {
        BFS::new(self, start)
    }

    /// Returns an iterator that traverses nodes reachable from `start`
    /// in **breadth-first search (BFS) order**.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArrayUndir::from_edges(2, [(0, 1)]);
    ///
    /// let order: Vec<_> = g.dfs(0).collect();
    /// assert_eq!(order, vec![0, 1]);
    /// ```
    fn dfs(&self, start: Node) -> DFS<'_, Self> {
        DFS::new(self, start)
    }

    /// Returns a BFS iterator starting from `start` that additionally
    /// yields the predecessor relation (edges traversed).
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::{*, traversal::SequencedItem}};
    ///
    /// let g = AdjArrayUndir::from_edges(2, [(0, 1)]);
    ///
    /// let mut it = g.bfs_with_predecessor(0);
    /// assert_eq!(it.next().unwrap().item(), 0);
    /// assert_eq!(it.next().unwrap().predecessor(), Some(0));
    /// ```
    fn bfs_with_predecessor(&self, start: Node) -> BFSWithPredecessor<'_, Self> {
        BFSWithPredecessor::new(self, start)
    }

    /// Returns a DFS iterator starting from `start` that additionally
    /// yields the predecessor relation (edges traversed).
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::{*, traversal::SequencedItem}};
    ///
    /// let g = AdjArrayUndir::from_edges(2, [(0, 1)]);
    ///
    /// let mut it = g.dfs_with_predecessor(0);
    /// assert_eq!(it.next().unwrap().item(), 0);
    /// assert_eq!(it.next().unwrap().predecessor(), Some(0));
    /// ```
    fn dfs_with_predecessor(&self, start: Node) -> DFSWithPredecessor<'_, Self> {
        DFSWithPredecessor::new(self, start)
    }

    /// Returns an iterator yielding nodes in a valid **topological order**.
    ///
    /// - Only available for directed graphs.
    /// - Terminates early if the graph contains a cycle.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArray::from_edges(3, [(0, 1), (1, 2)]);
    /// let order: Vec<_> = g.topo_search().collect();
    /// assert_eq!(order, vec![0, 1, 2]);
    /// ```
    fn topo_search(&self) -> TopoSearch<'_, Self>
    where
        Self: DirectedAdjacencyList,
    {
        TopoSearch::new(self)
    }

    /// Returns `true` if the directed graph is **acyclic**.
    ///
    /// Implementation: runs a topological search and checks whether
    /// all nodes were output.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArray::from_edges(3, [(0, 1), (1, 2)]);
    /// assert!(g.is_acyclic());
    /// ```
    fn is_acyclic(&self) -> bool
    where
        Self: DirectedAdjacencyList,
    {
        self.topo_search().count() == self.len()
    }

    /// Returns `true` if node `u` lies on a directed cycle
    /// (i.e. if there is a non-trivial strongly connected component
    /// containing `u`).
    ///
    /// Only available for directed graphs.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArray::from_edges(3, [(0, 1), (1, 2), (2, 1)]);
    /// assert!(!g.is_node_on_cycle(0));
    /// assert!(g.is_node_on_cycle(1));
    /// ```
    fn is_node_on_cycle(&self, u: Node) -> bool
    where
        Self: GraphType<Dir = Directed>,
    {
        self.bfs(u).is_node_reachable(u)
    }

    /// Returns `true` if node `u` lies on a directed cycle **after
    /// removing the given set of nodes** from the graph.
    ///
    /// - `deleted` specifies nodes to exclude from the search.
    /// - If `u` itself is in `deleted`, it is treated as if it were not.
    ///
    /// Only available for directed graphs.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    ///
    /// let g = AdjArray::from_edges(3, [(0, 1), (1, 2), (2, 1)]);
    /// assert!(g.is_node_on_cycle_after_deleting(1, [0]));
    /// assert!(!g.is_node_on_cycle_after_deleting(1, [2]));
    /// ```
    fn is_node_on_cycle_after_deleting<I>(&self, u: Node, deleted: I) -> bool
    where
        I: IntoIterator<Item = Node>,
        Self: GraphType<Dir = Directed>,
    {
        self.bfs(u)
            .with_nodes_excluded(deleted)
            .is_node_reachable(u)
    }

    /// Computes the **shortest path** from `start` to `end` using BFS.
    ///
    /// - Returns `Some(path)` if a path exists, where `path` is the sequence
    ///   of intermediate nodes (excluding `start`, ending before `end`).
    /// - Returns `None` if no path exists.
    ///
    /// # Note
    /// This method uses BFS with explicit predecessor tracking to reconstruct
    /// the path.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*};
    /// use fxhash::FxHashMap;
    ///
    /// let g = AdjArrayUndir::from_edges(3, [(0, 1), (1, 2)]);
    ///
    /// let path = g.shortest_path::<NodeBitSet, FxHashMap<Node, Node>>(0, 2);
    /// assert_eq!(path, Some(vec![1]));
    /// ```
    fn shortest_path<S, M>(&self, start: Node, end: Node) -> Option<Vec<Node>>
    where
        S: Set<Node> + FromCapacity,
        M: Map<Node, Node> + FromCapacity,
    {
        let mut bfs =
            TraversalSearch::<'_, Self, VecDeque<PredecessorOfNode>, PredecessorOfNode, S>::new(
                self, start,
            );
        let len = self.len();
        let mut parent = M::from_total_used_capacity(len, len);

        parent.insert(start, start);

        if start == end {
            bfs.visited.remove(&start);
        }

        // `bfs` first returns `start` which we can skip here
        // (and MUST as this item in `bfs` has no predecessor)
        bfs.next();

        for item in bfs {
            parent.insert(item.item(), item.predecessor().unwrap());
            if item.item() == end {
                let mut path = Vec::new();

                let mut node = item.predecessor().unwrap();
                while node != start {
                    path.push(node);
                    node = *parent.get(&node).unwrap();
                }

                path.reverse();
                return Some(path);
            }
        }

        None
    }
}

impl<G> Traversal for G where G: AdjacencyList + Sized {}

#[cfg(test)]
pub mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn bfs_order() {
        //  / 2 --- \
        // 1         4 - 3
        //  \ 0 - 5 /
        let graph = AdjArrayMatrix::from_edges(6, [(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)]);

        {
            let order: Vec<Node> = graph.bfs(1).collect();
            assert_eq!(order.len(), 6);

            assert_eq!(order[0], 1);
            assert!((order[1] == 0 && order[2] == 2) || (order[2] == 0 && order[1] == 2));
            assert!((order[3] == 4 && order[4] == 5) || (order[4] == 4 && order[3] == 5));
            assert_eq!(order[5], 3);
        }

        {
            let order: Vec<Node> = BFS::new(&graph, 5).collect();
            assert_eq!(order, [5, 4, 3]);
        }
    }

    #[test]
    fn bfs_with_predecessor() {
        let graph = AdjArrayMatrix::from_edges(6, [(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)]);

        let mut edges: Vec<_> = graph
            .bfs_with_predecessor(1)
            .map(|x| x.predecessor_with_item())
            .collect();
        edges.sort();
        assert_eq!(
            edges,
            vec![
                (None, 1),
                (Some(0), 5),
                (Some(1), 0),
                (Some(1), 2),
                (Some(2), 4),
                (Some(4), 3)
            ]
        );
    }

    #[test]
    fn test_stopper() {
        let graph = AdjArrayMatrix::from_edges(4, [(0, 1), (1, 2), (2, 3)]);
        assert_eq!(graph.bfs(0).collect_vec(), vec![0, 1, 2, 3]);

        assert_eq!(graph.bfs(0).stop_at(1).collect_vec(), vec![0, 1]);
    }

    #[test]
    fn bfs_tree() {
        let graph = AdjArrayMatrix::from_edges(6, [(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)]);
        let tree = graph.bfs_with_predecessor(1).parent_array();
        assert_eq!(tree, vec![1, 1, 1, 4, 2, 0]);
    }

    #[test]
    fn dfs_order() {
        //  / 2
        // 1         4 - 3
        //  \ 0 - 5 /
        let graph = AdjArrayMatrix::from_edges(6, [(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);

        {
            let order: Vec<Node> = DFS::new(&graph, 1).collect();
            assert_eq!(order.len(), 6);

            assert_eq!(order[0], 1);

            if order[1] == 2 {
                assert_eq!(order[2..6], [0, 5, 4, 3]);
            } else {
                assert_eq!(order[1..6], [0, 5, 4, 3, 2]);
            }
        }

        {
            let order: Vec<Node> = graph.dfs(5).collect();
            assert_eq!(order, [5, 4, 3]);
        }
    }

    #[test]
    fn dfs_tree() {
        let graph = AdjArrayMatrix::from_edges(6, [(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);
        let tree = graph.dfs_with_predecessor(1).parent_array();
        assert_eq!(tree, vec![1, 1, 1, 4, 5, 0]);
    }

    #[test]
    fn dfs_with_predecessor() {
        let graph = AdjArrayMatrix::from_edges(6, [(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);

        let mut edges: Vec<_> = graph
            .dfs_with_predecessor(1)
            .map(|x| x.predecessor_with_item())
            .collect();
        edges.sort();
        assert_eq!(
            edges,
            vec![
                (None, 1),
                (Some(0), 5),
                (Some(1), 0),
                (Some(1), 2),
                (Some(4), 3),
                (Some(5), 4)
            ]
        );
    }

    #[test]
    fn topology_rank() {
        let mut graph =
            AdjArrayMatrix::from_edges(7, [(2, 0), (1, 0), (0, 3), (0, 4), (0, 5), (3, 6)]);

        {
            let ranks = graph.topo_search().ranking().unwrap();
            assert_eq!(*ranks.iter().min().unwrap(), 0);
            assert_eq!(*ranks.iter().max().unwrap(), graph.number_of_nodes() - 1);
            for Edge(u, v) in graph.edges(false) {
                assert!(ranks[u as usize] < ranks[v as usize]);
            }
        }

        graph.add_edge(6, 2); // introduce cycle
        {
            let topo = graph.topo_search().ranking();
            assert!(topo.is_none());
        }
    }

    #[test]
    fn is_acyclic() {
        let mut graph =
            AdjArrayMatrix::from_edges(7, [(2, 0), (1, 0), (0, 3), (0, 4), (0, 5), (3, 6)]);
        assert!(graph.is_acyclic());
        graph.add_edge(6, 2); // introduce cycle
        assert!(!graph.is_acyclic());
    }

    #[test]
    fn node_on_cycle() {
        let mut graph =
            AdjArrayMatrix::from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5)]);
        assert!(graph.is_node_on_cycle(0));
        assert!(graph.is_node_on_cycle(1));
        assert!(graph.is_node_on_cycle(2));
        assert!(graph.is_node_on_cycle(3));
        assert!(!graph.is_node_on_cycle(4));
        assert!(!graph.is_node_on_cycle(5));

        graph.add_edge(5, 2);
        assert!(graph.vertices().all(|u| graph.is_node_on_cycle(u)));
    }
}
