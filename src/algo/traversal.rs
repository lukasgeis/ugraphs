use crate::{
    ops::*,
    utils::{FromCapacity, Map, Set},
    *,
};
use std::{collections::VecDeque, marker::PhantomData};

pub trait WithGraphRef<G> {
    fn graph(&self) -> &G;
}

pub trait TraversalState<S>
where
    S: Set<Node>,
{
    fn visited(&self) -> &S;

    fn did_visit_node(&self, u: Node) -> bool {
        self.visited().contains(&u)
    }
}

pub trait SequencedItem: Clone + Copy {
    fn new_with_predecessor(predecessor: Node, item: Node) -> Self;
    fn new_without_predecessor(item: Node) -> Self;
    fn item(&self) -> Node;
    fn predecessor(&self) -> Option<Node>;
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

// We use an ordinary Edge to encode the item and the optional predecessor to safe some
// memory. We can easily accomplish this by exploiting that the traversal algorithms do
// not take self-loops. So "None" is encoded by setting the predecessor as the node itself.
type PredecessorOfNode = (Node, Node);
impl SequencedItem for PredecessorOfNode {
    fn new_with_predecessor(predecessor: Node, item: Node) -> Self {
        (predecessor, item)
    }
    fn new_without_predecessor(item: Node) -> Self {
        (item, item)
    }
    fn item(&self) -> Node {
        self.1
    }
    fn predecessor(&self) -> Option<Node> {
        if self.0 == self.1 { None } else { Some(self.0) }
    }
}

pub trait NodeSequencer<T> {
    // would prefer this to be private
    fn init(u: T) -> Self;
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
    fn peek(&self) -> Option<T>;
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

////////////////////////////////////////////////////////////////////////////////////////// BFS & DFS
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

pub type BFSWithSet<'a, G, V> = TraversalSearch<'a, G, VecDeque<Node>, Node, V>;
pub type DFSWithSet<'a, G, V> = TraversalSearch<'a, G, Vec<Node>, Node, V>;

pub type BFS<'a, G> = TraversalSearch<'a, G, VecDeque<Node>, Node, NodeBitSet>;
pub type DFS<'a, G> = TraversalSearch<'a, G, Vec<Node>, Node, NodeBitSet>;
pub type BFSWithPredecessor<'a, G> =
    TraversalSearch<'a, G, VecDeque<PredecessorOfNode>, PredecessorOfNode, NodeBitSet>;
pub type DFSWithPredecessor<'a, G> =
    TraversalSearch<'a, G, Vec<PredecessorOfNode>, PredecessorOfNode, NodeBitSet>;

impl<G, S, I, V> WithGraphRef<G> for TraversalSearch<'_, G, S, I, V>
where
    G: AdjacencyList,
    S: NodeSequencer<I>,
    I: SequencedItem,
    V: Set<Node>,
{
    fn graph(&self) -> &G {
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
    pub fn stop_at(&mut self, stopper: Node) {
        self.stop_at = Some(stopper);
    }

    /// Excludes a node from the search. It will be treated as if it was already visited,
    /// i.e. no edges to or from that node will be taken. If the node was already visited,
    /// this is a non-op.
    ///
    /// # Warning
    /// Calling this method has no effect if the node is already on the stack. It is therefore highly
    /// recommended to call this method directly after the constructor.
    pub fn exclude_node(&mut self, u: Node) -> &mut Self {
        self.visited.insert(u);
        self
    }

    /// Exclude multiple nodes from traversal. It is functionally equivalent to repeatedly
    /// calling [`TraversalSearch::exclude_node`].
    ///
    /// # Warning
    /// Calling this method has no effect for nodes that are already on the stack. It is
    /// therefore highly recommended to call this method directly after the constructor.
    pub fn exclude_nodes(&mut self, us: impl IntoIterator<Item = Node>) -> &mut Self {
        for u in us {
            self.exclude_node(u);
        }
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

//////////////////////////////////////////////////////////////////////////////////////// Convenience
pub trait RankFromOrder<'a, G>: WithGraphRef<G> + Iterator<Item = Node> + Sized
where
    G: 'a + AdjacencyList,
{
    /// Consumes a graph traversal iterator and returns a mapping, where the i-th
    /// item contains the rank (starting from 0) as which it was iterated over.
    /// Returns None iff not all nodes were iterated
    fn ranking(mut self) -> Option<Vec<Node>> {
        let mut ranking = vec![INVALID_NODE; self.graph().len()];
        let mut rank: Node = 0;

        for u in self.by_ref() {
            assert_eq!(ranking[u as usize], INVALID_NODE); // assert no item is repeated by iterator
            ranking[u as usize] = rank;
            rank += 1;
        }

        if rank == self.graph().number_of_nodes() {
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

pub trait TraversalTree<'a, G>:
    WithGraphRef<G> + Iterator<Item = PredecessorOfNode> + Sized
where
    G: 'a + AdjacencyList,
{
    /// Consumes the underlying graph traversal iterator and records the implied tree structure
    /// into an parent-array, i.e. `result[i]` stores the predecessor of node `i`. It is the
    /// calling code's responsibility to ensure that the slice `tree` is sufficiently large to
    /// store all reachable nodes (i.e. in general of size at least `graph.len()`).
    fn parent_array_into(&mut self, tree: &mut [Node]) {
        for pred_with_item in self.by_ref() {
            if let Some(p) = pred_with_item.predecessor() {
                tree[pred_with_item.item() as usize] = p;
            }
        }
    }

    /// Calls allocates a vector of size [`graph.len()`] and calls [self.parent_array_into] on it.
    /// Unvisited nodes have themselves as parents.
    fn parent_array(&mut self) -> Vec<Node> {
        let mut tree: Vec<_> = self.graph().vertices_range().collect();
        self.parent_array_into(&mut tree);
        tree
    }

    /// Consumes the underlying graph traversal iterator and depth of nodes in the implied
    /// tree structure, i.e. `result[i]` stores the depth of node `i` where a root has depth 0.
    /// It is the calling code's responsibility to ensure that the slice `depths` is sufficiently
    /// large to store all reachable nodes (i.e. in general of size at least `graph.len()`).
    fn depths_into(&mut self, depths: &mut [Node]) {
        for pred_with_item in self.by_ref() {
            depths[pred_with_item.item() as usize] = pred_with_item
                .predecessor()
                .map_or(0, |p| depths[p as usize] + 1);
        }
    }

    /// Calls allocates a vector of size [`graph.len()`] and calls [self.parent_array_into] on it.
    /// Unvisited nodes have themselves as parents.
    fn depths(&mut self) -> Vec<Node> {
        let mut depths: Vec<_> = vec![0; self.graph().number_of_nodes() as usize];
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

///////////////////////////////////////////////////////////////////////////////////////// TopoSearch
pub struct TopoSearch<'a, G> {
    graph: &'a G,
    in_degs: Vec<Node>,
    stack: Vec<Node>,
}

impl<'a, G> WithGraphRef<G> for TopoSearch<'a, G>
where
    G: DirectedAdjacencyList,
{
    fn graph(&self) -> &G {
        self.graph
    }
}

impl<'a, G> Iterator for TopoSearch<'a, G>
where
    G: DirectedAdjacencyList,
{
    type Item = Node;

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

/// Offers graph traversal algorithms as methods of the graph representation
pub trait Traversal: AdjacencyList + Sized {
    /// Returns an iterator traversing nodes reachable from `start` in breadth-first-search order
    fn bfs(&self, start: Node) -> BFS<'_, Self> {
        BFS::new(self, start)
    }

    /// Returns an iterator traversing nodes reachable from `start` in depth-first-search order
    fn dfs(&self, start: Node) -> DFS<'_, Self> {
        DFS::new(self, start)
    }

    /// Returns an iterator traversing nodes reachable from `start` in breadth-first-search order
    /// The items returned are the edges taken
    fn bfs_with_predecessor(&self, start: Node) -> BFSWithPredecessor<'_, Self> {
        BFSWithPredecessor::new(self, start)
    }

    /// Returns an iterator traversing nodes reachable from `start` in depth-first-search order
    /// The items returned are the edges taken
    fn dfs_with_predecessor(&self, start: Node) -> DFSWithPredecessor<'_, Self> {
        DFSWithPredecessor::new(self, start)
    }

    /// Returns an iterator traversing nodes in acyclic order. The iterator stops prematurely
    /// iff the graph is not acyclic (see `is_acyclic`); only implemented for directed graphs
    fn topo_search(&self) -> TopoSearch<'_, Self>
    where
        Self: DirectedAdjacencyList,
    {
        TopoSearch::new(self)
    }

    /// Returns true iff the graph is acyclic, i.e. there exists a order f of nodes, such that
    /// for all edge (u, v) we have f(u) < f(v); only implemented for directed graphs
    fn is_acyclic(&self) -> bool
    where
        Self: DirectedAdjacencyList,
    {
        self.topo_search().count() == self.len()
    }

    /// Returns true iff there exists a directed path from u to u itself, i.e. if u is part of a
    /// non-trivial SCC; only implemented for directed graphs
    fn is_node_on_cycle(&self, u: Node) -> bool
    where
        Self: GraphType<Dir = Directed>,
    {
        self.bfs(u).is_node_reachable(u)
    }

    /// Returns true iff there exists a directed path from u to u itself without using any nodes in
    /// deleted. The method ignores a potential entry u in deleted, i.e. behaves as if it were not
    /// in deleted.
    ///
    /// Only implemented for directed graphs
    fn is_node_on_cycle_after_deleting<I>(&self, u: Node, deleted: I) -> bool
    where
        I: IntoIterator<Item = Node>,
        Self: GraphType<Dir = Directed>,
    {
        let mut bfs = self.bfs(u);
        bfs.exclude_nodes(deleted);
        bfs.is_node_reachable(u)
    }

    /// Computes the shortest path from `start` to `end` using BFS and returns the path if it exists.
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
            bfs.next();
        }

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

impl<T> Traversal for T where T: AdjacencyList + Sized {}

#[cfg(test)]
pub mod tests {
    use crate::repr::AdjArrayMatrix;

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

        let mut bfs = graph.bfs(0);
        bfs.stop_at(1);
        assert_eq!(bfs.collect_vec(), vec![0, 1]);
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
