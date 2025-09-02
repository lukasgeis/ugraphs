/*!
# Connectivity Algorithms

This module provides traits and iterators for analyzing the connectivity of graphs.

It includes algorithms for:
- **Connected components** in undirected graphs
- **Strongly connected components (SCCs)** in directed graphs (via Tarjan’s algorithm)

The functionality is designed both as **iterators** (to lazily enumerate components) and as
**partitions** (to group nodes into disjoint sets).
*/

use std::iter::FusedIterator;

use itertools::Itertools;

use super::{traversal::*, *};

/// A trait providing algorithms for analyzing connectivity in graphs.
///
/// Supports:
/// - Connected components in undirected graphs
/// - Strongly connected components in directed graphs
///
/// Many methods return lazy iterators (`ConnectedComponents` or `StronglyConnectedComponents`),
/// while convenience methods are available to directly produce a `Partition`.
pub trait Connectivity: AdjacencyList + Traversal + Sized {
    /// Returns an iterator over all connected components of an undirected graph.
    fn connected_components(&self) -> ConnectedComponents<'_, Self>
    where
        Self: AdjacencyList + GraphType<Dir = Undirected>;

    /// Returns an iterator over connected components, excluding trivial single-node components.
    fn connected_components_no_singletons(&self) -> ConnectedComponents<'_, Self>
    where
        Self: AdjacencyList + GraphType<Dir = Undirected>;

    /// Returns an iterator over connected components, excluding trivial components and/or nodes from a given list.
    ///
    /// - `skip_trivial`: if true, omits single-node components
    /// - `ignore`: an iterable of nodes to exclude from all components
    fn connected_components_exclude_nodes<I>(
        &self,
        skip_trivial: bool,
        ignore: I,
    ) -> ConnectedComponents<'_, Self>
    where
        I: IntoIterator<Item = Node>,
        Self: AdjacencyList + GraphType<Dir = Undirected>;

    /// Returns a `Partition` of the nodes into connected components of an undirected graph.
    fn partition_into_connected_components(&self) -> Partition
    where
        Self: GraphType<Dir = Undirected>,
    {
        self.connected_components()
            .into_partition(self.number_of_nodes())
    }

    /// Returns a `Partition` of nodes into connected components, omitting single-node components.
    fn partition_into_connected_components_no_singletons(&self) -> Partition
    where
        Self: GraphType<Dir = Undirected>,
    {
        self.connected_components_no_singletons()
            .into_partition(self.number_of_nodes())
    }

    /// Returns a `Partition` of nodes into connected components, optionally excluding nodes and/or trivial components.
    fn partition_into_connected_components_exclude_nodes<I>(
        &self,
        skip_trivial: bool,
        ignore: I,
    ) -> Partition
    where
        I: IntoIterator<Item = Node>,
        Self: GraphType<Dir = Undirected>,
    {
        self.connected_components_exclude_nodes(skip_trivial, ignore)
            .into_partition(self.number_of_nodes())
    }

    /// Returns an iterator over strongly connected components (SCCs) in a directed graph.
    ///
    /// Each SCC is returned as a vector of nodes.
    fn strongly_connected_components(&self) -> StronglyConnectedComponents<'_, Self>
    where
        Self: DirectedAdjacencyList;

    /// Returns an iterator over SCCs, excluding trivial single-node SCCs that do not have a self-loop.
    fn strongly_connected_components_no_singletons(&self) -> StronglyConnectedComponents<'_, Self>
    where
        Self: DirectedAdjacencyList;

    /// Returns a `Partition` of nodes into strongly connected components.
    fn partition_into_strongly_connected_components(&self) -> Partition
    where
        Self: DirectedAdjacencyList,
    {
        self.strongly_connected_components()
            .into_partition(self.number_of_nodes())
    }

    /// Returns a `Partition` of nodes into non-trivial SCCs (singletons excluded unless they have self-loops).
    fn partition_into_strongly_connected_components_no_singletons(&self) -> Partition
    where
        Self: DirectedAdjacencyList,
    {
        self.strongly_connected_components_no_singletons()
            .into_partition(self.number_of_nodes())
    }
}

impl<G> Connectivity for G
where
    G: AdjacencyList + Sized,
{
    fn connected_components(&self) -> ConnectedComponents<'_, Self>
    where
        Self: AdjacencyList + GraphType<Dir = Undirected>,
    {
        ConnectedComponents::new(self, false)
    }

    fn connected_components_no_singletons(&self) -> ConnectedComponents<'_, Self>
    where
        Self: AdjacencyList + GraphType<Dir = Undirected>,
    {
        ConnectedComponents::new(self, true)
    }

    fn connected_components_exclude_nodes<I>(
        &self,
        skip_trivial: bool,
        ignore: I,
    ) -> ConnectedComponents<'_, Self>
    where
        I: IntoIterator<Item = Node>,
        Self: AdjacencyList + GraphType<Dir = Undirected>,
    {
        ConnectedComponents::new(self, skip_trivial).exclude_nodes(ignore)
    }

    fn strongly_connected_components(&self) -> StronglyConnectedComponents<'_, Self>
    where
        Self: DirectedAdjacencyList,
    {
        StronglyConnectedComponents::new(self)
    }

    fn strongly_connected_components_no_singletons(&self) -> StronglyConnectedComponents<'_, Self>
    where
        Self: DirectedAdjacencyList,
    {
        StronglyConnectedComponents::new(self).include_singletons(false)
    }
}

/// Iterator over connected components of an undirected graph.
///
/// Each iteration yields one connected component as a vector of nodes.
/// Optionally supports skipping trivial (singleton) components or excluding
/// a set of nodes entirely.
pub struct ConnectedComponents<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    bfs: BFS<'a, G>,
}

impl<'a, G> ConnectedComponents<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    /// Constructs a new iterator over connected components for the given graph.
    ///
    /// - `skip_trivial`: if true, single-node components are excluded
    pub fn new(graph: &'a G, skip_trivial: bool) -> Self {
        assert!(
            !graph.is_empty(),
            "Can't iterate connected components in a graph with no nodes!"
        );
        if skip_trivial {
            if let Some(start_node) = graph.vertices_no_singletons().next() {
                Self {
                    bfs: graph
                        .bfs(start_node)
                        .with_nodes_excluded(graph.vertices().filter(|&u| graph.is_singleton(u))),
                }
            } else {
                let mut bfs = graph.bfs(0);
                bfs.exclude_nodes(graph.vertices());
                bfs.next(); // Consume falsely inserted starting node
                Self { bfs }
            }
        } else {
            Self { bfs: graph.bfs(0) }
        }
    }

    /// Exclude the given nodes from all components (in-place).
    pub fn set_exclude_nodes<I>(&mut self, exclude: I)
    where
        I: IntoIterator<Item = Node>,
    {
        self.bfs.exclude_nodes(exclude);
    }

    /// Exclude the given nodes from all components (returns the updated iterator).
    pub fn exclude_nodes<I>(mut self, exclude: I) -> Self
    where
        I: IntoIterator<Item = Node>,
    {
        self.set_exclude_nodes(exclude);
        self
    }
}

impl<'a, G> Iterator for ConnectedComponents<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let cc = self.bfs.by_ref().collect_vec();
            if !cc.is_empty() {
                return Some(cc);
            }

            if !self.bfs.try_restart_at_unvisited() {
                return None;
            }
        }
    }
}

/// Iterator over strongly connected components (SCCs) of a directed graph.
///
/// Implements Tarjan’s algorithm in an iterative (non-recursive) style to avoid stack overflows.
/// Each iteration yields one SCC as a vector of nodes.
///
/// The SCCs are returned in reverse topological order of the component DAG.
pub struct StronglyConnectedComponents<'a, G>
where
    G: DirectedAdjacencyList,
{
    graph: &'a G,
    idx: Node,

    states: Vec<NodeState>,
    potentially_unvisited: usize,

    include_singletons: bool,

    path_stack: Vec<Node>,

    call_stack: Vec<StackFrame<'a, G>>,
}

impl<'a, G> StronglyConnectedComponents<'a, G>
where
    G: DirectedAdjacencyList,
{
    /// Construct a new iterator for strongly connected components of the given graph.
    pub fn new(graph: &'a G) -> Self {
        Self {
            graph,
            idx: 0,
            states: vec![Default::default(); graph.len()],
            potentially_unvisited: 0,

            include_singletons: true,

            path_stack: Vec::with_capacity(32),
            call_stack: Vec::with_capacity(32),
        }
    }

    /// Specify whether singletons (isolated nodes without self-loops) should be included.
    /// Excluding them may improve performance.
    pub fn set_include_singletons(&mut self, include: bool) {
        self.include_singletons = include;
    }

    /// Builder-style version of [`Self::set_include_singletons`].
    pub fn include_singletons(mut self, include: bool) -> Self {
        self.set_include_singletons(include);
        self
    }

    /// Just like in a classic DFS where we want to compute a spanning-forest, we will need to
    /// to visit each node at least once. We start we node 0, and cover all nodes reachable from
    /// there in `search`. Then, we search for an untouched node here, and start over.
    fn next_unvisited_node(&mut self) -> Option<Node> {
        while self.potentially_unvisited < self.graph.len() {
            if !self.states[self.potentially_unvisited].visited {
                let v = self.potentially_unvisited as Node;
                self.push_node(v, None);
                return Some(v);
            }

            self.potentially_unvisited += 1;
        }
        None
    }

    /// Put a pristine stack frame on the call stack. Roughly speaking, this is the first step
    /// to a recursive call of search.
    fn push_node(&mut self, node: Node, parent: Option<Node>) {
        self.call_stack.push(StackFrame {
            node,
            parent: parent.unwrap_or(node),
            initial_stack_len: 0,
            first_call: true,
            has_loop: false,
            neighbors: self.graph.out_neighbors_of(node),
        });
    }

    fn search(&mut self) -> Option<Vec<Node>> {
        /*
        Tarjan's algorithm is typically described in a recursive fashion similarly to DFS
        with some extra steps. This design has two issues:
         1.) We cannot easily build an iterator from it
         2.) For large graphs we get stack overflows

        To overcome these issues, we use the explicit call stack `self.call_stack` that simulates
        recursive calls. On first visit of a node v it is assigned a "DFS rank"ish index and
        additionally the same low_link value. This value stores the smallest known index of any node known to be
        reachable from v. We then process all of its neighbors (which may trigger recursive calls).
        Eventually, all nodes in an SCC will have the same low_link and the unique node with this
        index becomes the arbitrary representative of this SCC (known as root).

        The key design is that the whole computation is wrapped in a `while` loop and all state
        (including iterators) is stored in `self.call_stack`. So we continue execution directly with
        another iteration. Alternative, we can pause processing, return an value and resume by
        reentering the function.
        */

        'recurse: while let Some(frame) = self.call_stack.last_mut() {
            let v = frame.node;

            if frame.first_call {
                frame.first_call = false;
                frame.initial_stack_len = self.path_stack.len() as Node;

                self.states[v as usize].visit(self.idx);
                self.idx += 1;

                self.path_stack.push(v);
            }

            for w in frame.neighbors.by_ref() {
                let w_state = self.states[w as usize];
                frame.has_loop |= w == v;

                if !w_state.visited {
                    self.push_node(w, Some(v));
                    continue 'recurse;
                } else if w_state.on_stack {
                    self.states[frame.node as usize].try_lower_link(w_state.index);
                }
            }

            let frame = self.call_stack.pop().unwrap();
            let state = self.states[v as usize];

            self.states[frame.parent as usize].try_lower_link(state.low_link);

            if state.is_root() {
                if !self.include_singletons
                    && *self.path_stack.last().unwrap() == v
                    && !frame.has_loop
                {
                    // skip producing component descriptor, since we have a singleton node
                    // but we need to undo
                    self.states[v as usize].on_stack = false;
                    self.path_stack.pop();
                } else {
                    // this component goes into the result, so produce a descriptor and clean-up stack
                    // while doing so
                    let component = self.path_stack
                        [frame.initial_stack_len as usize..self.path_stack.len()]
                        .iter()
                        .copied()
                        .collect_vec();

                    self.path_stack.truncate(frame.initial_stack_len as usize);

                    for &w in &component {
                        self.states[w as usize].on_stack = false;
                    }

                    debug_assert_eq!(*component.first().unwrap(), v);

                    return Some(component);
                }
            }
        }

        None
    }
}

impl<'a, G> Iterator for StronglyConnectedComponents<'a, G>
where
    G: DirectedAdjacencyList,
{
    type Item = Vec<Node>;

    /// Returns either a vector of node ids that form an SCC or None if no further SCC was found
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(x) = self.search() {
                return Some(x);
            }

            self.next_unvisited_node()?;
        }
    }
}

impl<'a, G> FusedIterator for StronglyConnectedComponents<'a, G> where G: DirectedAdjacencyList {}

/// Internal helper structure representing a stack frame in the non-recursive
/// implementation of Tarjan’s algorithm.
///
/// Stores per-node state during DFS traversal.
#[derive(Debug, Clone)]
struct StackFrame<'a, T>
where
    T: DirectedAdjacencyList + 'a,
{
    node: Node,
    parent: Node,
    initial_stack_len: Node,
    first_call: bool,
    has_loop: bool,
    neighbors: T::NeighborIter<'a>,
}

/// Internal helper representing the DFS state of a single node in Tarjan’s algorithm.
#[derive(Debug, Clone, Copy, Default)]
struct NodeState {
    visited: bool,
    on_stack: bool,
    index: Node,
    low_link: Node,
}

impl NodeState {
    /// Marks the node as visited and assigns it an index and low-link value.
    fn visit(&mut self, u: Node) {
        debug_assert!(!self.visited);
        self.index = u;
        self.low_link = u;
        self.visited = true;
        self.on_stack = true;
    }

    /// Attempt to lower the node’s low-link value based on a neighbor’s index.
    fn try_lower_link(&mut self, l: Node) {
        self.low_link = self.low_link.min(l);
    }

    /// Returns true if the node is the root of its strongly connected component.
    fn is_root(&self) -> bool {
        self.index == self.low_link
    }
}

/// Sorts the nodes within each component in ascending order,
/// then sorts the components themselves lexicographically by their first element.
pub fn sort_components(mut components: Vec<Vec<Node>>) -> Vec<Vec<Node>> {
    components.iter_mut().for_each(|comp| comp.sort_unstable());
    components.sort_by(|a, b| a[0].cmp(&b[0]));
    components
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    use super::*;
    use crate::gens::{GeneratorSubstructures, RandomGraph};

    #[test]
    fn partition_into_connected_components() {
        let mut graph = AdjArrayUndir::new(7);
        graph.add_edges([(1, 2), (2, 3), (4, 5)]);

        {
            let part = graph.partition_into_connected_components_no_singletons();
            assert_eq!(part.number_of_classes(), 2);
            assert_eq!(part.number_of_unassigned(), 2);

            assert_eq!(part.class_of_node(1), part.class_of_node(2));
            assert_eq!(part.class_of_node(1), part.class_of_node(3));
            assert_eq!(part.class_of_node(4), part.class_of_node(5));
            assert_ne!(part.class_of_node(1), part.class_of_node(5));
            assert!(part.class_of_node(0).is_none());
            assert!(part.class_of_node(6).is_none());
        }

        {
            let part = graph.partition_into_connected_components();
            assert_eq!(part.number_of_classes(), 4);
            assert_eq!(part.number_of_unassigned(), 0);
            assert!(part.class_of_node(0).is_some());
            assert!(part.class_of_node(6).is_some());
        }
    }

    #[test]
    pub fn scc() {
        let graph = AdjArrayMatrix::from_edges(
            8,
            [
                (0, 1),
                (1, 2),
                (1, 4),
                (1, 5),
                (2, 6),
                (2, 3),
                (3, 2),
                (3, 7),
                (4, 0),
                (4, 5),
                (5, 6),
                (6, 5),
                (7, 3),
                (7, 6),
            ],
        );

        let sccs = graph.strongly_connected_components().collect_vec();
        assert_eq!(sccs.len(), 3);
        assert!(!sccs[0].is_empty());
        assert!(!sccs[1].is_empty());
        assert!(!sccs[2].is_empty());

        let sccs = sort_components(sccs);
        assert_eq!(sccs[0], [0, 1, 4]);
        assert_eq!(sccs[1], [2, 3, 7]);
        assert_eq!(sccs[2], [5, 6]);
    }

    #[test]
    pub fn scc_singletons() {
        // {0,1} and {4,5} are scc pairs, 2 is a loop, 3 is a singleton
        let graph = AdjArrayMatrix::from_edges(
            6,
            [
                (0, 1),
                (1, 0),
                (2, 2),
                // 3 is missing
                (4, 5),
                (5, 4),
            ],
        );

        {
            let sccs = graph.strongly_connected_components().collect_vec();
            assert_eq!(sccs.len(), 4);

            let sccs = sort_components(sccs);
            assert_eq!(sccs[0], [0, 1]);
            assert_eq!(sccs[1], [2]);
            assert_eq!(sccs[2], [3]); // 3 is included
            assert_eq!(sccs[3], [4, 5]);
        }

        {
            let sccs = graph
                .strongly_connected_components_no_singletons()
                .collect_vec();
            assert_eq!(sccs.len(), 3);
            let sccs = sort_components(sccs);

            assert_eq!(sccs[0], [0, 1]);
            assert_eq!(sccs[1], [2]);
            assert_eq!(sccs[2], [4, 5]);
        }
    }

    #[test]
    pub fn scc_tree() {
        let graph = AdjArrayMatrix::from_edges(7, [(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (3, 6)]);

        let mut sccs = graph.strongly_connected_components().collect_vec();
        // in a directed tree each vertex is a strongly connected component
        assert_eq!(sccs.len(), 7);

        sccs.sort_by(|a, b| a[0].cmp(&b[0]));
        for (i, scc) in sccs.iter().enumerate() {
            assert_eq!(i as Node, scc[0]);
        }
    }

    #[test]
    fn scc_gnp() {
        let rng = &mut Pcg64::seed_from_u64(1234);

        for i in 0..10 {
            let n = 10000;
            let graph: AdjArrayMatrix = AdjArrayMatrix::gnp(rng, n, 0.5 / (n as f64) * (i as f64));
            assert_eq!(
                StronglyConnectedComponents::new(&graph)
                    .map(|x| x.len())
                    .sum::<usize>(),
                n as usize
            );
        }
    }

    #[test]
    fn scc_long_cycle() {
        // assert that we can deal with very deep stacks
        let n: Node = 10_000;
        let mut graph = AdjArrayMatrix::new(n);
        graph.connect_cycle((0..n).into_iter());
        let sccs = graph.strongly_connected_components().collect_vec();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs.first().unwrap().len(), n as usize);
    }
}
