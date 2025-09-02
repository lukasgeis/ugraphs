/*!
# Vertex Cuts (Minimum and Balanced)

This module provides algorithms to compute **minimum vertex cuts** and **approximate balanced cuts**
in directed graphs. Unlike the heuristic cuts in [`GraphCutBuilder`](super::GraphCutBuilder),
the algorithms in this module are based on **flow-based methods** (Edmonds–Karp style).

## Core concepts
- A **vertex cut** is a set of vertices whose removal disconnects a chosen source `s` from a
  target `t` in a directed graph.
- A **minimum vertex cut** has the smallest possible size among all such cuts.
- A **balanced cut** attempts to partition the graph into two reasonably sized subsets while
  disconnecting `s` from `t`.

## Implementations
- [`MinVertexCut`] provides flow-based routines for computing (s, t) vertex cuts and for
  approximating balanced cuts.
- [`EdmondsKarpGeneric`] implements the Edmonds–Karp augmenting path algorithm for finding
  **edge-disjoint or vertex-disjoint paths**, which form the basis for computing vertex cuts.
- [`STFlow`] is a utility trait for constructing and running (s, t)-flow computations with
  the option to **undo or persist changes** to the graph.

## Use cases
- Exact minimum vertex cut computation (flow-based).
- Approximate balanced partitioning subject to vertex cut constraints.
- Analysis of connectivity and disjoint paths in directed graphs.
*/

use super::{traversal::*, *};
use num::Integer;
use rand::Rng;
use std::{collections::HashSet, ops::Range};
use stream_bitset::{bitset::BitsetStream, prelude::*};

/// Implementation of the Edmonds–Karp algorithm for finding
/// edge-disjoint (or vertex-disjoint, depending on network construction)
/// paths between a given source and target in a directed graph.
///
/// Internally, it maintains a residual network and a predecessor array
/// used for BFS. It also supports optionally tracking modifications
/// to the residual network, which is useful when computing
/// special structures such as *petals* that require undoing changes.
pub struct EdmondsKarp {
    residual_network: ResidualBitMatrix,
    predecessor: Vec<Node>,
    changes_on_bitmatrix: Vec<(Node, Node)>,
    remember_changes: bool,
}

/// A residual network data structure for flow algorithms.
///
/// Provides primitives for reversing edges and constructing networks
/// specialized for computing:
/// - edge-disjoint paths,
/// - vertex-disjoint paths,
/// - and *petals* (vertex-disjoint cycles all sharing a root vertex).
///
/// Additionally, it allows creating networks directly from an
/// externally provided capacity/label structure and supports undoing
/// changes when flows are rolled back.
pub trait ResidualNetwork: SourceTarget + DirectedAdjacencyList + Label<Node> {
    /// Reverses the directed edge `(u, v)` into `(v, u)`
    /// in the residual network.
    fn reverse(&mut self, u: Node, v: Node);

    /// Constructs a residual network to compute **edge-disjoint paths**
    /// between source `s` and target `t`.
    fn edge_disjoint<G>(graph: &G, s: Node, t: Node) -> Self
    where
        G: GraphNodeOrder + DirectedAdjacencyList;

    /// Constructs a residual network to compute **vertex-disjoint paths**.
    /// Each vertex `v` is split into `v_in` and `v_out` with an edge between them,
    /// except for `s` and `t` which are handled specially.
    fn vertex_disjoint<G>(graph: &G, s: Node, t: Node) -> Self
    where
        G: GraphNodeOrder + DirectedAdjacencyList;

    /// Constructs a residual network for finding **petals**,
    /// i.e. vertex-disjoint cycles all passing through vertex `s`.
    ///
    /// Similar to vertex-disjoint construction, except `s_in` and `s_out`
    /// are not connected, and the source is `s_out` while the target is `s_in`.
    fn petals<G>(graph: &G, s: Node) -> Self
    where
        G: GraphNodeOrder + DirectedAdjacencyList;

    /// Builds a residual network directly from a provided set of
    /// capacity bitsets and vertex labels. Useful when reusing
    /// previously computed capacities.
    fn from_capacity_and_labels(
        capacity: Vec<NodeBitSet>,
        labels: Vec<Node>,
        s: Node,
        t: Node,
    ) -> Self;

    /// Undoes previously applied changes (edge reversals) to restore
    /// the residual network to an earlier state.
    fn undo_changes<I>(&mut self, changes_on_bitmatrix: I)
    where
        I: IntoIterator<Item = (Node, Node)>;
}

/// Residual network implementation based on a bit-matrix adjacency structure.
///
/// Stores:
/// - the source and target nodes,
/// - the number of nodes and edges,
/// - adjacency represented as `NodeBitSet` capacities,
/// - and per-node labels.
///
/// Efficiently supports adjacency queries needed by max-flow/path-packing
/// algorithms such as Edmonds–Karp.
pub struct ResidualBitMatrix {
    s: Node,
    t: Node,
    n: NumNodes,
    m: NumEdges,
    capacity: Vec<NodeBitSet>,
    labels: Vec<Node>,
}

/// Provides access to per-vertex labels in the residual network.
///
/// Labels are used to map auxiliary vertices (e.g. `v_in`, `v_out`)
/// back to their corresponding original graph vertices.
pub trait Label<T> {
    /// Returns the label associated with vertex `u`.
    fn label(&self, u: Node) -> &T;
}

impl Label<Node> for ResidualBitMatrix {
    fn label(&self, u: Node) -> &Node {
        &self.labels[u as usize]
    }
}

/// Provides accessors for the source and target nodes
/// of a residual network.
pub trait SourceTarget {
    /// Returns a reference to the source node.
    fn source(&self) -> &Node;

    /// Returns a reference to the target node.
    fn target(&self) -> &Node;

    /// Returns a mutable reference to the source node.
    fn source_mut(&mut self) -> &mut Node;

    /// Returns a mutable reference to the target node.
    fn target_mut(&mut self) -> &mut Node;
}

impl SourceTarget for ResidualBitMatrix {
    fn source(&self) -> &Node {
        &self.s
    }

    fn target(&self) -> &Node {
        &self.t
    }

    fn source_mut(&mut self) -> &mut Node {
        &mut self.t
    }

    fn target_mut(&mut self) -> &mut Node {
        &mut self.t
    }
}

impl GraphType for ResidualBitMatrix {
    type Dir = Directed;
}

impl GraphNodeOrder for ResidualBitMatrix {
    type VertexIter<'a>
        = Range<Node>
    where
        Self: 'a;

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }

    fn number_of_nodes(&self) -> NumNodes {
        self.n
    }
}

impl GraphEdgeOrder for ResidualBitMatrix {
    fn number_of_edges(&self) -> NumEdges {
        self.m
    }
}

impl AdjacencyList for ResidualBitMatrix {
    type NeighborIter<'a>
        = BitmaskStreamToIndices<BitmaskSliceStream<'a>, Node, true>
    where
        Self: 'a;

    type ClosedNeighborIter<'a>
        = std::iter::Chain<std::iter::Once<Node>, Self::NeighborIter<'a>>
    where
        Self: 'a;

    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
        self.capacity[u as usize].bitmask_stream().iter_set_bits()
    }

    fn closed_neighbors_of(&self, u: Node) -> Self::ClosedNeighborIter<'_> {
        std::iter::once(u).chain(self.neighbors_of(u))
    }

    fn degree_of(&self, u: Node) -> Node {
        self.capacity[u as usize].cardinality() as Node
    }

    type NeighborsStream<'a>
        = BitmaskSliceStream<'a>
    where
        Self: 'a;

    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
        self.capacity[u as usize].bitmask_stream()
    }
}

/// Helper struct to allow implementing [`DirectedAdjacencyList::in_neighbors_of`].
///
/// While we currently do not use this method on [`ResidualBitMatrix`], we implement it for
/// completeness and to adhere to trait-bounds.
pub struct ResidualInNeighbors<'a> {
    graph: &'a ResidualBitMatrix,
    node: Node,
    lb: Node,
}

impl<'a> Iterator for ResidualInNeighbors<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        while self.lb < self.graph.number_of_nodes() {
            self.lb += 1;

            if self.graph.capacity[self.lb as usize - 1].get_bit(self.node) {
                return Some(self.lb - 1);
            }
        }

        None
    }
}

/// Implementation for completeness even though we only need functionality already provided by [`AdjacencyList`]
impl DirectedAdjacencyList for ResidualBitMatrix {
    type InNeighborIter<'a>
        = ResidualInNeighbors<'a>
    where
        Self: 'a;

    fn in_neighbors_of(&self, u: Node) -> Self::InNeighborIter<'_> {
        ResidualInNeighbors {
            graph: self,
            node: u,
            lb: 0,
        }
    }

    fn in_degree_of(&self, u: Node) -> NumNodes {
        self.in_neighbors_of(u).count() as NumNodes
    }

    type InNeighborsStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;

    fn in_neighbors_of_as_stream(&self, u: Node) -> Self::InNeighborsStream<'_> {
        NodeBitSet::new_with_bits_set(self.number_of_nodes(), self.in_neighbors_of(u))
            .into_bitmask_stream()
    }
}

impl ResidualNetwork for ResidualBitMatrix {
    fn reverse(&mut self, u: Node, v: Node) {
        assert!(self.capacity[u as usize].get_bit(v));
        self.capacity[u as usize].clear_bit(v);
        self.capacity[v as usize].set_bit(u);
    }

    fn edge_disjoint<G>(graph: &G, s: Node, t: Node) -> Self
    where
        G: GraphNodeOrder + DirectedAdjacencyList,
    {
        let n = graph.number_of_nodes();
        Self {
            s,
            t,
            n,
            m: 0,
            capacity: graph
                .vertices()
                .map(|u| NodeBitSet::new_with_bits_set(n, graph.out_neighbors_of(u)))
                .collect(),
            labels: graph.vertices().collect(),
        }
    }

    fn vertex_disjoint<G>(graph: &G, s: Node, t: Node) -> Self
    where
        G: GraphNodeOrder + DirectedAdjacencyList,
    {
        let n = graph.number_of_nodes() * 2; // duplicate
        let labels: Vec<_> = graph.vertices().chain(graph.vertices()).collect();

        let mut capacity = vec![NodeBitSet::new(n); n as usize];
        for v in graph.vertices() {
            // handle s and t
            if v == s || v == t {
                for u in graph.out_neighbors_of(v) {
                    // add edge from v to u
                    capacity[v as usize].set_bit(u);
                }
                continue;
            }

            let v_out = graph.number_of_nodes() + v;
            // from v_in to v_out
            capacity[v as usize].set_bit(v_out);

            for u in graph.out_neighbors_of(v) {
                // this also handles s and t
                // add edge from v_out to u_in
                capacity[v_out as usize].set_bit(u);
            }
        }

        Self {
            s,
            t,
            n,
            m: 0,
            capacity,
            labels,
        }
    }

    fn petals<G>(graph: &G, s: Node) -> Self
    where
        G: GraphNodeOrder + DirectedAdjacencyList,
    {
        let n = graph.number_of_nodes() * 2; // duplicate
        let labels: Vec<_> = graph.vertices().chain(graph.vertices()).collect();

        let mut capacity = vec![NodeBitSet::new(n); n as usize];
        for v in graph.vertices() {
            // handle s and t
            let v_out = graph.number_of_nodes() + v;
            // from v_in to v_out. Unless the vertex is s.
            if v != s {
                capacity[v as usize].set_bit(v_out);
            }

            for u in graph.out_neighbors_of(v) {
                // add edge from v_out to u
                capacity[v_out as usize].set_bit(u);
            }
        }

        Self {
            s: graph.number_of_nodes() + s,
            t: s,
            n,
            m: capacity.iter().map(|v| v.cardinality()).sum(),
            capacity,
            labels,
        }
    }

    fn from_capacity_and_labels(
        capacity: Vec<NodeBitSet>,
        labels: Vec<Node>,
        s: Node,
        t: Node,
    ) -> Self {
        Self {
            s,
            t,
            n: capacity.len() as NumNodes,
            m: capacity.iter().map(|v| v.cardinality() as NumEdges).sum(),
            capacity,
            labels,
        }
    }

    fn undo_changes<I>(&mut self, changes: I)
    where
        I: IntoIterator<Item = (Node, Node)>,
    {
        changes.into_iter().for_each(|(u, v)| self.reverse(v, u));
    }
}

impl ResidualBitMatrix {
    /// Consumes the residual network and returns its capacity
    /// and label vectors for reuse.
    pub fn take(self) -> (Vec<NodeBitSet>, Vec<Node>) {
        (self.capacity, self.labels)
    }
}

impl EdmondsKarp {
    /// Creates a new Edmonds–Karp solver from a given residual network.
    pub fn new(residual_network: ResidualBitMatrix) -> Self {
        let n = residual_network.len();
        Self {
            residual_network,
            predecessor: vec![0; n],
            changes_on_bitmatrix: vec![],
            remember_changes: false,
        }
    }

    /// Performs BFS to find an augmenting path from source to target.
    /// Updates the predecessor array and returns whether the target was reached.
    fn bfs(&mut self) -> bool {
        let s = *self.residual_network.source();
        let t = *self.residual_network.target();

        let mut bfs = self.residual_network.bfs_with_predecessor(s);
        bfs.set_stop_at(t);
        bfs.parent_array_into(self.predecessor.as_mut_slice());
        bfs.did_visit_node(t)
    }

    /// Returns the total number of disjoint paths between source and target.
    pub fn num_disjoint(&mut self) -> usize {
        self.count()
    }

    /// Returns the number of disjoint paths found, but stops early once
    /// `k` disjoint paths are reached.
    pub fn count_num_disjoint_upto(&mut self, k: Node) -> Node {
        self.take(k as usize).count() as Node
    }

    /// Returns all disjoint paths between source and target.
    /// Each path is represented as a vector of vertices.
    /// When constructed with vertex-disjoint gadgets, the paths
    /// are vertex-disjoint in the original graph.
    pub fn disjoint_paths(&mut self) -> Vec<Vec<Node>> {
        self.collect()
    }

    /// Enables or disables remembering residual network modifications.
    /// Useful when changes must later be undone.
    pub fn set_remember_changes(&mut self, remember_changes: bool) {
        self.remember_changes = remember_changes;
    }

    /// Chainable version of [`Self::set_remember_changes`].
    pub fn remember_changes(mut self, remember_changes: bool) -> Self {
        self.set_remember_changes(remember_changes);
        self
    }

    /// Reverts all recorded changes in the residual network.
    /// Used after computing petals or other structures that modify edges.
    pub fn undo_changes(&mut self) {
        self.residual_network
            .undo_changes(self.changes_on_bitmatrix.iter().rev().copied());
    }

    /// Consumes the solver and returns the underlying residual
    /// capacity and labels for reuse.
    pub fn take(self) -> (Vec<NodeBitSet>, Vec<Node>) {
        self.residual_network.take()
    }
}

/// Records modifications to the residual network, so they
/// can be undone later.
pub trait RememberChanges {
    /// Remembers that edge `(u, v)` was reversed in the residual network.
    fn remember_change(&mut self, u: Node, v: Node);
}

impl RememberChanges for EdmondsKarp {
    fn remember_change(&mut self, u: Node, v: Node) {
        self.changes_on_bitmatrix.push((u, v));
    }
}

/// Iterates over disjoint paths found by the Edmonds–Karp algorithm.
/// Each iteration returns a path from source to target as a vector of nodes.
/// The iterator terminates when no further augmenting paths exist.
impl Iterator for EdmondsKarp {
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.bfs() {
            return None;
        }

        let t = *self.residual_network.target();
        let s = *self.residual_network.source();
        let mut path = vec![t];
        let mut v = t;
        while v != s {
            let u = self.predecessor[v as usize];
            // when trying to find vertex disjoint this skips edges inside 'gadgets'
            if self.residual_network.label(u) != self.residual_network.label(v) {
                path.push(u);
            }
            self.residual_network.reverse(u, v);

            if self.remember_changes {
                self.remember_change(u, v);
            }

            v = u;
        }

        Some(
            path.iter()
                .map(|v| *self.residual_network.label(*v))
                .rev()
                .collect(),
        )
    }
}

/// Algorithms for computing minimum (s, t) vertex cuts and approximating balanced cuts in directed graphs.
pub trait MinVertexCut: DirectedAdjacencyList {
    /// Computes a **minimum (s, t) vertex cut** using a flow-based method.
    ///
    /// Returns:
    /// - The set of cut vertices.
    /// - The number of vertices reachable from and including `s` after the cut.
    ///
    /// A maximal acceptable cut size (`max_size`) may be supplied as a performance bound.
    /// Returns `None` iff the minimum cut exceeds `max_size`.
    fn min_st_vertex_cut(
        &self,
        s: Node,
        t: Node,
        max_size: Option<Node>,
    ) -> Option<(Vec<Node>, Node)>;

    /// Approximates a **balanced minimum vertex cut** by repeatedly sampling source-target pairs.
    ///
    /// Algorithm:
    /// - Perform `attempts` iterations.
    /// - In each iteration, choose random `s` and `t`.
    /// - Compute a minimum (s, t) cut.
    /// - If the cut splits the graph such that both sides have size at least
    ///   `self.len() * imbalance - 1`, declare the cut "legal".
    ///
    /// Returns the smallest legal cut found.
    ///
    /// # Warning
    /// If `imbalance > 0.5`, no legal solution can exist.
    fn approx_min_balanced_cut<R>(
        &self,
        rng: &mut R,
        attempts: usize,
        imbalance: f64,
        max_size: Option<Node>,
        greedy: bool,
    ) -> Option<Vec<Node>>
    where
        R: Rng;
}

impl<G> MinVertexCut for G
where
    G: DirectedAdjacencyList,
{
    fn min_st_vertex_cut(
        &self,
        s: Node,
        t: Node,
        max_size: Option<Node>,
    ) -> Option<(Vec<Node>, Node)> {
        let max_size = max_size.unwrap_or(self.number_of_nodes() - 2);

        let mut ek = EdmondsKarp::new(ResidualBitMatrix::vertex_disjoint(self, s, t));
        let flow_lower_bound = ek.count_num_disjoint_upto(max_size + 1);
        if flow_lower_bound > max_size {
            return None;
        }

        let mut cut_candidates = HashSet::with_capacity(2 * flow_lower_bound as usize);
        let mut size_s_cut: usize = 2; // count s itself

        let mut dfs = ek.residual_network.dfs(*ek.residual_network.source());
        dfs.next(); // skip source

        for v in dfs.by_ref() {
            debug_assert_ne!(v, *ek.residual_network.target());
            size_s_cut += 1;

            let v = if v < self.number_of_nodes() {
                v
            } else {
                v - self.number_of_nodes()
            };

            if !cut_candidates.insert(v) {
                // already contained: remove!
                cut_candidates.remove(&v);
            }
        }

        size_s_cut -= cut_candidates.len();

        // Add a cut-vertex for isolated s->t paths
        for v in self.out_neighbors_of(s) {
            // Todo: here we arbitrarily choose a node closest to the source; this might not be a good
            // choice towards a balanced cut. But given the density of our kernels, it should not
            // matter too much

            if dfs.did_visit_node(v) {
                continue;
            }
            if !ek.residual_network.capacity[v as usize].get_bit(s) {
                continue;
            }
            let success = cut_candidates.insert(v);
            debug_assert!(success);
        }

        assert_eq!(cut_candidates.len(), flow_lower_bound as usize);
        assert!(size_s_cut.is_even());

        Some((
            cut_candidates.iter().copied().collect(),
            (size_s_cut / 2) as Node,
        ))
    }

    fn approx_min_balanced_cut<R>(
        &self,
        rng: &mut R,
        attempts: usize,
        imbalance: f64,
        max_size: Option<Node>,
        greedy: bool,
    ) -> Option<Vec<Node>>
    where
        R: Rng,
    {
        assert!(self.len() > 2);

        // we are not using (0..attempts). ... .min_by_key(|c| c.len()) since each we keep track
        // of the current best solution and pass it as an upper bound into min_st
        let mut current_upper_bound = max_size;
        let mut current_best = None;

        for _ in 0..attempts {
            let s = rng.random_range(self.vertices_range());
            let t = loop {
                let t = rng.random_range(self.vertices_range());
                if t != s {
                    break t;
                }
            };

            if let Some((cut_vertices, size)) = self.min_st_vertex_cut(s, t, current_upper_bound) {
                if cut_vertices.is_empty() {
                    return None;
                }

                let smaller_partition_size =
                    size.min(self.number_of_nodes() - size - cut_vertices.len() as Node);

                if smaller_partition_size as f64 + 1.0 >= self.number_of_nodes() as f64 * imbalance
                {
                    current_upper_bound = Some(cut_vertices.len() as Node - 1);
                    current_best = Some(cut_vertices);

                    if greedy {
                        return current_best;
                    }

                    if current_upper_bound.unwrap() == 1 {
                        break;
                    }
                }
            }
        }

        current_best
    }
}

/// Enum representing structural modifications applied during flow computations
/// when path augmentations are carried out.
enum Change {
    /// Indicates an edge `(u, v)` was added during augmentation.
    Add(Node, Node),
    /// Indicates an edge `(u, v)` was removed during augmentation.
    Remove(Node, Node),
}

/// Flow-based algorithm for computing edge/vertex-disjoint paths between `s` and `t` in a directed graph.
///
/// This generic Edmonds–Karp variant supports both **vertex-disjoint** and **edge-disjoint** flow
/// formulations depending on the input graph construction. It can optionally remember all graph
/// modifications (edge additions/removals) in order to undo them when dropped.
pub struct EdmondsKarpGeneric<'a, G, T, L>
where
    G: DirectedAdjacencyList + GraphEdgeEditing,
    T: Fn(Node) -> L,
    L: Eq + Copy,
{
    graph: &'a mut G,
    labels: T,
    predecessor: Vec<Node>,
    source: Node,
    target: Node,
    changes: Option<Vec<Change>>,
}

/// Utility trait for computing (s, t)-flows in graphs with **undoable or permanent modifications**.
pub trait STFlow: DirectedAdjacencyList + GraphEdgeEditing {
    /// Runs an Edmonds–Karp style (s, t)-flow computation while remembering changes
    /// to the graph, which will be automatically undone when the flow object is dropped.
    fn st_flow_undo_changes<T, L>(
        &mut self,
        labels: T,
        s: Node,
        t: Node,
    ) -> EdmondsKarpGeneric<'_, Self, T, L>
    where
        T: Fn(Node) -> L,
        L: Eq + Copy;

    /// Runs an Edmonds–Karp style (s, t)-flow computation, keeping changes to the graph
    /// permanently applied (i.e., without rollback).
    fn st_flow_keep_changes<T, L>(
        &mut self,
        labels: T,
        s: Node,
        t: Node,
    ) -> EdmondsKarpGeneric<'_, Self, T, L>
    where
        T: Fn(Node) -> L,
        L: Eq + Copy;
}

impl<G> STFlow for G
where
    G: DirectedAdjacencyList + GraphEdgeEditing,
{
    fn st_flow_undo_changes<T, L>(
        &mut self,
        labels: T,
        s: Node,
        t: Node,
    ) -> EdmondsKarpGeneric<'_, Self, T, L>
    where
        T: Fn(Node) -> L,
        L: Eq + Copy,
    {
        let mut res = self.st_flow_keep_changes(labels, s, t);
        res.set_remember_changes(true);
        res
    }

    fn st_flow_keep_changes<T, L>(
        &mut self,
        labels: T,
        s: Node,
        t: Node,
    ) -> EdmondsKarpGeneric<'_, Self, T, L>
    where
        T: Fn(Node) -> L,
        L: Eq + Copy,
    {
        EdmondsKarpGeneric::new(self, labels, s, t)
    }
}

impl<'a, G, T, L> EdmondsKarpGeneric<'a, G, T, L>
where
    G: DirectedAdjacencyList + GraphEdgeEditing,
    T: Fn(Node) -> L,
    L: Eq + Copy,
{
    /// Creates a new instance of the Edmonds–Karp generic flow computation.
    pub fn new(graph: &'a mut G, labels: T, source: Node, target: Node) -> Self {
        let n = graph.len();
        Self {
            graph,
            labels,
            predecessor: vec![0; n],
            source,
            target,
            changes: None,
        }
    }

    /// Performs a BFS in the residual network to find an augmenting path.
    fn bfs(&mut self) -> bool {
        let mut bfs = self.graph.bfs_with_predecessor(self.source);
        bfs.set_stop_at(self.target);
        bfs.parent_array_into(self.predecessor.as_mut_slice());
        bfs.did_visit_node(self.target)
    }

    /// Computes the **number of disjoint (s, t)-paths** by fully exhausting augmentations.
    pub fn num_disjoint(&mut self) -> usize {
        self.count()
    }

    /// Computes the number of disjoint (s, t)-paths, but stops counting early at a threshold `k`.
    pub fn count_num_disjoint_upto(&mut self, k: Node) -> Node {
        self.take(k as usize).count() as Node
    }

    /// Returns all edge/vertex-disjoint (s, t)-paths as vectors of labeled vertices.
    pub fn disjoint_paths(&mut self) -> Vec<Vec<L>> {
        self.collect()
    }

    /// Configures whether changes to the residual network should be remembered
    /// (to allow undoing them later).
    pub fn set_remember_changes(&mut self, remember_changes: bool) {
        if remember_changes {
            assert!(self.changes.as_ref().is_none_or(|v| v.is_empty()));
            self.changes = Some(Vec::new());
        } else {
            self.changes = None;
        }
    }

    /// Builder-style version of [`Self::set_remember_changes`].
    pub fn remember_changes(mut self, remember_changes: bool) -> Self {
        self.set_remember_changes(remember_changes);
        self
    }

    /// Undoes all graph modifications performed during augmentation.
    /// Requires that remembering changes was enabled.
    pub fn undo_changes(&mut self) {
        let stack = self.changes.as_mut().unwrap();

        while let Some(change) = stack.pop() {
            match change {
                Change::Add(u, v) => self.graph.add_edge(u, v),
                Change::Remove(u, v) => self.graph.remove_edge(u, v),
            }
        }
    }
}

impl<'a, G, T, L> Drop for EdmondsKarpGeneric<'a, G, T, L>
where
    G: DirectedAdjacencyList + GraphEdgeEditing,
    T: Fn(Node) -> L,
    L: Eq + Copy,
{
    fn drop(&mut self) {
        if self.changes.is_some() {
            self.undo_changes();
        }
    }
}

impl<'a, G, T, L> Iterator for EdmondsKarpGeneric<'a, G, T, L>
where
    G: DirectedAdjacencyList + GraphEdgeEditing,
    T: Fn(Node) -> L,
    L: Eq + Copy,
{
    type Item = Vec<L>;

    /// Extracts the next disjoint (s, t)-path from the residual network by performing
    /// one Edmonds–Karp augmentation step. Returns `None` once no augmenting path exists.
    fn next(&mut self) -> Option<Self::Item> {
        if !self.bfs() {
            return None;
        }

        let s = self.source;
        let t = self.target;
        let mut path = vec![t];
        let mut v = t;

        while v != s {
            let u = self.predecessor[v as usize];
            // when trying to find vertex disjoint this skips edges inside 'gadgets'
            if (self.labels)(u) != (self.labels)(v) {
                path.push(u);
            }

            self.graph.remove_edge(u, v);
            let added = self.graph.try_add_edge(v, u);

            if let Some(changes) = self.changes.as_mut() {
                if added {
                    changes.push(Change::Remove(v, u));
                }
                changes.push(Change::Add(u, v));
            }

            v = u;
        }

        Some(path.iter().map(|&v| (self.labels)(v)).rev().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand::prelude::IteratorRandom;
    use rand_pcg::Pcg64;

    const EDGES: [(Node, Node); 13] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
        (2, 6),
        (3, 6),
        (4, 2),
        (4, 7),
        (5, 1),
        (5, 7),
        (6, 7),
        (6, 5),
    ];

    #[test]
    fn edmonds_karp() {
        let mut g = AdjMatrix::new(8);
        g.add_edges(&EDGES);
        let edges_reverse: Vec<_> = EDGES.iter().map(|(u, v)| (*v, *u)).collect();
        g.add_edges(&edges_reverse);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 3);
    }

    #[test]
    fn edmonds_karp_vertex_disjoint() {
        let mut g = AdjMatrix::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::vertex_disjoint(&g, 0, 7));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 1);
    }

    #[test]
    fn edmonds_karp_petals() {
        let mut g = AdjMatrix::new(8);
        g.add_edges(&EDGES);
        g.add_edge(3, 7);
        g.add_edge(7, 0);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 2);
    }

    #[test]
    fn edmonds_karp_no_co_arcs() {
        let mut g = AdjMatrix::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 2);
    }

    #[test]
    fn undo_changes() {
        let mut g = AdjMatrix::new(8);
        g.add_edges(&EDGES);
        let ec_before = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        let mut ec_after = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        ec_after.set_remember_changes(true);
        ec_after.disjoint_paths();
        ec_after.undo_changes();

        let capacity_before = ec_before.residual_network.capacity;
        let capacity_after = ec_after.residual_network.capacity;

        for out_nodes in 0..capacity_after.len() {
            for out_node in 0..capacity_after[out_nodes].cardinality() {
                let node_before = capacity_before[out_nodes].get_bit(out_node);
                let node_after = capacity_after[out_nodes].get_bit(out_node);
                assert_eq!(node_before, node_after);
            }
        }
    }

    #[test]
    fn count_num_disjoint_upto() {
        fn min(a: u32, b: u32) -> u32 {
            if a < b {
                return a;
            }
            return b;
        }
        // for node 3
        let petals_after_edge_added: [((Node, Node), u32); 13] = [
            ((0, 3), 0),
            ((3, 2), 0),
            ((2, 1), 0),
            ((1, 0), 1),
            ((3, 5), 1),
            ((5, 4), 1),
            ((4, 2), 1),
            ((4, 3), 2),
            ((4, 7), 2),
            ((7, 6), 2),
            ((6, 1), 2),
            ((6, 3), 2),
            ((3, 7), 3),
        ];
        let mut g = AdjMatrix::new(8);
        for edge in petals_after_edge_added {
            g.add_edge(edge.0.0, edge.0.1);
            for k in 0..3 {
                let mut ec = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
                let guess = ec.count_num_disjoint_upto(k);
                let actual = min(k, edge.1);
                assert_eq!(guess, actual);
            }
        }
    }

    #[test]
    fn min_st_cut() {
        // 1 cut vertex
        {
            let graph = AdjArray::from_edges(3, [(0, 1), (2, 1)]); // 2 is not reachable from 0
            let (cut, size_s) = graph.min_st_vertex_cut(0, 2, None).unwrap();
            assert_eq!(size_s, 2); // only s is reachable
            assert_eq!(cut, vec![]);
        }

        // 1 cut vertex
        {
            let graph = AdjArray::from_edges(3, [(0, 1), (1, 2)]);
            let (cut, size_s) = graph.min_st_vertex_cut(0, 2, None).unwrap();
            assert_eq!(size_s, 1); // only s is reachable
            assert_eq!(cut, vec![1]);
        }

        // 2 cut vertices
        {
            let graph = AdjArray::from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)]);
            let (cut, size_s) = graph.min_st_vertex_cut(0, 3, None).unwrap();
            assert_eq!(size_s, 1); // only s is reachable
            assert!(cut == vec![1, 2] || cut == vec![2, 1]);

            // upper bound
            assert!(graph.min_st_vertex_cut(0, 3, Some(1)).is_none());
        }
    }

    #[test]
    fn approx_min_cut() {
        for (n0, n1, c) in [(20, 20, 2), (20, 20, 4), (20, 20, 8)] {
            // generate two cliques with n0 and n1 nodes respectively, that are connected with a
            // 2 path. The consists of all nodes [n0+n1..n0+n1+c] if c is sufficently small

            let mut rng = Pcg64::seed_from_u64(1234);

            let mut graph = AdjArray::new(n0 + n1 + c);

            // plant cliques
            (0..n0)
                .into_iter()
                .cartesian_product((0..n0).into_iter())
                .filter(|(u, v)| u != v)
                .for_each(|(u, v)| graph.add_edge(u, v));
            (0..n1)
                .into_iter()
                .cartesian_product((0..n1).into_iter())
                .filter(|(u, v)| u != v)
                .for_each(|(u, v)| graph.add_edge(u + n0, v + n0));

            // plant connections between cliques
            for v in 0..2 * c {
                let mut us = (0..n0).into_iter().choose_multiple(&mut rng, 3);
                let mut ws = (n0..n0 + n1).into_iter().choose_multiple(&mut rng, 3);

                if v.is_even() {
                    std::mem::swap(&mut us, &mut ws);
                }

                let v = n0 + n1 + v / 2;

                us.into_iter().for_each(|u| graph.add_edge(u, v));
                ws.into_iter().for_each(|w| graph.add_edge(v, w));
            }

            let imbalance = n0.min(n1) as f64 / ((n0 + n1 + c) as f64);

            // to succeed, we need to select s from one clique and t from the other. Thus, we have
            // error rate of roughly 2^-100
            let mut cut = graph
                .approx_min_balanced_cut(&mut rng, 100, imbalance, None, false)
                .unwrap();

            assert_eq!(cut.len(), c as usize);
            cut.sort_unstable();
            assert_eq!(cut, (n0 + n1..n0 + n1 + c).into_iter().collect_vec());

            // cannot find a solution with such a large minimal component
            assert!(
                graph
                    .approx_min_balanced_cut(&mut rng, 100, imbalance, Some(c - 1), false)
                    .is_none()
            );

            // cannot find a solution with such a large minimal component
            assert!(
                graph
                    .approx_min_balanced_cut(&mut rng, 100, imbalance * 1.1, None, false)
                    .is_none()
            );
        }
    }
}
