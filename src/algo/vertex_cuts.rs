/*!
# Vertex Cuts

This module provides algorithms for computing **vertex cuts** in undirected graphs.

A *vertex cut* is a set of nodes whose removal increases the number of
connected components of the graph. This is important for analyzing graph
connectivity, robustness, and fault tolerance.

## Features
- Exact computation of articulation points (1-vertex cuts).
- Heuristic computation of 2- and 3-vertex cuts using articulation search.
- Builder interface for enabling/disabling cut types and applying
  filtering constraints.

## Notes
- Only articulation points (1-cuts) are computed exactly.
- Higher-order cuts (2- and 3-cuts) are found heuristically by
  repeatedly marking nodes as visited and recomputing articulation points.
  These are **not guaranteed to be complete**.
*/

use fxhash::FxHashSet;

use super::*;

/// Provides functionality to compute **articulation points**
/// (1-vertex cuts) in undirected graphs.
///
/// An articulation point is a vertex whose removal increases the number
/// of connected components in the graph.
pub trait ArticulationPoint: GraphType<Dir = Undirected> {
    /// Computes the articulation points of the graph.
    ///
    /// Returns a [`NodeBitSet`] where each set bit corresponds to
    /// a node that is an articulation point.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::{prelude::*, algo::*, gens::*};
    ///
    /// let mut g = AdjArrayUndir::new(10);
    /// g.connect_path(0..10 as Node);
    ///
    /// let mut aps = g.vertex_bitset_set();
    /// aps.clear_bit(0);
    /// aps.clear_bit(9);
    ///
    /// assert_eq!(g.compute_articulation_points(), aps);
    /// ```
    fn compute_articulation_points(&self) -> NodeBitSet;

    /// Computes articulation points with some nodes already considered visited
    /// in the underlying DFS used.
    ///
    /// You will probably do not need to interact with this method directly as it
    /// it is mostly useful for higher-order cuts (e.g., 2-cut, 3-cut), where some
    /// vertices are pre-excluded from the search.
    fn compute_articulation_points_with_visited(&self, visited: NodeBitSet) -> NodeBitSet;
}

impl<G> ArticulationPoint for G
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    fn compute_articulation_points(&self) -> NodeBitSet {
        ArticulationPointSearch::new(self).compute()
    }
    fn compute_articulation_points_with_visited(&self, visited: NodeBitSet) -> NodeBitSet {
        let mut ap = ArticulationPointSearch::new(self);
        ap.visited = visited;
        ap.compute()
    }
}

/// Internal DFS-based search for computing articulation points.
///
/// Implements the standard lowpoint-based DFS algorithm.
/// Keeps track of discovery times, lowpoints, and parent relations
/// for identifying articulation points.
pub struct ArticulationPointSearch<'a, T>
where
    T: AdjacencyList + GraphType<Dir = Undirected>,
{
    /// The graph being analyzed.
    graph: &'a T,
    /// Lowpoint values of nodes in DFS tree.
    low_point: Vec<Node>,
    /// Discovery times of nodes in DFS tree.
    dfs_num: Vec<Node>,
    /// Tracks visited nodes.
    visited: NodeBitSet,
    /// Stores articulation points found during the search.
    articulation_points: NodeBitSet,
    /// Current DFS counter.
    current_dfs_num: Node,
    /// Parent nodes in DFS tree.
    parent: Vec<Option<Node>>,
}

impl<'a, T> ArticulationPointSearch<'a, T>
where
    T: AdjacencyList + GraphType<Dir = Undirected>,
{
    /// Creates a new articulation point search for the given graph.
    ///
    /// Assumes:
    /// - The graph is connected.
    /// - For every edge `(u, v)`, the reverse edge `(v, u)` exists.
    pub fn new(graph: &'a T) -> Self {
        let n = graph.number_of_nodes();
        Self {
            graph,
            low_point: vec![0; n as usize],
            dfs_num: vec![0; n as usize],
            visited: NodeBitSet::new(n),
            parent: vec![None; n as usize],
            articulation_points: NodeBitSet::new(n),
            current_dfs_num: 0,
        }
    }

    /// Computes all articulation points of the graph.
    ///
    /// Returns a [`NodeBitSet`] with articulation points marked.
    pub fn compute(mut self) -> NodeBitSet {
        let start = self.visited.iter_cleared_bits().next().unwrap();
        let _ = self.compute_recursive(start, 0);
        self.articulation_points
    }

    /// Recursive DFS function for computing lowpoints and articulation points.
    ///
    /// # Arguments
    /// * `u` - Current node.
    /// * `depth` - Current recursion depth (with safety cutoff).
    fn compute_recursive(&mut self, u: Node, depth: Node) -> Result<(), ()> {
        if depth > 10000 {
            return Err(());
        }

        self.visited.set_bit(u);
        self.current_dfs_num += 1;
        self.dfs_num[u as usize] = self.current_dfs_num;
        self.low_point[u as usize] = self.current_dfs_num;

        // counts number of tree neighbors
        let mut tree_neighbors = 0;
        for v in self.graph.neighbors_of(u) {
            // tree edge
            if !self.visited.get_bit(v) {
                tree_neighbors += 1;
                self.parent[v as usize] = Some(u);
                self.compute_recursive(v, depth + 1)?;
                self.low_point[u as usize] =
                    self.low_point[u as usize].min(self.low_point[v as usize]);

                if self.parent[u as usize].is_some()
                    && self.low_point[v as usize] >= self.dfs_num[u as usize]
                {
                    self.articulation_points.set_bit(u);
                }
            } else {
                // back edge, update value if v is not the parent
                if self.parent[u as usize].is_none() || self.parent[u as usize].unwrap() != v {
                    self.low_point[u as usize] =
                        self.low_point[u as usize].min(self.dfs_num[v as usize]);
                }
            }
        }

        if self.parent[u as usize].is_none() && tree_neighbors > 1 {
            self.articulation_points.set_bit(u);
        }

        Ok(())
    }
}

/// Builder for computing vertex cuts of size 1, 2, or 3.
///
/// Provides a configurable interface to:
/// - Enable or disable specific cut sizes.
/// - Apply minimum connected component size filtering.
/// - Compute all enabled cuts at once.
pub struct GraphCutBuilder<'a, G>
where
    G: AdjacencyList + ArticulationPoint + Traversal + GraphType<Dir = Undirected>,
{
    /// Reference to the input graph.
    graph: &'a G,
    /// Which cut types are enabled.
    enabled_cuts: Vec<CutType>,
    /// Minimum connected component size to enforce after cuts.
    min_cc_size: Option<NumNodes>,
}

/// Types of vertex cuts that can be computed.
///
/// - `OneCut`: Exact articulation points.
/// - `TwoCut`: Heuristic two-node cuts.
/// - `ThreeCut`: Heuristic three-node cuts.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CutType {
    /// Single-node cuts (articulation points, exact).
    OneCut,
    /// Two-node cuts (heuristic).
    TwoCut,
    /// Three-node cuts (heuristic).
    ThreeCut,
}

/// Builder for computing **heuristic vertex cuts** of size 1, 2, or 3.
///
/// Provides a configurable interface to:
/// - Enable or disable specific cut sizes.
/// - Apply minimum connected component size filtering.
/// - Compute all enabled cuts at once.
///
/// ## Notes
/// - 1-cuts (articulation points) are computed exactly.
/// - 2-cuts and 3-cuts are computed heuristically by excluding nodes
///   and recomputing articulation points. The results are not guaranteed
///   to enumerate all possible minimal cuts.
impl<'a, G> GraphCutBuilder<'a, G>
where
    G: AdjacencyList + ArticulationPoint + Traversal + GraphType<Dir = Undirected>,
{
    /// Creates a new GraphCutBuilder with all cut types enabled by default.
    ///
    /// # Arguments
    /// * `graph` - The graph to compute cuts on
    pub fn new(graph: &'a G) -> Self {
        Self {
            graph,
            enabled_cuts: vec![CutType::OneCut, CutType::TwoCut, CutType::ThreeCut],
            min_cc_size: None,
        }
    }

    /// Enables a specific cut type if it's not already enabled.
    ///
    /// # Arguments
    /// * `cut_type` - The cut type to enable
    pub fn set_enable_cut(&mut self, cut_type: CutType) {
        if !self.enabled_cuts.contains(&cut_type) {
            self.enabled_cuts.push(cut_type);
        }
    }

    /// Chainable version of [`Self::set_enable_cut`].
    pub fn enable_cut(mut self, cut_type: CutType) -> Self {
        self.set_enable_cut(cut_type);
        self
    }

    /// Disables a specific cut type if it's currently enabled.
    ///
    /// # Arguments
    /// * `cut_type` - The cut type to disable
    pub fn set_disable_cut(&mut self, cut_type: CutType) {
        if let Some(index) = self
            .enabled_cuts
            .iter()
            .position(|value| *value == cut_type)
        {
            self.enabled_cuts.swap_remove(index);
        }
    }

    /// Chainable version of [`Self::set_disable_cut`].
    pub fn disable_cut(mut self, cut_type: CutType) -> Self {
        self.set_disable_cut(cut_type);
        self
    }

    /// Sets the minimum connected component size requirement for cuts.
    ///
    /// # Arguments
    /// * `size` - Minimum number of nodes required in each connected component
    ///   after removing the cut. Use `None` for no minimum requirement.
    pub fn set_min_cc_size(&mut self, size: Option<NumNodes>) {
        self.min_cc_size = size;
    }

    pub fn min_cc_size(mut self, size: Option<NumNodes>) -> Self {
        self.set_min_cc_size(size);
        self
    }

    /// Computes all enabled cuts.
    ///
    /// Returns a vector of cuts, where each cut is represented as a
    /// vector of nodes. Only cuts that meet the `min_cc_size` requirement
    /// (if set) are included.
    ///
    /// ## Notes
    /// - 1-cuts are exact.
    /// - 2-cuts and 3-cuts are **heuristic** and may not include all
    ///   minimal cuts.
    pub fn compute(self) -> Vec<Vec<Node>> {
        let mut all_cuts: Vec<Vec<Node>> = Vec::new();

        for &cut_type in &self.enabled_cuts {
            let cut = match cut_type {
                CutType::OneCut => self.compute_one_cut(),
                CutType::TwoCut => self.compute_two_cut(),
                CutType::ThreeCut => self.compute_three_cut(),
            };
            all_cuts.extend(cut.into_iter());
        }
        all_cuts
    }

    /// Filters candidate cuts by connected component size.
    fn filter_cuts(&self, candidates: Vec<Vec<Node>>) -> Vec<Vec<Node>> {
        let mut good_cuts: Vec<Vec<Node>> = vec![];
        if let Some(min_cc_size) = self.min_cc_size {
            for candidate in candidates {
                let partition = self
                    .graph
                    .partition_into_connected_components_exclude_nodes(true, candidate.clone());
                let mut largest = 0;
                let mut second_largest = 0;
                if partition.number_of_classes() > 1 {
                    for i in 0..partition.number_of_classes() {
                        let a = partition.number_in_class(i);
                        if a > largest {
                            second_largest = largest;
                            largest = a;
                        } else if a < second_largest {
                            second_largest = a;
                        }
                    }
                    if second_largest >= min_cc_size {
                        good_cuts.push(candidate);
                    }
                }
            }
        }
        good_cuts
    }

    /// Computes all 1-cuts (articulation points).
    fn compute_one_cut(&self) -> Vec<Vec<Node>> {
        let candidates: Vec<Vec<Node>> = self
            .graph
            .compute_articulation_points()
            .iter_set_bits()
            .map(|u| vec![u])
            .collect();
        self.filter_cuts(candidates)
    }

    /// Computes heuristic 2-cuts by iteratively excluding nodes and
    /// recomputing articulation points.
    fn compute_two_cut(&self) -> Vec<Vec<Node>> {
        let mut cuts: FxHashSet<(Node, Node)> = Default::default();
        let n = self.graph.number_of_nodes();
        for u in 0..n {
            let mut visited = NodeBitSet::new(n);
            visited.set_bit(u);
            cuts.extend(
                self.graph
                    .compute_articulation_points_with_visited(visited)
                    .iter_set_bits()
                    .map(|v| (u.min(v), u.max(v))),
            );
        }
        let vec_cuts: Vec<Vec<Node>> = cuts.iter().map(|(u, v)| vec![*u, *v]).collect();
        self.filter_cuts(vec_cuts)
    }

    /// Computes heuristic 3-cuts by iteratively excluding pairs of nodes
    /// and recomputing articulation points.
    fn compute_three_cut(&self) -> Vec<Vec<Node>> {
        let mut cuts: FxHashSet<(Node, Node, Node)> = Default::default();
        let n = self.graph.number_of_nodes();
        for u in 0..n {
            for v in (u + 1)..n {
                let mut visited = NodeBitSet::new(n);
                visited.set_bit(u);
                visited.set_bit(v);
                cuts.extend(
                    self.graph
                        .compute_articulation_points_with_visited(visited)
                        .iter_set_bits()
                        .map(|w| {
                            let mut tmp = [u, v, w];
                            tmp.sort();
                            (tmp[0], tmp[1], tmp[2])
                        }),
                );
            }
        }
        let vec_cuts: Vec<Vec<Node>> = cuts.iter().map(|(u, v, w)| vec![*u, *v, *w]).collect();
        self.filter_cuts(vec_cuts)
    }
}
