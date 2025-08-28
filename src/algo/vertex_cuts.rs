use fxhash::FxHashSet;

use super::*;

pub trait ArticluationPoint: GraphType<Dir = Undirected> {
    fn compute_articulation_points(&self) -> NodeBitSet;
    fn compute_articulation_points_with_visited(&self, visited: NodeBitSet) -> NodeBitSet;
}

impl<G> ArticluationPoint for G
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

pub struct ArticulationPointSearch<'a, T>
where
    T: AdjacencyList + GraphType<Dir = Undirected>,
{
    graph: &'a T,
    low_point: Vec<Node>,
    dfs_num: Vec<Node>,
    visited: NodeBitSet,
    articulation_points: NodeBitSet,
    current_dfs_num: Node,
    parent: Vec<Option<Node>>,
}

impl<'a, T> ArticulationPointSearch<'a, T>
where
    T: AdjacencyList + GraphType<Dir = Undirected>,
{
    /// Assumes the graph is connected, and for each edge (u, v) the edge (v, u) exists
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

    pub fn compute(mut self) -> NodeBitSet {
        let start = self.visited.iter_cleared_bits().next().unwrap();
        let _ = self.compute_recursive(start, 0);
        self.articulation_points
    }

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

pub struct GraphCutBuilder<'a, G>
where
    G: AdjacencyList + ArticluationPoint + Traversal + GraphType<Dir = Undirected>,
{
    graph: &'a G,
    enabled_cuts: Vec<CutType>,
    min_cc_size: Option<NumNodes>, // Minimum connected component size to keep
}

// Enum representing the different types of graph cuts that can be computed.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CutType {
    OneCut, // Single-node cut (articulation points)
    TwoCut,
    ThreeCut,
}

/// Builder for computing various graph cuts with configurable parameters.
///
/// The builder allows enabling/disabling specific cut types and setting
/// minimum connected component size requirements.
/// If a cut produces >=2 components where the second largest component does
/// not contain > threshold nodes, it is filtered out.
impl<'a, G> GraphCutBuilder<'a, G>
where
    G: AdjacencyList + ArticluationPoint + Traversal + GraphType<Dir = Undirected>,
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
    pub fn enable_cut(mut self, cut_type: CutType) -> Self {
        if self.enabled_cuts.contains(&cut_type) {
            self
        } else {
            self.enabled_cuts.push(cut_type);
            self
        }
    }

    /// Disables a specific cut type if it's currently enabled.
    ///
    /// # Arguments
    /// * `cut_type` - The cut type to disable
    pub fn disable_cut(mut self, cut_type: CutType) -> Self {
        if let Some(index) = self
            .enabled_cuts
            .iter()
            .position(|value| *value == cut_type)
        {
            self.enabled_cuts.swap_remove(index);
        }
        self
    }

    /// Sets the minimum connected component size requirement for cuts.
    ///
    /// # Arguments
    /// * `size` - Minimum number of nodes required in each connected component
    ///   after removing the cut. Use `None` for no minimum requirement.
    pub fn min_cc_size(mut self, size: Option<NumNodes>) -> Self {
        self.min_cc_size = size;
        self
    }

    /// Computes all enabled cuts and returns them as a single collection.
    ///
    /// The result is a vector of cut sets, where each cut set is a vector of nodes.
    /// only cuts that meet the minimum cc`size (if set)
    /// are included in the results.
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

    fn compute_one_cut(&self) -> Vec<Vec<Node>> {
        let candidates: Vec<Vec<Node>> = self
            .graph
            .compute_articulation_points()
            .iter_set_bits()
            .map(|u| vec![u])
            .collect();
        self.filter_cuts(candidates)
    }

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
