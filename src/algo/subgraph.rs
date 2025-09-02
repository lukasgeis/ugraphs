/*!
# Subgraph Algorithms

Provides traits and implementations for extracting subgraphs from graphs
(e.g., vertex-induced subgraphs, subgraphs without singletons) and for
combining multiple graphs into a single disjoint union.
*/

use super::*;

/// A trait for creating different kinds of subgraphs from a graph.
///
/// Subgraphs may be vertex-induced, restricted to certain nodes, or
/// variants that filter out special structures like singletons.
pub trait Subgraph: Sized {
    /// Creates a **vertex-induced subgraph** from the current graph,
    /// restricted to the nodes in `vertices`.
    ///
    /// The result contains:
    /// - A new graph of type `GO`
    /// - A mapping of old node IDs to new ones
    ///
    /// # Type Parameters
    /// - `M`: Node map implementation
    /// - `GO`: Output graph type
    /// - `S`: Node set type
    ///
    /// # Returns
    /// A tuple `(graph, mapping)` with the induced subgraph and the
    /// node remapping.
    fn vertex_induced_as<M, GO, S>(&self, vertices: &S) -> (GO, M)
    where
        M: NodeMapGetter + NodeMapSetter,
        GO: GraphFromScratch + GraphType,
        S: Set<Node>;

    /// Creates a vertex-induced subgraph of the same type as `Self`.
    ///
    /// This is shorthand for [`Subgraph::vertex_induced_as`] where
    /// the output graph type matches the input.
    fn vertex_induced<M, S>(&self, vertices: &S) -> (Self, M)
    where
        Self: GraphFromScratch + GraphType,
        M: NodeMapGetter + NodeMapSetter,
        S: Set<Node>,
    {
        self.vertex_induced_as(vertices)
    }

    /// Creates a subgraph of the current graph that excludes all **singleton
    /// nodes** (nodes of degree 0).
    ///
    /// This method keeps only vertices with at least one incident edge
    /// and returns both the new graph and a node mapping.
    fn subgraph_without_singletons<M>(&self) -> (Self, M)
    where
        Self: GraphFromScratch + GraphType + Singletons,
        M: NodeMapGetter + NodeMapSetter,
    {
        self.vertex_induced(&NodeBitSet::new_with_bits_cleared(
            self.number_of_nodes(),
            self.vertices_no_singletons(),
        ))
    }

    /// Creates a subgraph that only contains edges where **both endpoints**
    /// are included in the provided set `vertices`.
    ///
    /// Unlike [`Subgraph::vertex_induced_as`], this does not remove nodes
    /// outside the set — it only filters edges.
    fn subgraph_of<GO, S>(&self, vertices: &S) -> GO
    where
        GO: GraphFromScratch + GraphType,
        S: Set<Node>;
}

impl<G> Subgraph for G
where
    G: GraphFromScratch + AdjacencyList + GraphType,
{
    fn vertex_induced_as<M, GO, S>(&self, vertices: &S) -> (GO, M)
    where
        M: NodeMapGetter + NodeMapSetter,
        GO: GraphFromScratch + GraphType,
        S: Set<Node>,
    {
        let new_n = vertices.len() as NumNodes;
        let mut mapping = M::with_capacity(new_n);

        for (new, old) in vertices.iter().enumerate() {
            mapping.map_node_to(old, new as Node);
        }

        // Prevent moving mapping into the closure
        let mapping_ref = &mapping;
        let graph = GO::from_edges(
            new_n,
            vertices.iter().flat_map(|u| {
                let new_u = mapping_ref.new_id_of(u).unwrap();
                self.neighbors_of(u).filter_map(move |v| {
                    if let Some(new_v) = mapping_ref.new_id_of(v) {
                        let e = Edge(new_u, new_v);
                        if GO::is_directed() || Self::is_directed() || e.is_normalized() {
                            return Some(e);
                        }
                    }
                    None
                })
            }),
        );

        (graph, mapping)
    }

    fn subgraph_of<GO, S>(&self, vertices: &S) -> GO
    where
        GO: GraphFromScratch + GraphType,
        S: Set<Node>,
    {
        GO::from_edges(
            self.number_of_nodes(),
            vertices.iter().flat_map(|u| {
                self.neighbors_of(u).filter_map(move |v| {
                    let e = Edge(u, v);
                    (vertices.contains(&v)
                        && (GO::is_directed() || Self::is_directed() || e.is_normalized()))
                    .then_some(e)
                })
            }),
        )
    }
}

/// A trait for combining multiple graphs into a **single disjoint graph**.
///
/// Each input graph is copied into the new graph, with all node IDs
/// shifted to avoid collisions. This allows concatenating graphs into
/// a single larger graph where their node sets are disjoint.
pub trait Concat {
    /// Concatenates a collection of graphs into one disjoint union graph.
    ///
    /// Node relabeling rule:  
    /// If the input graphs have sizes `n_1, ..., n_k`, then node `i` of
    /// graph `G_j` is mapped to
    /// ```text
    /// i + (n_1+ n_2 + ... + n_(j - 1))
    /// ```
    ///
    /// # Arguments
    /// - `graphs`: An iterable of references to graphs
    ///
    /// # Returns
    /// A single graph of the same type as `Self` containing all edges
    /// and nodes from the inputs, shifted to be disjoint.
    fn concat<'a, IG, T>(graphs: T) -> Self
    where
        IG: 'a + AdjacencyList + GraphType,
        T: IntoIterator<Item = &'a IG> + Clone;
}

impl<G> Concat for G
where
    G: GraphFromScratch + GraphType,
{
    fn concat<'a, IG, T>(graphs: T) -> Self
    where
        IG: 'a + AdjacencyList + GraphType,
        T: IntoIterator<Item = &'a IG> + Clone,
    {
        let total_n = graphs
            .clone()
            .into_iter()
            .map(|g| g.number_of_nodes())
            .sum();

        let mut node_shift: Node = 0;
        Self::from_edges(
            total_n,
            graphs.into_iter().flat_map(|g| {
                let prev_shift = node_shift;
                node_shift += g.number_of_nodes() as Node;
                g.edges(IG::is_undirected() && Self::is_undirected())
                    .map(move |Edge(u, v)| Edge(u + prev_shift, v + prev_shift))
            }),
        )
    }
}
