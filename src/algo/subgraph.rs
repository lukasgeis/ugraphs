use crate::{
    ops::{AdjacencyList, GraphFromScratch, GraphType},
    utils::*,
    *,
};

/// Trait for creating subgraphs
pub trait Subgraph: Sized {
    /// Create a new vertex-induced subgraph with remapped nodes
    fn vertex_induced_as<M: Getter + Setter, GO: GraphFromScratch + GraphType, S: Set<Node>>(
        &self,
        vertices: &S,
    ) -> (GO, M);

    /// Like `vertex_induced_as` but the output graph has the same type as `Self`
    fn vertex_induced<M: Getter + Setter, S: Set<Node>>(&self, vertices: &S) -> (Self, M)
    where
        Self: GraphFromScratch + GraphType,
    {
        self.vertex_induced_as(vertices)
    }

    /// Remaps the graph to only include nodes with degree greater than 0
    fn subgraph_without_singletons<M: Getter + Setter>(&self) -> (Self, M)
    where
        Self: GraphFromScratch + AdjacencyList + GraphType,
    {
        self.vertex_induced(&NodeBitSet::new_with_bits_cleared(
            self.number_of_nodes(),
            self.vertices_with_neighbors(),
        ))
    }

    /// Creates a subgraph of `Self` that only contains edges between specified nodes without
    /// deleting other nodes from the graph.
    fn subgraph_of<GO: GraphFromScratch + GraphType, S: Set<Node>>(&self, vertices: &S) -> GO;
}

impl<G: GraphFromScratch + AdjacencyList + GraphType> Subgraph for G {
    fn vertex_induced_as<M: Getter + Setter, GO: GraphFromScratch + GraphType, S: Set<Node>>(
        &self,
        vertices: &S,
    ) -> (GO, M) {
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

    fn subgraph_of<GO: GraphFromScratch + GraphType, S: Set<Node>>(&self, vertices: &S) -> GO {
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

/// Trait for combining multiple graphs into one
pub trait Concat {
    /// Takes a list of graphs and outputs a single graph containing disjoint copies of them all
    /// Let n_1, ..., n_k be the number of nodes in the input graph. Then node i of graph G_j becomes
    /// sum(n_1 + .. + n_{j-1}) + i
    fn concat<'a, IG: 'a + AdjacencyList + GraphType, T: IntoIterator<Item = &'a IG> + Clone>(
        graphs: T,
    ) -> Self;
}

impl<G: GraphFromScratch + GraphType> Concat for G {
    fn concat<'a, IG: 'a + AdjacencyList + GraphType, T: IntoIterator<Item = &'a IG> + Clone>(
        graphs: T,
    ) -> Self {
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
