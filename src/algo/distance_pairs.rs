use crate::{ops::*, *};

pub struct DistancePairsIterator<'a, G: AdjacencyList + GraphType> {
    graph: &'a G,
    node: Node,
    distance: usize,

    neighbors: NodeBitSet,
    neighbor_lb: Node,
}

pub trait DistancePairs: AdjacencyList + GraphType {
    fn distance_pairs(&self, distance: usize) -> DistancePairsIterator<'_, Self>;
}

impl<G: AdjacencyList + GraphType> DistancePairs for G {
    fn distance_pairs(&self, distance: usize) -> DistancePairsIterator<'_, Self> {
        assert!(distance > 1);
        DistancePairsIterator::new(self, distance)
    }
}

impl<'a, G: AdjacencyList + GraphType> DistancePairsIterator<'a, G> {
    pub fn new(graph: &'a G, distance: usize) -> Self {
        let n = graph.number_of_nodes();

        let mut inst = Self {
            graph,
            node: 0,
            neighbors: NodeBitSet::new(n),
            neighbor_lb: 0,
            distance,
        };

        inst.setup_node();

        inst
    }

    fn setup_node(&mut self) {
        self.neighbors.clear_all();

        for v in self.graph.neighbors_of(self.node) {
            self.neighbors.set_bits(self.graph.neighbors_of(v));
            self.neighbors.set_bit(v);
        }

        for _ in 3..=self.distance {
            let mut next_dist = self.neighbors.clone();
            for x in self.neighbors.iter_set_bits() {
                next_dist.set_bits(self.graph.neighbors_of(x));
            }
            self.neighbors = next_dist;
        }

        // In undirected graphs, we normalize the edges and can thus skip all neighbors that are
        // smaller than or equal to the current node
        self.neighbor_lb = if G::is_undirected() { self.node + 1 } else { 0 };
    }
}

impl<G: AdjacencyList + GraphType> Iterator for DistancePairsIterator<'_, G> {
    type Item = (Node, Node);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.neighbors.get_first_set_index_atleast(self.neighbor_lb) {
            self.neighbor_lb = (v + 1) as Node;
            return Some((self.node, v as Node));
        }

        loop {
            self.node += 1;

            if self.node + 1 >= self.graph.number_of_nodes() {
                return None;
            }

            self.setup_node();

            if self.neighbors.cardinality() > 0 {
                return self.next();
            }
        }
    }
}
