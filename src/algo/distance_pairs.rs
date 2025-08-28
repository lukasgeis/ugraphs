use super::*;

pub struct DistancePairsIterator<'a, G>
where
    G: AdjacencyList + GraphType,
{
    graph: &'a G,
    node: Node,
    distance: usize,

    neighbors: NodeBitSet,
    neighbor_lb: Node,
}

pub trait DistancePairs: AdjacencyList + GraphType {
    fn distance_pairs(&self, distance: usize) -> DistancePairsIterator<'_, Self>;
}

impl<G> DistancePairs for G
where
    G: AdjacencyList + GraphType,
{
    fn distance_pairs(&self, distance: usize) -> DistancePairsIterator<'_, Self> {
        assert!(distance > 1);
        DistancePairsIterator::new(self, distance)
    }
}

impl<'a, G> DistancePairsIterator<'a, G>
where
    G: AdjacencyList + GraphType,
{
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
        }

        for _ in 3..=self.distance {
            let mut next_dist = self.graph.vertex_bitset_unset();
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

impl<G> Iterator for DistancePairsIterator<'_, G>
where
    G: AdjacencyList + GraphType,
{
    type Item = (Node, Node);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.neighbors.get_first_set_index_atleast(self.neighbor_lb) {
            self.neighbor_lb = (v + 1) as Node;
            return Some((self.node, v as Node));
        }

        loop {
            self.node += 1;

            if self.node >= self.graph.number_of_nodes() {
                return None;
            }

            self.setup_node();

            if self.neighbors.cardinality() > 0 {
                return self.next();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn path() {
        // Directed Path
        for n in 3..(20 as NumNodes) {
            let mut graph = AdjArray::new(n);
            for u in 0..(n - 1) {
                graph.add_edge(u, u + 1);
            }

            for distance in 2..(n - 1) {
                let mut dist_pairs = graph.distance_pairs(distance as usize).collect_vec();
                dist_pairs.sort_unstable();

                let real_dist_pairs = (0..n)
                    .filter_map(|u| (u + distance < n).then_some((u, u + distance)))
                    .collect_vec();
                assert_eq!(dist_pairs, real_dist_pairs);
            }
        }

        // Undirected Path
        for n in 3..(20 as NumNodes) {
            let mut graph = AdjArrayUndir::new(n);
            for u in 0..(n - 1) {
                graph.add_edge(u, u + 1);
            }

            for distance in 2..(n - 1) {
                let mut dist_pairs = graph.distance_pairs(distance as usize).collect_vec();
                dist_pairs.sort_unstable();

                let real_dist_pairs = (0..n)
                    .flat_map(|u| {
                        (0..=(distance / 2)).filter_map(move |d| {
                            let step = d * 2 + (distance % 2);
                            (step > 0 && u + step < n).then_some((u, u + step))
                        })
                    })
                    .collect_vec();
                assert_eq!(dist_pairs, real_dist_pairs);
            }
        }

        // Directed Double-Linked Path
        for n in 3..(20 as NumNodes) {
            let mut graph = AdjArray::new(n);
            for u in 0..(n - 1) {
                graph.add_edge(u, u + 1);
                graph.add_edge(u + 1, u);
            }

            for distance in 2..(n - 1) {
                let mut dist_pairs = graph.distance_pairs(distance as usize).collect_vec();
                dist_pairs.sort_unstable();

                let mut real_dist_pairs = (0..n)
                    .flat_map(|u| {
                        (0..=(distance / 2)).flat_map(move |d| {
                            let mut pairs = Vec::with_capacity(2);
                            let step = d * 2 + (distance % 2);

                            if u + step < n {
                                pairs.push((u, u + step));
                            }
                            if u >= step && step != 0 {
                                pairs.push((u, u - step));
                            }
                            pairs
                        })
                    })
                    .collect_vec();
                real_dist_pairs.sort_unstable();
                assert_eq!(dist_pairs, real_dist_pairs);
            }
        }
    }
}
