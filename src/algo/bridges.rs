use super::*;

pub trait Bridges: GraphType<Dir = Undirected> {
    fn compute_bridges(&self) -> Vec<Edge>;
}

impl<G> Bridges for G
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    fn compute_bridges(&self) -> Vec<Edge> {
        BridgeSearch::new(self).compute()
    }
}

struct BridgeSearch<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    graph: &'a G,
    visited: NodeBitSet,
    nodes_info: Vec<NodeInfo>,
    time: Node,
    bridges: Vec<Edge>,
}

impl<'a, G> BridgeSearch<'a, G>
where
    G: AdjacencyList + GraphType<Dir = Undirected>,
{
    fn new(graph: &'a G) -> Self {
        let n = graph.number_of_nodes();
        Self {
            graph,
            visited: NodeBitSet::new(n),
            nodes_info: vec![NodeInfo::default(); n as usize],
            time: 0,
            bridges: Vec::new(),
        }
    }

    fn compute(mut self) -> Vec<Edge> {
        for u in self.graph.vertices_no_singletons() {
            if self.visited.set_bit(u) {
                continue;
            }

            self.compute_node(u, u);
        }

        self.bridges
    }

    fn compute_node(&mut self, parent: Node, u: Node) -> NodeInfo {
        self.time += 1;

        self.nodes_info[u as usize] = NodeInfo {
            parent,
            discovery: self.time,
            low: self.time,
        };

        for v in self.graph.neighbors_of(u) {
            if !self.visited.set_bit(v) {
                let info_v = self.compute_node(u, v);

                self.nodes_info[u as usize].update_low(info_v.low);

                if info_v.low > self.nodes_info[u as usize].discovery {
                    self.bridges.push(Edge(u, v));
                }
            } else if v != self.nodes_info[u as usize].parent {
                let v_disc = self.nodes_info[v as usize].discovery;
                self.nodes_info[u as usize].update_low(v_disc);
            }
        }

        self.nodes_info[u as usize]
    }
}

#[derive(Clone, Copy, Default)]
struct NodeInfo {
    low: Node,
    discovery: Node,
    parent: Node,
}

impl NodeInfo {
    fn update_low(&mut self, value: Node) {
        self.low = self.low.min(value);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn bridges_in_path() {
        for n in [1, 5, 10, 15] {
            let mut graph = AdjArrayUndir::new(n);
            for u in 0..(n - 1) {
                graph.add_edge(u, u + 1);
            }

            let mut bridges = graph.compute_bridges();
            bridges.sort();

            assert_eq!(bridges, graph.ordered_edges(true).collect_vec());
        }
    }

    #[test]
    fn bridge_in_example() {
        let mut graph = AdjArrayUndir::new(6);
        graph.add_edges([(0, 1), (0, 2), (2, 1), (1, 3), (3, 4), (4, 5), (5, 3)]);

        assert_eq!(graph.compute_bridges(), vec![Edge(1, 3)]);
    }
}
