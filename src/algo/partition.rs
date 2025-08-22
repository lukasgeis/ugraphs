use itertools::Itertools;

use crate::{
    ops::*,
    utils::{Getter, NodeMapper, Setter},
    *,
};

/// Internally, we store PartitionClasses as Options whereas we expose NumNodes as PartitionClasses
/// to the user
type PartitionClass = NumNodes;

/// A partition splits a graph into node-disjoint substructures (think SCCs, bipartite classes, etc)
pub struct Partition {
    classes: Vec<Option<OptionalNode>>,
    class_sizes: Vec<NumNodes>,
    unassigned: NumNodes,
}

impl Partition {
    /// Creates a partition for `nodes` nodes which are initially all unassigned
    pub fn new(nodes: Node) -> Self {
        Self {
            classes: vec![None; nodes as usize],
            class_sizes: vec![],
            unassigned: nodes as NumNodes,
        }
    }

    /// Creates a new partition class and assigns all provided nodes to it; we require that these
    /// nodes were previously unassigned.
    pub fn add_class<I: IntoIterator<Item = Node>>(&mut self, nodes: I) -> PartitionClass {
        let raw_class_id = self.class_sizes.len();
        let class_id = OptionalNode::new(raw_class_id as Node);
        self.class_sizes.push(0);

        let size = &mut self.class_sizes[raw_class_id];
        for u in nodes {
            assert_eq!(self.classes[u as usize], None); // check that node is unassigned
            self.classes[u as usize] = class_id;
            *size += 1;
        }

        self.unassigned -= *size;

        // self.class_sizes.len() should never be Node::MAX
        raw_class_id as NumNodes
    }

    /// Moves node into an existing partition class. The node may or may not have been previously assigned.
    pub fn move_node(&mut self, node: Node, new_class: PartitionClass) {
        if let Some(old_class) = self.classes[node as usize].map(|old_class| old_class.get()) {
            self.class_sizes[old_class as usize] -= 1;
        } else {
            self.unassigned -= 1;
        }
        self.classes[node as usize] = OptionalNode::new(new_class);
        self.class_sizes[new_class as usize] += 1;
    }

    /// Returns the class identifier of node `node` or `None` if `node` is unassigned
    pub fn class_of_node(&self, node: Node) -> Option<PartitionClass> {
        self.classes[node as usize].map(|class| class.get() as NumNodes)
    }

    /// Returns the class identifier if both nodes `u` and `v` are assigned to the same class
    /// and `None` otherwise.
    pub fn class_of_edge(&self, u: Node, v: Node) -> Option<PartitionClass> {
        let cu = self.class_of_node(u)?;
        let cv = self.class_of_node(v)?;
        if cu == cv { Some(cu) } else { None }
    }

    /// Returns the number of unassigned nodes
    pub fn number_of_unassigned(&self) -> NumNodes {
        self.unassigned
    }

    /// Returns the number of nodes in class `class_id`
    pub fn number_in_class(&self, class_id: PartitionClass) -> NumNodes {
        self.class_sizes[class_id as usize]
    }

    /// Returns the number of partition classes (0 if all nodes are unassigned)
    pub fn number_of_classes(&self) -> NumNodes {
        self.class_sizes.len() as NumNodes
    }

    /// Returns the members of a partition class in order.
    ///
    /// # Warning
    /// This operation is expensive and requires time linear in the total number of nodes, i.e. it
    /// is roughly independent of the actual size of partition class `class_id`.
    pub fn members_of_class(&self, class_id: Node) -> impl Iterator<Item = Node> + '_ {
        let class = OptionalNode::new(class_id);
        assert!(self.class_sizes.len() > class_id as usize);
        self.classes
            .iter()
            .enumerate()
            .filter_map(move |(i, &c)| (c == class).then_some(i as Node))
    }

    /// Splits the input graph `graph` (has to have the same number of nodes as `self`) into
    /// one subgraph per partition class; the `result[i]` corresponds to partition class `i`.
    pub fn split_into_subgraphs_as<GI, GO, M>(&self, graph: &GI) -> Vec<(GO, M)>
    where
        GI: AdjacencyList,
        GO: GraphNew + GraphEdgeEditing,
        M: Setter + Getter,
    {
        assert_eq!(graph.len(), self.classes.len());

        // Create an empty graph and mapper with the capacity for each partition class
        let mut result = (0..self.number_of_classes())
            .map(|class_id| {
                let n = self.number_in_class(class_id);
                (GO::new(n as NumNodes), M::with_capacity(n))
            })
            .collect_vec();

        // Iterator over all (assigned) nodes and map them into their respective subgraph
        let mut nodes_mapped_in_class = vec![0; self.number_of_classes() as usize];
        for (u, class_id) in self
            .classes
            .iter()
            .enumerate()
            .filter_map(|(i, &class)| class.map(|c| (i, c.get() as usize)))
        {
            result[class_id]
                .1
                .map_node_to(u as Node, nodes_mapped_in_class[class_id]);
            nodes_mapped_in_class[class_id] += 1;
        }

        // Iterate over all edges incident to assigned nodes
        for (u, class_id) in self
            .classes
            .iter()
            .enumerate()
            .filter_map(|(i, &class)| class.map(|c| (i, c.get())))
        {
            let class = OptionalNodeImpl::new(class_id);

            let u = u as Node;
            let result_containg_u = &mut result[class_id as usize];

            let mapped_u = result_containg_u.1.new_id_of(u).unwrap();

            // Iterate over all out-neighbors of u that are in the same partition class
            for Edge(_, v) in graph
                .edges_of(u, true)
                .filter(|Edge(_, v)| self.classes[*v as usize] == class)
            {
                let mapped_v = result_containg_u.1.new_id_of(v).unwrap();
                result_containg_u.0.add_edge(mapped_u, mapped_v);
            }
        }

        result
    }

    /// Shorthand for [`Partition::split_into_subgraphs_as`]
    pub fn split_into_subgraphs<G>(&self, graph: &G) -> Vec<(G, NodeMapper)>
    where
        G: AdjacencyList + GraphNew + GraphEdgeEditing,
    {
        self.split_into_subgraphs_as(graph)
    }
}

impl From<NodeBitSet> for Partition {
    fn from(set: NodeBitSet) -> Self {
        let mut part = Partition::new(set.number_of_bits());

        assert_eq!(part.add_class(std::iter::empty()), 0);
        assert_eq!(part.add_class(std::iter::empty()), 1);

        for i in 0..set.number_of_bits() {
            part.move_node(i, set.get_bit(i) as u32);
        }

        part
    }
}
