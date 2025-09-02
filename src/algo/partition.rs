/*!
# Partitioning of Nodes

This module provides data structures and utilities to partition the nodes of a graph
into disjoint **classes** (also called "blocks" or "subsets").

A partition is a common concept in graph algorithms, e.g.:

- Strongly Connected Components (SCCs)
- Bipartite classes
- Communities in clustering

The [`Partition`] struct allows:
- Creating and managing classes of nodes
- Moving nodes between classes
- Querying class membership
- Splitting a graph into subgraphs along partition boundaries

Helper traits like [`IntoPartition`] simplify construction from higher-level
representations (e.g. collections of node sets).

# Example

```rust
use ugraphs::algo::Partition;

let mut part = Partition::new(5);

// Add first class with nodes 0, 1
let c0 = part.add_class([0, 1]);

// Add second class with nodes 2, 3
let c1 = part.add_class([2, 3]);

// Move node 4 into class 0
part.move_node(4, c0);

assert_eq!(part.number_of_classes(), 2);
assert_eq!(part.number_in_class(c0), 3);
assert_eq!(part.class_of_edge(0, 4), Some(c0));
```
*/

use std::{iter::Enumerate, slice::Iter};

use itertools::Itertools;

use super::*;

/// Internally, we store PartitionClasses as Options whereas we expose NumNodes as PartitionClasses to the user
type PartitionClass = NumNodes;

/// Represents a **partition** of the node set into disjoint classes.
///
/// Each node can belong to at most one class, or remain **unassigned**.
/// Classes are internally identified by integer IDs and stored along with
/// their sizes.
pub struct Partition {
    classes: Vec<Option<OptionalNode>>,
    class_sizes: Vec<NumNodes>,
    unassigned: NumNodes,
}

/// Iterator over the members of a single partition class.
///
/// Returned by [`Partition::members_of_class`].
pub struct ClassMemberIter<'a> {
    classes: Enumerate<Iter<'a, Option<OptionalNode>>>,
    class_id: Option<OptionalNode>,
}

impl<'a> Iterator for ClassMemberIter<'a> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.classes
            .find(|(_, c)| **c == self.class_id)
            .map(|(u, _)| u as Node)
    }
}

impl Partition {
    /// Creates a new partition over `nodes` nodes, all initially unassigned.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::Partition;
    ///
    /// let part = Partition::new(4);
    /// assert_eq!(part.number_of_unassigned(), 4);
    /// assert_eq!(part.number_of_classes(), 0);
    /// ```
    pub fn new(nodes: Node) -> Self {
        Self {
            classes: vec![None; nodes as usize],
            class_sizes: vec![],
            unassigned: nodes as NumNodes,
        }
    }

    /// Creates a new class and assigns the given nodes to it.
    ///
    /// All nodes must be **previously unassigned**.
    /// Returns the new class identifier.
    ///
    /// # Panics
    /// - If any provided node was already assigned to another class.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::Partition;
    ///
    /// let mut part = Partition::new(4);
    /// let c0 = part.add_class([0, 1]);
    ///
    /// assert_eq!(part.number_in_class(c0), 2);
    /// assert_eq!(part.number_of_unassigned(), 2);
    /// ```
    pub fn add_class<I>(&mut self, nodes: I) -> PartitionClass
    where
        I: IntoIterator<Item = Node>,
    {
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

    /// Moves a node into an existing partition class.
    ///
    /// - If the node was already in a class, it is removed from its old class.
    /// - If the node was unassigned, it becomes assigned.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::Partition;
    ///
    /// let mut part = Partition::new(3);
    /// let c0 = part.add_class([0]);
    /// let c1 = part.add_class([1]);
    ///
    /// part.move_node(2, c0);
    /// assert_eq!(part.class_of_node(2), Some(c0));
    /// ```
    pub fn move_node(&mut self, node: Node, new_class: PartitionClass) {
        if let Some(old_class) = self.classes[node as usize].map(|old_class| old_class.get()) {
            self.class_sizes[old_class as usize] -= 1;
        } else {
            self.unassigned -= 1;
        }
        self.classes[node as usize] = OptionalNode::new(new_class);
        self.class_sizes[new_class as usize] += 1;
    }

    /// Returns the class identifier of a node, or `None` if the node is unassigned.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::Partition;
    ///
    /// let mut part = Partition::new(2);
    /// let c0 = part.add_class([0]);
    ///
    /// assert_eq!(part.class_of_node(0), Some(c0));
    /// assert_eq!(part.class_of_node(1), None);
    /// ```
    pub fn class_of_node(&self, node: Node) -> Option<PartitionClass> {
        self.classes[node as usize].map(|class| class.get() as NumNodes)
    }

    /// Returns the class identifier if both endpoints of an edge belong
    /// to the same class, or `None` otherwise.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::Partition;
    ///
    /// let mut part = Partition::new(3);
    /// let c0 = part.add_class([0, 1]);
    ///
    /// assert_eq!(part.class_of_edge(0, 1), Some(c0));
    /// assert_eq!(part.class_of_edge(1, 2), None);
    /// ```
    pub fn class_of_edge(&self, u: Node, v: Node) -> Option<PartitionClass> {
        let cu = self.class_of_node(u)?;
        let cv = self.class_of_node(v)?;
        if cu == cv { Some(cu) } else { None }
    }

    /// Returns the number of currently unassigned nodes.
    pub fn number_of_unassigned(&self) -> NumNodes {
        self.unassigned
    }

    /// Returns the number of nodes in the specified class.
    pub fn number_in_class(&self, class_id: PartitionClass) -> NumNodes {
        self.class_sizes[class_id as usize]
    }

    /// Returns the number of partition classes (0 if all nodes are unassigned)
    pub fn number_of_classes(&self) -> NumNodes {
        self.class_sizes.len() as NumNodes
    }

    /// Returns an iterator over all members of a given class.
    ///
    /// # Warning
    /// This operation is **linear in the total number of nodes**,
    /// not the size of the class itself.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::Partition;
    ///
    /// let mut part = Partition::new(3);
    /// let c0 = part.add_class([0, 2]);
    ///
    /// let members: Vec<_> = part.members_of_class(c0).collect();
    /// assert_eq!(members, vec![0, 2]);
    /// ```
    pub fn members_of_class(&self, class_id: Node) -> ClassMemberIter<'_> {
        let class = OptionalNode::new(class_id);
        assert!(self.class_sizes.len() > class_id as usize);
        ClassMemberIter {
            classes: self.classes.iter().enumerate(),
            class_id: class,
        }
    }

    /// Splits the graph into one subgraph per partition class.
    ///
    /// - The input graph must have the same number of nodes as the partition.
    /// - Returns a vector where `result[i]` corresponds to the subgraph induced
    ///   by class `i` and its node mapping.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::{prelude::*, algo::Partition, utils::NodeMapper};
    ///
    /// let mut g = AdjArray::new(4);
    /// g.add_edge(0, 1);
    /// g.add_edge(2, 3);
    ///
    /// let mut part = Partition::new(4);
    /// part.add_class([0, 1]);
    /// part.add_class([2, 3]);
    ///
    /// let subs = part.split_into_subgraphs_as::<_, AdjArray, NodeMapper>(&g);
    /// assert_eq!(subs.len(), 2);
    /// assert_eq!(subs[0].0.len(), 2);
    /// assert_eq!(subs[1].0.len(), 2);
    /// ```
    pub fn split_into_subgraphs_as<GI, GO, M>(&self, graph: &GI) -> Vec<(GO, M)>
    where
        GI: AdjacencyList,
        GO: GraphNew + GraphEdgeEditing,
        M: NodeMapSetter + NodeMapGetter,
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
    /// using the same graph type and [`NodeMapper`].
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

/// Convenience trait for converting a collection of classes into a [`Partition`].
///
/// Each inner collection is interpreted as one partition class.
pub trait IntoPartition {
    /// Consumes the collection and builds a [`Partition`] with `n` total nodes.
    ///
    /// # Example
    /// ```rust
    /// use ugraphs::algo::{Partition, IntoPartition};
    ///
    /// let classes = vec![vec![0, 1], vec![2, 3]];
    /// let part = classes.into_partition(4);
    ///
    /// assert_eq!(part.number_of_classes(), 2);
    /// assert_eq!(part.class_of_edge(0, 1), Some(0));
    /// assert_eq!(part.class_of_edge(2, 3), Some(1));
    /// ```
    fn into_partition(self, n: NumNodes) -> Partition;
}

impl<N, I> IntoPartition for I
where
    N: IntoIterator<Item = Node>,
    I: IntoIterator<Item = N>,
{
    fn into_partition(self, n: NumNodes) -> Partition {
        let mut partition = Partition::new(n);
        for class in self {
            partition.add_class(class);
        }
        partition
    }
}
