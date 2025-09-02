/*!
# Node Mapper

Provides functionality to map nodes between graphs or subgraphs.
Includes utilities to store, query, and manipulate node mappings efficiently.
Can handle identity mappings, forward mappings, inverse mappings, and compositions.
*/
use crate::{edge::*, node::*, ops::*};

use fxhash::FxHashMap;
use itertools::Itertools;
use std::cmp::Ordering;
use std::fmt;

/// A trait for constructing node mappings.
///
/// Provides functions for building a [`NodeMapper`] from explicit data sources,  
/// such as rank vectors or custom node orderings.
pub trait NodeMapSetter: Sized {
    /// Creates a mapper where the largest node that can be inserted is `n-1`.
    fn with_capacity(n: NumNodes) -> Self;

    /// Creates a mapper where the largest node that can be handled is `n-1`.
    /// Each mapping is of form `x <-> x` for all `x`.
    /// Subsequent calls to [`NodeMapSetter::map_node_to`] are forbidden.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::identity(5);
    /// assert_eq!(mapper.new_id_of(2), Some(2));
    /// ```
    fn identity(n: NumNodes) -> Self;

    /// Stores a mapping `old <-> new`.
    fn map_node_to(&mut self, old: Node, new: Node);

    /// Constructs a mapper from a Node-slice `new_ids` where `new_ids[i]` stores the new id of the old node `i`.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::from_rank(&[1, 0]);
    /// assert_eq!(mapper.new_id_of(0), Some(1));
    /// assert_eq!(mapper.new_id_of(1), Some(0));
    /// ```
    fn from_rank(new_ids: &[Node]) -> Self {
        let cap = new_ids
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(new_ids.len() as Node);
        let mut res = Self::with_capacity(cap + 1);
        for (old, &new) in new_ids.iter().enumerate() {
            res.map_node_to(old as Node, new);
        }
        res
    }

    /// Constructs a mapper from a sequence of tuples `(old, new)`.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::from_sequence(&[(1, 0), (0, 1)]);
    /// assert_eq!(mapper.new_id_of(0), Some(1));
    /// assert_eq!(mapper.new_id_of(1), Some(0));
    /// ```
    fn from_sequence(seq: &[(Node, Node)]) -> Self {
        if seq.is_empty() {
            return Self::with_capacity(0);
        }

        let cap = seq.iter().map(|&(o, n)| o.max(n)).max().unwrap();

        let mut res = Self::with_capacity(cap + 1);
        for &(old, new) in seq {
            res.map_node_to(old as Node, new);
        }
        res
    }
}

/// Helper struct that is returned by `Getters` in [`NodeMapGetter`] such as
/// [`NodeMapGetter::get_old_ids`], [`NodeMapGetter::get_filtered_old_ids`],
/// [`NodeMapGetter::get_new_ids`], [`NodeMapGetter::get_filtered_new_ids`].
pub struct GetIdIter<'a, G, I, const FILTER: bool>
where
    I: Iterator<Item = Node> + 'a,
{
    ids: I,
    getter: &'a G,
    mapper: fn(&'a G, Node) -> Option<Node>,
}

impl<'a, G, I, const FILTER: bool> Iterator for GetIdIter<'a, G, I, FILTER>
where
    I: Iterator<Item = Node> + 'a,
{
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_id = self.ids.next()?;
            if let Some(mapped_id) = (self.mapper)(self.getter, next_id) {
                return Some(mapped_id);
            } else if !FILTER {
                return Some(next_id);
            }
        }
    }
}

/// Iterator type returned by [`NodeMapGetter::get_old_ids`].
///
/// Produces the original (old) node IDs corresponding to some iterable  
/// of new node IDs.  
///
/// This variant **does not filter out** invalid or unmapped IDs.
pub type OldIds<'a, G, I> = GetIdIter<'a, G, <I as IntoIterator>::IntoIter, false>;

/// Iterator type returned by [`NodeMapGetter::get_filtered_old_ids`].
///
/// Produces the original (old) node IDs corresponding to some iterable  
/// of new node IDs.  
///
/// This variant **filters out invalid or unmapped IDs**.
pub type FilteredOldIds<'a, G, I> = GetIdIter<'a, G, <I as IntoIterator>::IntoIter, true>;

/// Iterator type returned by [`NodeMapGetter::get_new_ids`].
///
/// Produces the mapped (new) node IDs corresponding to some iterable  
/// of original node IDs.  
///
/// This variant **does not filter out** invalid or unmapped IDs.
pub type NewIds<'a, G, I> = GetIdIter<'a, G, <I as IntoIterator>::IntoIter, false>;

/// Iterator type returned by [`NodeMapGetter::get_filtered_new_ids`].
///
/// Produces the mapped (new) node IDs corresponding to some iterable  
/// of original node IDs.  
///
/// This variant **filters out invalid or unmapped IDs**.
pub type FilteredNewIds<'a, G, I> = GetIdIter<'a, G, <I as IntoIterator>::IntoIter, true>;

/// A trait for accessing node mappings.
///
/// Provides read-only access to the underlying mapping data, useful when  
/// algorithms need to inspect how nodes were reordered or relabeled.
pub trait NodeMapGetter {
    /// If the mapping `(old, new)` exists, returns `Some(new)`, otherwise `None`.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mut mapper = NodeMapper::with_capacity(3);
    /// mapper.map_node_to(0, 2);
    /// assert_eq!(mapper.new_id_of(0), Some(2));
    /// ```
    fn new_id_of(&self, old: Node) -> Option<Node>;

    /// If the mapping `(old, new)` exists, returns `Some(old)`, otherwise `None`.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mut mapper = NodeMapper::with_capacity(3);
    /// mapper.map_node_to(0, 2);
    /// assert_eq!(mapper.old_id_of(2), Some(0));
    /// ```
    fn old_id_of(&self, new: Node) -> Option<Node>;

    /// Returns the number of explicitly stored mappings; returns `0` for identity mapping.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::identity(5);
    /// assert_eq!(mapper.len(), 0);
    /// ```
    fn len(&self) -> Node;

    /// Returns `true` if no mapping is stored; returns `true` for identity mapping.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Applies [`NodeMapGetter::old_id_of`] to each iterator item. Uses the iterator item (new) as a fallback
    /// if the mapping does not exist.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::identity(3);
    /// let old_ids: Vec<_> = mapper.get_old_ids(vec![0, 1, 2]).collect();
    /// assert_eq!(old_ids, vec![0, 1, 2]);
    /// ```
    fn get_old_ids<'a, I>(&self, new_ids: I) -> OldIds<'_, Self, I>
    where
        I: IntoIterator<Item = Node> + 'a,
        Self: Sized,
    {
        GetIdIter {
            ids: new_ids.into_iter(),
            getter: self,
            mapper: Self::old_id_of,
        }
    }

    /// Applies [`NodeMapGetter::old_id_of`] to each iterator item and returns only items for which a mapping exists.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::from_sequence(&[(1, 0), (2, 1)]);
    /// let old_ids: Vec<_> = mapper.get_filtered_old_ids(vec![0, 1, 2]).collect();
    /// assert_eq!(old_ids, vec![1, 2]);
    /// ```
    fn get_filtered_old_ids<'a, I>(&self, new_ids: I) -> FilteredOldIds<'_, Self, I>
    where
        I: IntoIterator<Item = Node> + 'a,
        Self: Sized,
    {
        GetIdIter {
            ids: new_ids.into_iter(),
            getter: self,
            mapper: Self::old_id_of,
        }
    }

    /// Applies [`NodeMapGetter::new_id_of`] to each iterator item. Uses the iterator item (old) as a fallback
    /// if the mapping does not exist.
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::identity(3);
    /// let new_ids: Vec<_> = mapper.get_old_ids(vec![0, 1, 2]).collect();
    /// assert_eq!(new_ids, vec![0, 1, 2]);
    /// ```
    fn get_new_ids<'a, I>(&self, old_ids: I) -> NewIds<'_, Self, I>
    where
        I: IntoIterator<Item = Node> + 'a,
        Self: Sized,
    {
        GetIdIter {
            ids: old_ids.into_iter(),
            getter: self,
            mapper: Self::new_id_of,
        }
    }

    /// Applies [`NodeMapGetter::new_id_of`] to each iterator item and returns only items for which a mapping exists.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mapper = NodeMapper::from_sequence(&[(1, 0), (2, 1)]);
    /// let new_ids: Vec<_> = mapper.get_filtered_new_ids(vec![0, 1, 2]).collect();
    /// assert_eq!(new_ids, vec![0, 1]);
    /// ```
    fn get_filtered_new_ids<'a, I>(&self, old_ids: I) -> FilteredNewIds<'_, Self, I>
    where
        I: IntoIterator<Item = Node> + 'a,
        Self: Sized,
    {
        GetIdIter {
            ids: old_ids.into_iter(),
            getter: self,
            mapper: Self::new_id_of,
        }
    }

    /// Create a 'copy' of type `GO` from the input graph where all nodes are relabelled according to this mapper.
    /// Any node `u` (and its incident edges) is dropped if there is no mapping.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, utils::*};
    ///
    /// let g = AdjArray::from_edges(2, [(0, 1)]);
    /// assert!(g.has_edge(0, 1));
    /// assert!(!g.has_edge(1, 0));
    ///
    /// let mapper = NodeMapper::from_rank(&[1, 0]);
    /// let gm: AdjMatrix = mapper.relabelled_graph_as(&g);
    /// assert!(!gm.has_edge(0, 1));
    /// assert!(gm.has_edge(1, 0));
    /// ```
    fn relabelled_graph_as<GI, GO>(&self, input: &GI) -> GO
    where
        GI: GraphNodeOrder + AdjacencyList + GraphType,
        GO: GraphFromScratch,
    {
        let max_node = input
            .vertices()
            .flat_map(|u| self.new_id_of(u).map(|x| x + 1))
            .max()
            .unwrap_or(0);

        if max_node == 0 {
            return GO::from_edges(0, std::iter::empty::<Edge>());
        }

        GO::from_edges(
            max_node as NumNodes,
            input
                .vertices()
                .filter_map(|old_u| self.new_id_of(old_u).map(|new_u| (old_u, new_u)))
                .flat_map(|(old_u, new_u)| {
                    input
                        .edges_of(old_u, GI::is_undirected())
                        .filter_map(move |Edge(_, old_v)| {
                            self.new_id_of(old_v).map(|new_v| Edge(new_u, new_v))
                        })
                }),
        )
    }

    /// Short-hand for [`NodeMapGetter::relabelled_graph_as`] where the output type matches the input type.
    fn relabelled_graph<G>(&self, input: &G) -> G
    where
        G: AdjacencyList + GraphType + GraphFromScratch,
    {
        self.relabelled_graph_as::<G, G>(input)
    }
}

/// A trait for composing two node mappings.
///
/// Allows combining two [`NodeMapper`] implementations into a single one,  
/// so that applying the composed mapper is equivalent to applying the first  
/// mapping and then the second in sequence.
///
/// This is useful for building complex node reorderings step by step.
pub trait NodeMapCompose {
    /// Takes two mappers `M1` (original -> intermediate) and `M2` (intermediate -> final)
    /// and produces a new mapper (original -> final). All mappings without a correspondence
    /// in the other mapper are dropped.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mut m1 = NodeMapper::with_capacity(5);
    /// m1.map_node_to(0, 1);
    /// let mut m2 = NodeMapper::with_capacity(5);
    /// m2.map_node_to(1, 3);
    /// let composed = NodeMapper::compose(&m1, &m2);
    /// assert_eq!(composed.new_id_of(0), Some(3));
    /// ```
    fn compose(first: &Self, second: &Self) -> Self;
}

/// A trait for inverting a node mapping.
pub trait NodeMapInverse {
    /// Returns a new mapper where for each mapping `(a, b)` of the original,
    /// there exists a mapping `(b, a)` in the new mapper.
    ///
    /// # Example
    /// ```
    /// use ugraphs::utils::*;
    ///
    /// let mut mapper = NodeMapper::with_capacity(3);
    /// mapper.map_node_to(0, 2);
    /// let inv = mapper.inverse();
    /// assert_eq!(inv.new_id_of(2), Some(0));
    /// ```
    #[must_use]
    fn inverse(&self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A node mapper that only stores forward mappings (`old -> new`).
///
/// Does not support inverse queries.
/// Useful when only relabelling in one direction is required.
pub struct WriteOnlyNodeMapper {}

impl NodeMapSetter for WriteOnlyNodeMapper {
    fn with_capacity(_: Node) -> Self {
        Self {}
    }

    fn identity(_n: Node) -> Self {
        Self {}
    }

    fn map_node_to(&mut self, _old: Node, _new: Node) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A bidirectional node mapping between "old" and "new" nodes.
///
/// Supports querying both directions (`old -> new`, `new -> old`).
/// Can represent identity mappings (all nodes map to themselves) or explicit mappings.
/// Provides efficient relabelling utilities for graphs.
#[derive(Clone)]
pub struct NodeMapper {
    new_to_old: FxHashMap<Node, Node>,
    old_to_new: FxHashMap<Node, Node>,
    is_identity: bool,
}

impl NodeMapSetter for NodeMapper {
    fn with_capacity(n: Node) -> Self {
        Self {
            new_to_old: FxHashMap::with_capacity_and_hasher(n as usize, Default::default()),
            old_to_new: FxHashMap::with_capacity_and_hasher(n as usize, Default::default()),
            is_identity: false,
        }
    }

    fn identity(_n: Node) -> Self {
        let mut res = Self::with_capacity(0);
        res.is_identity = true;
        res
    }

    fn map_node_to(&mut self, old: Node, new: Node) {
        assert!(!self.is_identity);
        let success = self.old_to_new.insert(old, new).is_none()
            && self.new_to_old.insert(new, old).is_none();
        assert!(success);
    }
}

impl NodeMapGetter for NodeMapper {
    fn new_id_of(&self, old: Node) -> Option<Node> {
        if self.is_identity {
            Some(old)
        } else {
            Some(*self.old_to_new.get(&old)?)
        }
    }

    fn old_id_of(&self, new: Node) -> Option<Node> {
        if self.is_identity {
            Some(new)
        } else {
            Some(*self.new_to_old.get(&new)?)
        }
    }

    fn len(&self) -> Node {
        self.old_to_new.len() as Node
    }
}

impl NodeMapCompose for NodeMapper {
    fn compose(first: &Self, second: &Self) -> Self {
        if first.is_identity {
            return second.clone();
        }

        if second.is_identity {
            return first.clone();
        }

        let mut composition = Self::with_capacity(second.len() as Node);
        for (&original, &intermediate) in first.old_to_new.iter() {
            if let Some(new) = second.new_id_of(intermediate) {
                composition.map_node_to(original, new);
            }
        }
        composition
    }
}

impl NodeMapInverse for NodeMapper {
    fn inverse(&self) -> Self {
        Self {
            old_to_new: self.new_to_old.clone(),
            new_to_old: self.old_to_new.clone(),
            is_identity: self.is_identity,
        }
    }
}

impl fmt::Debug for NodeMapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            format!(
                "[{}]",
                self.old_to_new
                    .iter()
                    .map(|(&o, &n)| format!("{o}<->{n}"))
                    .join(", ")
            )
            .as_str(),
        )?;

        Ok(())
    }
}

/// A mapping based on the index order of nodes in a given iterator.
///
/// Each node is assigned a new id based on its position in the sequence.
/// Less overhead than [`NodeMapper`] but only useful for complete (dense) mappings.
#[derive(Clone)]
pub struct IndexMapper {
    new_to_old: Vec<Node>,
    old_to_new: Vec<Node>,
}

impl IndexMapper {
    /// Creates a new `IndexMapper` from two `Vec`-mappings.
    /// Does not assert that these value are correct at runtime.
    pub fn from_vecs(old_to_new: Vec<Node>, new_to_old: Vec<Node>) -> Self {
        Self {
            old_to_new,
            new_to_old,
        }
    }
}

impl NodeMapGetter for IndexMapper {
    fn new_id_of(&self, old: Node) -> Option<Node> {
        if (old as usize) < self.old_to_new.len() {
            Some(self.old_to_new[old as usize])
        } else {
            None
        }
    }

    fn old_id_of(&self, new: Node) -> Option<Node> {
        if (new as usize) < self.new_to_old.len() {
            Some(self.new_to_old[new as usize])
        } else {
            None
        }
    }

    fn len(&self) -> Node {
        self.new_to_old.len() as Node
    }
}

/// A mapper that assigns new ids based on node ranks or orderings.
///
/// Provides only forward mappings (`old -> new`) without inverse lookup.
#[derive(Clone)]
pub struct RankingForwardMapper {
    new_ids: Vec<Node>,
}

impl RankingForwardMapper {
    /// Constructs a [`RankingForwardMapper`] directly from a vector `new_ids`.
    ///
    /// Functionally equivalent to [`NodeMapSetter::from_rank`], but consumes the vector
    /// instead of borrowing a slice.  
    ///
    /// Each entry `new_ids[i]` represents the new id assigned to the old node `i`.
    pub fn from_vec(new_ids: Vec<Node>) -> Self {
        Self { new_ids }
    }

    /// Constructs a [`RankingForwardMapper`] by ranking all nodes of the given graph
    /// using a custom comparator.
    ///
    /// The comparator receives two nodes `(a, b)` and must return an [`Ordering`]
    /// that determines their relative order.  
    ///
    /// The resulting mapping assigns lower new ids to nodes ranked earlier.
    ///
    /// # Example
    /// ```ignore
    /// let mapper = RankingForwardMapper::from_graph_sort_by(&graph, |a, b| degree[a].cmp(&degree[b]));
    /// ```
    pub fn from_graph_sort_by<G, F>(graph: &G, mut compare: F) -> Self
    where
        G: GraphNodeOrder,
        F: FnMut(Node, Node) -> Ordering,
    {
        let mut vertices: Vec<Node> = graph.vertices_range().collect();
        vertices.sort_by(|&a, &b| compare(a, b));
        Self::from_vec(vertices)
    }

    /// Constructs a [`RankingForwardMapper`] by ranking all nodes of the given graph
    /// increasingly according to the values of a `key` function.
    ///
    /// The `key` function maps each node to a value of type `K` (which must implement [`Ord`]).  
    /// Nodes with smaller keys receive smaller new ids.
    ///
    /// # Example
    /// ```ignore
    /// let mapper = RankingForwardMapper::from_graph_sort_by_key(&graph, |u| weight[u]);
    /// ```
    pub fn from_graph_sort_by_key<G, F, K>(graph: &G, mut key: F) -> Self
    where
        G: GraphNodeOrder,
        F: FnMut(Node) -> K,
        K: Ord,
    {
        Self::from_graph_sort_by(graph, |a, b| key(a).cmp(&key(b)))
    }

    /// Constructs a [`RankingForwardMapper`] by ranking all nodes of the given graph
    /// decreasingly according to the values of a `key` function.
    ///
    /// The `key` function maps each node to a value of type `K` (which must implement [`Ord`]).  
    /// Nodes with larger keys receive smaller new ids.
    ///
    /// # Example
    /// ```ignore
    /// let mapper = RankingForwardMapper::from_graph_sort_by_key_reverse(&graph, |u| centrality[u]);
    /// ```
    pub fn from_graph_sort_by_key_reverse<G, F, K>(graph: &G, mut key: F) -> Self
    where
        G: GraphNodeOrder,
        F: FnMut(Node) -> K,
        K: Ord,
    {
        Self::from_graph_sort_by(graph, |a, b| key(b).cmp(&key(a)))
    }
}

impl NodeMapGetter for RankingForwardMapper {
    fn new_id_of(&self, old: Node) -> Option<Node> {
        if (old as usize) < self.new_ids.len() {
            Some(self.new_ids[old as usize])
        } else {
            None
        }
    }

    /// If the mapping (old, new) exists, returns Some(old), otherwise None.
    ///
    /// # Warning
    /// This implementation is provided for compatibility reasons but takes linear time.
    /// If this method is required, consider using a `NodeMapper` instance instead.
    fn old_id_of(&self, new: Node) -> Option<Node> {
        self.new_ids
            .iter()
            .find_position(|&&x| x == new)
            .map(|(i, _)| i as Node)
    }

    fn len(&self) -> Node {
        self.new_ids.len() as Node
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::repr::{AdjArray, AdjArrayUndir};

    #[test]
    fn test_node_mapper() {
        let mut m1 = NodeMapper::with_capacity(10);
        assert!(m1.is_empty());
        for x in 0..5 {
            m1.map_node_to(2 * x, x);
            assert!(!m1.is_empty());
            assert_eq!(m1.len(), x + 1);
        }

        let mut m2 = NodeMapper::with_capacity(5);
        m2.map_node_to(0, 3);
        m2.map_node_to(1, 2);
        m2.map_node_to(2, 1);

        let m = NodeMapper::compose(&m1, &m2);

        let result: Vec<Option<Node>> = (0..10).map(|x| m.new_id_of(x)).collect();
        assert_eq!(
            result,
            [
                Some(3),
                None,
                Some(2),
                None,
                Some(1),
                None,
                None,
                None,
                None,
                None
            ]
        );
    }

    #[test]
    fn test_compose() {
        let mappings = vec![(0, 3), (10, 2), (20, 1)];

        let mut map = NodeMapper::with_capacity(5);
        for &(u, v) in &mappings {
            map.map_node_to(u, v);
        }

        let id = NodeMapper::identity(50);

        {
            let comp = NodeMapper::compose(&id, &map);
            for &(u, v) in &mappings {
                assert_eq!(comp.new_id_of(u), Some(v));
                assert_eq!(comp.old_id_of(v), Some(u));
            }
        }

        {
            let comp = NodeMapper::compose(&map, &id);
            for &(u, v) in &mappings {
                assert_eq!(comp.new_id_of(u), Some(v));
                assert_eq!(comp.old_id_of(v), Some(u));
            }
        }
    }

    #[test]
    fn test_node_mapper_inverse() {
        let mut m2 = NodeMapper::with_capacity(5);
        assert!(m2.is_empty());
        m2.map_node_to(0, 3);
        m2.map_node_to(10, 2);
        m2.map_node_to(20, 1);
        assert!(!m2.is_empty());

        let inv = m2.inverse();
        let result: Vec<Option<Node>> = (0..5).map(|x| inv.new_id_of(x)).collect();
        assert_eq!(result, vec![None, Some(20), Some(10), Some(0), None]);

        for i in 0..20 {
            assert_eq!(inv.new_id_of(i), m2.old_id_of(i));
            assert_eq!(m2.new_id_of(i), inv.old_id_of(i));
        }
    }

    #[test]
    fn test_identity() {
        let map = NodeMapper::identity(123);
        assert_eq!(map.old_id_of(12), Some(12));
        assert_eq!(map.new_id_of(5), Some(5));
    }

    #[test]
    fn test_format() {
        let mut map = NodeMapper::with_capacity(10);
        map.map_node_to(2, 3);
        map.map_node_to(5, 4);
        let text = format!("{map:?}");
        assert!(text.contains("2<->3"));
        assert!(text.contains("5<->4"));
    }

    #[test]
    #[should_panic]
    fn test_collision_on_old() {
        let mut map = NodeMapper::with_capacity(10);
        map.map_node_to(2, 3);
        map.map_node_to(2, 4);
    }

    #[test]
    #[should_panic]
    fn test_collision_on_new() {
        let mut map = NodeMapper::with_capacity(10);
        map.map_node_to(1, 3);
        map.map_node_to(2, 3);
    }

    #[test]
    fn relabelling() {
        // keep all
        {
            let graph = AdjArrayUndir::from_edges(3, [(0, 1), (1, 2)]);
            let mut map = NodeMapper::with_capacity(10);
            map.map_node_to(0, 2);
            map.map_node_to(1, 1);
            map.map_node_to(2, 0);
            let graph = map.relabelled_graph(&graph);

            assert_eq!(
                graph.ordered_edges(true).collect_vec(),
                vec![Edge(0, 1), Edge(1, 2)]
            );
            assert_eq!(graph.len(), 3);
        }

        // drop 1
        {
            let graph = AdjArray::from_edges(3, [(0, 1), (1, 2)]);
            let mut map = NodeMapper::with_capacity(10);
            map.map_node_to(0, 1);
            map.map_node_to(2, 0);
            let graph = map.relabelled_graph(&graph);

            assert_eq!(graph.number_of_edges(), 0);
            assert_eq!(graph.len(), 2);
        }

        // loop at node 0 (previous bug)
        {
            let graph = AdjArray::from_edges(2, [(1, 1)]);
            let mut map = NodeMapper::with_capacity(10);
            map.map_node_to(1, 0);
            let graph = map.relabelled_graph(&graph);

            assert_eq!(graph.ordered_edges(false).collect_vec(), vec![Edge(0, 0)]);
            assert_eq!(graph.len(), 1);
        }
    }
}
