/*!
# Graph Algorithms

This module provides a suite of **graph algorithms** built on top of the graph representations in this crate.
All algorithms are re-exported at the top level of this module, so you can simply do:
```
use ugraphs::algo::*;
```
and gain access to traversal, connectivity, matching, flow, and many other classical graph routines.
If possible, algorithms are provided as **iterators**, making it easy to consume results lazily.
*/

pub mod bipartite;
pub mod bridges;
pub mod connectivity;
pub mod cuthill_mckee;
pub mod distance_pairs;
pub mod matching;
pub mod network_flow;
pub mod path_iterator;
pub mod subgraph;
pub mod traversal;
pub mod vertex_cuts;

use crate::{prelude::*, utils::*};

pub use bipartite::{BipartiteEdit, BipartiteTest, Bipartition};
pub use bridges::Bridges;
pub use connectivity::Connectivity;
pub use cuthill_mckee::CuthillMcKee;
pub use distance_pairs::DistancePairs;
pub use matching::Matching;
pub use network_flow::{MinVertexCut, STFlow};
pub use path_iterator::PathIterator;
pub use subgraph::{GraphConcat, Subgraph};
pub use traversal::{RankFromOrder, Traversal};
pub use vertex_cuts::ArticluationPoint;

/// Most graph algorithms take an immutable reference to a graph while executing.
/// This helper trait allows getting that reference.
/// Useful when using another algorithm as a subroutine.
pub trait WithGraphRef<G> {
    /// Return the reference to the graph.
    fn graph_ref(&self) -> &G;
}
