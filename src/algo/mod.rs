/*!
# Graph Algorithms

This module provides a suite of **graph algorithms** built on top of the graph representations in this crate.
All algorithms are re-exported at the top level of this module, so you can simply do:
```rust
use ugraphs::algo::*;
```
and gain access to traversal, connectivity, matching, flow, and many other classical graph routines.
If possible, algorithms are provided as **iterators**, making it easy to consume results lazily.
*/

mod bipartite;
mod bridges;
mod connectivity;
mod cuthill_mckee;
mod distance_pairs;
mod matching;
mod network_flow;
mod partition;
mod path_iterator;
mod subgraph;
mod traversal;
mod vertex_cuts;

use crate::{prelude::*, utils::*};

pub use bipartite::*;
pub use bridges::*;
pub use connectivity::*;
pub use cuthill_mckee::*;
pub use distance_pairs::*;
pub use matching::*;
pub use network_flow::*;
pub use partition::*;
pub use path_iterator::*;
pub use subgraph::*;
pub use traversal::*;
pub use vertex_cuts::*;
