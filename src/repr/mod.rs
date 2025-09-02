/*!
# Graph Representation

This module contains the concrete graph data structures one can use.
Each structure balances **memory usage** and **access performance** differently, making them suitable for different algorithmic settings.

### Representations

- [`AdjArray`]
  Stores adjacency lists in a `Vec<Vec<Node>>`.
  - Good for sparse graphs.
  - Fast iteration over neighbors.
  - Moderate memory overhead.
  - Also available as [`AdjArrayUndir`] for undirected graphs and [`AdjArrayIn`] for directed graphs that also store incoming neighbors in addition.

- [`SparseAdjArray`]
  Like `AdjArray`, but uses `Vec<SmallVec<Node>>` for adjacency lists.
  - Optimized for graphs where most nodes have very few neighbors.
  - Reduces heap allocations by storing small lists inline.
  - Also available as [`SparseAdjArrayUndir`] for undirected graphs and [`SparseAdjArrayIn`] for directed graphs that also store incoming neighbors in addition.

- [`AdjMatrix`]
  Stores edges in an `n × n` boolean matrix.
  - Best for dense graphs or when `has_edge(u, v)` queries dominate.
  - High memory cost: `O(n^2)`.
  - Also available as [`AdjMatrixUndir`] for undirected graphs and [`AdjMatrixIn`] for directed graphs that also store incoming neighbors in addition.

- [`CsrGraph`] (Compressed Sparse Row)
  Stores adjacency lists in a single flattened array with offset indices.
  - Memory-efficient for sparse graphs.
  - Good cache locality and iteration speed.
  - Construction cost is higher, but structure is compact and immutable.
  - Also available as [`CsrGraphUndir`] for undirected graphs and [`CsrGraphIn`] for directed graphs that also store incoming neighbors in addition.
  - For use-cases where one wants cross-pointers to the 'reverse'-edge in undirected graphs, [`CrossCsrGraph`] can be used.

- [`AdjArrayMatrix`] *(directed only)*
  Hybrid structure combining adjacency arrays and an adjacency matrix.
  - Adjacency array: fast neighbor iteration (stores outgoing neighbors).
  - Matrix: constant-time edge existence queries (stores incoming neighbors).
  - Useful when both traversal and fast membership testing are needed.

## Choosing a Representation

- Use **`AdjMatrix`** if the graph is dense and you need constant-time `has_edge(u, v)` checks.
- Use **`CsrGraph`** for large, static, sparse graphs where memory efficiency and fast iteration matter.
- Use **`AdjArray`** or **`SparseAdjArray`** if you need a flexible, general-purpose adjacency list.
- Use **`AdjArrayMatrix`** if you’re working with **directed** graphs and need both fast adjacency iteration and `O(1)` membership testing.
*/

use crate::{edge::*, node::*, ops::*};
use stream_bitset::prelude::BitmaskStream;

pub mod csr;
pub mod directed;
pub mod neighborhood;
pub mod undirected;

pub mod digest;

use neighborhood::*;

pub use csr::{CrossCsrGraph, CsrGraph, CsrGraphIn, CsrGraphUndir};
pub use directed::{
    AdjArray, AdjArrayIn, AdjArrayMatrix, AdjMatrix, AdjMatrixIn, DirectedGraph, DirectedGraphIn,
    SparseAdjArray, SparseAdjArrayIn,
};
pub use undirected::{AdjArrayUndir, AdjMatrixUndir, SparseAdjArrayUndir, UndirectedGraph};
