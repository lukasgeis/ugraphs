/*!
`ugraphs` is a graph data structure & algorithms library designed for graphs that are
- **u**nlabelled & **u**nsigned : Nodes are numbered `0` to `n - 1`
- **u**nweighted : Neither nodes nor edges have a weight attached to them
- **u**ndirected : This one is **optional** (but fits the naming scheme)

# Representation

We represent **nodes** as `u32` in the range `0..n` if `n` is the number of nodes in the graph.
As most common graphs do not exceed `2^32` nodes, this should normally suffice and save space as compared to `u64/usize`.
For **edges**, we use a simple tuple-struct `Edge(Node, Node)`.

### Directed vs Undirected

We support both **directed** and **undirected** graphs:

- In an **undirected** graph, `Edge(u, v)` is treated as equivalent to `Edge(v, u)` (although we normalize edges often).
- In a **directed** graph, the edge has orientation, so `Edge(u, v)` and `Edge(v, u)` are distinct.

### Available Representations

See the [`repr`] module for the full list of graph storage backends:

- [`AdjArray`](crate::repr::AdjArray)
- [`AdjMatrix`](crate::repr::AdjMatrix)
- [`CsrGraph`](crate::repr::CsrGraph)
- [`SparseAdjArray`](crate::repr::SparseAdjArray)
- [`AdjArrayMatrix`](crate::repr::AdjArrayMatrix) (directed only)

Each representation makes different trade-offs in terms of memory usage and lookup/iteration performance.
There are also `DirectedIn`-Variants for directed graphs that store not only out-neighbors

# Design

All algorithms/generators are provided as configurable structs that one can alter to their needs using either the *Builder* / *Setter* pattern before calling the configured algorithm on a provided graph.
Alternatively, most important and commonly used functionalities should already be implemented via traits on graphs itself, making them usable without configuring the algorithm beforehand.

# Usage




# When to use
You should only use this library if the following apply:
- Your graphs are unlabelled & unweighted
- You want to work in *Rust*
- You require only basic functionality for graphs.
- Performance is important

In all other cases, it might make sense for you to check out [petgraph](https://crates.io/crates/petgraph) who provide a more extensive library for general graphs in *Rust* or [NetworKit](https://networkit.github.io/) who provide high-performance graph algorithms in *C++* and *Python*.


# Credits
Originally developed by [Manuel Penschuck](https://github.com/manpen) & [Johannes Meintrup](https://github.com/jmeintrup) as part of the [BreakingTheCycle](https://github.com/goethe-tcs/breaking-the-cycle) solver for the [Pace2022-Challenge](https://pacechallenge.org/2022/), it was then further developed for the [Pace2025-Challenge](https://pacechallenge.org/2025/) as part of the [PaceYourself](https://github.com/manpen/pace25/tree/master) solver.
*/

pub mod algo;
pub mod edge;
pub mod gens;
pub mod io;
pub mod node;
pub mod ops;
pub mod repr;
pub(crate) mod testing;
pub mod utils;

/// `ugraphs::prelude` includes definitions for nodes and edges, all basic graph operation traits as well as all implemented representations.
pub mod prelude {
    pub use super::{edge::*, node::*, ops::*, repr::*};
}
