/*!
# Graph Generators

This module provides a suite of traits and builder patterns for constructing random graph generators.

Each graph generator allows parameterized control over structural properties of the graph (e.g., number
of nodes or edges, average degree), and can produce either a complete collection of edges or a stream
of them through iterators.

Generators are designed to support a builder-style pattern for fluent graph configuration. The typical
usage workflow is:

1. Create a generator instance (e.g., `Gnp::new()`).
2. Set parameters using trait methods (e.g., `.nodes(n).prob(p)`).
3. Generate edges via `generate()` or `stream()`.

In addition, the `RandomGraph` trait abstracts the generation of whole graph instances (e.g., for
`G(n,p)` or `G(n,m)` models) into reusable constructors. These implementations internally rely on the
edge generators to create graph structure according to the type’s requirements (directed, undirected, etc.).

Supported models include:
- G(n,m): Uniform random graphs with a fixed number of nodes and edges
- G(n,p): Erdős–Rényi model with independent edge probability
- G(n): Uniform random graphs with a fixed number of nodes
- Rhg: Random hyperbolic graphs in the threshold case (T = 0)

All graph types implementing `GraphFromScratch` and `GraphType` can leverage the `RandomGraph` trait
for convenient random graph construction.
*/

use fxhash::FxHashMap;
use rand::Rng;

use crate::prelude::*;

mod gnm;
mod gnp;
mod mst;
mod rhg;
mod substructures;

pub use gnm::*;
pub use gnp::*;
pub use mst::*;
pub use rhg::*;
pub use substructures::*;

/// Trait for generators that allow setting the number of nodes.
///
/// This is the most common builder trait across all generators.
/// Allows a fluent interface when configuring generators.
pub trait NumNodesGen {
    /// Sets the number of nodes in the graph generator.
    fn nodes(self, n: NumNodes) -> Self;
}

/// Trait for generators that allow setting the number of edges.
///
/// Often used in models like G(n, m) where the edge count is fixed.
pub trait NumEdgesGen {
    /// Sets the number of edges in the graph generator.
    fn edges(self, m: NumEdges) -> Self;
}

/// Trait for generators that allow setting the average degree.
///
/// Common in scale-free and random geometric graph models.
pub trait AverageDegreeGen {
    /// Set the average degree of this generator.
    fn avg_deg(self, deg: f64) -> Self;
}

/// General trait for a configurable random edge generator.
///
/// Types implementing this trait can produce a complete edge list
/// or a lazily-evaluated stream (iterator) of edges.
pub trait GraphGenerator {
    /// Generates a list of random edges.
    ///
    /// This collects the full result from `stream()` into a `Vec<Edge>` as default.
    fn generate<R>(&self, rng: &mut R) -> Vec<Edge>
    where
        R: Rng,
    {
        self.stream(rng).collect()
    }

    /// Creates a lazy iterator (stream) over generated edges.
    ///
    /// Preferred for large graphs or pipelined filtering. Depending on the underlying graph
    /// model, this might also be just an iterator over the already generated list of edges if
    /// a direct iterator is not feasible in the model.
    fn stream<R>(&self, rng: &mut R) -> impl Iterator<Item = Edge>
    where
        R: Rng;
}

/// Trait for building full graph instances from common random models.
///
/// Requires that the implementing type supports construction from a set of edges.
/// Provided implementations use the corresponding edge generators under the hood.
pub trait RandomGraph: Sized {
    /// Creates a random `G(n)` graph.
    fn gn<R>(rng: &mut R, n: NumNodes) -> Self
    where
        R: Rng;

    /// Creates a random `G(n,p)` graph using edge probability `p`.
    fn gnp<R>(rng: &mut R, n: NumNodes, p: f64) -> Self
    where
        R: Rng;

    /// Creates a `G(n,p)` graph with no self-loops.
    fn gnp_no_loops<R>(rng: &mut R, n: NumNodes, p: f64) -> Self
    where
        R: Rng;

    /// Creates a random `G(n,m)` graph with exactly `m` edges.
    ///
    /// Internally uses a default hash map-based edge selector.
    fn gnm<R>(rng: &mut R, n: NumNodes, m: NumEdges) -> Self
    where
        R: Rng,
    {
        Self::gnm_with_map::<R, FxHashMap<u64, OptionalU64>>(rng, n, m)
    }

    /// Creates a random `G(n,m)` graph using a custom hash map-like type for sampling.
    ///
    /// This allows external control over memory layout or hashing strategy.
    fn gnm_with_map<R, H>(rng: &mut R, n: NumNodes, m: NumEdges) -> Self
    where
        R: Rng,
        H: GnmMap;

    /// Creates a random `Rhg(alpha = 1.0, T = 0)` graph with `n` nodes and specified average degree.
    fn rhg<R>(rng: &mut R, n: NumNodes, avg_deg: f64) -> Self
    where
        R: Rng;

    /// Creates a random Mst with `n` nodes and root node `0`
    fn mst<R>(rng: &mut R, n: NumNodes) -> Self
    where
        R: Rng;
}

impl<G> RandomGraph for G
where
    G: GraphFromScratch + GraphType,
{
    fn gn<R>(rng: &mut R, n: NumNodes) -> Self
    where
        R: Rng,
    {
        Self::from_edges(
            n,
            Gn::new()
                .nodes(n)
                .stream(rng)
                .filter(|e| Self::is_directed() || e.is_normalized()),
        )
    }

    fn gnp<R>(rng: &mut R, n: NumNodes, p: f64) -> Self
    where
        R: Rng,
    {
        Self::from_edges(
            n,
            Gnp::new()
                .nodes(n)
                .prob(p)
                .stream(rng)
                .filter(|e| Self::is_directed() || e.is_normalized()),
        )
    }

    fn gnp_no_loops<R>(rng: &mut R, n: NumNodes, p: f64) -> Self
    where
        R: Rng,
    {
        Self::from_edges(
            n,
            Gnp::new()
                .nodes(n)
                .prob(p)
                .stream(rng)
                .filter(|e| !e.is_loop() && (Self::is_directed() || e.is_normalized())),
        )
    }

    fn gnm_with_map<R, H>(rng: &mut R, n: NumNodes, m: NumEdges) -> Self
    where
        R: Rng,
        H: GnmMap,
    {
        Self::from_edges(
            n,
            Gnm::<H>::new()
                .nodes(n)
                .edges(m)
                .undirected(Self::is_undirected())
                .stream(rng),
        )
    }

    fn rhg<R>(rng: &mut R, n: NumNodes, avg_deg: f64) -> Self
    where
        R: Rng,
    {
        Self::from_edges(n, Rhg::new().nodes(n).avg_deg(avg_deg).stream(rng))
    }

    fn mst<R>(rng: &mut R, n: NumNodes) -> Self
    where
        R: Rng,
    {
        Self::from_edges(n, Mst::new().nodes(n).stream(rng))
    }
}
