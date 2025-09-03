/*!
# Graph Generators

This module provides traits for constructing random graph generators.

Generators encapsulate common random graph models (e.g., Erdős–Rényi G(n, p), G(n, m), Random Hyperbolic Graphs, Minimum Spanning Tree, etc.) and provide both:

- **Model configuration**: e.g., `.set_nodes(n) / .nodes(n)`, `.set_edges(m) / .edges(m)`, `.set_avg_deg(d) / .avg_deg(d)`.
- **Edge production**: either as a collected list (`generate`) or a lazy stream (`stream`).

Instead of manually configuring a model step by step, you can also create a graph directly using the [`RandomGraph`] trait:
```
use ugraphs::{prelude::*, gens::*};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

let rng = &mut Pcg64Mcg::seed_from_u64(42);
let g = AdjArray::gnp(rng, 50, 0.1); // Erdős–Rényi G(n,p)
assert_eq!(g.number_of_nodes(), 50);
```

Supported models include:
- G(n,m): Uniform random graphs with a fixed number of nodes and edges.
- G(n,p): Erdős–Rényi model with independent edge probability
- G(n): Uniform random graphs with a fixed number of nodes
- Rhg: Random hyperbolic graphs in the threshold case (T = 0)
*/

use fxhash::FxHashMap;
use rand::Rng;

use crate::prelude::*;

pub mod gnm;
pub mod gnp;
pub mod mst;
pub mod rhg;
pub mod substructures;

use gnm::*;
use gnp::*;
use mst::*;
use rhg::*;
pub use substructures::*;

/// Trait for generators that allow specifying the number of nodes.
///
/// Common across all random graph generators.
pub trait NumNodesGen: Sized {
    /// Sets the number of nodes in the generator (mutable setter).
    fn set_nodes(&mut self, n: NumNodes);

    /// Sets the number of nodes and returns the generator (builder style).
    fn nodes(mut self, n: NumNodes) -> Self {
        self.set_nodes(n);
        self
    }

    /// Constructs a generator with `n` nodes using `Default` + builder pattern.
    fn with_nodes(n: NumNodes) -> Self
    where
        Self: Default,
    {
        Self::default().nodes(n)
    }
}

/// Trait for generators that allow specifying the number of edges.
///
/// Used in models like G(n, m) where the total edge count is fixed.
pub trait NumEdgesGen: Sized {
    /// Sets the number of edges in the generator (mutable setter).
    fn set_edges(&mut self, m: NumEdges);

    /// Sets the number of edges and returns the generator (builder style).
    fn edges(mut self, m: NumEdges) -> Self {
        self.set_edges(m);
        self
    }

    /// Constructs a generator with `m` edges using `Default` + builder pattern.
    fn with_edges(m: NumEdges) -> Self
    where
        Self: Default,
    {
        Self::default().edges(m)
    }
}

/// Trait for generators that allow specifying the expected average degree.
///
/// Useful in models like Random Hyperbolic Graphs.
/// Provides builder-style configuration.
pub trait AverageDegreeGen: Sized {
    /// Sets the average degree in the generator (mutable setter).
    fn set_avg_deg(&mut self, deg: f64);

    /// Sets the average degree and returns the generator (builder style).
    fn avg_deg(mut self, deg: f64) -> Self {
        self.set_avg_deg(deg);
        self
    }

    /// Constructs a generator with average degree `deg` using `Default` + builder pattern.
    fn with_avg_deg(deg: f64) -> Self
    where
        Self: Default,
    {
        Self::default().avg_deg(deg)
    }
}

/// General trait for a configurable random edge generator.
///
/// Types implementing this trait can produce:
/// - a **full list of edges** via [`GraphGenerator::generate`]
/// - a **lazy iterator** of edges via [`GraphGenerator::stream`]
///
/// The lazy stream form is preferred for large graphs.
///
/// # Example
/// ```
/// use ugraphs::{prelude::*, gens::{*, gnp::*}};
/// use rand::SeedableRng;
/// use rand_pcg::Pcg64Mcg;
///
/// let rng = &mut Pcg64Mcg::seed_from_u64(5);
/// let edges: Vec<Edge> = Gnp::new().nodes(5).prob(0.5).generate(rng);
/// assert!(edges.into_iter().all(|Edge(u, v)| u < 5 && v < 5));
/// ```
pub trait GraphGenerator {
    /// Generates a `Vec<Edge>` by fully materializing the edge stream.
    ///
    /// Default implementation collects from [`GraphGenerator::stream`].
    fn generate<R>(&self, rng: &mut R) -> Vec<Edge>
    where
        R: Rng,
    {
        self.stream(rng).collect()
    }

    /// Type of the streaming iterator over edges.
    /// Bound to a specific [`Rng`] type `R`.
    type EdgeStream<'a, R>: Iterator<Item = Edge> + 'a
    where
        R: Rng + 'a,
        Self: 'a;

    /// Produces a lazy stream (iterator) of edges.
    ///
    /// This is usually more efficient for large graphs or when edges
    /// are consumed incrementally (if the model allows it).
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::{*, gnp::*}};
    /// use rand::SeedableRng;
    /// use rand_pcg::Pcg64Mcg;
    ///
    /// let rng = &mut Pcg64Mcg::seed_from_u64(11);
    /// let gnp = Gnp::new().nodes(4).prob(0.3);
    /// assert!(gnp.stream(rng).all(|Edge(u, v)| u < 4 && v < 4));
    /// ```
    fn stream<'a, R>(&'a self, rng: &'a mut R) -> Self::EdgeStream<'a, R>
    where
        R: Rng;
}

/// Trait for constructing full random graph instances.
///
/// Implemented for any type that supports [`GraphFromScratch`] + [`GraphType`].
/// Provides standard models like:
/// - `G(n), G(n,p), G(n,m)`
/// - Random Hyperbolic Graphs (`Rhg`)
/// - Minimum Spanning Tree (`Mst`)
///
/// # Example
/// ```
/// use ugraphs::{prelude::*, gens::*};
/// use rand::SeedableRng;
/// use rand_pcg::Pcg64Mcg;
///
/// let rng = &mut Pcg64Mcg::seed_from_u64(13);
/// let g = AdjArray::gnm(rng, 10, 20);
/// assert_eq!(g.number_of_nodes(), 10);
/// assert_eq!(g.number_of_edges(), 20);
/// ```
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
        R: Rng,
        Self: GraphType<Dir = Directed>;

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
        Self: GraphType<Dir = Undirected>,
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
                .undirected(Self::is_undirected())
                .stream(rng),
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
                .undirected(Self::is_undirected())
                .stream(rng),
        )
    }

    fn gnp_no_loops<R>(rng: &mut R, n: NumNodes, p: f64) -> Self
    where
        R: Rng,
        Self: GraphType<Dir = Directed>,
    {
        Self::from_edges(
            n,
            Gnp::new()
                .nodes(n)
                .prob(p)
                .directed(true)
                .stream(rng)
                .filter(|e| !e.is_loop()),
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
        Self: GraphType<Dir = Undirected>,
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

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use crate::algo::Connectivity;

    use super::*;

    #[test]
    fn random_graph() {
        let rng = &mut Pcg64Mcg::seed_from_u64(3);
        let repeats = 1000;

        // G(n)
        {
            for n in [10 as NumNodes, 20, 50, 100] {
                // Directed
                let mean_edges = (0..repeats)
                    .map(|_| {
                        let g = AdjArray::gn(rng, n);
                        assert_eq!(g.number_of_nodes(), n);
                        g.number_of_edges() as f64
                    })
                    .sum::<f64>()
                    / repeats as f64;
                let expected = (n as f64) * (n as f64) / 2.0;

                assert!((0.75 * expected..1.25 * expected).contains(&mean_edges));

                // Undirected
                let mean_edges = (0..repeats)
                    .map(|_| {
                        let g = AdjArrayUndir::gn(rng, n);
                        assert_eq!(g.number_of_nodes(), n);
                        g.number_of_edges() as f64
                    })
                    .sum::<f64>()
                    / repeats as f64;
                let expected = (n as f64) * ((n - 1) as f64) / 2.0 / 2.0;

                assert!((0.75 * expected..1.25 * expected).contains(&mean_edges));
            }
        }

        // G(n,p)
        {
            for n in [10 as NumNodes, 20, 50, 100] {
                for p in [0.001f64, 0.01, 0.1] {
                    // Directed
                    let mean_edges = (0..repeats)
                        .map(|_| {
                            let g = AdjArray::gnp(rng, n, p);
                            assert_eq!(g.number_of_nodes(), n);
                            g.number_of_edges() as f64
                        })
                        .sum::<f64>()
                        / repeats as f64;
                    let expected = (n as f64) * (n as f64) * p;

                    assert!((0.75 * expected..1.25 * expected).contains(&mean_edges));

                    // Undirected
                    let mean_edges = (0..repeats)
                        .map(|_| {
                            let g = AdjArrayUndir::gnp(rng, n, p);
                            assert_eq!(g.number_of_nodes(), n);
                            g.number_of_edges() as f64
                        })
                        .sum::<f64>()
                        / repeats as f64;
                    let expected = (n as f64) * ((n - 1) as f64) * p / 2.0;

                    assert!((0.75 * expected..1.25 * expected).contains(&mean_edges));

                    // No Loops
                    assert!(!AdjMatrix::gnp_no_loops(rng, n, p).has_self_loops());
                }
            }
        }

        // G(n,m)
        {
            for (n, m) in [
                (10 as NumNodes, 20 as NumEdges),
                (20, 60),
                (50, 1000),
                (100, 2000),
            ] {
                for _ in 0..100 {
                    // Directed
                    let g = AdjArray::gnm(rng, n, m);
                    assert_eq!(g.number_of_nodes(), n);
                    assert_eq!(g.number_of_edges(), m);

                    // Undirected
                    let g = AdjArrayUndir::gnm(rng, n, m);
                    assert_eq!(g.number_of_nodes(), n);
                    assert_eq!(g.number_of_edges(), m);
                }
            }
        }

        // Rhg
        {
            for n in [20 as NumNodes, 50, 100] {
                for d in [2.0, 5.0, 10.0] {
                    let mean_degree = (0..repeats)
                        .map(|_| {
                            let g = AdjArrayUndir::rhg(rng, n, d);
                            assert_eq!(g.number_of_nodes(), n);
                            g.number_of_edges() as f64 * 2.0 / g.number_of_nodes() as f64
                        })
                        .sum::<f64>()
                        / repeats as f64;

                    assert!((0.75 * d..1.25 * d).contains(&mean_degree));
                }
            }
        }

        // Mst
        {
            for n in [10 as NumNodes, 20, 50, 100] {
                for _ in 0..100 {
                    let g = AdjArrayUndir::mst(rng, n);
                    assert_eq!(g.number_of_nodes(), n);
                    assert_eq!(g.number_of_edges(), n - 1);
                    assert_eq!(g.connected_components().count(), 1);
                }
            }
        }
    }
}
