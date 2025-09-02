/*!
# Erdős–Rényi Graph Generators: `G(n,p)` and `G(n)`

This module provides random graph generators for the classical Erdős–Rényi models:

- **`G(n,p)`**: each possible edge is included independently with probability `p`.
- **`G(n)`**: shorthand for `G(n, 0.5)` — uniform distribution over all possible graphs with `n` nodes.

Generation is implemented lazily using the [`GeometricJumper`] inversion method, which skips absent edges efficiently instead of flipping a coin for every possible edge.

Both directed and undirected graphs are supported, depending on which graph type consumes the generated edges.

# Examples
```
use ugraphs::{prelude::*, gens::*};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

// Generate a G(n,p) graph with n=10, p=0.2
let rng = &mut Pcg64Mcg::seed_from_u64(42);
let g = AdjArray::gnp(rng, 10, 0.2);
assert_eq!(g.number_of_nodes(), 10);

// Generate a uniform G(n) graph with n=5
let rng = &mut Pcg64Mcg::seed_from_u64(7);
let g = AdjArray::gn(rng, 5);
assert_eq!(g.number_of_nodes(), 5);
```
*/

use std::{fmt::Debug, iter::Empty, ops::Range};

use super::*;
use crate::utils::{Probability, geometric::*, multi_traits::TripleIter};

/// Internal representation of how a `G(n,p)` generator is parameterized.
///
/// A `Gnp` generator can be configured either with:
/// - an explicit edge probability (`p`), or
/// - an expected average degree (`d`), internally converted into `p = d / n`.
///
/// This enum is used internally to delay conversion until generation time.
#[derive(Debug, Copy, Clone, Default)]
enum GnpType {
    /// No parameters set yet.  
    /// Using this will panic when attempting to generate edges.
    #[default]
    NotSet,
    /// Explicit edge probability value.
    Prob(f64),
    /// Average degree per node (converted internally to `p = d/n`).
    AvgDeg(f64),
}

/// Generator for Erdős–Rényi `G(n,p)` random graphs.
///
/// In a `G(n,p)` graph, each of the `n * n` possible directed/self-loop edges
/// (or `n * (n-1) / 2` possible undirected edges, depending on graph type)
/// is included independently with probability `p`.
///
/// The generator can be parameterized either by:
/// - setting a direct probability `p`, or
/// - specifying an average degree `d`, which is converted to `p = d / n`.
///
/// Generation is efficient thanks to the [`GeometricJumper`], which skips
/// non-edges instead of flipping `n^2` coins.
///
/// > **Note:** No filtering is done automatically.
/// > Callers are responsible for filtering if needed.
///
/// # Examples
/// ```
/// use ugraphs::{prelude::*, gens::{*, gnp::*}};
/// use rand::SeedableRng;
/// use rand_pcg::Pcg64Mcg;
///
/// let rng = &mut Pcg64Mcg::seed_from_u64(1);
/// let edges: Vec<Edge> = Gnp::new().nodes(4).prob(0.5).generate(rng);
/// assert!(edges.into_iter().all(|Edge(u, v)| u < 4 && v < 4));
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct Gnp {
    n: u64,
    p: GnpType,
}

impl Gnp {
    /// Creates a new, empty `G(n,p)` generator.
    ///
    /// By default, this has no parameters set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the probability `p` for including each edge independently.
    ///
    /// # Panics
    /// Panics if `p` is not in the valid range `[0.0, 1.0]`.
    pub fn set_prob(&mut self, prob: f64) {
        assert!(prob.is_valid_probility());
        self.p = GnpType::Prob(prob);
    }

    /// Sets the probability `p` for including each edge independently (builder-style).
    ///
    /// # Panics
    /// Panics if `p` is not in the valid range `[0.0, 1.0]`.
    pub fn prob(mut self, prob: f64) -> Self {
        self.set_prob(prob);
        self
    }
}

impl NumNodesGen for Gnp {
    fn set_nodes(&mut self, n: NumNodes) {
        self.n = n as u64;
    }
}

impl AverageDegreeGen for Gnp {
    /// Sets the average degree `d` for the generator.
    ///
    /// This is internally converted to a probability `p = d/n` during edge generation.
    fn set_avg_deg(&mut self, deg: f64) {
        self.p = GnpType::AvgDeg(deg);
    }
}

impl GraphGenerator for Gnp {
    /// Iterator type returned by the generator’s [`GraphGenerator::stream`] method.
    ///
    /// Internally a [`TripleIter`] which chooses between:
    /// - empty iterator (`p = 0`),
    /// - full range iterator (`p = 1`),
    /// - geometric skipping iterator (`0 < p < 1`).
    type EdgeStream<'a, R>
        = TripleIter<
        Edge,
        Empty<Edge>,
        MapToEdgeIter<Range<u64>>,
        MapToEdgeIter<GeometricJumperIter<'a, R>>,
    >
    where
        R: Rng + 'a,
        Self: 'a;

    /// Produces a lazily-evaluated iterator over randomly generated edges.
    ///
    /// Behavior depends on the configured probability:
    /// - `p = 0.0` -> no edges
    /// - `p = 1.0` -> all edges
    /// - `0.0 < p < 1.0` -> edges sampled via [`GeometricJumper`]
    ///
    /// # Panics
    /// - If node count `n` is zero
    /// - If no probability or average degree is set
    /// - If average degree is invalid for the given `n`
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, gens::{*, gnp::*}};
    /// use rand::SeedableRng;
    /// use rand_pcg::Pcg64Mcg;
    ///
    /// let rng = &mut Pcg64Mcg::seed_from_u64(10);
    /// let gnp = Gnp::new().nodes(3).prob(0.5);
    /// assert!(gnp.stream(rng).all(|Edge(u, v)| u < 3 && v < 3));
    /// ```
    fn stream<'a, R>(&'a self, rng: &'a mut R) -> Self::EdgeStream<'a, R>
    where
        R: Rng,
    {
        assert!(self.n > 0, "At least one node must be generated!");
        let p = match self.p {
            GnpType::NotSet => panic!("Probility of Gnp was not set!"),
            GnpType::Prob(p) => p,
            GnpType::AvgDeg(d) => {
                let p = d / self.n as f64;
                assert!(
                    p.is_valid_probility(),
                    "The average degree is invalid for the given n!"
                );
                p
            }
        };

        // The maximum possible value an edge can be mapped to
        let max_value = self.n * self.n;

        // Different between easy and hard cases with little overhead to ensure maximum efficiency
        match p {
            0.0 => TripleIter::IterA(std::iter::empty()),
            1.0 => TripleIter::IterB(MapToEdgeIter {
                iter: 0..max_value,
                n: self.n,
            }),
            // We verified that `p` is a valid probability at this point
            _ => TripleIter::IterC(MapToEdgeIter {
                iter: GeometricJumper::new(p).stop_at(max_value).iter(rng),
                n: self.n,
            }),
        }
    }
}

/// Helper iterator adapter that maps `u64` indices to [`Edge`] values.
///
/// Used internally to translate sampled positions into `(source, target)` edges.
pub struct MapToEdgeIter<I>
where
    I: Iterator<Item = u64>,
{
    iter: I,
    n: u64,
}

impl<I> Iterator for MapToEdgeIter<I>
where
    I: Iterator<Item = u64>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| Edge::from_u64(x, self.n))
    }
}

/// Generator for uniform `G(n)` graphs, equivalent to `G(n, 0.5)`.
///
/// Each possible edge is included independently with `50%` probability.
///
/// This is a convenience wrapper around `Gnp::new().prob(0.5)`.
#[derive(Debug, Copy, Clone, Default)]
pub struct Gn {
    n: u64,
}

impl Gn {
    /// Creates a new `Gn` generator with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
}

impl NumNodesGen for Gn {
    fn set_nodes(&mut self, n: NumEdges) {
        self.n = n as u64;
    }
}

impl GraphGenerator for Gn {
    type EdgeStream<'a, R>
        = MapToEdgeIter<GeometricJumperIter<'a, R>>
    where
        R: Rng + 'a,
        Self: 'a;

    fn stream<'a, R>(&'a self, rng: &'a mut R) -> Self::EdgeStream<'a, R>
    where
        R: Rng,
    {
        assert!(self.n > 0, "At least one node must be generated!");
        MapToEdgeIter {
            iter: GeometricJumper::new(0.5).stop_at(self.n * self.n).iter(rng),
            n: self.n,
        }
    }
}
