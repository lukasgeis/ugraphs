use std::fmt::Debug;

use super::*;
use crate::utils::{GeometricJumper, Probability, TripleIter};

/// Internal representation of how a `G(n,p)` generator is parameterized.
///
/// This enum allows the generator to be configured using either:
/// - a direct edge probability (`p`), or
/// - an average node degree (`d`), which is internally converted to a probability.
///
/// Used internally by [`Gnp`] to delay parameter conversion until generation time.
#[derive(Debug, Copy, Clone, Default)]
enum GnpType {
    /// No parameters set yet; using this will panic at runtime.
    #[default]
    NotSet,
    /// Explicit edge probability value.
    Prob(f64),
    /// Average degree per node (converted internally to `p = d/n`).
    AvgDeg(f64),
}

/// Generator for `G(n,p)` random graphs.
///
/// In a `G(n,p)` graph, each of the `n*(n-1)/2` or `n*n` possible directed/undirected/self-loop
/// edge combinations is included independently with probability `p`.
///
/// This generator can be parameterized either by setting a direct probability `p`
/// or by specifying an average degree `d`, which will be converted to `p = d/n`.
///
/// > **Note:** No filtering is done automatically.
/// > Callers are responsible for filtering if needed.
///
/// This generator produces edges lazily using the inversion method of [`GeometricJumper`] to
/// sample edges efficiently.
#[derive(Debug, Copy, Clone, Default)]
pub struct Gnp {
    n: u64,
    p: GnpType,
}

impl Gnp {
    /// Creates a new, empty `G(n,p)` generator.
    ///
    /// By default, this has no parameters set. Use builder methods to configure it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the probability `p` used for generating each edge independently.
    ///
    /// # Panics
    /// Panics if `p` is not within the valid probability range `[0.0, 1.0]`.
    pub fn prob(mut self, prob: f64) -> Self {
        assert!(prob.is_valid_probility());
        self.p = GnpType::Prob(prob);
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
    /// Returns a lazily-evaluated iterator over randomly generated `G(n,p)` edges.
    ///
    /// The generator handles the edge generation strategy depending on the value of `p`:
    /// - `p = 0.0`: produces no edges
    /// - `p = 1.0`: generates all possible edges
    /// - `0.0 < p < 1.0`: samples edges using a [`GeometricJumper`] for efficiency
    ///
    /// # Panics
    /// - If no node count is set (`n == 0`)
    /// - If no valid probability is configured
    /// - If average degree is invalid for the configured `n`
    fn stream<R>(&self, rng: &mut R) -> impl Iterator<Item = Edge>
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

        let to_edge = |x: u64| Edge::from_u64(x, self.n);

        // Different between easy and hard cases with little overhead to ensure maximum efficiency
        match p {
            0.0 => TripleIter::IterA(std::iter::empty()),
            1.0 => TripleIter::IterB((0..max_value).map(to_edge)),
            // We verified that `p` is a valid probability at this point
            _ => TripleIter::IterC(
                GeometricJumper::new(p)
                    .stop_at(max_value)
                    .iter(rng)
                    .map(to_edge),
            ),
        }
    }
}

/// Generator for uniform `G(n)` graphs, equivalent to `G(n,0.5)`.
///
/// In `G(n)`, each possible edge is included independently with 50% probability,
/// representing a uniform distribution over all graphs with `n` nodes.
///
/// This is a convenience wrapper for symmetric `G(n,p)` generation with `p = 0.5`.
#[derive(Debug, Copy, Clone, Default)]
pub struct Gn {
    n: u64,
}

impl Gn {
    /// Creates a new `G(n)` generator.
    ///
    /// Defaults to zero nodes. Use `.nodes(n)` to set the node count.
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
    /// Returns a lazily-evaluated iterator over edges generated with 50% probability.
    ///
    /// Internally equivalent to `Gnp::new().nodes(n).prob(0.5).stream(rng)`.
    fn stream<R>(&self, rng: &mut R) -> impl Iterator<Item = Edge>
    where
        R: Rng,
    {
        assert!(self.n > 0, "At least one node must be generated!");
        GeometricJumper::new(0.5)
            .stop_at(self.n * self.n)
            .iter(rng)
            .map(|x| Edge::from_u64(x, self.n))
    }
}
