use std::fmt::Debug;

use crate::{gens::*, utils::*};

/// A G(n, p) graph can be defined by either a probability or the average degree which is more
/// common in practice
#[derive(Debug, Copy, Clone, Default)]
enum GnpType {
    /// No value has been set yet
    #[default]
    NotSet,
    /// Direct probability value
    Prob(f64),
    /// Average degree of a node
    AvgDeg(f64),
}

/// `G(n,p)` graphs generate every possible edge in a graph with `n` nodes with probability `p`
/// independent from each other.
///
/// Due to this independence, we do not need to incorporate normalized-checks for undirected graphs
/// or self-loop checks in the generator itself as the overhead is minimal (`2 * n/(n - 1)` at most).
///
/// Filterings of this sort are thus up to the caller.
#[derive(Debug, Copy, Clone, Default)]
pub struct Gnp {
    n: u64,
    p: GnpType,
}

impl Gnp {
    /// Creates a new empty `G(n,p)` generator
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates `p` directly
    pub fn prob(mut self, prob: f64) -> Self {
        assert!(prob.is_valid_probility());
        self.p = GnpType::Prob(prob);
        self
    }
}

impl NumNodesGen for Gnp {
    /// Updates `n`
    fn nodes(mut self, n: usize) -> Self {
        self.n = n as u64;
        self
    }
}

impl AverageDegreeGen for Gnp {
    /// Updates `p` such that `p = d/n`.
    /// Note that this conversion will only be done when calling `stream/generate`.
    fn avg_deg(mut self, deg: f64) -> Self {
        self.p = GnpType::AvgDeg(deg);
        self
    }
}

impl StreamingGenerator for Gnp {
    /// Creates a streaming generator over random `G(n,p)` edges
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge> {
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

/// `G(n) = G(n,1/2)` generators are uniform distributions over all graphs with `n` nodes.
#[derive(Debug, Copy, Clone, Default)]
pub struct Gn {
    n: u64,
}

impl Gn {
    /// Creates a new `G(n)` generator
    pub fn new() -> Self {
        Self::default()
    }
}

impl NumNodesGen for Gn {
    fn nodes(mut self, n: usize) -> Self {
        self.n = n as u64;
        self
    }
}

impl StreamingGenerator for Gn {
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge> {
        assert!(self.n > 0, "At least one node must be generated!");
        GeometricJumper::new(0.5)
            .stop_at(self.n * self.n)
            .iter(rng)
            .map(|x| Edge::from_u64(x, self.n))
    }
}
