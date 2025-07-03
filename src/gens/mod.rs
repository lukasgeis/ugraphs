//! # Graph Generators
//!
//! Module probiding multiple (random) graph generators.

use rand::Rng;

use crate::*;

mod gnp;

pub use gnp::*;

// Builder trait for generators that allow setting the number of nodes.
// Should be almost (if not) all but we abstract anyway to keep the pattern.
pub trait NumNodesGen {
    /// Set the number of nodes of this generator.
    fn nodes(self, n: usize) -> Self;
}

/// Builder trait for generators that allow setting the number of edges.
pub trait NumEdgesGen {
    /// Set the number of edges of this generator.
    fn edges(self, m: usize) -> Self;
}

/// Builder trait for generators that allow setting the average AverageDegree
pub trait AverageDegreeGen {
    /// Set the average degree of this generator.
    fn avg_deg(self, deg: f64) -> Self;
}

/// General trait for a random graph generator: sample a list of edges.
pub trait Generator {
    /// Create a list of random edges of this generator
    fn generate<R: Rng>(&self, rng: &mut R) -> Vec<Edge>;
}

/// General trait for a random graph generator that allows sampling edges as an iterator.
pub trait StreamingGenerator {
    /// Create a stream over random edges of this generator
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge>;
}

impl<SG: StreamingGenerator> Generator for SG {
    #[inline]
    fn generate<R: Rng>(&self, rng: &mut R) -> Vec<Edge> {
        self.stream(rng).collect()
    }
}
