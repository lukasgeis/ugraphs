//! # Graph Generators
//!
//! Module probiding multiple (random) graph generators.

use rand::Rng;

use crate::{
    ops::{GraphFromScratch, GraphType},
    *,
};

mod gnp;

pub use gnp::*;

// Builder trait for generators that allow setting the number of nodes.
// Should be almost (if not) all but we abstract anyway to keep the pattern.
pub trait NumNodesGen {
    /// Set the number of nodes of this generator.
    fn nodes(self, n: NumNodes) -> Self;
}

/// Builder trait for generators that allow setting the number of edges.
pub trait NumEdgesGen {
    /// Set the number of edges of this generator.
    fn edges(self, m: NumEdges) -> Self;
}

/// Builder trait for generators that allow setting the average AverageDegree
pub trait AverageDegreeGen {
    /// Set the average degree of this generator.
    fn avg_deg(self, deg: f64) -> Self;
}

/// General trait for a random graph generator
pub trait GraphGenerator {
    /// Create a list of random edges of this generator
    fn generate<R: Rng>(&self, rng: &mut R) -> Vec<Edge> {
        self.stream(rng).collect()
    }

    /// Create a stream over random edges of this generator
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge>;
}

/// Trait for generating a random graph
pub trait RandomGraph {
    /// Creates a random `G(n,p)` graph
    fn new_gnp_graph<R: Rng>(rng: &mut R, n: NumNodes, p: f64) -> Self;

    /// Creates a new `G(n,p)` graph with no loops
    fn new_gnp_graph_no_loops<R: Rng>(rng: &mut R, n: NumNodes, p: f64) -> Self;
}

impl<G: GraphFromScratch + GraphType> RandomGraph for G {
    fn new_gnp_graph<R: Rng>(rng: &mut R, n: NumNodes, p: f64) -> Self {
        Self::from_edges(
            n,
            Gnp::new()
                .nodes(n)
                .prob(p)
                .stream(rng)
                .filter(|e| Self::is_directed() || e.is_normalized()),
        )
    }

    fn new_gnp_graph_no_loops<R: Rng>(rng: &mut R, n: NumNodes, p: f64) -> Self {
        Self::from_edges(
            n,
            Gnp::new()
                .nodes(n)
                .prob(p)
                .stream(rng)
                .filter(|e| !e.is_loop() && (Self::is_directed() || e.is_normalized())),
        )
    }
}
