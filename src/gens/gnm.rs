use std::marker::PhantomData;

use fxhash::FxHashMap;

use super::*;
use crate::utils::{FromCapacity, Map};

/// Internal representation of how a `G(n,m)` generator is parameterized.
///
/// This can be configured in two ways:
/// - by specifying the exact number of edges `m`
/// - by specifying an average degree `d`, which will be converted to an edge count
#[derive(Debug, Copy, Clone, Default)]
enum GnmType {
    /// No parameters set yet; using this will panic at runtime.
    #[default]
    NotSet,
    /// Fixed number of edges `m`.
    Edges(NumEdges),
    /// Average degree `d`, converted internally to `m = d*n`.
    AvgDeg(f64),
}

/// Marker trait abstracting over possible map implementations used for edge sampling.
///
/// A `GnmMap` is responsible for tracking edge remappings during generation.  
/// Different backing implementations can be used depending on performance needs:
/// - `FxHashMap` (default) for general (sparse) graphs
/// - `Vec`-like structures for dense graphs
///
/// Any implementor must provide both:
/// - [`FromCapacity`] — to preallocate capacity efficiently
/// - [`Map<K, V>`] — a map-like interface with `K = u64` and `V = OptionalU64`
pub trait GnmMap: FromCapacity + Map<u64, OptionalU64> {}
impl<H> GnmMap for H where H: FromCapacity + Map<u64, OptionalU64> {}

/// Generator for uniform random graphs of type `G(n,m)`.
///
/// In a `G(n,m)` graph:
/// - there are exactly `n` nodes
/// - exactly `m` distinct edges are chosen uniformly at random from the possible edge set
///
/// The generator allows customization via methods:
/// - `.nodes(n) / .set_nodes(n)` — set number of nodes
/// - `.edges(m) / .set_edges(m)` or `.avg_deg(d) / .set_avg_deg(d)` — set edge count or average degree
/// - `.directed(bool) / .set_directed(bool)` / `.undirected(bool) / .set_undirected(bool)` — toggle directedness
/// - `.with_mapper<T>()` — override the internal edge-mapping structure
///
/// Internally, edge sampling uses the Batagelj–Brandes algorithm (2005),
/// which efficiently simulates a partial shuffle without replacement.
#[derive(Debug, Copy, Clone)]
pub struct Gnm<H = FxHashMap<u64, OptionalU64>>
where
    H: GnmMap,
{
    n: u64,
    m: GnmType,
    /// Whether the graph is undirected or not.
    ///
    /// This affects how edges are interpreted and mapped.
    undirected: bool,
    _phantom: PhantomData<H>,
}

impl<H> Default for Gnm<H>
where
    H: GnmMap,
{
    fn default() -> Self {
        Self {
            n: 0,
            m: GnmType::NotSet,
            undirected: false,
            _phantom: Default::default(),
        }
    }
}

impl<H> Gnm<H>
where
    H: GnmMap,
{
    /// Creates a new, empty `G(n,m)` generator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the graph as directed or not.
    pub fn set_directed(&mut self, directed: bool) {
        self.undirected = !directed;
    }

    /// Marks the graph as directed or not (builder-style).
    pub fn directed(mut self, directed: bool) -> Self {
        self.set_directed(directed);
        self
    }

    /// Marks the graph as undirected or not.
    pub fn set_undirected(&mut self, undirected: bool) {
        self.undirected = undirected;
    }

    /// Marks the graph as undirected or not (builder-style).
    pub fn undirected(mut self, undirected: bool) -> Self {
        self.set_undirected(undirected);
        self
    }

    /// Replaces the internal map type used to track edge samples.
    ///
    /// Defaults to [`FxHashMap`], but can be swapped for other implementations
    /// (e.g., `Vec`-based maps for dense graphs).
    pub fn with_mapper<M: FromCapacity + Map<u64, OptionalU64>>(self) -> Gnm<M> {
        Gnm {
            n: self.n,
            m: self.m,
            undirected: self.undirected,
            _phantom: Default::default(),
        }
    }
}

impl<H> NumNodesGen for Gnm<H>
where
    H: GnmMap,
{
    fn set_nodes(&mut self, n: NumNodes) {
        self.n = n as u64;
    }
}

impl<H> NumEdgesGen for Gnm<H>
where
    H: GnmMap,
{
    fn set_edges(&mut self, m: NumEdges) {
        self.m = GnmType::Edges(m);
    }
}

impl<H> AverageDegreeGen for Gnm<H>
where
    H: GnmMap,
{
    /// Sets the average degree `d` in the graph.
    ///
    /// Internally converted to an edge count: `m = d*n`.
    /// Note that this is only an approximation.
    fn set_avg_deg(&mut self, deg: f64) {
        self.m = GnmType::AvgDeg(deg);
    }
}

impl<H> GraphGenerator for Gnm<H>
where
    H: GnmMap,
{
    type EdgeStream<'a, R>
        = GnmGenerator<'a, R, H>
    where
        R: Rng + 'a,
        Self: 'a;

    /// Returns a lazily-evaluated iterator over uniformly sampled `G(n,m)` edges.
    ///
    /// Edges are drawn without replacement using the Batagelj–Brandes method.
    ///
    /// # Panics
    /// - if no nodes were set (`n == 0`)
    /// - if neither edge count nor average degree was specified
    fn stream<'a, R>(&'a self, rng: &'a mut R) -> Self::EdgeStream<'a, R>
    where
        R: Rng,
    {
        assert!(self.n > 0, "At least one node must be generated!");
        let m = match self.m {
            GnmType::NotSet => panic!("Probility of Gnp was not set!"),
            GnmType::Edges(m) => m,
            GnmType::AvgDeg(d) => (self.n as f64 * d) as NumEdges,
        };

        // TBD: make edge cases faster

        let end = if self.undirected {
            self.n * (self.n - 1) / 2
        } else {
            self.n * self.n
        };

        GnmGenerator::new(
            rng,
            self.n,
            m as u64,
            H::from_total_used_capacity(end as usize, m as usize),
            self.undirected,
        )
    }
}

/// Streaming generator for `G(n,m)` edges using Batagelj–Brandes sampling.
///
/// Produces exactly `m` distinct edges chosen uniformly without replacement.
/// The algorithm avoids a full shuffle by using a partial mapping scheme.
///
/// # References
/// - V. Batagelj and U. Brandes, *Efficient Generation of Large Random Networks*,
///   Physical Review E 71.3 (2005): 036113.
pub struct GnmGenerator<'a, R, H>
where
    R: Rng,
    H: Map<u64, OptionalU64>,
{
    n: u64,
    rem: u64,
    cur: u64,
    end: u64,
    map: H,
    rng: &'a mut R,
    undirected: bool,
}

impl<'a, R, H> GnmGenerator<'a, R, H>
where
    R: Rng,
    H: Map<u64, OptionalU64>,
{
    /// Creates a new `GnmGenerator`.
    ///
    /// # Panics
    /// Panics if `m > end`, i.e., if more edges are requested than possible.
    pub fn new(rng: &'a mut R, n: u64, m: u64, map: H, undirected: bool) -> Self {
        let end = if undirected { n * (n - 1) / 2 } else { n * n };
        debug_assert!(m <= end);

        Self {
            n,
            rem: m,
            cur: 0,
            end,
            map,
            rng,
            undirected,
        }
    }

    /// Performs one sampling step of the Batagelj–Brandes method.
    ///
    /// Emulates a Fisher–Yates shuffle lazily via a sparse mapping,
    /// ensuring each edge index is chosen uniformly without replacement.
    fn next_step(&mut self) -> Option<u64> {
        // Stop if `m` values were generated
        if self.rem == 0 {
            return None;
        }

        // Draw value and check if it already exists
        let next_rng = self.rng.random_range(self.cur..self.end);
        let next_u64 = match self.map.get(&next_rng) {
            Some(v) => v.get(),
            None => next_rng,
        };

        // Store possible replacements for later
        if let Some(v) = self.map.get(&self.cur) {
            self.map.insert(next_rng, *v);
        } else {
            self.map
                .insert(next_rng, OptionalU64::new(self.cur).unwrap());
        }

        self.cur += 1;
        self.rem -= 1;

        Some(next_u64)
    }
}

impl<'a, R, H> Iterator for GnmGenerator<'a, R, H>
where
    R: Rng,
    H: Map<u64, OptionalU64>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_step().map(|x| {
            if self.undirected {
                Edge::from_u64_undir(x, self.n)
            } else {
                Edge::from_u64(x, self.n)
            }
        })
    }

    /// Returns the number of edges remaining to be generated.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rem as usize, Some(self.rem as usize))
    }
}

impl<'a, R, H> ExactSizeIterator for GnmGenerator<'a, R, H>
where
    R: Rng,
    H: Map<u64, OptionalU64>,
{
}
