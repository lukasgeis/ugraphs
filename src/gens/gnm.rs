use std::marker::PhantomData;

use fxhash::FxHashMap;

use super::*;
use crate::utils::{FromCapacity, Map};

/// Configuration type used by [`Gnm`] to determine how the graph should be parameterized.
///
/// This can be either:
/// - a fixed number of edges, or
/// - an average degree value, which is converted into an edge count during generation.
#[derive(Debug, Copy, Clone, Default)]
enum GnmType {
    /// No value has been set yet; using this will panic at runtime.
    #[default]
    NotSet,
    /// Fixed number of edges `m`.
    Edges(NumEdges),
    /// Average degree `d`, to be converted to `m = d*n`.
    AvgDeg(f64),
}

/// Marker trait to generalize over internal map implementations for tracking chosen edges.
///
/// This allows users to customize the underlying structure used to perform the edge shuffle
/// in `G(n,m)` generation (e.g., [`FxHashMap`], `Vec`, or any other custom map-like container).
///
/// Must implement both [`FromCapacity`] and [`Map<K, V>`] with `K = u64`, `V = OptionalU64`.
pub trait GnmMap: FromCapacity + Map<u64, OptionalU64> {}
impl<H> GnmMap for H where H: FromCapacity + Map<u64, OptionalU64> {}

/// Generator for uniform `G(n,m)` random graphs with `n` nodes and `m` edges.
///
/// The generator can be parameterized via:
/// - `.nodes(n)` — total number of nodes
/// - `.edges(m)` or `.avg_deg(d)` — total number of edges or average degree
/// - `.directed(bool)` or `.undirected(bool)` — whether the graph is directed
/// - `.with_mapper<T>()` — optionally override the internal map type
///
/// The choice of internal map structure (`GnmMap`) allows performance tuning:
/// - [`FxHashMap`] (default) for general use
/// - `Vec`-based maps for dense graphs
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
    /// Creates a new empty `G(n,m)` generator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the graph as directed (or not).
    ///
    /// Note: `.undirected(bool)` is preferred for clarity.
    pub fn directed(mut self, directed: bool) -> Self {
        self.undirected = !directed;
        self
    }

    /// Marks the graph as undirected (or not).
    ///
    /// Affects how edges are interpreted internally (e.g., `u64 -> Edge` mapping).
    pub fn undirected(mut self, undirected: bool) -> Self {
        self.undirected = undirected;
        self
    }

    /// Switches the internal map implementation used for edge sampling.
    ///
    /// This allows replacing the default [`FxHashMap`] with a faster or more memory-efficient
    /// structure for specific scenarios (e.g., `Vec` for dense graphs).
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
    fn set_avg_deg(&mut self, deg: f64) {
        self.m = GnmType::AvgDeg(deg);
    }
}

impl<H> GraphGenerator for Gnm<H>
where
    H: GnmMap,
{
    /// Returns a streaming iterator over a random `G(n,m)` edge set.
    ///
    /// Internally, edges are uniformly sampled without replacement.
    ///
    /// # Panics
    /// - If `n == 0`
    /// - If neither `edges(m)` nor `avg_deg(d)` was set
    fn stream<R>(&self, rng: &mut R) -> impl Iterator<Item = Edge>
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

/// Given `n` nodes and a total edge space of size `end` (depending on whether the graph is
/// directed or undirected), this iterator produces exactly `m` uniformly random and distinct
/// edges without replacement.
///
/// The algorithm used is based on:
/// > *V. Batagelj and U. Brandes. Efficient Generation of Large Random Networks.
/// > Physical Review E 71.3 (2005): 036113.*
///
/// The implementation avoids full shuffling by using a partial mapping technique
/// (sometimes called "hash-based sampling") to simulate an in-place permutation.
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
    /// This generator yields exactly `m` random edge values in `[0, end)`, avoiding duplicates.
    ///
    /// # Panics
    /// Panics if `m > end`, which would violate sampling without replacement.
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

    /// Selects the next unique edge index using the Batagelj–Brandes partial mapping method.
    ///
    /// This method emulates a Fisher-Yates shuffle on-the-fly using a sparse map structure
    /// to track remappings. Ensures `m` unique samples from `[0, end)`.
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
