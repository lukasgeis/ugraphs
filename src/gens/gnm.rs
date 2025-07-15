use std::marker::PhantomData;

use fxhash::FxHashMap;

use crate::{
    NumEdges,
    gens::*,
    utils::{FromCapacity, Map},
};

/// A G(n,m) graph can be defined by either the number of edges or the average degree
#[derive(Debug, Copy, Clone, Default)]
enum GnmType {
    /// No value has been set yet
    #[default]
    NotSet,
    /// Number of Edges
    Edges(NumEdges),
    /// Average degree of a node
    AvgDeg(f64),
}

/// `G(n,m)` graphs are uniform graphs with `n` nodes and `m` edges.
#[derive(Debug, Copy, Clone)]
pub struct Gnm<H: FromCapacity + Map<u64, OptionalU64> = FxHashMap<u64, OptionalU64>> {
    n: u64,
    m: GnmType,
    /// It is important to know beforehand whether the graph is supposed to be undirected or
    /// not as this changes the workings of the generator itself.
    undirected: bool,
    _phantom: PhantomData<H>,
}

impl<H: FromCapacity + Map<u64, OptionalU64>> Default for Gnm<H> {
    fn default() -> Self {
        Self {
            n: 0,
            m: GnmType::NotSet,
            undirected: false,
            _phantom: Default::default(),
        }
    }
}

impl<H: FromCapacity + Map<u64, OptionalU64>> Gnm<H> {
    /// Creates a new empty `G(n,p)` generator
    pub fn new() -> Self {
        Self::default()
    }

    /// Change whether the graph is directed or not
    pub fn directed(mut self, directed: bool) -> Self {
        self.undirected = !directed;
        self
    }

    /// Change whether the graph is undirected or not
    pub fn undirected(mut self, undirected: bool) -> Self {
        self.undirected = undirected;
        self
    }

    /// Change the underlying mapper used
    pub fn with_mapper<M: FromCapacity + Map<u64, OptionalU64>>(self) -> Gnm<M> {
        Gnm {
            n: self.n,
            m: self.m,
            undirected: self.undirected,
            _phantom: Default::default(),
        }
    }
}

impl<H: FromCapacity + Map<u64, OptionalU64>> NumNodesGen for Gnm<H> {
    /// Updates `n`
    fn nodes(mut self, n: NumNodes) -> Self {
        self.n = n as u64;
        self
    }
}

impl<H: FromCapacity + Map<u64, OptionalU64>> NumEdgesGen for Gnm<H> {
    /// Updates `m`
    fn edges(mut self, m: NumEdges) -> Self {
        self.m = GnmType::Edges(m);
        self
    }
}

impl<H: FromCapacity + Map<u64, OptionalU64>> AverageDegreeGen for Gnm<H> {
    /// Updates `p` such that `m = d * n` (accounts for direction of graph).
    /// Note that this conversion will only be done when calling `stream/generate`.
    fn avg_deg(mut self, deg: f64) -> Self {
        self.m = GnmType::AvgDeg(deg);
        self
    }
}

impl<H: FromCapacity + Map<u64, OptionalU64>> GraphGenerator for Gnm<H> {
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge> {
        assert!(self.n > 0, "At least one node must be generated!");
        let m = match self.m {
            GnmType::NotSet => panic!("Probility of Gnp was not set!"),
            GnmType::Edges(m) => m,
            GnmType::AvgDeg(d) => (self.n as f64 * d) as NumEdges,
        };

        // TBD: make edge cases faster

        GnmGenerator::new(
            rng,
            self.n,
            m as u64,
            H::from_capacity(m as usize),
            self.undirected,
        )
    }
}

pub struct GnmGenerator<'a, R: Rng, H: Map<u64, OptionalU64>> {
    n: u64,
    rem: u64,
    cur: u64,
    end: u64,
    map: H,
    rng: &'a mut R,
    undirected: bool,
}

impl<'a, R: Rng, H: Map<u64, OptionalU64>> GnmGenerator<'a, R, H> {
    /// Creates a new generator
    pub fn new(rng: &'a mut R, n: u64, m: u64, map: H, undirected: bool) -> Self {
        // Make sure that `m` is not too big
        if undirected {
            assert!(m <= (n * (n - 1) / 2));
        } else {
            assert!(m <= n * n);
        }

        Self {
            n,
            rem: m,
            cur: 0,
            end: n * n,
            map,
            rng,
            undirected,
        }
    }

    /// Draws a random value from `self.cur..self.end` and returns it if it represents an
    /// undirected edge or we are okay with directed edges
    fn draw_value(&mut self) -> u64 {
        // If `self.undirected = false`, this returns after one iteration, otherwise the expected
        // number of iterations is `~2` as roughly half of all values represent normalized edges.
        loop {
            let x = self.rng.random_range(self.cur..self.end);
            if !self.undirected || Edge::from_u64(x, self.n).is_normalized() {
                return x;
            }
        }
    }

    /// Tries to increment `self.cur`
    /// Panics if `self.cur >= self.end - 1`
    fn increment_cur(&mut self) {
        assert!(self.cur + 1 < self.end);

        // If directed edges are allowed, an increment is just `+1`
        if !self.undirected {
            self.cur += 1;
            return;
        }

        let Edge(u, v) = Edge::from_u64(self.cur, self.n);

        // A `+1` increment is valid as the 'current node-range' is not exhausted
        if v as u64 == self.n - 1 {
            self.cur += 1;
            return;
        }

        // Increment `u` by 1 and set `v = u + 1`.
        // This is safe as `self.cur < self.end - 1` and thus actually `self.cur < self.end - self.n` since `v == self.n - 1`
        self.cur = (u as u64 + 1) * self.n + (u as u64) + 2;
    }

    /// Draws the next random edge as `u64` or returns None if no edges remain
    fn next_step(&mut self) -> Option<u64> {
        // Stop if `m` values were generated
        if self.rem == 0 {
            return None;
        }

        // Draw value and check if it already exists
        let next_rng = self.draw_value();
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

        self.increment_cur();
        self.rem -= 1;

        Some(next_u64)
    }
}

impl<'a, R: Rng, H: Map<u64, OptionalU64>> Iterator for GnmGenerator<'a, R, H> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_step().map(|x| Edge::from_u64(x, self.n))
    }

    /// The exact size of a valid iterator is given by the number of remaining edges to be drawn
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rem as usize, Some(self.rem as usize))
    }
}

impl<'a, R: Rng, H: Map<u64, OptionalU64>> ExactSizeIterator for GnmGenerator<'a, R, H> {}
