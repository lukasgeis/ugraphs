/*!
# Random Hyperbolic Graph Generator (`RHG`)

This module provides an implementation of the **Random Hyperbolic Graph (RHG)** model, a widely used
random graph model that generates graphs with power-law degree distributions, high clustering, and
small-world properties.

RHGs are constructed by placing nodes in a hyperbolic disk and connecting pairs of nodes whose
hyperbolic distance is below a given threshold. The generator supports tuning of key parameters
such as:

- **Number of nodes (`n`)**
- **Power-law exponent (`γ`)**
- **Temperature (`T`)**, controlling clustering vs. randomness
- **Radius scaling (`R`)**

The implementation supports both directed and undirected graphs and is optimized with spatial
partitioning and streaming edge generation.

The implementation follows ideas from:
- “Efficient Generation of Large-Scale Random Hyperbolic Graphs” by von Looz et al.
- “Communication-free Massively Distributed Graph Generation” by Funke et al.
- NetworKIT’s RHG generator (including safe region optimization)
*/

use super::*;
use std::f64::consts::{PI, TAU};

use itertools::Itertools;

/// Struct representing a node coordinate in hyperbolic space with precomputed values.
///
/// We precompute `sinh(rad)`, `cosh(rad)`, `sin(phi)`, `cos(phi)`, and assign the node to a
/// radial band (`bid`). This avoids repeated computation and improves edge generation efficiency.
#[derive(Debug, Clone, Copy)]
struct Coord {
    id: Node,
    phi: f64,
    bid: usize,
    rad_cosh: f64,
    rad_sinh: f64,
    phi_cos: f64,
    phi_sin: f64,
}

impl PartialEq for Coord {
    fn eq(&self, other: &Self) -> bool {
        (self.bid, self.phi) == (other.bid, other.phi)
    }
}

impl PartialOrd for Coord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.bid, self.phi).partial_cmp(&(other.bid, other.phi))
    }
}

impl Eq for Coord {}

/// Radius of a Rhg-Generator can be set manually or derived from alpha and an average degree
#[derive(Debug, Copy, Clone, Default)]
pub enum RhgRadius {
    /// Radius has not been set yet
    #[default]
    NotSet,
    /// Radius defined in terms of an average degree
    AvgDeg(f64),
    /// Radius defined manually
    Radius(f64),
}

/// Random Hyperbolic Graph generator.
///
/// This struct encapsulates configuration parameters for generating an RHG.
/// The generator supports tuning the number of nodes, power-law exponent,
/// average degree, and/or radius. Edges are streamed using [`RhgGenerator`].
#[derive(Debug, Copy, Clone)]
pub struct Rhg {
    /// Number of nodes
    nodes: NumNodes,
    /// Radial dispersion parameter alpha controlling node distribution
    alpha: f64,
    /// Radius of hyperbolic disk, either set manually or computed from average degree
    radius: RhgRadius,
    /// Optional number of radial bands used for partitioning nodes
    num_bands: Option<usize>,
}

impl Default for Rhg {
    fn default() -> Self {
        Self {
            nodes: 0,
            alpha: 1.0,
            radius: RhgRadius::NotSet,
            num_bands: None,
        }
    }
}

impl Rhg {
    /// Creates a new `Rhg` generator with default parameters.
    ///
    /// Equivalent to calling `Rhg::default()`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the alpha parameter, which controls radial node distribution
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Sets the alpha parameter, which controls radial node distribution
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.set_alpha(alpha);
        self
    }

    /// Manually sets the radius of the hyperbolic disk
    pub fn set_radius(&mut self, radius: f64) {
        self.radius = RhgRadius::Radius(radius);
    }

    /// Manually sets the radius of the hyperbolic disk
    pub fn radius(mut self, radius: f64) -> Self {
        self.set_radius(radius);
        self
    }

    /// Sets the number of radial bands used for node partitioning
    pub fn set_num_bands(&mut self, num_bands: usize) {
        self.num_bands = Some(num_bands);
    }

    /// Sets the number of radial bands used for node partitioning
    pub fn num_bands(mut self, num_bands: usize) -> Self {
        self.set_num_bands(num_bands);
        self
    }
}

impl NumNodesGen for Rhg {
    fn set_nodes(&mut self, n: NumNodes) {
        self.nodes = n;
    }
}

impl AverageDegreeGen for Rhg {
    fn set_avg_deg(&mut self, deg: f64) {
        self.radius = RhgRadius::AvgDeg(deg);
    }
}

impl GraphGenerator for Rhg {
    type EdgeStream<'a, R>
        = RhgGenerator
    where
        R: Rng + 'a,
        Self: 'a;

    fn stream<'a, R>(&'a self, rng: &'a mut R) -> Self::EdgeStream<'a, R>
    where
        R: Rng,
    {
        RhgGenerator::new(rng, self.nodes, self.radius, self.alpha, self.num_bands)
    }
}

/// Streaming edge generator for the Random Hyperbolic Graph.
///
/// Produces edges one-by-one according to RHG rules without materializing
/// the full adjacency list in memory.
///
/// Implements [`Iterator`] yielding [`Edge`]s.
///
/// The generator uses geometric partitioning to efficiently sample
/// node pairs below the threshold distance.
#[derive(Debug, Clone)]
pub struct RhgGenerator {
    /// Coordinates of nodes in hyperbolic space, sorted by band and angular coordinate
    coordinates: Vec<Coord>,
    /// Prefix sums over the number of nodes in each radial band (used for indexing)
    band_bounds: Vec<NumNodes>,
    /// Precomputed `(cosh(band_limit), sinh(band_limit))` for each band boundary
    band_cosh_sinh: Vec<(f64, f64)>,
    /// `cosh(radius)` of the hyperbolic disk radius (distance threshold)
    radius_cosh: f64,

    // --- Internal iteration state ---
    /// Index of the current node being processed
    curr_coord_index: usize,
    /// Angular range (safe_values) within which neighbors are guaranteed to be connected:
    /// (angle_size, lower_bound_phi, upper_bound_phi), modulo TAU
    safe_values: (f64, f64, f64),
    /// Current radial band ID being searched for neighbors
    curr_bid: usize,
    /// Bounds (indices) in `coordinates` for nodes in the current band
    slab_bounds: (usize, usize),
    /// Current search window within `slab_bounds` for candidate neighbors,
    /// accounts for modular wrap-around of the angular coordinate
    slab_pointer: (usize, usize),
}

impl RhgGenerator {
    /// Creates a new `RhgGenerator` with the given parameters.
    ///
    /// This basically already does all the necessary preprocessing, including:
    /// - sampling the coordinate values & computing associated constants
    /// - partitioning coordinates into bands and sorting these bands by coordinate
    ///
    /// The returned iterator only finds the next pair of nodes which must be connected
    /// as their hyperbolic distance is less than `radius`.
    ///
    /// # Panics
    /// Panics if:
    /// - `n == 0`
    /// - `alpha == 0`
    /// - incompatible parameter configurations were set in the [`Rhg`] instance
    pub fn new<R>(
        rng: &mut R,
        n: NumNodes,
        radius: RhgRadius,
        alpha: f64,
        num_bands: Option<usize>,
    ) -> Self
    where
        R: Rng,
    {
        assert!(alpha > 0.0);

        // Determine radius either from provided value or compute from avg degree and alpha
        let radius = match radius {
            RhgRadius::Radius(radius) => radius,
            RhgRadius::AvgDeg(deg) => {
                assert!(deg + 1.0 < n as f64);
                Self::get_target_radius(n as f64, deg, alpha).expect(
                    "RHG: Provided values for n, avg-deg, alpha can not produce a valid radius!",
                )
            }
            _ => unimplemented!("No correct radius identifier set for Rhg"),
        };

        let num_bands = if let Some(b) = num_bands {
            assert!(b > 1);
            b
        } else {
            // inspired by "Communication-free Massively Distributed Graph Generation" [Funke et al.]
            2.max((radius * alpha / 2.0 / std::f64::consts::LN_2).ceil() as usize)
        };

        let band_limits: Vec<f64> = [0.0, radius / 2.0]
            .into_iter()
            .chain(
                (1..num_bands)
                    .map(|i| radius / 2.0 / (num_bands - 1) as f64 * i as f64 + radius / 2.0),
            )
            .collect();

        let band_cosh_sinh = band_limits
            .iter()
            .map(|b| (b.cosh(), b.sinh()))
            .collect_vec();
        let radius_cosh = band_cosh_sinh.last().unwrap().0;

        let (mut coordinates, band_sizes) =
            Self::sample_coordinates(rng, n, radius, alpha, &band_limits);
        coordinates.sort_unstable_by(|u, v| u.partial_cmp(v).unwrap());

        let mut cur_prefix = 0;
        let mut band_bounds = Vec::with_capacity(band_limits.len() + 1);
        band_bounds.push(0);
        for &band_size in &band_sizes {
            cur_prefix += band_size;
            band_bounds.push(cur_prefix);
        }

        let init_bid = coordinates[0].bid;
        let mut initial_state = Self {
            coordinates,
            band_bounds,
            band_cosh_sinh,
            radius_cosh,

            curr_coord_index: 0,
            safe_values: (0.0, 0.0, 0.0),
            curr_bid: init_bid,
            slab_bounds: (0, 0),
            slab_pointer: (0, 0),
        };
        initial_state.recompute_safe_values(initial_state.curr_bid);
        initial_state.recompute_slab();
        initial_state.recompute_safe_values(initial_state.curr_bid + 1);

        initial_state
    }

    /// Computes the radius needed to achieve a target average degree `k` for given `n` and `alpha`.
    ///
    /// Uses a numerical bisection method based on expected degree formula from NetworKit.
    fn get_target_radius(n: f64, k: f64, alpha: f64) -> Option<f64> {
        let gamma = 2.0 * alpha + 1.0;
        let xi_inv = (gamma - 2.0) / (gamma - 1.0);
        let v = k * (PI / 2.0) * xi_inv * xi_inv;
        let current_r = 2.0 * (n / v).ln();
        let mut lower_bound = current_r / 2.0;
        let mut upper_bound = current_r * 2.0;

        if expected_degree(n, alpha, lower_bound) <= k
            || expected_degree(n, alpha, upper_bound) >= k
        {
            return None;
        }

        fn expected_degree(n: f64, alpha: f64, rad: f64) -> f64 {
            let gamma = 2.0 * alpha + 1.0;
            let xi = (gamma - 1.0) / (gamma - 2.0);
            let first_sum_term = (-rad / 2.0).exp();
            let second_sum_term = (-alpha * rad).exp()
                * (alpha
                    * (rad / 2.0)
                    * ((PI / 4.0) * (1.0 / alpha).powi(2) - (PI - 1.0) * (1.0 / alpha)
                        + (PI - 2.0))
                    - 1.0);
            (2.0 / PI) * xi * xi * n * (first_sum_term + second_sum_term)
        }

        loop {
            let current_r = (lower_bound + upper_bound) / 2.0;
            let current_k = expected_degree(n, alpha, current_r);

            if current_k < k {
                upper_bound = current_r;
            } else {
                lower_bound = current_r;
            }

            if (expected_degree(n, alpha, current_r) - k).abs() < 1e-5 {
                return Some(current_r);
            }
        }
    }

    /// Samples `n` node coordinates uniformly at random in hyperbolic space with given radius and alpha.
    /// Returns the vector of `Coord` and a vector of counts of points per band.
    fn sample_coordinates<R>(
        rng: &mut R,
        n: NumNodes,
        disk_rad: f64,
        alpha: f64,
        band_limits: &[f64],
    ) -> (Vec<Coord>, Vec<NumNodes>)
    where
        R: Rng,
    {
        let min = 1.0_f64.next_up();
        let max = (alpha * disk_rad).cosh();
        let mut band_sizes = vec![0; band_limits.len()];
        (
            (0..n)
                .map(|id| {
                    let phi = rng.random_range(0.0..TAU);
                    let rad = rng.random_range(min..max).acosh() / alpha;

                    assert!(0.0 <= rad);
                    assert!(rad <= disk_rad);

                    // Linear reverse search is fastest as we only have very few bands normally and
                    // expect a exponential decrease in points for lower bands
                    let bid = band_limits
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, limit)| rad >= **limit)
                        .unwrap()
                        .0;

                    band_sizes[bid] += 1;
                    Coord {
                        id,
                        phi,
                        bid,
                        rad_cosh: rad.cosh(),
                        rad_sinh: rad.sinh(),
                        phi_sin: phi.sin(),
                        phi_cos: phi.cos(),
                    }
                })
                .collect(),
            band_sizes,
        )
    }

    /// Performs a custom binary search over coordinates partitioned by phi to find the partition index of `val`.
    /// Previously, benchmarks showed this to be faster than standard binary search for this use case.
    #[inline]
    fn binary_search_partition(val: f64, points: &[Coord]) -> usize {
        if points.is_empty() {
            return 0;
        }

        let mut left = 0usize;
        let mut right = points.len() - 1;

        while left < right {
            let mid = (left + right) / 2;

            if mid == 0 {
                return (points[1].phi <= val) as usize;
            }

            if points[mid].phi <= val && points[mid + 1].phi > val {
                return mid;
            } else if points[mid].phi > val {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        left
    }

    /// Computes safe angular bounds around the current node that guarantee neighbors within radius.
    ///
    /// This avoids expensive distance checks for neighbors inside this range.
    fn recompute_safe_values(&mut self, bid: usize) {
        let u = &self.coordinates[self.curr_coord_index];

        self.safe_values.0 = ((u.rad_cosh * self.band_cosh_sinh[bid].0 - self.radius_cosh)
            / (u.rad_sinh * self.band_cosh_sinh[bid].1))
            .acos();

        if self.safe_values.0.is_nan() {
            self.safe_values.1 = -1.0;
            self.safe_values.2 = -1.0;
        } else {
            self.safe_values.1 = u.phi - self.safe_values.0;
            if self.safe_values.1 < 0.0 {
                self.safe_values.1 += TAU;
            }

            self.safe_values.2 = u.phi + self.safe_values.0;
            if self.safe_values.2 >= TAU {
                self.safe_values.2 -= TAU;
            }
        }
    }

    /// Searches for the next edge in the current slab.
    /// If it finds one, it returns it, otherwise return None;
    fn next_edge(&mut self) -> Option<Edge> {
        let u = &self.coordinates[self.curr_coord_index];

        loop {
            if self.slab_pointer.0 == self.slab_pointer.1 {
                return None;
            }

            let v = &self.coordinates[self.slab_pointer.0];

            self.slab_pointer.0 += 1;
            // `(self.slab_bounds.0..self.slab_bounds.1).contains(&self.slab_pointer.{0,1})` is guaranteed
            if self.slab_pointer.0 > self.slab_pointer.1
                && self.slab_pointer.0 >= self.slab_bounds.1
            {
                self.slab_pointer.0 = self.slab_bounds.0;
            }

            if self.curr_bid > u.bid || u.id < v.id {
                let within_inner = if self.safe_values.1 <= self.safe_values.2 {
                    self.safe_values.1 < v.phi && v.phi < self.safe_values.2
                } else {
                    self.safe_values.1 < v.phi || v.phi < self.safe_values.2
                };

                if within_inner {
                    return Some(Edge(u.id, v.id).normalized());
                } else {
                    let dist_cosh = u.rad_cosh * v.rad_cosh
                        - u.rad_sinh * v.rad_sinh * (u.phi_cos * v.phi_cos + u.phi_sin * v.phi_sin);
                    if dist_cosh < self.radius_cosh {
                        return Some(Edge(u.id, v.id).normalized());
                    }
                }
            }
        }
    }

    /// Recomputes the index range (slab) of coordinates in the current radial band.
    fn recompute_slab(&mut self) {
        self.slab_bounds = (
            self.band_bounds[self.curr_bid] as usize,
            self.band_bounds[self.curr_bid + 1] as usize,
        );

        if self.safe_values.0.is_nan() {
            self.slab_pointer = self.slab_bounds;
        } else {
            let slab = &self.coordinates[self.slab_bounds.0..self.slab_bounds.1];

            self.slab_pointer.0 =
                self.slab_bounds.0 + Self::binary_search_partition(self.safe_values.1, slab);
            self.slab_pointer.1 =
                self.slab_bounds.0 + Self::binary_search_partition(self.safe_values.2, slab) + 1;

            // Edge cases where the whole slab is viable
            if self.safe_values.1 >= self.safe_values.2
                && self.slab_pointer.0 <= self.slab_pointer.1
            {
                self.slab_pointer = self.slab_bounds;
            }
        }
    }
}

impl Iterator for RhgGenerator {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Stop if no node is left to consider
            if self.curr_coord_index >= self.coordinates.len() {
                return None;
            }

            // If there still is an edge in the current slab, return it
            if let Some(next_edge) = self.next_edge() {
                return Some(next_edge);
            }

            // Increase the band-id to be searched for the next edge and recompute the respective
            // slab if in bounds
            self.curr_bid += 1;
            if self.curr_bid + 2 < self.band_bounds.len() {
                self.recompute_slab();
                self.recompute_safe_values(self.curr_bid + 1);
                continue;
            }

            // Go to the next coordinate to consider and recompute initial slab if in bounds
            self.curr_coord_index += 1;
            if self.curr_coord_index >= self.coordinates.len() {
                return None;
            }

            self.curr_bid = self.coordinates[self.curr_coord_index].bid;
            self.recompute_safe_values(self.curr_bid);
            self.recompute_slab();
            self.recompute_safe_values(self.curr_bid + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use super::*;

    fn compute_edges_naive(coords: &[Coord], radius_cosh: f64) -> Vec<Edge> {
        let n = coords.len();

        let mut edges = Vec::new();
        for i in 0..n {
            let u = &coords[i];
            for j in (i + 1)..n {
                let v = &coords[j];

                let dist_cosh = u.rad_cosh * v.rad_cosh
                    - u.rad_sinh * v.rad_sinh * (u.phi_cos * v.phi_cos + u.phi_sin * v.phi_sin);
                if dist_cosh < radius_cosh {
                    edges.push(Edge(u.id, v.id).normalized());
                }
            }
        }

        edges
    }

    #[test]
    fn compare_to_naive() {
        let rng = &mut Pcg64Mcg::seed_from_u64(3);

        for n in [10, 30, 100] {
            for deg_mult in [0.05, 0.10, 0.25] {
                for alpha in [0.6, 0.8, 1.2] {
                    let deg = (n as f64) * deg_mult;
                    let rhg_gen = RhgGenerator::new(rng, n, RhgRadius::AvgDeg(deg), alpha, None);

                    let mut naive_edges =
                        compute_edges_naive(&rhg_gen.coordinates, rhg_gen.radius_cosh);
                    naive_edges.sort_unstable();

                    let mut rhg_edges = rhg_gen.collect_vec();
                    rhg_edges.sort_unstable();

                    assert_eq!(naive_edges, rhg_edges);
                }
            }
        }
    }
}
