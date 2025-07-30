use super::*;
use std::f64::consts::{PI, TAU};

use itertools::Itertools;

/// A coordinate in hyperbolic space consists of
/// - an angle `phi`
/// - a radius `rad`
/// - an `id` to identify the coordinate
///
/// We precompute values such as `sinh(rad), cosh(rad), sin(phi), cos(phi)` as well as the id of
/// the band in which `rad` lies. Thus we do not need to store `rad`.
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

/// A RandomHyperbolicGraph-Generator for the threshold-case
#[derive(Debug, Copy, Clone, Default)]
pub struct Rhg {
    /// Number of nodes
    nodes: NumNodes,
    /// Radial dispersion
    alpha: f64,
    /// Radius of hyperbolic disk
    radius: RhgRadius,
    /// Optional specified number of bands used
    num_bands: Option<usize>,
}

impl Rhg {
    /// Creates a new generator
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Changes the alpha value
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Manually sets the radius of the disk
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = RhgRadius::Radius(radius);
        self
    }

    /// Manually sets the number of bands used
    pub fn num_bands(mut self, num_bands: usize) -> Self {
        self.num_bands = Some(num_bands);
        self
    }
}

impl NumNodesGen for Rhg {
    fn nodes(mut self, n: NumNodes) -> Self {
        self.nodes = n;
        self
    }
}

impl AverageDegreeGen for Rhg {
    fn avg_deg(mut self, deg: f64) -> Self {
        self.radius = RhgRadius::AvgDeg(deg);
        self
    }
}

impl GraphGenerator for Rhg {
    fn stream<R: Rng>(&self, rng: &mut R) -> impl Iterator<Item = Edge> {
        RhgGenerator::new(rng, self.nodes, self.radius, self.alpha, self.num_bands)
    }
}

/// Provides an iterator over edges between coordinates in hyperbolic space that are close to each other
#[derive(Debug, Clone)]
pub struct RhgGenerator {
    /// List of already sampled coordinates for each node the graph
    coordinates: Vec<Coord>,
    /// Prefix-Sum over number of points in each band
    band_bounds: Vec<NumNodes>,
    /// Pre-computed `cosh(x), sinh(x)` values for all band limits
    band_cosh_sinh: Vec<(f64, f64)>,
    /// Last `cosh` entry of `band_cosh_sinh`
    radius_cosh: f64,

    // --- Inner Iterator State
    /// Current coordinate (index) to be considered
    curr_coord_index: usize,
    /// Phi-Range in the current band which has to be considered for the current coordinate
    /// Structured as (size, u.phi - size, u.phi + size) with respect to module TAU
    safe_values: (f64, f64, f64),
    /// Current band that is considered
    curr_bid: usize,
    /// Range bounds for self.coordinates for the current bound
    slab_bounds: (usize, usize),
    /// Range bounds for self.coordinates for coordinates that must still be checked.
    /// As the phi-range is to be taken modulo TAU, slab_pointer.0 > slab_pointer.1 is possible
    slab_pointer: (usize, usize),
}

impl RhgGenerator {
    /// Creates a new RhgGenerator by computing a radius and a set of coordinates in hyperbolic space
    pub fn new<R: Rng>(
        rng: &mut R,
        n: NumNodes,
        radius: RhgRadius,
        alpha: f64,
        num_bands: Option<usize>,
    ) -> Self {
        assert!(alpha > 0.0);

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

    /// Computes for a given number of nodes `n`, average degree `k` and `alpha` the fitting radius
    ///
    /// Adopted from `NetworKIT`
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

    /// Samples `n` random coordinates (`Coord`) in hyperbolic space.
    /// Returns these coordinates as well as a list that indicates how many points lie on a
    /// specific band for given band limits.
    fn sample_coordinates(
        rng: &mut impl Rng,
        n: NumNodes,
        disk_rad: f64,
        alpha: f64,
        band_limits: &[f64],
    ) -> (Vec<Coord>, Vec<NumNodes>) {
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

    /// Binary-search a partition of points to find `val`
    /// In previous benchmarks, this showed to be faster than the `std`-implementation
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

    /// `safe_values` are used to find the borders of the inner circle, wherein every node is definitely near enough to u.
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

    /// Searches for the next edge in the current State.
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

    /// Computes the bounds (in terms of indexes in `self.coordinates`) for which we have to
    /// consider neighbors for the current node
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
