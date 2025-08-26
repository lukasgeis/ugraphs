use rand::Rng;
use rand_distr::{Distribution, Geometric, StandardGeometric};

use crate::utils::Probability;

/// A geometric distribution.
/// As the case for `p = 1/2` can be siginificantly sped up by using `StandardGeometric` instead of
/// `Geometric`, we abstract over both using an enum as `p = 1/2` is the base value for `G(n)`
/// graphs and thus used often.
#[derive(Debug, Copy, Clone)]
pub enum GeometricDistribution {
    /// General geometric distribution
    General(Geometric),
    /// Geometric distribution for `p = 1/2`
    OneHalf(StandardGeometric),
}

impl Distribution<u64> for GeometricDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        match self {
            GeometricDistribution::General(distr) => distr.sample(rng),
            GeometricDistribution::OneHalf(distr) => distr.sample(rng),
        }
    }
}

impl GeometricDistribution {
    /// Creates a new geometric distribution from a given probability
    pub fn from_prob(prob: f64) -> Self {
        if prob == 0.5 {
            Self::OneHalf(StandardGeometric)
        } else {
            Self::General(Geometric::new(prob).unwrap())
        }
    }

    /// Given a probability, returns a geometric distribution and *true* if the distribution is inversed:
    /// For `p > 0.5`, it is beneficial to invert (`1.0 - p`) the distribution and iterate over the
    /// non-successes in order to minimize draws from the distribution itself.
    pub fn from_prob_with_inv(prob: f64) -> (Self, bool) {
        let mut inv = false;

        let distr = if prob == 0.5 {
            Self::OneHalf(StandardGeometric)
        } else if prob < 0.5 {
            Self::General(Geometric::new(prob).unwrap())
        } else {
            inv = true;
            Self::General(Geometric::new(1.0 - prob).unwrap())
        };

        (distr, inv)
    }
}

/// A geometric jumper starts at `0` and a step size from a geometric distribution.
/// It keeps doing that until an optional stop value is reached.
#[derive(Debug, Copy, Clone)]
pub struct GeometricJumper {
    /// Probability of the geometric distribution
    prob: f64,
    /// Stop if this value is exceeded
    stop: Option<u64>,
}

impl GeometricJumper {
    /// Creates a new geometric jumper from a probability with no stop value
    pub fn new(prob: f64) -> Self {
        assert!(prob.is_valid_probility());

        Self { prob, stop: None }
    }

    /// Updates the stop value of the jumper
    pub fn stop_at(mut self, stop: u64) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Creates an iterator of geometric jumps starting at `0`
    pub fn iter<'a, R: Rng>(self, rng: &'a mut R) -> GeometricJumperIter<'a, R> {
        let (distr, inv) = GeometricDistribution::from_prob_with_inv(self.prob);

        GeometricJumperIter {
            geom_distr: distr,
            rng,
            stop: self.stop,
            inv,
            cur: 0,
            next_inv: 0,
        }
    }
}

/// An iterator over geometric jumps starting at `0` with an optional stop value
#[derive(Debug)]
pub struct GeometricJumperIter<'a, R>
where
    R: Rng,
{
    geom_distr: GeometricDistribution,
    rng: &'a mut R,
    stop: Option<u64>,
    cur: u64,
    inv: bool,
    next_inv: u64,
}

impl<'a, R> GeometricJumperIter<'a, R>
where
    R: Rng,
{
    /// Updates the stop value (inplace)
    pub fn change_stop(&mut self, stop: u64) {
        self.stop = Some(stop);
    }

    /// Updates the probability (inplace)
    pub fn change_prob(&mut self, prob: f64) {
        assert!(prob.is_valid_probility());

        let (distr, inv) = GeometricDistribution::from_prob_with_inv(prob);
        self.geom_distr = distr;
        self.inv = inv;
        self.next_inv = self.cur;
    }

    /// Performs a geometric jump
    pub fn jump(&mut self) -> Option<u64> {
        if self.cur == u64::MAX {
            return None;
        }

        // Check if stop has been exceeded
        let stop = self.stop.unwrap_or(u64::MAX);
        if self.cur > stop {
            return None;
        }

        // Check if the next value are already queued
        if self.cur < self.next_inv {
            self.cur += 1;
            if self.cur > stop {
                return None;
            }
            return Some(self.cur - 1);
        }

        // Compute jump
        let next = self.rng.sample(self.geom_distr);
        if next == u64::MAX {
            if self.inv {
                self.next_inv = u64::MAX;
                self.cur += 1;
                return Some(self.cur - 1);
            }
            self.cur = u64::MAX;
            return None;
        }

        // Invert jump if needed
        if self.inv {
            self.next_inv = match self.next_inv.checked_add(next + 1) {
                Some(x) => x,
                None => u64::MAX,
            };

            self.cur += 1;
            return Some(self.cur - 1);
        }

        // Jump
        self.cur = match self.cur.checked_add(next + 1) {
            Some(x) => x,
            None => {
                self.cur = u64::MAX;
                return None;
            }
        };

        if self.cur > stop {
            return None;
        }

        Some(self.cur - 1)
    }
}

impl<'a, R> Iterator for GeometricJumperIter<'a, R>
where
    R: Rng,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.jump()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use super::*;

    #[test]
    fn wrong_prob() {
        for prob in [-10.0, -0.001, 1.0001, 3.4] {
            assert!(std::panic::catch_unwind(|| GeometricJumper::new(prob)).is_err());
        }
    }

    #[test]
    fn edge_cases() {
        let rng = &mut Pcg64Mcg::seed_from_u64(3);

        // p = 1.0
        for stop in [3, 10] {
            assert_eq!(
                GeometricJumper::new(1.0).stop_at(stop).iter(rng).count(),
                stop as usize
            );

            assert_eq!(
                GeometricJumper::new(1.0)
                    .iter(rng)
                    .take(stop as usize)
                    .count(),
                stop as usize
            );
        }

        // p = 0.0
        assert_eq!(GeometricJumper::new(0.0).iter(rng).count(), 0);
    }

    #[test]
    fn change_prob() {
        let rng = &mut Pcg64Mcg::seed_from_u64(4);
        let mut iter = GeometricJumper::new(1.0).iter(rng);

        let mut res = Vec::new();
        for _ in 0..100 {
            res.push(iter.next().unwrap());
        }

        iter.change_prob(0.0);
        for x in iter {
            res.push(x);
        }

        assert_eq!(res, (0..100).collect::<Vec<u64>>());
    }

    #[test]
    fn occurences() {
        let rng = &mut Pcg64Mcg::seed_from_u64(5);

        let stop = 100u64;
        let mut occurences = vec![0; stop as usize];
        for _ in 0..1000 {
            for x in GeometricJumper::new(0.25).stop_at(stop).iter(rng) {
                occurences[x as usize] += 1;
            }
        }

        assert!(occurences.into_iter().all(|x| (150..350).contains(&x)));
    }
}
