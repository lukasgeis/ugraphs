/*!
# Geometric Jumper

This module provides utilities for efficiently generating *geometric jumps*,
which are used in the `G(n,p)` random graph model. In such models, the presence
of each edge is determined by independent Bernoulli trials with probability `p`.

Instead of drawing `n^2` independent coin flips, one can **jump directly between
successes** by sampling from a geometric distribution. This greatly reduces the
number of random draws and improves efficiency.

The [`GeometricJumper`] is the main utility here:
- It starts at index `0` and repeatedly adds a *geometric step size* drawn
  from a [`GeometricDistribution`].
- Optionally, a stop value can be set to avoid overshooting.
- The jumper produces an iterator (`GeometricJumperIter`) over successive jumps.

The implementation also optimizes the common case of `p = 1/2` by using the
faster [`StandardGeometric`] distribution.
*/

use rand::Rng;
use rand_distr::{Distribution, Geometric, StandardGeometric};

use super::Probability;

/// Wrapper around different variants of a geometric distribution.
///
/// Normally, a geometric distribution is parametrized by a success probability `p`.
/// For the common case `p = 1/2`, this type uses [`StandardGeometric`], which is
/// significantly faster than the general [`Geometric`] distribution.
///
/// This abstraction is useful in `G(n)` random graphs, where `p = 1/2` is used.
#[derive(Debug, Copy, Clone)]
pub enum GeometricDistribution {
    /// General case geometric distribution with arbitrary probability `p`.
    General(Geometric),
    /// Specialized geometric distribution for the case `p = 1/2`.
    OneHalf(StandardGeometric),
}

impl Distribution<u64> for GeometricDistribution {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        match self {
            GeometricDistribution::General(distr) => distr.sample(rng),
            GeometricDistribution::OneHalf(distr) => distr.sample(rng),
        }
    }
}

impl GeometricDistribution {
    /// Constructs a [`GeometricDistribution`] from a probability `p`.
    ///
    /// Uses [`StandardGeometric`] if `p == 0.5`, otherwise falls back to a general
    /// [`Geometric`] distribution.
    ///
    /// # Panics
    /// Panics if `p` is not in `[0, 1]`.
    pub fn from_prob(prob: f64) -> Self {
        assert!(prob.is_valid_probility());

        if prob == 0.5 {
            Self::OneHalf(StandardGeometric)
        } else {
            Self::General(Geometric::new(prob).unwrap())
        }
    }

    /// Constructs a [`GeometricDistribution`] from a probability `p`, possibly inverted.
    ///
    /// For `p > 0.5`, it is more efficient to sample from the complementary
    /// distribution `Geometric(1.0 - p)` and invert the meaning of successes.
    /// This reduces the expected number of random draws.
    ///
    /// Returns the distribution and a `bool` flag indicating whether inversion
    /// should be applied.
    ///
    /// # Panics
    /// Panics if `p` is not in `[0, 1]`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let (geom, inv) = GeometricDistribution::from_prob_with_inv(0.8);
    /// assert!(inv);
    /// ```
    pub fn from_prob_with_inv(prob: f64) -> (Self, bool) {
        assert!(prob.is_valid_probility());

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

/// A geometric jumper that generates indices by repeatedly adding
/// steps drawn from a geometric distribution.
///
/// The jumper starts at `0` and continues generating jumps until
/// an optional stop value is reached. It is mainly used for
/// efficiently sampling edges in `G(n,p)` random graphs.
///
/// Use [`GeometricJumper::iter`] to create an iterator over jumps.
///
/// # Examples
/// ```
/// use ugraphs::utils::geometric::*;
///
/// let mut rng = rand::rng();
/// let jumper = GeometricJumper::new(0.5).stop_at(10);
/// let jumps: Vec<_> = jumper.iter(&mut rng).collect();
/// assert!(jumps.iter().all(|&x| x <= 10));
/// ```
#[derive(Debug, Copy, Clone)]
pub struct GeometricJumper {
    /// Probability of the geometric distribution
    prob: f64,
    /// Stop if this value is exceeded
    stop: Option<u64>,
}

impl GeometricJumper {
    /// Creates a new [`GeometricJumper`] with probability `p` and no stop value.
    ///
    /// # Panics
    /// Panics if `p` is not a valid probability (not in `[0, 1]`).
    pub fn new(prob: f64) -> Self {
        assert!(prob.is_valid_probility());

        Self { prob, stop: None }
    }

    /// Updates the stop value of the jumper in-place.
    ///
    /// After the stop value is exceeded, the iterator terminates.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let mut rng = rand::rng();
    /// let mut jumper = GeometricJumper::new(0.5);
    /// jumper.set_stop_at(5);
    /// let vals: Vec<_> = jumper.iter(&mut rng).collect();
    /// assert!(vals.iter().all(|&x| x <= 5));
    /// ```
    pub fn set_stop_at(&mut self, stop: u64) {
        self.stop = Some(stop);
    }

    /// Builder-style version of [`GeometricJumper::set_stop_at`].
    ///
    /// Consumes the jumper, sets the stop value, and returns it.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let mut rng = rand::rng();
    /// let jumper = GeometricJumper::new(0.5).stop_at(7);
    /// let vals: Vec<_> = jumper.iter(&mut rng).collect();
    /// assert!(vals.iter().all(|&x| x <= 7));
    /// ```
    pub fn stop_at(mut self, stop: u64) -> Self {
        self.set_stop_at(stop);
        self
    }

    /// Creates an iterator over geometric jumps starting at `0`.
    ///
    /// The iterator yields successive positions until the stop value (if any)
    /// is reached.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let mut rng = rand::rng();
    /// let jumper = GeometricJumper::new(0.5).stop_at(5);
    /// let vals: Vec<_> = jumper.iter(&mut rng).collect();
    /// assert!(vals.iter().all(|&x| x <= 5));
    /// ```
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

/// Iterator returned by [`GeometricJumper::iter`].
///
/// Internally tracks the current position, the geometric distribution,
/// and an optional stop value.
///
/// Implements [`Iterator<Item = u64>`].
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
    /// Updates the stop value in-place.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let mut rng = rand::rng();
    /// let mut iter = GeometricJumper::new(0.5).iter(&mut rng);
    /// iter.change_stop(3);
    /// let vals: Vec<_> = iter.collect();
    /// assert!(vals.iter().all(|&x| x <= 3));
    /// ```
    pub fn change_stop(&mut self, stop: u64) {
        self.stop = Some(stop);
    }

    /// Updates the probability of the underlying geometric distribution in-place.
    ///
    /// The jumper will continue from the current position but use the new distribution.
    ///
    /// # Panics
    /// Panics if `p` is not a valid probability (not in `[0, 1]`).
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let mut rng = rand::rng();
    /// let mut iter = GeometricJumper::new(0.5).iter(&mut rng);
    /// iter.change_prob(0.25);
    /// ```
    pub fn change_prob(&mut self, prob: f64) {
        assert!(prob.is_valid_probility());

        let (distr, inv) = GeometricDistribution::from_prob_with_inv(prob);
        self.geom_distr = distr;
        self.inv = inv;
        self.next_inv = self.cur;
    }

    /// Performs a single geometric jump and returns the next index, or `None` if
    /// the stop value has been exceeded.
    ///
    /// Usually called internally by the iterator, but can also be invoked directly.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::geometric::*;
    ///
    /// let mut rng = rand::rng();
    /// let mut iter = GeometricJumper::new(0.5).stop_at(5).iter(&mut rng);
    /// while let Some(val) = iter.jump() {
    ///     assert!(val <= 5);
    /// }
    /// ```
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
            if self.cur > stop {
                return None;
            }
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
