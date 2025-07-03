use num::{One, Zero};

mod geometric;
mod multi_traits;
mod set;
mod sliced_buffer;

pub use geometric::*;
pub use multi_traits::*;
pub use set::*;
pub use sliced_buffer::*;

/// Helper trait for probalities
pub trait Probability {
    /// Returns *true* if the probality is valid (ie. between `0` and `1`)
    fn is_valid_probility(&self) -> bool;
}

impl<P: Zero + One + PartialOrd> Probability for P {
    fn is_valid_probility(&self) -> bool {
        Self::zero().le(self) && Self::one().ge(self)
    }
}
