use std::{
    fmt::{Debug, Display},
    num::{NonZero, ParseIntError},
    ops::*,
};

#[cfg(feature = "node_range")]
use std::iter::Step;

#[cfg(not(feature = "node_range"))]
pub use node_range::*;

use num::*;
use stream_bitset::bitset::BitSetImpl;

/// We use a NonZero-Wrapper around `RawNode` to allow `Option<Node>` to be same-sized as `RawNode`
/// as many of our algorithms use `Option<Node>`.
///
/// Thus, Nodes are 1-indexed and and store the original value plus one instead, meaning the
/// `RawNode::MAX` is no longer an allowed value a node can have.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node(NonZero<RawNode>);

/// 0-indexed counterpart of `Node`
pub type RawNode = u32;

/// There can be at most `2^32 - 1` nodes in a graph!
/// If wanting to use `Option<NumNodes>`, consider using `Option<Node>` instead to save memory.
pub type NumNodes = RawNode;

/// BitSet for Nodes
pub type NodeBitSet = BitSetImpl<Node>;

/// BitSet for NumNodes
pub type NumNodesBitSet = BitSetImpl<NumNodes>;

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // We want to have 1-indexed outputs
        write!(f, "{}", self.raw())
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::MIN
    }
}

impl PartialEq<RawNode> for Node {
    fn eq(&self, other: &RawNode) -> bool {
        self.raw().eq(other)
    }
}

impl PartialOrd<RawNode> for Node {
    fn partial_cmp(&self, other: &RawNode) -> Option<std::cmp::Ordering> {
        self.raw().partial_cmp(other)
    }
}

impl Node {
    /// The maximum possible value a node can have
    pub const MAX: Self = Node(NonZero::new(RawNode::MAX).unwrap());

    /// The minimum possible value a node can have
    pub const MIN: Self = Node(NonZero::new(1).unwrap());

    /// 0 is the smallest value (but stored as 1)
    pub const ZERO: Self = Self::MIN;

    /// 1 is stored as 2
    pub const ONE: Self = Node(NonZero::new(2).unwrap());

    /// Creates a new node from a 0-indexed RawNode
    /// If the raw value is RawNode::MAX, this will panic.
    #[inline]
    pub const fn new(u: RawNode) -> Self {
        Node(NonZero::new(u.wrapping_add(1)).unwrap())
    }

    /// Tries to create a new node and returns *None* if the value is too big (ie. equals `RawNode::MAX`)
    #[inline]
    pub const fn new_checked(u: RawNode) -> Option<Self> {
        if u == RawNode::MAX {
            return None;
        }
        // SAFETY: `1 <= u + 1` for any `u` as well as `u < RawNode::MAX`
        unsafe { Some(Node(NonZero::new(u.unchecked_add(1)).unwrap_unchecked())) }
    }

    /// Tries to create a new node without increasing the value and returns *None* if `u == 0`
    #[inline]
    pub const fn new_unchanged(u: RawNode) -> Option<Self> {
        if u == 0 {
            return None;
        }
        // SAFETY: at this point, `u > 0` is guaranteed
        unsafe { Some(Node(NonZero::new(u).unwrap_unchecked())) }
    }

    /// Creates a new Node without checking for overflow
    ///
    /// # SAFETY
    /// The caller is responsible for making sure that `u != RawNode::MAX`
    #[inline]
    pub const unsafe fn new_unchecked(u: RawNode) -> Self {
        // SAFETY: caller is responsible for SAFETY
        unsafe { Node(NonZero::new(u.unchecked_add(1)).unwrap_unchecked()) }
    }

    /// Creates a new Node saturating the value if `u == RawNode::MAX`
    #[inline]
    pub const fn new_saturated(u: RawNode) -> Self {
        // SAFETY: `1 <= u + 1`
        unsafe { Node(NonZero::new(u.saturating_add(1)).unwrap_unchecked()) }
    }

    /// Gets the 0-indexed value of the node
    #[inline]
    pub const fn raw(&self) -> RawNode {
        // SAFETY: `self.0` has type `NonZeroU32` and `self.0.raw()` is at least 1.
        unsafe { self.0.get().unchecked_sub(1) }
    }

    /// Gets the inner 1-indexed value of the node
    #[inline]
    pub const fn inner(&self) -> RawNode {
        self.0.get()
    }
}

impl From<RawNode> for Node {
    fn from(value: RawNode) -> Self {
        Self::new(value)
    }
}

impl From<&RawNode> for Node {
    fn from(value: &RawNode) -> Self {
        Self::new(*value)
    }
}

// ---------- std::ops Implementors ----------

/// Implement Ops-Traits that might produce `RawNode::MAX`
macro_rules! impl_checked_ops {
    ($($trait:ident $fn:ident),+) => {
        $(
            impl $trait for Node {
                type Output = Self;

                #[inline]
                fn $fn(self, other: Self) -> Self::Output {
                    #[cfg(not(feature = "saturate_node_overflow"))]
                    {
                        Self::new_checked(self.raw().$fn(other.raw())).unwrap_or_else(|| {
                            panic!("Overflow when trying to apply {}.{}({})", self.raw(), stringify!($fn), other.raw())
                        })
                    }

                    #[cfg(feature = "saturate_node_overflow")]
                    Self::new_saturated(self.raw().$fn(other.raw()))
                }
            }

            // Also allow applying RawNode directly
            impl $trait<RawNode> for Node {
                type Output = Self;

                #[inline]
                fn $fn(self, other: RawNode) -> Self::Output {
                    #[cfg(not(feature = "saturate_node_overflow"))]
                    {
                        Self::new_checked(self.raw().$fn(other)).unwrap_or_else(|| {
                            panic!("Overflow when trying to apply {}.{}({other})", self.raw(), stringify!($fn))
                        })
                    }

                    #[cfg(feature = "saturate_node_overflow")]
                    Self::new_saturated(self.raw().$fn(other))
                }
            }
        )+
    };
}

/// Implements Ops-Traits that are guaranteed to not produce `RawNode::MAX`
macro_rules! impl_unchecked_ops {
    ($($trait:ident $fn:ident),+) => {
        $(
            impl $trait for Node {
                type Output = Self;

                #[inline]
                fn $fn(self, other: Self) -> Self::Output {
                    // SAFETY: See macro-implemetors for why this is safe!
                    unsafe { Self::new_unchecked(self.raw().$fn(other.raw())) }
                }
            }

            impl $trait<RawNode> for Node {
                type Output = Self;

                #[inline]
                fn $fn(self, other: RawNode) -> Self::Output {
                    // SAFETY: See macro-implemetors for why this is safe!
                    unsafe { Self::new_unchecked(self.raw().$fn(other)) }
                }
            }
        )+
    };
}

impl_checked_ops!(Add add, Mul mul, BitOr bitor, BitXor bitxor, Shl shl);

// SAFETY: all those operations can only decrease `self` which is valid
impl_unchecked_ops!(Sub sub, Div div, Rem rem, BitAnd bitand, Shr shr);

/// Implementors for `std::ops::*Assign`-Traits
macro_rules! impl_ops_assigns {
    ($($trait:ident $fn:ident $org_fn:ident),+) => {
        $(
            impl $trait for Node {
                #[inline]
                fn $fn(&mut self, rhs: Self) {
                    // As NonZero<T> prohibits mutable access to the underlying value, we must
                    // extract the value and overwrite the original value itself
                    *self = self.$org_fn(rhs);
                }
            }

            impl $trait<RawNode> for Node {
                #[inline]
                fn $fn(&mut self, rhs: RawNode) {
                    // As NonZero<T> prohibits mutable access to the underlying value, we must
                    // extract the value and overwrite the original value itself
                    *self = self.$org_fn(rhs);
                }
            }
        )+
    };
}

impl_ops_assigns!(
    AddAssign add_assign add,
    BitAndAssign bitand_assign bitand,
    BitOrAssign bitor_assign bitor,
    BitXorAssign bitxor_assign bitxor,
    DivAssign div_assign div,
    MulAssign mul_assign mul,
    RemAssign rem_assign rem,
    ShlAssign shl_assign shl,
    ShrAssign shr_assign shr,
    SubAssign sub_assign sub
);

impl Shl<usize> for Node {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        #[cfg(not(feature = "saturate_node_overflow"))]
        {
            Self::new_checked(self.raw().shr(rhs)).unwrap_or_else(|| {
                panic!("Overflow when trying to apply {}.shl({rhs})", self.raw())
            })
        }

        #[cfg(feature = "saturate_node_overflow")]
        Self::new_saturated(self.raw().shl(rhs))
    }
}

impl ShlAssign<usize> for Node {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        // As NonZeroU32 prohibits mutable access to the underlying value, we must
        // extract the value and overwrite the original value itself
        *self = self.shl(rhs);
    }
}

impl Shr<usize> for Node {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        // SAFETY: `shr` can only decrease the value of the Node thus not leading to overflow
        unsafe { Self::new_unchecked(self.raw().shl(rhs)) }
    }
}

impl ShrAssign<usize> for Node {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        // As NonZeroU32 prohibits mutable access to the underlying value, we must
        // extract the value and overwrite the original value itself
        *self = self.shl(rhs);
    }
}

impl Not for Node {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        // The only case where this fails if `self` stores `0`
        // TBD: maybe set to `RawNode::MAX` if feature `saturate_node_overflow` is enabled
        Node::new_checked(self.raw().not()).expect("Can not apply Not on `0`")
    }
}

// ---------- num Implementors ----------

macro_rules! impl_checked_ops {
    ($($trait:ident $fn:ident),+) => {
        $(
            impl $trait for Node {
                #[inline]
                fn $fn(&self, v: &Self) -> Option<Self> {
                    Self::new_checked(self.raw().$fn(v.raw())?)
                }
            }
        )+
    };
}

impl_checked_ops!(
    CheckedAdd checked_add,
    CheckedSub checked_sub,
    CheckedMul checked_mul,
    CheckedDiv checked_div
);

impl Saturating for Node {
    #[inline]
    fn saturating_add(self, v: Self) -> Self {
        Self::new_saturated(self.raw().saturating_add(v.raw()))
    }

    #[inline]
    fn saturating_sub(self, v: Self) -> Self {
        // SAFETY: saturating-subtracting a valid value from another value can never underflow/overflow
        unsafe { Self::new_unchecked(self.raw().saturating_sub(v.raw())) }
    }
}

impl FromPrimitive for Node {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        Self::new_checked(RawNode::from_i64(n)?)
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        Self::new_checked(RawNode::from_u64(n)?)
    }
}

impl ToPrimitive for Node {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.raw().to_i64()
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.raw().to_u64()
    }
}

impl NumCast for Node {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        // Abstract over `u64` in case that `RawNode` will be set to `u64`
        Self::new_checked(RawNode::from_u64(n.to_u64()?)?)
    }
}

impl Zero for Node {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

impl One for Node {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl Bounded for Node {
    #[inline]
    fn min_value() -> Self {
        Self::MIN
    }

    #[inline]
    fn max_value() -> Self {
        Self::MAX
    }
}

pub enum NodeParseError {
    Overflow,
    Parse(ParseIntError),
}

impl Num for Node {
    type FromStrRadixErr = NodeParseError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Node::new_checked(RawNode::from_str_radix(str, radix).map_err(NodeParseError::Parse)?)
            .ok_or(NodeParseError::Overflow)
    }
}

impl Unsigned for Node {}

macro_rules! impl_fn_self_to_u32 {
    ($($fn:ident),+) => {
        $(
            #[inline]
            fn $fn(self) -> u32 {
                self.raw().$fn()
            }
        )+
    };
}

macro_rules! impl_fn_self_to_self {
    ($($fn:ident),+) => {
        $(
            #[inline]
            fn $fn(self) -> Self {
                #[cfg(not(feature = "saturate_node_overflow"))]
                {
                    Self::new_checked(self.raw().$fn()).unwrap_or_else(|| {
                        panic!("Overflow when trying to apply {}.{}()", self.raw(), stringify!($fn))
                    })
                }

                #[cfg(feature = "saturate_node_overflow")]
                Self::new_saturated(self.raw().$fn())
            }
        )+
    };
}

macro_rules! impl_fn_self_arg_to_self {
    ($($fn:ident),+) => {
        $(
            #[inline]
            fn $fn(x: Self) -> Self {
                #[cfg(not(feature = "saturate_node_overflow"))]
                {
                    Self::new_checked(RawNode::$fn(x.raw())).unwrap_or_else(|| {
                        panic!("Overflow when trying to apply Node::{}({})", stringify!($fn), x.raw())
                    })
                }

                #[cfg(feature = "saturate_node_overflow")]
                Self::new_saturated(RawNode::$fn(x.raw()))
            }
        )+
    };
}

macro_rules! impl_fn_self_u32_to_self {
    ($($fn:ident),+) => {
        $(
            #[inline]
            fn $fn(self, n: u32) -> Self {
                #[cfg(not(feature = "saturate_node_overflow"))]
                {
                    Self::new_checked(self.raw().$fn(n)).unwrap_or_else(|| {
                        panic!("Overflow when trying to apply {}.{}({n})", self.raw(), stringify!($fn))
                    })
                }

                #[cfg(feature = "saturate_node_overflow")]
                Self::new_saturated(self.raw().$fn(n))
            }
        )+
    };
}

impl PrimInt for Node {
    impl_fn_self_to_u32!(count_ones, count_zeros, leading_zeros, trailing_zeros);
    impl_fn_self_to_self!(swap_bytes, to_be, to_le);
    impl_fn_self_arg_to_self!(from_be, from_le);
    impl_fn_self_u32_to_self!(
        rotate_left,
        rotate_right,
        signed_shl,
        signed_shr,
        unsigned_shl,
        unsigned_shr,
        pow
    );
}

macro_rules! impl_fn_self_self_to_self {
    ($($fn:ident),+) => {
        $(
            fn $fn(&self, other: &Self) -> Self {
                // SAFETY: See macro-implemetors for why this is safe!
                //
                // Use `Integer::$fn` instead of `self.$fn` as `div_floor` might be added to the
                // standard library in the future
                unsafe { Node::new_unchecked(Integer::$fn(&self.raw(), &other.raw())) }
            }
        )+
    };
}

impl Integer for Node {
    // SAFETY: all these function satisfy `self >= self.fn(x)`
    impl_fn_self_self_to_self!(div_floor, mod_floor, gcd);

    #[inline]
    fn lcm(&self, other: &Self) -> Self {
        #[cfg(not(feature = "saturate_node_overflow"))]
        {
            Self::new_checked(self.raw().lcm(&other.raw())).unwrap_or_else(|| {
                panic!(
                    "Overflow when trying to apply {}.lcm({})",
                    self.raw(),
                    other.raw()
                )
            })
        }

        // Saturation does not really make sense for `lcm`
        #[cfg(feature = "saturate_node_overflow")]
        Self::new_saturated(self.raw().lcm(&other.raw()))
    }

    #[inline]
    fn is_multiple_of(&self, other: &Self) -> bool {
        self.raw().is_multiple_of(other.raw())
    }

    #[inline]
    fn is_even(&self) -> bool {
        self.raw().is_even()
    }

    #[inline]
    fn is_odd(&self) -> bool {
        self.raw().is_odd()
    }

    #[inline]
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (q, r) = self.raw().div_rem(&other.raw());
        // SAFETY: `q <= self, r <= self`
        unsafe { (Node::new_unchecked(q), Node::new_unchecked(r)) }
    }
}

/// Implement `Step` to enable `Range<Node>` as a viable type
#[cfg(feature = "node_range")]
impl Step for Node {
    #[inline]
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        <RawNode as Step>::steps_between(&start.raw(), &end.raw())
    }

    #[inline]
    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Node::new_unchanged(<RawNode as Step>::forward_checked(start.raw(), count)?)
    }

    #[inline]
    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Node::new_checked(<RawNode as Step>::backward_checked(start.raw(), count)?)
    }
}

/// Custom Range<Node> Implementation
#[cfg(not(feature = "node_range"))]
mod node_range {
    use std::iter::FusedIterator;

    use super::*;

    /// Custom RangeInclusive<Node> Implementation
    ///
    /// # SAFETY
    /// `self.end` should **never** be manually modified as it is always assumed to be a valid
    /// state for `Node`
    #[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
    pub struct NodeRange {
        cur: RawNode,
        end: RawNode,
    }

    impl NodeRange {
        /// Returns `self.cur` as a Node without checking for bounds
        #[inline]
        const unsafe fn cur_node(&self) -> Node {
            unsafe { Node::new_unchecked(self.cur) }
        }

        /// Returns `self.end` as a Node without checking for bounds
        #[inline]
        const unsafe fn end_node(&self) -> Node {
            unsafe { Node::new_unchecked(self.end) }
        }

        /// Creates a new exclusive NodeRange
        #[inline]
        pub const fn new(start: Node, end: Node) -> Self {
            // An exclusive range `(X, 0)` is always empty
            if end.raw() == 0 {
                return Self { cur: 1, end: 0 };
            }

            Self {
                cur: start.raw(),
                // SAFETY: `end.raw() > 0` is guaranteed here
                end: unsafe { end.raw().unchecked_sub(1) },
            }
        }

        /// Creates a new inclusive NodeRange
        #[inline]
        pub const fn new_inclusive(start: Node, end: Node) -> Self {
            Self {
                cur: start.raw(),
                end: end.raw(),
            }
        }

        /// Creates a new exclusive range from `0` to `end`
        #[inline]
        pub const fn new_to(end: Node) -> Self {
            Self::new(Node::ZERO, end)
        }

        /// Creates a new inclusive range from `0` to `end`
        #[inline]
        pub const fn new_to_inclusive(end: Node) -> Self {
            Self::new_inclusive(Node::ZERO, end)
        }

        /// Creates a new inclusive range from `start` to `Node::MAX`
        #[inline]
        pub const fn new_from(start: Node) -> Self {
            Self::new_inclusive(start, Node::MAX)
        }

        /// Returns *true* if `item` is contained in the NodeRange.
        #[inline]
        pub fn contains<U>(&self, item: &U) -> bool
        where
            Node: PartialOrd<U>,
            U: PartialOrd<Node>,
        {
            if self.is_empty() {
                return false;
            }

            // SAFETY: `self.cur <= self.end <= Node::MAX` as we assume `self.end` to be a valid Node
            unsafe { &self.cur_node() <= item && item <= &self.end_node() }
        }

        /// Returns *true* if the range is empty  
        #[inline]
        pub const fn is_empty(&self) -> bool {
            self.cur > self.end
        }
    }

    impl Iterator for NodeRange {
        type Item = Node;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            (self.cur <= self.end).then_some(unsafe {
                // SAFETY: `self.cur <= self.end <= Node::MAX` implying that
                // (1) `self.cur` is a valid state for `Node`
                // (2) `self.cur < RawNode::MAX`
                let cur = self.cur_node();
                self.cur = self.cur.unchecked_add(1);

                cur
            })
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            if self.cur <= self.end {
                let size = (self.end - self.cur + 1) as usize;
                (size, Some(size))
            } else {
                (0, Some(0))
            }
        }
    }

    impl DoubleEndedIterator for NodeRange {
        fn next_back(&mut self) -> Option<Self::Item> {
            (self.cur <= self.end).then_some(unsafe {
                // SAFETY: `self.cur <= self.end <= Node::MAX`
                let end = self.end_node();

                if self.end > 0 {
                    // SAFETY: `self.end > 0`
                    self.end = self.end.unchecked_sub(1);
                } else {
                    // SAFETY: `self.cur <= self.end <= 0` implying `self.cur == self.end == 0` and
                    // we increase `self.cur` instead as the Range will be empty in the next step
                    self.cur = self.cur.unchecked_add(1);
                }

                end
            })
        }
    }

    impl ExactSizeIterator for NodeRange {}
    impl FusedIterator for NodeRange {}
}
