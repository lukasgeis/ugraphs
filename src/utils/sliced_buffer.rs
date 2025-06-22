use std::ops::{Index, IndexMut};

use stream_bitset::PrimIndex;

// Implements the core data structure of CSR (i.e. a data vector `buffer`
// and an non-decreasing index vector offsets, where the start of the
// i-th slice in `buffer` is stored at `offsets[i]`.
//
// This implementation verifies the following invariants at construction
// and to avoid repeated checks during accesses:
//  (0) `offset` has at least two elements
//  (1) `offset` is non-decreasing (i.e. produce a valid range) and
//  (2) `offset` stays within bounds of `buffer`
//
// The implementation is its own module to prevent the other data structures
// from manipulating the offsets vector directly, which may invalidate the aforementioned
// invariants.
#[derive(Debug, Clone)]
pub struct SlicedBuffer<T, I: PrimIndex> {
    buffer: Vec<T>,
    offsets: Vec<I>,
}

impl<T, I: PrimIndex> Default for SlicedBuffer<T, I> {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            offsets: vec![I::zero(), I::zero()],
        }
    }
}

impl<T, I: PrimIndex> SlicedBuffer<T, I> {
    /// Constructs the SlicedBuffer and panics if one of the three
    /// invariants on offset are violated.
    pub fn new(buffer: Vec<T>, offsets: Vec<I>) -> Self {
        assert!(offsets.len() > 1);
        assert!(offsets.len() - 1 <= I::max_value().to_usize().unwrap());
        assert!(offsets.is_sorted());
        assert!(offsets.last().unwrap().to_usize().unwrap() <= buffer.len());

        Self { buffer, offsets }
    }

    /// Returns the number of slices as `usize`#
    #[allow(clippy::len_without_is_empty)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        // Cannot underflow since `self.offset` has at least two entries
        unsafe { self.offsets.len().unchecked_sub(1) }
    }

    /// Returns the number of slices as `Idx: PrimInt`
    #[inline(always)]
    pub fn number_of_slices<Idx: PrimIndex>(&self) -> Idx {
        Idx::from_usize(self.len()).unwrap()
    }

    /// Returns the number of entries in the buffer as `I`
    #[inline(always)]
    pub fn number_of_entries(&self) -> I {
        I::from_usize(self.buffer.len()).unwrap()
    }

    /// Returns the size of a slice of a given index
    #[inline(always)]
    pub fn size_of<Idx: PrimIndex>(&self, u: Idx) -> I {
        self.offsets[u.to_usize().unwrap() + 1] - self.offsets[u.to_usize().unwrap()]
    }

    /// Returns the complete buffer
    #[inline(always)]
    pub fn raw_buffer_slice(&self) -> &[T] {
        &self.buffer
    }

    /// Returns the complete `offsets`-slice
    #[inline(always)]
    pub fn raw_offset_slice(&self) -> &[I] {
        &self.offsets
    }

    /// Grants mutable access to two different slices concurrently
    #[inline(always)]
    pub fn double_mut<Idx: PrimIndex>(&mut self, u: Idx, v: Idx) -> (&mut [T], &mut [T]) {
        let (u, v) = (u.to_usize().unwrap(), v.to_usize().unwrap());

        assert_ne!(u, v);
        if u < v {
            let v_off = self.offsets[v].to_usize().unwrap();
            let (beg, end) = self.buffer.split_at_mut(v_off);

            let u_start = self.offsets[u].to_usize().unwrap();
            let u_end = self.offsets[u + 1].to_usize().unwrap();

            let v_len = self.offsets[v + 1].to_usize().unwrap() - v_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    beg.get_unchecked_mut(u_start..u_end),
                    end.get_unchecked_mut(0..v_len),
                )
            }
        } else {
            let u_off = self.offsets[u].to_usize().unwrap();
            let (beg, end) = self.buffer.split_at_mut(u_off);

            let v_start = self.offsets[v].to_usize().unwrap();
            let v_end = self.offsets[v + 1].to_usize().unwrap();

            let u_len = self.offsets[u + 1].to_usize().unwrap() - u_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    end.get_unchecked_mut(0..u_len),
                    beg.get_unchecked_mut(v_start..v_end),
                )
            }
        }
    }
}

impl<T, I: PrimIndex, Idx: PrimIndex> Index<Idx> for SlicedBuffer<T, I> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, idx: Idx) -> &Self::Output {
        let end = self.offsets[idx.to_usize().unwrap() + 1]
            .to_usize()
            .unwrap();
        let start = self.offsets[idx.to_usize().unwrap()].to_usize().unwrap();

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        unsafe { self.buffer.get_unchecked(start..end) }
    }
}

impl<T, I: PrimIndex, Idx: PrimIndex> IndexMut<Idx> for SlicedBuffer<T, I> {
    #[inline(always)]
    fn index_mut(&mut self, idx: Idx) -> &mut Self::Output {
        let end = self.offsets[idx.to_usize().unwrap() + 1]
            .to_usize()
            .unwrap();
        let start = self.offsets[idx.to_usize().unwrap()].to_usize().unwrap();

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        unsafe { self.buffer.get_unchecked_mut(start..end) }
    }
}

// Extends on the core data structure of CSR (i.e. a data vector `buffer`
// and an non-decreasing index vector offsets, where the start of the
// i-th slice in `buffer` is stored at `offsets[i]`.
//
// Stores an additional default-buffer which is immutable and the same length
// (and offsets) as the main buffer. Allows restoring of data by copying the
// default data into the main buffer.
//
// This implementation verifies the following invariants at construction
// and to avoid repeated checks during accesses:
//  (0) `offset` has at least two elements
//  (1) `offset` is non-decreasing (i.e. produce a valid range) and
//  (2) `offset` stays within bounds of `buffer`
//  (3) `buffer` and `default` have the same length
//
// The implementation is its own module to prevent other data structures
// from manipulating the offsets vector, which may invalidate the aforementioned
// invariants.
#[derive(Debug, Clone)]
pub struct SlicedBufferWithDefault<T: Clone, I: PrimIndex> {
    buffer: Vec<T>,
    default: Vec<T>,
    offsets: Vec<I>,
}

impl<T: Clone, I: PrimIndex> Default for SlicedBufferWithDefault<T, I> {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            default: Vec::new(),
            offsets: vec![I::zero(), I::zero()],
        }
    }
}

impl<T: Clone, I: PrimIndex> SlicedBufferWithDefault<T, I> {
    /// Constructs the SlicedBuffer and panics if one of the three
    /// invariants on offset are violated.
    pub fn new(buffer: Vec<T>, offsets: Vec<I>) -> Self {
        assert!(offsets.len() > 1);
        assert!(offsets.len() - 1 <= I::max_value().to_usize().unwrap());
        assert!(offsets.is_sorted());
        assert!(offsets.last().unwrap().to_usize().unwrap() <= buffer.len());

        Self {
            default: buffer.clone(),
            buffer,
            offsets,
        }
    }

    /// Returns the number of slices as `usize`
    #[allow(clippy::len_without_is_empty)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        // Cannot underflow since `self.offset` has at least two entries
        unsafe { self.offsets.len().unchecked_sub(1) }
    }

    /// Returns the number of slices as `Idx: PrimInt`
    #[inline(always)]
    pub fn number_of_slices<Idx: PrimIndex>(&self) -> Idx {
        Idx::from_usize(self.len()).unwrap()
    }

    /// Returns the number of entries in the buffer as `I`
    #[inline(always)]
    pub fn number_of_entries(&self) -> I {
        I::from_usize(self.buffer.len()).unwrap()
    }

    /// Returns the size of a slice of a given index
    #[inline(always)]
    pub fn size_of<Idx: PrimIndex>(&self, u: Idx) -> I {
        self.offsets[u.to_usize().unwrap() + 1] - self.offsets[u.to_usize().unwrap()]
    }

    /// Returns the complete buffer
    #[inline(always)]
    pub fn raw_buffer_slice(&self) -> &[T] {
        &self.buffer
    }

    /// Returns the complete `offsets`-slice
    #[inline(always)]
    pub fn raw_offset_slice(&self) -> &[I] {
        &self.offsets
    }

    /// Grants mutable access to two different slices concurrently
    #[inline(always)]
    pub fn double_mut<Idx: PrimIndex>(&mut self, u: Idx, v: Idx) -> (&mut [T], &mut [T]) {
        let (u, v) = (u.to_usize().unwrap(), v.to_usize().unwrap());

        assert_ne!(u, v);
        if u < v {
            let v_off = self.offsets[v].to_usize().unwrap();
            let (beg, end) = self.buffer.split_at_mut(v_off);

            let u_start = self.offsets[u].to_usize().unwrap();
            let u_end = self.offsets[u + 1].to_usize().unwrap();

            let v_len = self.offsets[v + 1].to_usize().unwrap() - v_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    beg.get_unchecked_mut(u_start..u_end),
                    end.get_unchecked_mut(0..v_len),
                )
            }
        } else {
            let u_off = self.offsets[u].to_usize().unwrap();
            let (beg, end) = self.buffer.split_at_mut(u_off);

            let v_start = self.offsets[v].to_usize().unwrap();
            let v_end = self.offsets[v + 1].to_usize().unwrap();

            let u_len = self.offsets[u + 1].to_usize().unwrap() - u_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    end.get_unchecked_mut(0..u_len),
                    beg.get_unchecked_mut(v_start..v_end),
                )
            }
        }
    }

    /// Restores the values of slice `u` to its default values
    #[inline(always)]
    pub fn restore_node<Idx: PrimIndex>(&mut self, u: Idx) {
        let u = u.to_usize().unwrap();
        let offset = self.offsets[u].to_usize().unwrap();
        let len = self.offsets[u + 1].to_usize().unwrap() - offset;

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        //  (iii) `self.buffer` and `self.default` have the same length
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.default.as_ptr().add(offset),
                self.buffer.as_mut_ptr().add(offset),
                len,
            );
        }
    }

    #[inline(always)]
    pub fn default_values<Idx: PrimIndex>(&self, idx: Idx) -> &[T] {
        let end = self.offsets[idx.to_usize().unwrap() + 1]
            .to_usize()
            .unwrap();
        let start = self.offsets[idx.to_usize().unwrap()].to_usize().unwrap();

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.default`
        unsafe { self.default.get_unchecked(start..end) }
    }
}

impl<T: Clone, I: PrimIndex, Idx: PrimIndex> Index<Idx> for SlicedBufferWithDefault<T, I> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, idx: Idx) -> &Self::Output {
        let end = self.offsets[idx.to_usize().unwrap() + 1]
            .to_usize()
            .unwrap();
        let start = self.offsets[idx.to_usize().unwrap()].to_usize().unwrap();

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        unsafe { self.buffer.get_unchecked(start..end) }
    }
}

impl<T: Clone, I: PrimIndex, Idx: PrimIndex> IndexMut<Idx> for SlicedBufferWithDefault<T, I> {
    #[inline(always)]
    fn index_mut(&mut self, idx: Idx) -> &mut Self::Output {
        let end = self.offsets[idx.to_usize().unwrap() + 1]
            .to_usize()
            .unwrap();
        let start = self.offsets[idx.to_usize().unwrap()].to_usize().unwrap();

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        unsafe { self.buffer.get_unchecked_mut(start..end) }
    }
}
