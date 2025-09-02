/*!
# CSR-based Sliced Buffers

This module provides a **Compressed Sparse Row (CSR)**-like data structure for storing
variable-length slices efficiently.

The key idea:

- A contiguous `buffer: Vec<T>` stores all elements.
- A non-decreasing `offsets: Vec<I>` stores slice boundaries, where slice `i` is `buffer[offsets[i]..offsets[i+1]]`.

### Invariants
All constructions verify the following invariants:

1. `offsets.len() >= 2`
2. `offsets` is non-decreasing
3. `offsets` entries are within `buffer` bounds
4. In `SlicedBufferWithDefault`, `buffer.len() == default.len()`

These invariants allow **unchecked access** in methods for performance.

### Variants
- [`SlicedBuffer`] – basic CSR structure
- [`SlicedBufferWithDefault`] – CSR with a backup copy to restore original data
*/

use std::ops::{Index, IndexMut};

use stream_bitset::PrimIndex;

/// CSR-like structure storing slices of elements.
///
/// - `buffer`: all elements contiguously
/// - `offsets`: start indices of each slice
///
/// Provides indexed access to slices, and safe mutable access to two slices at a time.
///  
#[derive(Debug, Clone)]
pub struct SlicedBuffer<T, I>
where
    I: PrimIndex,
{
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

impl<T, I> SlicedBuffer<T, I>
where
    I: PrimIndex,
{
    /// Constructs a new `SlicedBuffer`.
    ///
    /// # Panics
    /// Panics if:
    /// - `offsets.len() < 2`
    /// - `offsets` is not sorted
    /// - `offsets` exceed `buffer` length
    pub fn new(buffer: Vec<T>, offsets: Vec<I>) -> Self {
        assert!(offsets.len() > 1);
        assert!(offsets.len() - 1 <= I::max_value().to_usize().unwrap());
        assert!(offsets.is_sorted());
        assert!(offsets.last().unwrap().to_usize().unwrap() <= buffer.len());

        Self { buffer, offsets }
    }

    /// Returns the number of slices as `usize`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.len(), 3);
    /// ```
    #[allow(clippy::len_without_is_empty)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        // Cannot underflow since `self.offset` has at least two entries
        unsafe { self.offsets.len().unchecked_sub(1) }
    }

    /// Returns the number of slices as type `Idx: PrimIndex`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.number_of_slices::<u8>(), 3u8);
    /// ```
    #[inline(always)]
    pub fn number_of_slices<Idx: PrimIndex>(&self) -> Idx {
        Idx::from_usize(self.len()).unwrap()
    }

    /// Returns the total number of entries in the buffer.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.number_of_entries(), 7u32);
    /// ```
    #[inline(always)]
    pub fn number_of_entries(&self) -> I {
        I::from_usize(self.buffer.len()).unwrap()
    }

    /// Returns the length of slice `u`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.size_of(2u8), 3u32);
    /// ```
    #[inline(always)]
    pub fn size_of<Idx: PrimIndex>(&self, u: Idx) -> I {
        self.offsets[u.to_usize().unwrap() + 1] - self.offsets[u.to_usize().unwrap()]
    }

    /// Returns a reference to the complete buffer.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let slice = sb.raw_buffer_slice();
    /// assert!(slice.is_sorted());
    /// ```
    #[inline(always)]
    pub fn raw_buffer_slice(&self) -> &[T] {
        &self.buffer
    }

    /// Returns a reference to the offsets array.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let slice = sb.raw_offset_slice();
    /// assert!(slice.is_sorted());
    /// ```
    #[inline(always)]
    pub fn raw_offset_slice(&self) -> &[I] {
        &self.offsets
    }

    /// Returns mutable references to two distinct slices simultaneously.
    ///
    /// # Panics
    /// Panics if `u == v`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBuffer;
    ///
    /// let mut sb = SlicedBuffer::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let (s1, s2) = sb.double_mut(0u16, 1);
    /// s1.reverse();
    /// s2.reverse();
    ///
    /// assert_eq!(s1[0], 2);
    /// assert_eq!(s2[1], 4);
    /// ```
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

impl<T, I, Idx> Index<Idx> for SlicedBuffer<T, I>
where
    I: PrimIndex,
    Idx: PrimIndex,
{
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

impl<T, I, Idx> IndexMut<Idx> for SlicedBuffer<T, I>
where
    I: PrimIndex,
    Idx: PrimIndex,
{
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

/// CSR-like structure with a default buffer.
///
/// - Stores an immutable `default` buffer alongside `buffer`.
/// - Can restore any slice to its default values using [`SlicedBufferWithDefault::restore_node`].
#[derive(Debug, Clone)]
pub struct SlicedBufferWithDefault<T, I>
where
    T: Clone,
    I: PrimIndex,
{
    buffer: Vec<T>,
    default: Vec<T>,
    offsets: Vec<I>,
}

impl<T, I> Default for SlicedBufferWithDefault<T, I>
where
    T: Clone,
    I: PrimIndex,
{
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            default: Vec::new(),
            offsets: vec![I::zero(), I::zero()],
        }
    }
}

impl<T, I> SlicedBufferWithDefault<T, I>
where
    T: Clone,
    I: PrimIndex,
{
    /// Constructs a new `SlicedBufferWithDefault`.
    ///
    /// # Panics
    /// Panics if offsets are invalid (same invariants as [`SlicedBuffer`])
    /// Copies `buffer` to `default`.
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

    /// Returns the number of slices as `usize`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.len(), 3);
    /// ```
    #[allow(clippy::len_without_is_empty)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        // Cannot underflow since `self.offset` has at least two entries
        unsafe { self.offsets.len().unchecked_sub(1) }
    }

    /// Returns the number of slices as type `Idx: PrimIndex`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.number_of_slices::<u8>(), 3u8);
    /// ```
    #[inline(always)]
    pub fn number_of_slices<Idx: PrimIndex>(&self) -> Idx {
        Idx::from_usize(self.len()).unwrap()
    }

    /// Returns the total number of entries in the buffer.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.number_of_entries(), 7u32);
    /// ```
    #[inline(always)]
    pub fn number_of_entries(&self) -> I {
        I::from_usize(self.buffer.len()).unwrap()
    }

    /// Returns the length of slice `u`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(sb.size_of(2u8), 3u32);
    /// ```
    #[inline(always)]
    pub fn size_of<Idx: PrimIndex>(&self, u: Idx) -> I {
        self.offsets[u.to_usize().unwrap() + 1] - self.offsets[u.to_usize().unwrap()]
    }

    /// Returns a reference to the complete buffer.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let slice = sb.raw_buffer_slice();
    /// assert!(slice.is_sorted());
    /// ```
    #[inline(always)]
    pub fn raw_buffer_slice(&self) -> &[T] {
        &self.buffer
    }

    /// Returns a reference to the offsets array.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let slice = sb.raw_offset_slice();
    /// assert!(slice.is_sorted());
    /// ```
    #[inline(always)]
    pub fn raw_offset_slice(&self) -> &[I] {
        &self.offsets
    }

    /// Returns mutable references to two distinct slices simultaneously.
    ///
    /// # Panics
    /// Panics if `u == v`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let mut sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let (s1, s2) = sb.double_mut(0u16, 1);
    /// s1.reverse();
    /// s2.reverse();
    ///
    /// assert_eq!(s1[0], 2);
    /// assert_eq!(s2[1], 4);
    /// ```
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

    /// Restores slice `u` to its default values from the `default` buffer.
    ///
    /// # Panics
    /// Panics if `u` exceeds number of slices.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let mut sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// let (s1, s2) = sb.double_mut(0u16, 1);
    /// s1.reverse();
    /// s2.reverse();
    ///
    /// assert_eq!(s1[0], 2);
    /// assert_eq!(s2[1], 4);
    ///
    /// sb.restore_node(0u8);
    ///
    /// assert_eq!(sb[0u32][0], 1);
    /// ```
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

    /// Returns a reference to the default values of slice `idx`.
    ///
    /// # Examples
    /// ```
    /// use ugraphs::utils::sliced_buffer::SlicedBufferWithDefault;
    ///
    /// let sb = SlicedBufferWithDefault::new(vec![1u32, 2, 4, 5, 6, 7, 8], vec![0u32, 2, 4, 7]);
    /// assert_eq!(&sb[0u32], sb.default_values(0u32))
    /// ```
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

impl<T, I, Idx> Index<Idx> for SlicedBufferWithDefault<T, I>
where
    T: Clone,
    I: PrimIndex,
    Idx: PrimIndex,
{
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

impl<T, I, Idx> IndexMut<Idx> for SlicedBufferWithDefault<T, I>
where
    T: Clone,
    I: PrimIndex,
    Idx: PrimIndex,
{
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
