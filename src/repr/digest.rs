/*!
# Graph Hash Digests

This module provides the [`GraphDigest`] trait, which allows computing
**hash-based digests** of graphs that are independent of the underlying
data structure.

The digest encodes:
- the number of nodes, and
- a sorted edge list,

before feeding them into a cryptographic hash function.

## Example
```
use ugraphs::{prelude::*, repr::digest::GraphDigest};

let mut graph = AdjArray::new(10);
graph.add_edge(4, 3);
graph.add_edge(1, 2);

// Computes a SHA-256 digest (hex string of length 64).
assert_eq!(
    graph.digest_sha256(),
    "73f9b526b0528f6a33e96b064f90dd9ad5b8fd646717d33e7ab1286361aa847a"
);
```
*/

use std::fmt::LowerHex;

use super::*;
use ::digest::{Digest, Output};

/// Trait for computing a **canonical hash digest** of a graph.
///
/// Digests are designed to be:
/// - **Graph-structure dependent**: Two isomorphic but differently stored
///   graphs will yield the same digest.
/// - **Representation independent**: Works with any [`AdjacencyList`] implementation.
/// - **Deterministic**: Edges are encoded in sorted order.
///
/// # Example
/// ```
/// use ugraphs::{prelude::*, repr::digest::GraphDigest};
///
/// let mut graph = AdjArray::new(5);
/// graph.add_edge(0, 1);
/// graph.add_edge(2, 3);
///
/// // Any digest implementing `Digest` can be used
/// let hex = graph.digest::<sha2::Sha256>();
/// assert_eq!(hex.len(), 64); // SHA256 -> 64 hex chars
/// ```
pub trait GraphDigest {
    /// Computes a digest of the graph using the provided hash function `D`.
    ///
    /// The result is returned as a **hexadecimal string**.
    ///
    /// # Type Parameters
    /// - `D`: A hash function implementing [`Digest`].
    fn digest<D>(&self) -> String
    where
        Output<D>: LowerHex,
        D: Digest;

    /// Computes a **SHA-256 digest** of the graph.
    ///
    /// The returned string is exactly 64 characters long.
    ///
    /// # Example
    /// ```
    /// use ugraphs::{prelude::*, repr::digest::GraphDigest};
    ///
    /// let mut graph = AdjArray::new(3);
    /// graph.add_edge(0, 1);
    ///
    /// let digest = graph.digest_sha256();
    /// assert_eq!(digest.len(), 64);
    /// ```
    fn digest_sha256(&self) -> String {
        self.digest::<sha2::Sha256>()
    }
}

impl<G> GraphDigest for G
where
    G: AdjacencyList,
{
    fn digest<D>(&self) -> String
    where
        Output<D>: LowerHex,
        D: Digest,
    {
        let mut hasher = D::new();
        let mut buffer = [0u8; 8];

        let encode = |buf: &mut [u8], u: Node| {
            for (i, c) in buf.iter_mut().enumerate().take(4) {
                *c = (u >> (8 * i)) as u8;
            }
        };

        // first encode the number of nodes in the graph
        encode(&mut buffer[0..4], self.number_of_nodes());
        hasher.update(buffer);

        // then append a sorted edge list
        for Edge(u, v) in self.edges(false) {
            encode(&mut buffer[0..], u);
            encode(&mut buffer[4..], v);
            hasher.update(buffer);
        }

        format!("{:x}", hasher.finalize())
    }
}
