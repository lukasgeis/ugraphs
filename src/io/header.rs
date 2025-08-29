/*!
# Headers

A header(-line) for a graph file is usually defined by a series of tokens (separated by " " most often),
defining the size of the graph and/or the problem, the graph was 'created' for.

For example, the Pace2025-Challenge for DominatingSet defined its header with
 "p ds {n} {m}"
where n is the number of nodes and m the number of edges in the graph.

This module provides flexible parsing and writing of such headers
via [`HeaderFormat`] and [`HeaderToken`].

# Examples

## Parsing a Header
```
use ugraphs::io::*;

let header = Header::new_problem("ds");
let line = "p ds 5 10".to_string();

let (n, m) = header.parse_header(line).unwrap();
assert_eq!(n, 5);
assert_eq!(m, 10);
```

## Writing a Header
```
use ugraphs::io::*;
use std::io::Cursor;

let header = Header::new_problem("ds");
let mut buffer = Cursor::new(Vec::new());

header.write_header(&mut buffer, 5, 10).unwrap();
let output = String::from_utf8(buffer.into_inner()).unwrap();
assert_eq!(output.trim(), "p ds 5 10");
```

## Defining a Custom Format
```
use ugraphs::io::*;

let format = HeaderFormat::new()
    .str("c") // comments
    .any() // ignore some token
    .str("graph") // must contain "graph"
    .number_of_nodes()
    .number_of_edges()
    .end();
```
*/

use itertools::Itertools;
use smallvec::{SmallVec, smallvec};

use super::*;

/// Defining a single token in a graph file header.
///
/// A header token describes how a specific part of the header line
/// should be interpreted when parsing a graph file.
///
/// Tokens can either be ignored, matched against a fixed string,
/// or interpreted as metadata like the number of nodes or edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderToken {
    /// Ignore this entry in the header.
    Any,
    /// Require this entry to match a given string.
    Str(String),
    /// Parse this entry as the number of nodes in the graph.
    NumNodes,
    /// Parse this entry as the number of edges in the graph.
    NumEdges,
    /// Ensure that there are no more entries after this token.
    End,
    /// Ignore all further entries, regardless of content.
    Rest,
}

impl HeaderToken {
    /// When writing a header, this is the representation for HeaderToken::Any
    fn any_string() -> String {
        "0".to_string()
    }
}

/// Defines the expected format of a graph file header.
///
/// A `HeaderFormat` is a sequence of [`HeaderToken`]s that describe
/// the structure of a valid header line. The type parameters (`NODES_SET`,
/// `EDGES_SET`, `END`) ensure at compile time that a well-formed format
/// is created before use.
///
/// You can build a format using the builder pattern:
/// ```
/// use ugraphs::io::*;
///
/// let format = HeaderFormat::new()
///     .str("p")
///     .str("ds")
///     .number_of_nodes()
///     .number_of_edges()
///     .end();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderFormat<
    // Set to *true* if the position for number of nodes was set
    const NODES_SET: bool = false,
    // Set to *true* if the position for number of edges was set
    const EDGES_SET: bool = false,
    // Set to *true* if `NODES_SET = true & EDGES_SET = true` and
    // the last entry matches `HeaderToken::End | HeaderToken::Rest`
    const END: bool = false,
>(SmallVec<[HeaderToken; 6]>);

/// A fully defined and valid header format.
///
/// This alias enforces that:
/// - the number of nodes position is set,
/// - the number of edges position is set,
/// - and the format is properly terminated (with `End` or `Rest`).
pub type Header = HeaderFormat<true, true, true>;

impl Default for HeaderFormat<true, true, true> {
    /// Creates a default header format commonly used in many graph benchmarks.
    ///
    /// The default format assumes that the number of nodes and edges
    /// appear as the third and fourth tokens in the header, while other
    /// tokens are ignored.
    fn default() -> Self {
        Self(smallvec![
            HeaderToken::Any,
            HeaderToken::Any,
            HeaderToken::NumNodes,
            HeaderToken::NumEdges,
            HeaderToken::Rest,
        ])
    }
}

impl Header {
    /// Creates a new header format in the *PACE-style*.
    ///
    /// This format starts with `"p"` followed by a problem name,
    /// then expects the number of nodes and edges, and finally
    /// terminates the header.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    ///
    /// let header = Header::new_problem("ds");
    /// ```
    pub fn new_problem<S>(problem: S) -> Self
    where
        S: Into<String>,
    {
        Self(smallvec![
            HeaderToken::Str("p".to_string()),
            HeaderToken::Str(problem.into()),
            HeaderToken::NumNodes,
            HeaderToken::NumEdges,
            HeaderToken::End,
        ])
    }

    /// Parses a header line and extracts the number of nodes and edges.
    ///
    /// This function checks that the given line matches the format
    /// defined in `self`. If successful, it returns `(NumNodes, NumEdges)`.
    ///
    /// # Errors
    /// - Returns an error if the line does not match the expected format.
    /// - Returns an error if numbers cannot be parsed.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    ///
    /// let header = Header::new_problem("ds");
    /// let line = "p ds 5 10".to_string();
    /// let (n, m) = header.parse_header(line).unwrap();
    /// assert_eq!(n, 5);
    /// assert_eq!(m, 10);
    /// ```
    pub fn parse_header(&self, line: String) -> std::io::Result<(NumNodes, NumEdges)> {
        let mut number_of_nodes = 0;
        let mut number_of_edges = 0;

        // TBD: include optional separator?
        let parts = line.split(' ').filter(|t| !t.is_empty());
        let mut index = 0;

        for entry in parts {
            match &self.0[index] {
                HeaderToken::Any => {
                    index += 1;
                    continue;
                }
                HeaderToken::Str(p) => {
                    raise_error_unless!(entry == p, ErrorKind::InvalidData, "Invalid header found");
                }
                HeaderToken::NumNodes => {
                    number_of_nodes = match entry.parse::<NumNodes>() {
                        Ok(n) => n,
                        Err(_) => {
                            return Err(std::io::Error::new(
                                ErrorKind::InvalidInput,
                                "Number of Nodes can not be parsed",
                            ));
                        }
                    };
                }
                HeaderToken::NumEdges => {
                    number_of_edges = match entry.parse::<NumEdges>() {
                        Ok(m) => m,
                        Err(_) => {
                            return Err(std::io::Error::new(
                                ErrorKind::InvalidInput,
                                "Number of Edges can not be parsed",
                            ));
                        }
                    };
                }
                // We don't care about the rest
                HeaderToken::Rest => {
                    index = self.0.len();
                    break;
                }
                // End should never be matched to a token
                HeaderToken::End => {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        "Header is longer than expected",
                    ));
                }
            };

            index += 1;
        }

        // Check if rest of header is either empty or matches an end-marker
        raise_error_unless!(
            self.0.len() == index
                || (self.0.len() == index + 1
                    && matches!(self.0[index], HeaderToken::Rest | HeaderToken::End)),
            ErrorKind::InvalidData,
            "Header Length does not match format"
        );

        Ok((number_of_nodes, number_of_edges))
    }

    /// Writes a header line to the given writer.
    ///
    /// The header is constructed based on the format tokens and
    /// the provided number of nodes and edges.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    /// use std::io::Cursor;
    ///
    /// let header = Header::new_problem("ds");
    /// let mut buffer = Cursor::new(Vec::new());
    ///
    /// header.write_header(&mut buffer, 5, 10).unwrap();
    /// let output = String::from_utf8(buffer.into_inner()).unwrap();
    /// assert_eq!(output.trim(), "p ds 5 10");
    /// ```
    pub fn write_header<W>(&self, writer: &mut W, n: NumNodes, m: NumEdges) -> Result<()>
    where
        W: Write,
    {
        let header_str = self
            .0
            .iter()
            .take(self.0.len() - 1)
            .map(|token| match token {
                HeaderToken::Any => HeaderToken::any_string(),
                HeaderToken::Str(s) => s.to_string(),
                HeaderToken::NumNodes => n.to_string(),
                HeaderToken::NumEdges => m.to_string(),
                _ => panic!("The order of header tokens is broken"),
            })
            .collect_vec();
        writeln!(writer, "{}", header_str.join(" "))?;

        Ok(())
    }
}

impl HeaderFormat<false, false, false> {
    /// Creates a new empty header format.
    ///
    /// Typically used with the builder pattern to construct
    /// a complete format step by step.
    pub fn new() -> Self {
        HeaderFormat(smallvec![])
    }
}

impl<const NODES_SET: bool, const EDGES_SET: bool> HeaderFormat<NODES_SET, EDGES_SET, false> {
    /// Adds a token that is ignored during parsing.
    pub fn any(mut self) -> Self {
        self.0.push(HeaderToken::Any);
        self
    }

    /// Adds a token that must match a specific string in the header.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    ///
    /// let format = HeaderFormat::new().str("p").str("ds");
    /// ```
    pub fn str<S>(mut self, s: S) -> Self
    where
        S: Into<String>,
    {
        self.0.push(HeaderToken::Str(s.into()));
        self
    }
}

impl<const EDGES_SET: bool> HeaderFormat<false, EDGES_SET, false> {
    /// Adds a token for the number of nodes in the graph.
    pub fn number_of_nodes(mut self) -> HeaderFormat<true, EDGES_SET, false> {
        self.0.push(HeaderToken::NumNodes);
        HeaderFormat(self.0)
    }
}

impl<const NODES_SET: bool> HeaderFormat<NODES_SET, false, false> {
    /// Adds a token for the number of edges in the graph.
    pub fn number_of_edges(mut self) -> HeaderFormat<NODES_SET, true, false> {
        self.0.push(HeaderToken::NumEdges);
        HeaderFormat(self.0)
    }
}

impl HeaderFormat<true, true, false> {
    /// Terminates the header format, ensuring that no further tokens follow.
    pub fn end(mut self) -> HeaderFormat<true, true, true> {
        self.0.push(HeaderToken::End);
        HeaderFormat(self.0)
    }

    /// Terminates the header format, ignoring any remaining tokens.
    pub fn ignore_rest(mut self) -> HeaderFormat<true, true, true> {
        self.0.push(HeaderToken::Rest);
        HeaderFormat(self.0)
    }
}
