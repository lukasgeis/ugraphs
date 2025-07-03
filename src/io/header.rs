//! # Headers
//!
//! A header(-line) for a graph file is usually defined by a series of tokens (separated by " " most often),
//! defining the size of the graph and/or the problem, the graph was 'created' for.
//!
//! For example, the Pace2025-Challenge for DominatingSet defined its header with
//!     "p ds {n} {m}"
//! where n is the number of nodes and m the number of edges in the graph.

use itertools::Itertools;
use smallvec::{SmallVec, smallvec};

use super::*;

/// Defining a single token in the header
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderToken {
    /// Ignore entry
    Any,
    /// Match entry to str
    Str(String),
    /// Parse number of nodes
    NumNodes,
    /// Parse number of edges
    NumEdges,
    /// Ensure that there are no more entries
    End,
    /// Ignore all further entries
    Rest,
}

impl HeaderToken {
    /// When writing a header, this is the representation for HeaderToken::Any
    fn any_string() -> String {
        "0".to_string()
    }
}

/// Defines the complete format of the header.
///
/// Use the Builder-Pattern to define the format.
/// The following defines the format of the Pace2025-Challenge:
/// ```ignore
/// let format = HeaderFormat::new()
///     .str("p").str("ds").number_of_nodes().number_of_edges().end();
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

/// A header is defined as a format that satisfies all prerequisites.
pub type Header = HeaderFormat<true, true, true>;

impl Default for HeaderFormat<true, true, true> {
    /// Often (Pace...), number of nodes and edges are the third and fourth token in the header.
    /// We also ignore every other entry to not hard-code problem definitions.
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
    /// Creates a new HeaderFormat in the Pace-Style which is always valid
    pub fn new_problem<S: Into<String>>(problem: S) -> Self {
        Self(smallvec![
            HeaderToken::Str("p".to_string()),
            HeaderToken::Str(problem.into()),
            HeaderToken::NumNodes,
            HeaderToken::NumEdges,
            HeaderToken::End,
        ])
    }

    /// Tries to parse a the header and extract the number of nodes and edges.
    pub fn parse_header(&self, line: String) -> std::io::Result<(NumNodes, NumEdges)> {
        let mut number_of_nodes = 0;
        let mut number_of_edges = 0;

        // TBD: include optional separator?
        let mut parts = line.split(' ').filter(|t| !t.is_empty());
        let mut index = 0;

        while let Some(entry) = parts.next() {
            match &self.0[index] {
                HeaderToken::Any => continue,
                HeaderToken::Str(p) => {
                    raise_error_unless!(entry == p, ErrorKind::InvalidData, "Invalid header found");
                }
                HeaderToken::NumNodes => {
                    number_of_nodes = parse_next_value!(parts, "Header>Number of nodes");
                }
                HeaderToken::NumEdges => {
                    number_of_edges = parse_next_value!(parts, "Header>Number of edges");
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

    pub fn write_header<W: Write>(&self, writer: &mut W, n: NumNodes, m: NumEdges) -> Result<()> {
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

impl<const NODES_SET: bool, const EDGES_SET: bool> HeaderFormat<NODES_SET, EDGES_SET, false> {
    /// Creates a new empty format
    pub fn new() -> HeaderFormat<false, false, false> {
        HeaderFormat(smallvec![])
    }

    /// Pushes a token that can be ignored onto the stack
    pub fn any(mut self) -> Self {
        self.0.push(HeaderToken::Any);
        self
    }

    /// Pushes a token that should match a string onto the stack
    pub fn str<S: Into<String>>(mut self, s: S) -> Self {
        self.0.push(HeaderToken::Str(s.into()));
        self
    }
}

impl<const EDGES_SET: bool> HeaderFormat<false, EDGES_SET, false> {
    /// Adds the number of nodes token onto the stack
    pub fn number_of_nodes(mut self) -> HeaderFormat<true, EDGES_SET, false> {
        self.0.push(HeaderToken::NumNodes);
        HeaderFormat(self.0)
    }
}

impl<const NODES_SET: bool> HeaderFormat<NODES_SET, false, false> {
    /// Adds the number of edges token onto the stack
    pub fn number_of_edges(mut self) -> HeaderFormat<NODES_SET, true, false> {
        self.0.push(HeaderToken::NumEdges);
        HeaderFormat(self.0)
    }
}

impl HeaderFormat<true, true, false> {
    /// Marks the header as finished ensuring that no further tokens follow
    pub fn end(mut self) -> HeaderFormat<true, true, true> {
        self.0.push(HeaderToken::End);
        HeaderFormat(self.0)
    }

    /// Marks the header as finished ignoring all further tokens
    pub fn ignore_rest(mut self) -> HeaderFormat<true, true, true> {
        self.0.push(HeaderToken::Rest);
        HeaderFormat(self.0)
    }
}
