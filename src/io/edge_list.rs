/*
# EdgeList

Module for reading and writing graphs in the EdgeList format.

The EdgeList format consists of a typical header, followed by `m` lines,
each line representing a directed edge as `u v`. Nodes in the file are
1-indexed, while internally the graph representation uses 0-indexing.

Lines starting with a comment identifier (default `"c"`) are ignored.

Example usage:
```text
use ugraphs::prelude::*;
use ugraphs::io::*;

let g: AdjArray = EdgeListReader::default()
    .try_read_graph_file("graph.gr")?;

g.try_write_edge_list_file("out.gr")?;
```
*/

use std::{
    fs::File,
    io::{BufRead, BufWriter, ErrorKind, Lines, Write},
    path::Path,
};

use super::*;

/// A reader for graphs in the EdgeList format.
///
/// Handles optional headers and comment lines. Uses `Header` to parse
/// the number of nodes and edges, and then parses each line as an edge
/// with 1-based indexing (converted internally to 0-based).
#[derive(Debug, Clone)]
pub struct EdgeListReader {
    /// HeaderFormat
    header: Header,
    /// Lines starting with `comment_identifier` are skipped when reading
    comment_identifier: String,
}

impl Default for EdgeListReader {
    /// Default to the Pace-Format
    fn default() -> Self {
        Self {
            header: Header::default(),
            comment_identifier: "c".to_string(),
        }
    }
}

impl EdgeListReader {
    /// Creates a new EdgeListReader with default header and comment identifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the header format used by the reader.
    pub fn set_header_format(&mut self, format: Header) {
        self.header = format;
    }

    /// Updates the header format (builder style).
    pub fn header_format(mut self, format: Header) -> Self {
        self.set_header_format(format);
        self
    }

    /// Updates the comment identifier used to skip lines.
    pub fn set_comment_identifier<S>(&mut self, c: S)
    where
        S: Into<String>,
    {
        self.comment_identifier = c.into();
    }

    /// Updates the comment identifier (builder style).
    pub fn comment_identifier<S>(mut self, c: S) -> Self
    where
        S: Into<String>,
    {
        self.set_comment_identifier(c);
        self
    }
}

impl<G> GraphReader<G> for EdgeListReader
where
    G: GraphFromScratch,
{
    fn try_read_graph<R>(&self, reader: R) -> std::io::Result<G>
    where
        R: BufRead,
    {
        let edges_reader =
            EdgeListEdgesReader::try_new(reader, &self.header, &self.comment_identifier)?;
        let n = edges_reader.number_of_nodes();
        Ok(G::from_try_edges(n, edges_reader))
    }
}

/// Trait for types that can be read from an EdgeList-formatted file.
///
/// Provides default implementations for reading from a file or any BufRead source.
pub trait EdgeListRead: Sized {
    /// Tries to read the graph from a `BufRead` reader.
    fn try_read_edge_list<R>(reader: R) -> Result<Self>
    where
        R: BufRead;

    /// Tries to read the graph from a file path.
    fn try_read_edge_list_file<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Self::try_read_edge_list(BufReader::new(File::open(path)?))
    }
}

impl<G> EdgeListRead for G
where
    G: GraphFromScratch,
{
    fn try_read_edge_list<R>(reader: R) -> Result<Self>
    where
        R: BufRead,
    {
        EdgeListReader::default().try_read_graph(reader)
    }
}

/// Consumes an EdgeList file and iterates over the edges.
///
/// Each line after the header represents a single edge. Lines starting
/// with the comment identifier are ignored. The reader tracks the
/// number of nodes and edges parsed from the header.
pub struct EdgeListEdgesReader<'a, R> {
    /// Lines in the reader
    lines: Lines<R>,
    /// Number of nodes parsed from header
    number_of_nodes: NumNodes,
    /// Number of edges parsed from header
    number_of_edges: NumEdges,
    /// Comment identifier
    comment_identifier: &'a str,
}

impl<'a, R> EdgeListEdgesReader<'a, R>
where
    R: BufRead,
{
    /// Creates a new EdgeListEdgesReader from a BufRead reader.
    ///
    /// Parses the first non-comment line as the header using the given
    /// `Header` format, returning an error if missing or invalid.
    pub fn try_new(reader: R, header_format: &Header, comment_identifier: &'a str) -> Result<Self> {
        let mut edge_list_reader = Self {
            lines: reader.lines(),
            number_of_nodes: 0,
            number_of_edges: 0,
            comment_identifier,
        };

        (
            edge_list_reader.number_of_nodes,
            edge_list_reader.number_of_edges,
        ) = header_format.parse_header(
            edge_list_reader
                .next_non_comment_line()?
                .ok_or(io_error!(ErrorKind::NotFound, "Header not found"))?,
        )?;

        Ok(edge_list_reader)
    }

    /// Returns the number of edges parsed from the header.
    pub fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }

    /// Returns the number of nodes parsed from the header.
    pub fn number_of_nodes(&self) -> NumNodes {
        self.number_of_nodes
    }
}

impl<'a, R> Iterator for EdgeListEdgesReader<'a, R>
where
    R: BufRead,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.parse_edge_line()
            .unwrap()
            // Consider making the decrement optional
            .map(|Edge(u, v)| Edge(u - 1, v - 1))
    }
}

impl<'a, R> EdgeListEdgesReader<'a, R>
where
    R: BufRead,
{
    /// Returns the next line that does not start with the comment identifier,
    /// or `Ok(None)` if EOF is reached.
    fn next_non_comment_line(&mut self) -> Result<Option<String>> {
        loop {
            let line = self.lines.next();
            match line {
                None => return Ok(None),
                Some(Err(x)) => return Err(x),
                Some(Ok(line)) if line.starts_with(self.comment_identifier) => continue,
                Some(Ok(line)) => return Ok(Some(line)),
            }
        }
    }

    /// Parses the next edge from the EdgeList file.
    ///
    /// Returns an `Edge(u-1, v-1)` for internal 0-based representation,
    /// or `Ok(None)` if EOF is reached.
    fn parse_edge_line(&mut self) -> Result<Option<Edge>> {
        let line = self.next_non_comment_line()?;
        if let Some(line) = line {
            let mut parts = line.split(' ').filter(|t| !t.is_empty());

            let from = parse_next_value!(parts, "Source node");
            let dest = parse_next_value!(parts, "Target node");

            debug_assert!((1..=self.number_of_nodes).contains(&from));
            debug_assert!((1..=self.number_of_nodes).contains(&dest));

            Ok(Some(Edge(from, dest)))
        } else {
            Ok(None)
        }
    }
}

/// Writes graphs to an EdgeList format file.
///
/// Writes the header first, then `m` lines of edges as `u v` with 1-based indexing.
#[derive(Debug, Clone, Default)]
pub struct EdgeListWriter {
    /// HeaderFormat
    header: Header,
}

impl EdgeListWriter {
    /// Creates a new EdgeListWriter with default header.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the header format used by the writer.
    pub fn set_header_format(&mut self, format: Header) {
        self.header = format;
    }

    /// Updates the header format (builder style).
    pub fn header_format(mut self, format: Header) -> Self {
        self.set_header_format(format);
        self
    }
}

impl<G> GraphWriter<G> for EdgeListWriter
where
    G: AdjacencyList + GraphEdgeOrder + GraphType,
{
    fn try_write_graph<W>(&self, graph: &G, mut writer: W) -> std::io::Result<()>
    where
        W: Write,
    {
        self.header.write_header(
            &mut writer,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )?;

        for Edge(u, v) in graph.edges(G::is_undirected()) {
            writeln!(writer, "{} {}", u + 1, v + 1)?;
        }

        Ok(())
    }
}

/// Trait for types that can be written to an EdgeList-formatted file.
///
/// Provides default implementations for writing to a file or any `Write` target.
pub trait EdgeListWrite {
    /// Tries to write the graph to a writer.
    fn try_write_edge_list<W>(&self, writer: W) -> Result<()>
    where
        W: Write;

    /// Tries to write the graph to a file path.
    fn try_write_edge_list_file<P>(&self, path: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let writer = BufWriter::new(File::create(path)?);
        self.try_write_edge_list(writer)
    }
}

impl<G> EdgeListWrite for G
where
    G: AdjacencyList + GraphEdgeOrder + GraphType,
{
    fn try_write_edge_list<W>(&self, writer: W) -> Result<()>
    where
        W: Write,
    {
        EdgeListWriter::default().try_write_graph(self, writer)
    }
}
