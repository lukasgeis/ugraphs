//! # EdgeList
//!
//! The EdgeList-Format consists of a typical header, followed by `m` non-comment-lines
//! `u v` representing an (directed) edge `Edge(u - 1, v - 1)`.

use std::{
    fs::File,
    io::{BufRead, BufWriter, ErrorKind, Lines, Write},
    path::Path,
};

use super::*;
use crate::{
    ops::{AdjacencyList, GraphEdgeOrder, GraphFromScratch, GraphType},
    *,
};

/// A GraphReader for the EdgeList-Format
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
    /// Creates a new (default) reader
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the header format
    pub fn header_format(mut self, format: Header) -> EdgeListReader {
        self.header = format;
        self
    }

    /// Updates the comment identifier
    pub fn comment_identifier<S: Into<String>>(mut self, c: S) -> EdgeListReader {
        self.comment_identifier = c.into();
        self
    }
}

impl<G: GraphFromScratch> GraphReader<G> for EdgeListReader {
    fn try_read_graph<R: BufRead>(&self, reader: R) -> std::io::Result<G> {
        let edges_reader =
            EdgeListEdgesReader::try_new(reader, &self.header, &self.comment_identifier)?;
        let n = edges_reader.number_of_nodes();
        Ok(G::from_edges(n, edges_reader))
    }
}

/// Trait for creating graphs form an EdgeListReader.
/// Used as shorthand for default EdgeListReader settings
pub trait EdgeListRead: Sized {
    /// Tries to read the graph from a given reader
    fn try_read_edge_list<R: BufRead>(reader: R) -> Result<Self>;

    /// Tries to read the graph from a given file
    fn try_read_edge_list_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::try_read_edge_list(BufReader::new(File::open(path)?))
    }
}

impl<G> EdgeListRead for G
where
    G: GraphFromScratch,
{
    fn try_read_edge_list<R: BufRead>(reader: R) -> Result<Self> {
        EdgeListReader::default().try_read_graph(reader)
    }
}

/// Real EdgeListReader that consumes the reader
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

impl<'a, R: BufRead> EdgeListEdgesReader<'a, R> {
    /// Creates a new EdgeListEdgesReader and tries to parse the first non-comment-line as the header
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

    /// Returns the parsed number of edges in the graph
    pub fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }

    /// Returns the parsed number of nodes in the graph
    pub fn number_of_nodes(&self) -> NumNodes {
        self.number_of_nodes
    }
}

impl<'a, R: BufRead> Iterator for EdgeListEdgesReader<'a, R> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.parse_edge_line()
            .unwrap()
            // Consider making the decrement optional
            .map(|Edge(u, v)| Edge(u - 1, v - 1))
    }
}

impl<'a, R: BufRead> EdgeListEdgesReader<'a, R> {
    /// Returns the next non-comment-line if it exists or propagate an error
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

    /// Tries to parse an edge from the next non-comment-line
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

/// A writer for the EdgeList-Format
#[derive(Debug, Clone, Default)]
pub struct EdgeListWriter {
    /// HeaderFormat
    header: Header,
}

impl EdgeListWriter {
    /// Shorthand for default
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the header format
    pub fn header_format(mut self, format: Header) -> EdgeListWriter {
        self.header = format;
        self
    }
}

impl<G: AdjacencyList + GraphEdgeOrder + GraphType> GraphWriter<G> for EdgeListWriter {
    fn try_write_graph<W: Write>(&self, graph: &G, mut writer: W) -> std::io::Result<()> {
        self.header.write_header(
            &mut writer,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )?;

        for Edge(u, v) in graph.edges(graph.is_undirected()) {
            writeln!(writer, "{} {}", u + 1, v + 1)?;
        }

        Ok(())
    }
}

/// Trait for writing a graph to a writer in the EdgeList-Format.
/// Shorthand for default settings.
pub trait EdgeListWrite {
    /// Tries to write the graph to a writer
    fn try_write_edge_list<W: Write>(&self, writer: W) -> Result<()>;

    /// Tries to write the graph to a file
    fn try_write_edge_list_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        self.try_write_edge_list(writer)
    }
}

impl<G: AdjacencyList + GraphEdgeOrder + GraphType> EdgeListWrite for G {
    fn try_write_edge_list<W: Write>(&self, writer: W) -> Result<()> {
        EdgeListWriter::default().try_write_graph(self, writer)
    }
}
