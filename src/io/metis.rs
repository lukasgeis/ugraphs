//! # Metis
//!
//! The Metis-Format consists of a typical header, followed by `n` non-comment-lines
//! `v1 v2 v3 v4 ...` representing (directed) edges `Edge(u, v1 - 1), Edge(u, v2 - 1), ...`
//! where `u` is the index of the current non-comment-line (i.e. first line are neighbors of `0`...).

use std::{
    fs::File,
    io::{BufRead, BufWriter, ErrorKind, Lines, Write},
    path::Path,
};

use itertools::Itertools;

use super::*;
use crate::{
    ops::{AdjacencyList, GraphEdgeOrder, GraphFromScratch, GraphType},
    *,
};

/// A GraphReader for the Metis-Format
#[derive(Debug, Clone)]
pub struct MetisReader {
    /// HeaderFormat
    header: Header,
    /// Lines starting with `comment_identifier` are skipped when reading
    comment_identifier: String,
}

impl Default for MetisReader {
    /// Default to the Pace-Format
    fn default() -> Self {
        Self {
            header: Header::default(),
            comment_identifier: "c".to_string(),
        }
    }
}

impl MetisReader {
    /// Creates a new (default) reader
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the header format
    pub fn header_format(mut self, format: Header) -> MetisReader {
        self.header = format;
        self
    }

    /// Updates the comment identifier
    pub fn comment_identifier<S>(mut self, c: S) -> MetisReader
    where
        S: Into<String>,
    {
        self.comment_identifier = c.into();
        self
    }
}

impl<G> GraphReader<G> for MetisReader
where
    G: GraphFromScratch,
{
    fn try_read_graph<R: BufRead>(&self, reader: R) -> std::io::Result<G> {
        let edges_reader =
            MetisEdgesReader::try_new(reader, &self.header, &self.comment_identifier)?;
        let n = edges_reader.number_of_nodes();
        Ok(G::from_edges(n, edges_reader))
    }
}

/// Trait for creating graphs form an MetisReader.
/// Used as shorthand for default MetisReader settings
pub trait MetisRead: Sized {
    /// Tries to read the graph from a given reader
    fn try_read_metis<R>(reader: R) -> Result<Self>
    where
        R: BufRead;

    /// Tries to read the graph from a given file
    fn try_read_metis_file<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Self::try_read_metis(BufReader::new(File::open(path)?))
    }
}

impl<G> MetisRead for G
where
    G: GraphFromScratch,
{
    fn try_read_metis<R>(reader: R) -> Result<Self>
    where
        R: BufRead,
    {
        MetisReader::default().try_read_graph(reader)
    }
}

/// Real MetisReader that consumes the reader
pub struct MetisEdgesReader<'a, R> {
    /// Lines in the reader
    lines: Lines<R>,
    /// Current node index
    curr_node: Node,
    /// Current line buffer,
    buffer: Vec<Node>,
    /// Number of nodes parsed from header
    number_of_nodes: NumNodes,
    /// Number of edges parsed from header
    number_of_edges: NumEdges,
    /// Comment identifier
    comment_identifier: &'a str,
}

impl<'a, R> MetisEdgesReader<'a, R>
where
    R: BufRead,
{
    /// Creates a new MetisEdgesReader and tries to parse the first non-comment-line as the header
    pub fn try_new(reader: R, header_format: &Header, comment_identifier: &'a str) -> Result<Self> {
        let mut metis_reader = Self {
            lines: reader.lines(),
            curr_node: 0,
            buffer: Vec::new(),
            number_of_nodes: 0,
            number_of_edges: 0,
            comment_identifier,
        };

        (metis_reader.number_of_nodes, metis_reader.number_of_edges) = header_format.parse_header(
            metis_reader
                .next_non_comment_line()?
                .ok_or(io_error!(ErrorKind::NotFound, "Header not found"))?,
        )?;

        Ok(metis_reader)
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

impl<'a, R> Iterator for MetisEdgesReader<'a, R>
where
    R: BufRead,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_edge()
            .unwrap()
            // Consider making the decrement optional
            .map(|Edge(u, v)| Edge(u - 1, v - 1))
    }
}

impl<'a, R> MetisEdgesReader<'a, R>
where
    R: BufRead,
{
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

    /// Tries to parse a neighborhood from the next non-comment-line
    fn next_line(&mut self) -> Result<Option<()>> {
        debug_assert!(self.buffer.is_empty());
        let line = self.next_non_comment_line()?;

        if let Some(line) = line {
            self.curr_node += 1;
            debug_assert!((1..=self.curr_node).contains(&self.curr_node));
            let neighbors = line.split(' ').filter_map(|v| {
                if v.is_empty() {
                    None
                } else {
                    let v = v.parse::<Node>().map_err(|_| {
                        io_error!(
                            ErrorKind::InvalidData,
                            format!("Invalid value found. Cannot parse {}.", v)
                        )
                    });
                    Some(v)
                }
            });

            for nb in neighbors {
                let nb = nb?;
                debug_assert!((1..=self.number_of_nodes).contains(&nb));
                self.buffer.push(nb);
            }

            Ok(Some(()))
        } else {
            Ok(None)
        }
    }

    /// Tries to get the next edge in the buffer or parse the next non-comment-line if empty
    fn next_edge(&mut self) -> Result<Option<Edge>> {
        while self.buffer.is_empty() {
            if self.next_line()?.is_none() {
                return Ok(None);
            };
        }

        Ok(Some(Edge(self.curr_node, self.buffer.pop().unwrap())))
    }
}

/// A writer for the Metis-Format
#[derive(Debug, Clone, Default)]
pub struct MetisWriter {
    /// HeaderFormat
    header: Header,
}

impl MetisWriter {
    /// Shorthand for default
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the header format
    pub fn header_format(mut self, format: Header) -> MetisWriter {
        self.header = format;
        self
    }
}

impl<G> GraphWriter<G> for MetisWriter
where
    G: AdjacencyList + GraphEdgeOrder + GraphType,
{
    fn try_write_graph<W: Write>(&self, graph: &G, mut writer: W) -> std::io::Result<()> {
        self.header.write_header(
            &mut writer,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )?;

        for u in graph.vertices() {
            let neighbors_str = graph
                .neighbors_of(u)
                .map(|v| (v + 1).to_string())
                .collect_vec();
            writeln!(writer, "{}", neighbors_str.join(" "))?;
        }

        Ok(())
    }
}

/// Trait for writing a graph to a writer in the Metis-Format.
/// Shorthand for default settings.
pub trait MetisWrite {
    /// Tries to write the graph to a writer
    fn try_write_metis<W>(&self, writer: W) -> Result<()>
    where
        W: Write;

    /// Tries to write the graph to a file
    fn try_write_metis_file<P>(&self, path: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let writer = BufWriter::new(File::create(path)?);
        self.try_write_metis(writer)
    }
}

impl<G> MetisWrite for G
where
    G: AdjacencyList + GraphEdgeOrder + GraphType,
{
    fn try_write_metis<W>(&self, writer: W) -> Result<()>
    where
        W: Write,
    {
        MetisWriter::default().try_write_graph(self, writer)
    }
}
