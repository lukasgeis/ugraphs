/*!
# Metis

This module provides readers and writers for the **Metis graph format**.

A Metis file consists of:
- a **header line**, which specifies the number of nodes and edges, and
- `n` non-comment lines, each line describing the adjacency list of a node.

Each line of the adjacency section has the form:
```text
v1 v2 v3 v4 ...
```
which represents edges
```text
Edge(u, v1 - 1), Edge(u, v2 - 1), ...
```
where `u` is the index of the current non-comment line (starting from `0`).

Lines starting with a configurable **comment identifier** (default: `"c"`) are ignored.

# Examples

## Reading a graph
```
use ugraphs::prelude::*;
use ugraphs::io::*;
use std::io::Cursor;

let data = b"p metis 3 4\n2 3\n1\n1\n";
let cursor = Cursor::new(&data[..]);

let g: AdjArray = MetisReader::new().try_read_graph(cursor).unwrap();

assert_eq!(g.number_of_nodes(), 3);
assert_eq!(g.number_of_edges(), 4);
```

## Writing a graph
```
use ugraphs::prelude::*;
use ugraphs::io::*;
use std::io::Cursor;

let mut g = AdjArray::new(3);
g.add_edge(0, 1);
g.add_edge(0, 2);

let mut buffer = Cursor::new(Vec::new());
g.try_write_metis(&mut buffer).unwrap();

let output = String::from_utf8(buffer.into_inner()).unwrap();
assert!(output.contains("3 2")); // header
assert!(output.contains("2 3")); // adjacency list of node 1
```
*/

use std::{
    fs::File,
    io::{BufRead, BufWriter, ErrorKind, Lines, Write},
    path::Path,
};

use itertools::Itertools;

use super::*;

/// A configurable reader for the **Metis format**.
///
/// Provides parsing of the header and adjacency lists,
/// while skipping comment lines starting with a given identifier (default: `"c"`).
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
    /// Creates a new [`MetisReader`] with default settings (PACE-compatible).
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the expected [`Header`] format for parsing the file.
    ///
    /// Typically used if the Metis header differs from the PACE standard.
    pub fn set_header_format(&mut self, format: Header) {
        self.header = format;
    }

    /// Updates the header format, consuming and returning `self` for chaining.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    ///
    /// let reader = MetisReader::new()
    ///     .header_format(Header::new_problem("metis"));
    /// ```
    pub fn header_format(mut self, format: Header) -> Self {
        self.set_header_format(format);
        self
    }

    /// Updates the identifier used for detecting comment lines.
    ///
    /// Default is `"c"`.
    pub fn set_comment_identifier<S>(&mut self, c: S)
    where
        S: Into<String>,
    {
        self.comment_identifier = c.into();
    }

    /// Updates the comment identifier, consuming and returning `self` for chaining.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    ///
    /// let reader = MetisReader::new()
    ///     .comment_identifier("%");
    /// ```
    pub fn comment_identifier<S>(mut self, c: S) -> Self
    where
        S: Into<String>,
    {
        self.set_comment_identifier(c);
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
        Ok(G::from_try_edges(n, edges_reader))
    }
}

/// Trait for creating graphs from the **Metis format**.
///
/// Provides shorthand methods for reading graphs using the default [`MetisReader`] settings.
pub trait MetisRead: Sized {
    /// Tries to read a graph from a given buffered reader in **Metis format**.
    ///
    /// # Errors
    /// Returns an error if the input cannot be parsed as a valid Metis graph.
    ///
    /// # Example
    /// ```
    /// use ugraphs::prelude::*;
    /// use ugraphs::io::*;
    /// use std::io::Cursor;
    ///
    /// let data = b"p metis 2 2\n2\n1\n";
    /// let cursor = Cursor::new(&data[..]);
    /// let g: AdjArray = AdjArray::try_read_metis(cursor).unwrap();
    ///
    /// assert_eq!(g.number_of_nodes(), 2);
    /// assert_eq!(g.number_of_edges(), 2);
    /// ```
    fn try_read_metis<R>(reader: R) -> Result<Self>
    where
        R: BufRead;

    /// Tries to read a graph from a file on disk in **Metis format**.
    ///
    /// # Errors
    /// Returns an error if the file does not exist or is not a valid Metis file.
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

/// Low-level reader that consumes a stream of lines
/// and yields edges in the **Metis format**.
///
/// Usually constructed internally by [`MetisReader`].
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
    /// Creates a new [`MetisEdgesReader`] and parses the first non-comment line as the header.
    ///
    /// # Errors
    /// Returns an error if no valid header is found.
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

    /// Returns the number of edges as specified in the header.
    pub fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }

    /// Returns the number of nodes as specified in the header.
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

/// A writer for exporting graphs in the **Metis format**.
///
/// Handles writing the header and adjacency lists.
#[derive(Debug, Clone, Default)]
pub struct MetisWriter {
    /// HeaderFormat
    header: Header,
}

impl MetisWriter {
    /// Creates a new [`MetisWriter`] with default settings (PACE-compatible).
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the [`Header`] format to be written at the beginning of the file.
    pub fn set_header_format(&mut self, format: Header) {
        self.header = format;
    }

    /// Updates the header format, consuming and returning `self` for chaining.
    ///
    /// # Example
    /// ```
    /// use ugraphs::io::*;
    ///
    /// let writer = MetisWriter::new()
    ///     .header_format(Header::new_problem("metis"));
    /// ```
    pub fn header_format(mut self, format: Header) -> Self {
        self.set_header_format(format);
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

/// Trait for writing a graph to a writer in the **Metis format**.
///
/// Provides shorthand methods using the default [`MetisWriter`] settings.
pub trait MetisWrite {
    /// Tries to write the graph to a given writer in **Metis format**.
    ///
    /// # Errors
    /// Returns an error if writing fails (e.g., due to I/O issues).
    ///
    /// # Example
    /// ```
    /// use ugraphs::prelude::*;
    /// use ugraphs::io::*;
    /// use std::io::Cursor;
    ///
    /// let mut g = AdjArray::new(2);
    /// g.add_edge(0, 1);
    ///
    /// let mut buffer = Cursor::new(Vec::new());
    /// g.try_write_metis(&mut buffer).unwrap();
    ///
    /// let output = String::from_utf8(buffer.into_inner()).unwrap();
    /// assert!(output.contains("2 1")); // header line
    /// ```
    fn try_write_metis<W>(&self, writer: W) -> Result<()>
    where
        W: Write;

    /// Tries to write the graph to a file on disk in **Metis format**.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or written to.
    ///
    /// # Example
    /// ```ignore
    /// use ugraphs::prelude::*;
    /// use ugraphs::io::*;
    ///
    /// let mut g = AdjArray::new(2);
    /// g.add_edge(0, 1);
    ///
    /// g.try_write_metis_file("graph.metis").unwrap();
    /// ```
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
