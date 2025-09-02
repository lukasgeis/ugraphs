/*!
# IO

Utilities for reading and writing graphs from and to different file formats.

## Input Formats

Currently supported input formats:
- **Metis**: Similar to `AdjArray`, represents the graph as a list of neighborhoods separated by line breaks.
- **EdgeList**: Represents the graph as a list of edges separated by line breaks.

Both formats require a `Header` as defined by a custom `HeaderFormat`.

## Output Formats

For writing graphs, in addition to the above formats, the following is supported:
- **Dot**: The [DOT language](https://graphviz.org/doc/info/lang.html) of [GraphViz](https://graphviz.org/).

The DOT format:
- is the only format that does not require a header,
- supports node labels (all others only allow non-negative integer nodes),
- requires labels to follow DOT’s naming conventions (no spaces, hyphens, or other special characters).

## Traits

To generalize over reading/writing:
- [`GraphReader`] and [`GraphWriter`] are implemented by readers and writers for a specific format.
- [`GraphRead`] and [`GraphWrite`] abstract over reading/writing using a given [`FileFormat`].

*/

pub mod dot;
pub mod edge_list;
pub mod header;
pub mod metis;

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, ErrorKind, Result, Write},
    path::Path,
    str::FromStr,
};

use crate::prelude::*;

pub use dot::*;
pub use edge_list::*;
pub use header::*;
pub use metis::*;

/// Identifier for a graph file format.
///
/// Used in [`GraphRead`] and [`GraphWrite`] to determine the
/// correct parser or writer to use.
///
/// Currently supported:
/// - [`FileFormat::Dot`]
/// - [`FileFormat::Metis`]
/// - [`FileFormat::EdgeList`]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FileFormat {
    /// DOT language of GraphViz
    Dot,
    /// Metis neighborhood-list format
    Metis,
    /// Edge list format
    EdgeList,
}

impl FromStr for FileFormat {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "dot" => Ok(FileFormat::Dot),
            "metis" => Ok(FileFormat::Metis),
            "edgelist" => Ok(FileFormat::EdgeList),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unknown FileFormat: {s}").as_str(),
            )),
        }
    }
}

/// Trait for types that can read graphs in a specific format.
///
/// This trait provides both a low-level method to read from any
/// [`BufRead`] instance and a convenience wrapper to read directly
/// from files.
///
/// Typically implemented by specific readers (e.g., [`MetisRead`],
/// [`EdgeListRead`]).
pub trait GraphReader<G> {
    /// Reads a graph from the given reader according to the settings in `self`.
    ///
    /// # Errors
    /// Returns an error if the input is not a valid representation
    /// of a graph in the expected format.
    fn try_read_graph<R>(&self, reader: R) -> Result<G>
    where
        R: BufRead;

    /// Reads a graph from a file according to the settings in `self`.
    ///
    /// Internally wraps the file in a buffered reader.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or if its contents
    /// are not a valid representation of a graph in the expected format.
    fn try_read_graph_file<P>(&self, path: P) -> Result<G>
    where
        P: AsRef<Path>,
    {
        self.try_read_graph(BufReader::new(File::open(path)?))
    }
}

/// Trait for types that can write graphs in a specific format.
///
/// This trait provides both a low-level method to write to any
/// [`Write`] instance and a convenience wrapper to write directly
/// to files.
///
/// Typically implemented by specific writers (e.g., [`MetisWrite`],
/// [`EdgeListWrite`], [`DotWrite`]).
pub trait GraphWriter<G> {
    /// Writes the given graph to the provided writer according to the settings in `self`.
    ///
    /// # Errors
    /// Returns an error if writing fails (e.g., IO errors).
    fn try_write_graph<W>(&self, graph: &G, writer: W) -> Result<()>
    where
        W: Write;

    /// Writes the given graph to a file according to the settings in `self`.
    ///
    /// Internally wraps the file in a buffered writer.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or if writing fails.
    fn try_write_graph_file<P>(&self, graph: &G, path: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        self.try_write_graph(graph, BufWriter::new(File::create(path)?))
    }
}

/// Trait for reading graphs when only a [`FileFormat`] is known.
///
/// Provides a unified interface to construct graphs from readers
/// or files by dispatching to the correct format-specific parser.
///
/// Automatically implemented for graphs that support all required
/// format-specific traits (e.g., [`MetisRead`], [`EdgeListRead`]).
pub trait GraphRead: Sized {
    /// Reads a graph from the given reader according to the specified [`FileFormat`].
    ///
    /// # Errors
    /// Returns an error if the format is unsupported for this graph type
    /// or if the input does not match the expected format.
    fn try_from_reader<R>(reader: R, format: FileFormat) -> Result<Self>
    where
        R: BufRead;

    /// Reads a graph from the given file according to the specified [`FileFormat`].
    ///
    /// Internally wraps the file in a buffered reader.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or if the input
    /// is invalid for the chosen format.
    fn try_from_file<P>(path: P, format: FileFormat) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Self::try_from_reader(BufReader::new(File::open(path)?), format)
    }
}

impl<G> GraphRead for G
where
    G: MetisRead + EdgeListRead,
{
    fn try_from_reader<R>(reader: R, format: FileFormat) -> Result<Self>
    where
        R: BufRead,
    {
        match format {
            FileFormat::Metis => Self::try_read_metis(reader),
            FileFormat::EdgeList => Self::try_read_edge_list(reader),
            _ => Err(io_error!(
                ErrorKind::InvalidInput,
                format!("{format:?} does not support GraphRead")
            )),
        }
    }
}

/// Trait for writing graphs when only a [`FileFormat`] is known.
///
/// Provides a unified interface to output graphs to writers or files
/// by dispatching to the correct format-specific writer.
///
/// Automatically implemented for graphs that support all required
/// format-specific traits (e.g., [`MetisWrite`], [`EdgeListWrite`], [`DotWrite`]).
pub trait GraphWrite {
    /// Writes the graph to the given writer according to the specified [`FileFormat`].
    ///
    /// # Errors
    /// Returns an error if the format is unsupported for this graph type
    /// or if writing fails (e.g., IO errors).
    fn try_write_to_writer<W>(&self, writer: W, format: FileFormat) -> Result<()>
    where
        W: Write;

    /// Writes the graph to the given file according to the specified [`FileFormat`].
    ///
    /// Internally wraps the file in a buffered writer.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or if writing fails.
    fn try_write_to_file<P>(&self, path: P, format: FileFormat) -> Result<()>
    where
        P: AsRef<Path>,
    {
        self.try_write_to_writer(BufWriter::new(File::create(path)?), format)
    }
}

impl<G> GraphWrite for G
where
    G: MetisWrite + EdgeListWrite + DotWrite,
{
    fn try_write_to_writer<W>(&self, writer: W, format: FileFormat) -> Result<()>
    where
        W: Write,
    {
        match format {
            FileFormat::Metis => self.try_write_metis(writer),
            FileFormat::EdgeList => self.try_write_edge_list(writer),
            FileFormat::Dot => self.try_write_dot(writer),
        }
    }
}

/// Shorthand for creating a new IO-error
macro_rules! io_error {
    ($kind: expr, $info: expr) => {
        std::io::Error::new($kind, $info)
    };
}

/// Shorthand for returning `Err(std::io::Error)` early when a condition fails
macro_rules! raise_error_unless {
    ($cond : expr, $kind : expr, $info : expr) => {
        if !($cond) {
            return Err(io_error!($kind, $info));
        }
    };
}

/// Tries to parse the next value in an iterator and returns early if it fails
macro_rules! parse_next_value {
    ($iterator : expr, $name : expr) => {{
        let next = $iterator.next();
        raise_error_unless!(
            next.is_some(),
            ErrorKind::InvalidData,
            format!("Premature end of line when parsing {}.", $name)
        );

        let parsed = next.unwrap().parse();
        raise_error_unless!(
            parsed.is_ok(),
            ErrorKind::InvalidData,
            format!("Invalid value found. Cannot parse {}.", $name)
        );

        parsed.unwrap()
    }};
}

use io_error;
use parse_next_value;
use raise_error_unless;
