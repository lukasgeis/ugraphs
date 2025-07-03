//! # IO
//!
//! Module for reading graphs from an input or writing graphs to an output.
//!
//! ## Input Formats
//!
//! The following input formats are supported
//! - **Metis**: Similar to `AdjArray`, this represents the graph as a list of neighborhoods separated by linebreaks.
//! - **EdgeList**: Similar to `Csv`, this represents the graph as a list of edges separated by linebreaks.
//!
//! Each format requires a `Header` defined by a custom `HeaderFormat` in the header module.
//!
//!
//! ## Output Formats
//!
//! In addition to the above input formats, an additional output format is supported which also
//! serves as the base-Debug-impl. for all graphs.
//! - **Dot**: The [Dot-Language](https://graphviz.org/doc/info/lang.html) of [GraphViz](https://graphviz.org/).
//!
//! The [Dot-Format](https://graphviz.org/doc/info/lang.html) is the only format that does not require a header
//! and supports node labels whereas every other format again only allows non-negative numbers as nodes. Note
//! that the string representation of node labels must follow the naming conventions of the  
//! [Dot-Language](https://graphviz.org/dot/info/lang.html) and hence can not contain special characters such as
//! spaces or hyphens for example.

mod dot;
mod edge_list;
mod header;
mod metis;

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, ErrorKind, Write},
    path::Path,
    str::FromStr,
};

use crate::*;

pub use dot::*;
pub use edge_list::*;
pub use header::*;
pub use metis::*;

type Result<T> = std::io::Result<T>;

/// Identifier for a FileFormat
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FileFormat {
    Dot,
    Metis,
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

/// Trait for Readers to implement
pub trait GraphReader<G> {
    /// Read a graph a reader according to settings in `self`
    fn try_read_graph<R: BufRead>(&self, reader: R) -> Result<G>;

    /// Read a graph from file according to settings in `self`
    fn try_read_graph_file<P: AsRef<Path>>(&self, path: P) -> Result<G> {
        self.try_read_graph(BufReader::new(File::open(path)?))
    }
}

/// Trait for Writers to implement
pub trait GraphWriter<G> {
    /// Write a graph to a writer according to settings in `self`
    fn try_write_graph<W: Write>(&self, graph: &G, writer: W) -> Result<()>;

    /// Write a graph to a file according to settings in `self`
    fn try_write_graph_file<P: AsRef<Path>>(&self, graph: &G, path: P) -> Result<()> {
        self.try_write_graph(graph, BufWriter::new(File::create(path)?))
    }
}

/// Trait for reading a graph specified by a given FileFormat
pub trait GraphRead: Sized {
    /// Read a graph from a reader according to FileFormat
    fn try_from_reader<R: BufRead>(reader: R, format: FileFormat) -> Result<Self>;

    /// Read a graph from file according to FileFormat
    fn try_from_file<P: AsRef<Path>>(path: P, format: FileFormat) -> Result<Self> {
        Self::try_from_reader(BufReader::new(File::open(path)?), format)
    }
}

impl<G: MetisRead + EdgeListRead> GraphRead for G {
    fn try_from_reader<R: BufRead>(reader: R, format: FileFormat) -> Result<Self> {
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

/// Trait for writing a graph specified by a given FileFormat
pub trait GraphWrite {
    /// Write a graph to a writer according to FileFormat
    fn try_write_to_writer<W: Write>(&self, writer: W, format: FileFormat) -> Result<()>;

    /// Write a graph to file according to FileFormat
    fn try_write_to_file<P: AsRef<Path>>(&self, path: P, format: FileFormat) -> Result<()> {
        self.try_write_to_writer(BufWriter::new(File::create(path)?), format)
    }
}

impl<G: MetisWrite + EdgeListWrite + DotWrite> GraphWrite for G {
    fn try_write_to_writer<W: Write>(&self, writer: W, format: FileFormat) -> Result<()> {
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
