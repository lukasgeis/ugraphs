//! # Dot
//!
//! The Dot-Format is a very extensive format used by [GraphViz](https://graphviz.org/) to allow
//! for detailed visualizations. We only use basic functionality to draw (colored) nodes and edges.
//!
//! For example, drawing an uncolored (directed) graph where neighbors of `1` are colored red can be
//! achieved via
//! ```ignore
//! let dot_writer = DotWriter::default();
//! dot_writer.start_graph(&mut writer, true)?;
//! dot_writer.write_edges(&mut writer, graph.edges(false), true, None)?;
//! dot_writer.color_nodes(&mut writer, graph.neighbors_of(1), DotColor::Red)?;
//! dot_writer.finish_graph(&mut writer)?;
//! ```
//!
//! Note that for nodes, the latest coloring is the one that will be applied in a visualizer,
//! whereas for edges, each new colored edge adds another edge to the graph. Use the inbuilt
//! `.filter()` method to selectively prevent drawing edges prematurely.
use std::{fmt::Display, io::Write};

use super::*;

/// A writer for the Dot-Format
#[derive(Debug, Clone)]
pub struct DotWriter {
    /// Increment nodes by 1 before writing
    inc_nodes: bool,
    /// Prefix of a node (default: 'u')
    prefix: String,
}

impl Default for DotWriter {
    fn default() -> Self {
        Self {
            inc_nodes: true,
            prefix: "u".to_string(),
        }
    }
}

impl DotWriter {
    /// Shorthand for default
    pub fn new() -> Self {
        Self::default()
    }

    /// If *false*, nodes retain their interval value (-1 that of input)
    pub fn inc_nodes(mut self, inc_nodes: bool) -> Self {
        self.inc_nodes = inc_nodes;
        self
    }

    /// Set the prefix of a node (`u` by default). Can also be changed while drawing to draw
    /// additional subgraphs apart from the original graph.
    pub fn node_prefix<S>(self, prefix: S) -> DotWriter
    where
        S: Into<String>,
    {
        DotWriter {
            inc_nodes: self.inc_nodes,
            prefix: prefix.into(),
        }
    }

    /// Writes the opening brackets of the graph.
    /// Must know if the graph is undirected
    pub fn start_graph<W>(&self, writer: &mut W, directed: bool) -> Result<()>
    where
        W: Write,
    {
        let graph_name = if directed { "digraph" } else { "graph" };

        writeln!(writer, "{graph_name} {{")
    }

    /// Formats a node depending on `self.prefix, self.inc_nodes`
    fn format_node(&self, u: Node) -> String {
        let u = u + self.inc_nodes as Node;
        format!("{}{u}", self.prefix)
    }

    /// Writes an iterator of edges to `writer`. Must know if the edges are directed and if they
    /// should be colored.
    pub fn write_edges<W, I>(
        &self,
        writer: &mut W,
        edges: I,
        directed: bool,
        color: Option<DotColor>,
    ) -> Result<()>
    where
        W: Write,
        I: IntoIterator<Item = Edge>,
    {
        let edge_dir = if directed { "->" } else { "--" };

        let edge_color = if let Some(c) = color {
            &format!("[color={c}]")
        } else {
            ""
        };

        for Edge(u, v) in edges.into_iter() {
            write!(
                writer,
                "{}{edge_dir}{}{edge_color};",
                self.format_node(u),
                self.format_node(v)
            )?;
        }
        writeln!(writer)
    }

    /// Writes a list of colored nodes to `writer`.
    /// This method should only be needed when wanting to color additional nodes which is why
    /// `color` is not optional.
    pub fn color_nodes<W, I>(&self, writer: &mut W, nodes: I, color: DotColor) -> Result<()>
    where
        W: Write,
        I: IntoIterator<Item = Node>,
    {
        for u in nodes.into_iter() {
            write!(
                writer,
                "{}[style=filled, color={color}]",
                self.format_node(u)
            )?;
        }
        writeln!(writer)
    }

    /// Closes the Dot-Graph, thus finishing the graph
    pub fn finish_graph<W>(&self, writer: &mut W) -> Result<()>
    where
        W: Write,
    {
        writeln!(writer, "}}")
    }
}

impl<G> GraphWriter<G> for DotWriter
where
    G: AdjacencyList + GraphType,
{
    fn try_write_graph<W>(&self, graph: &G, mut writer: W) -> std::io::Result<()>
    where
        W: Write,
    {
        let directed = G::is_directed();
        self.start_graph(&mut writer, directed)?;
        self.write_edges(&mut writer, graph.edges(!directed), directed, None)?;
        self.finish_graph(&mut writer)
    }
}

/// Trait for writing a graph to a writer in the Dot-Format.
/// Shorthand for default settings.
pub trait DotWrite {
    /// Tries to write the graph to a writer
    fn try_write_dot<W>(&self, writer: W) -> Result<()>
    where
        W: Write;

    /// Tries to write the graph to a file
    fn try_write_dot_file<P>(&self, path: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let writer = BufWriter::new(File::create(path)?);
        self.try_write_dot(writer)
    }
}

impl<G> DotWrite for G
where
    G: AdjacencyList + GraphType,
{
    fn try_write_dot<W>(&self, writer: W) -> Result<()>
    where
        W: Write,
    {
        DotWriter::default().try_write_graph(self, writer)
    }
}

impl Display for DotColor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("{self:?}").to_lowercase())
    }
}

/// List of all permitted Colors in Svg-Dot taken from
/// `https://graphviz.gitlab.io/doc/info/colors.html#svg`
#[derive(Debug, Copy, Clone)]
pub enum DotColor {
    AliceBlue,
    AntiqueWhite,
    Aqua,
    Aquamarine,
    Azure,
    Beige,
    Bisque,
    Black,
    BlanchedAlmond,
    Blue,
    BlueViolet,
    Brown,
    BurlyWood,
    CadetBlue,
    Chartreuse,
    Chocolate,
    Coral,
    CornflowerBlue,
    Cornsilk,
    Crimson,
    Cyan,
    DarkBlue,
    DarkCyan,
    DarkGoldenrod,
    DarkGray,
    DarkGreen,
    DarkGrey,
    DarkKhaki,
    DarkMagenta,
    DarkOliveGreen,
    DarkOrange,
    DarkOrchid,
    DarkRed,
    DarkSalmon,
    DarkSeaGreen,
    DarkSlateBlue,
    DarkSlateGray,
    DarkSlateGrey,
    DarkTurquoise,
    DarkViolet,
    DeepPink,
    DeepSkyBlue,
    DimGray,
    DimGrey,
    DodgerBlue,
    FireBrick,
    FloralWhite,
    ForestGreen,
    Fuchsia,
    Gainsboro,
    GhostWhite,
    Gold,
    Goldenrod,
    Gray,
    Grey,
    Green,
    GreenYellow,
    Honeydew,
    HotPink,
    IndianRed,
    Indigo,
    Ivory,
    Khaki,
    Lavender,
    LavenderBlush,
    LawnGreen,
    LemonChiffon,
    LightBlue,
    LightCoral,
    LightCyan,
    LightGoldenrodYellow,
    LightGray,
    LightGreen,
    LightGrey,
    LightPink,
    LightSalmon,
    LightSeaGreen,
    LightSkyBlue,
    LightSlateGray,
    LightSlateGrey,
    LightSteelBlue,
    LightYellow,
    Lime,
    LimeGreen,
    Linen,
    Magenta,
    Maroon,
    MediumAquamarine,
    MediumBlue,
    MediumOrchid,
    MediumPurple,
    MediumSeaGreen,
    MediumSlateBlue,
    MediumSpringGreen,
    MediumTurquoise,
    MediumVioletRed,
    MidnightBlue,
    MintCream,
    MistyRose,
    Moccasin,
    NavajoWhite,
    Navy,
    OldLace,
    Olive,
    OliveDrab,
    Orange,
    OrangeRed,
    Orchid,
    PaleGoldenrod,
    PaleGreen,
    PaleTurquoise,
    PaleVioletRed,
    PapayaWhip,
    PeachPuff,
    Peru,
    Pink,
    Plum,
    PowderBlue,
    Purple,
    Red,
    RosyBrown,
    RoyalBlue,
    SaddleBrown,
    Salmon,
    SandyBrown,
    SeaGreen,
    SeaShell,
    Sienna,
    Silver,
    SkyBlue,
    SlateBlue,
    SlateGray,
    SlateGrey,
    Snow,
    SpringGreen,
    SteelBlue,
    Tan,
    Teal,
    Thistle,
    Tomato,
    Turquoise,
    Violet,
    Wheat,
    White,
    WhiteSmoke,
    Yellow,
    YellowGreen,
}
