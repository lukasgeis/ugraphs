/*!
 # Dot

 Module for writing graphs in the [Dot-Format](https://graphviz.org/doc/info/lang.html).

 The Dot format allows detailed visualization of graphs. This module supports
 basic features: writing directed/undirected graphs, coloring nodes and edges,
 and customizing node prefixes.
 Nodes are by default incremented by 1 (`0` → `u1`) to conform to typical Dot usage.

 Example usage:
 ```ignore
 let dot_writer = DotWriter::default();
 dot_writer.start_graph(&mut writer, true)?;
 dot_writer.write_edges(&mut writer, graph.edges(false), true, None)?;
 dot_writer.color_nodes(&mut writer, graph.neighbors_of(1), DotColor::Red)?;
 dot_writer.finish_graph(&mut writer)?;
 ```
*/

use std::{fmt::Display, io::Write};

use super::*;

/// A writer for the Dot-Format.
///
/// Allows customizing node prefixes and node incrementing. Supports writing
/// edges (optionally colored) and coloring nodes independently.
#[derive(Debug, Clone)]
pub struct DotWriter {
    /// Increment nodes by 1 before writing
    inc_nodes: bool,
    /// Prefix of a node (default: 'u')
    prefix: String,
}

impl Default for DotWriter {
    /// Default DotWriter with node increment enabled and prefix `"u"`.
    fn default() -> Self {
        Self {
            inc_nodes: true,
            prefix: "u".to_string(),
        }
    }
}

impl DotWriter {
    /// Shorthand for creating a default DotWriter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether nodes should be incremented before writing.
    pub fn set_inc_nodes(&mut self, inc_nodes: bool) {
        self.inc_nodes = inc_nodes;
    }

    /// Builder-style setter for node increment.
    pub fn inc_nodes(mut self, inc_nodes: bool) -> Self {
        self.set_inc_nodes(inc_nodes);
        self
    }

    /// Set the prefix of a node.
    ///
    /// Can be used to draw additional subgraphs with a different prefix.
    pub fn set_node_prefix<S>(&mut self, prefix: S)
    where
        S: Into<String>,
    {
        self.prefix = prefix.into();
    }

    /// Builder-style setter for node prefix.
    pub fn node_prefix<S>(mut self, prefix: S) -> Self
    where
        S: Into<String>,
    {
        self.set_node_prefix(prefix);
        self
    }

    /// Writes the opening bracket of a Dot graph.
    ///
    /// `directed = true` writes `digraph {`, otherwise `graph {`.
    pub fn start_graph<W>(&self, writer: &mut W, directed: bool) -> Result<()>
    where
        W: Write,
    {
        let graph_name = if directed { "digraph" } else { "graph" };

        writeln!(writer, "{graph_name} {{")
    }

    /// Formats a node as a string using prefix and increment options.
    fn format_node(&self, u: Node) -> String {
        let u = u + self.inc_nodes as Node;
        format!("{}{u}", self.prefix)
    }

    /// Writes edges to the Dot file.
    ///
    /// # Arguments
    /// - `edges`: iterator over `Edge(u, v)`
    /// - `directed`: if true, use `->`, otherwise `--`
    /// - `color`: optional color for all edges
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

    /// Colors nodes in the Dot output.
    ///
    /// Latest coloring for a node overwrites previous coloring.
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

    /// Closes the Dot graph (`}`).
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

/// Trait for writing a graph to a Dot-Format writer.
///
/// Provides default implementations for writing to any `Write` target or file.
pub trait DotWrite {
    /// Writes the graph to a writer.
    fn try_write_dot<W>(&self, writer: W) -> Result<()>
    where
        W: Write;

    /// Writes the graph to a file.
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

/// Enum representing all supported colors for Dot nodes/edges.
///
/// See [GraphViz SVG colors](https://graphviz.gitlab.io/doc/info/colors.html#svg).
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
