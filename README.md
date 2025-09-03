# ugraphs

**`ugraphs`** is a graph data structure and algorithms library in Rust, designed for graphs that are:

- **unlabelled & unsigned**: nodes are numbered `0..n-1`  
- **unweighted**: no weights on nodes or edges  
- **undirected**: optionally supported (directed graphs work too)

---

## Representation

- **Nodes** are `u32` in the range `0..n`. This covers up to 2³² nodes while saving memory compared to `usize/u64`.  
- **Edges** are stored as `Edge(Node, Node)`.  

Both **directed** and **undirected** graphs are supported.  
In undirected graphs, `Edge(u, v)` ≡ `Edge(v, u)`.  
In directed graphs, the two are distinct.

---

## Graph Backends

Multiple storage backends are available (see [`repr`](src/repr)):  

- `AdjArray` – adjacency array  
- `AdjMatrix` – adjacency matrix  
- `CsrGraph` – compressed sparse row  
- `SparseAdjArray` – sparse adjacency array  
- `AdjArrayMatrix` – hybrid (directed only)  

`DirectedIn` variants additionally store in-neighbors for efficient reverse traversal.  
Each representation makes different trade-offs in memory and iteration speed.

---

## Algorithms

Most algorithms are available directly as **traits on graphs**, e.g.:

```rust
use ugraphs::{prelude::*, algo::*, gens::*};

let g = AdjArray::cycle(5); // 5-cycle
let bfs_order: Vec<_> = g.bfs(0).collect();
```

Implemented algorithms include:

- Breadth-first search (BFS) and depth-first search (DFS)  
- Topological search & cycle detection  
- Connected components  
- Matchings & network flow  
- Bipartite checks  
- Random graph generators ([`repr`](src/gens))
- Hashing ([`digest`](src/repr/digest.rs)) 

---

## When to Use

Use **ugraphs** if:

- You need **fast, lightweight graph algorithms** in Rust  
- Your graphs are **unlabelled and unweighted**  
- You want **basic but efficient graph representations** 
- You want to use *Rust*

If you need **labeled, weighted, or richer features**, consider:  
- [petgraph](https://crates.io/crates/petgraph) (general-purpose Rust library)  
- [NetworKit](https://networkit.github.io/) (high-performance C++/Python library)  

---

## Disclaimers

* This library is currently under production.
* [`stream-bitset`](stream-bitset) will be separated into its own crate in the (hopefully near) future

---

## Credits

Developed by [Manuel Penschuck](https://github.com/manpen) and [Johannes Meintrup](https://github.com/jmeintrup) for the 
- [PACE 2022 Challenge](https://pacechallenge.org/2022/) (solver: [BreakingTheCycle](https://github.com/goethe-tcs/breaking-the-cycle)),  
- [PACE 2025 Challenge](https://pacechallenge.org/2025/) (solver: [PaceYourself](https://github.com/manpen/pace25/tree/master)).  


