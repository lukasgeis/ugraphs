/// Every graph should implement `GraphNodeOrder` and `GraphEdgeOrder`
macro_rules! test_graph_ops {
    ($env:ident, $graph:ident, $undirected:literal, ($($trait:ident),*)) => {
        #[cfg(test)]
        mod $env {
            use crate::{ops::*, repr::*, testing::test_graph_ops};
            use rand::{Rng, SeedableRng};
            use rand_pcg::Pcg64Mcg;
            use itertools::Itertools;

            /// Creates a list of at most `m_ub` random edges for nodes `0..n`
            fn random_edges<R: Rng>(rng: &mut R, n: NumNodes, m_ub: NumEdges) -> Vec<Edge> {
                let mut edges: Vec<Edge> = (0..m_ub).map(|_| {
                    let u = rng.random_range(0..n);
                    let v = rng.random_range(0..n);

                    if $undirected {
                        Edge(u, v).normalized()
                    } else {
                        Edge(u, v)
                    }
                }).collect_vec();
                edges.sort_unstable();
                edges.dedup();

                edges
            }

            $(
                test_graph_ops!($graph<$undirected>: $trait);
            )*
        }
    };
    ($graph:ident<$undirected:literal>: GraphNew) => {
        #[test]
        fn graph_new() {
            for n in 1..50 {
                let graph = <$graph>::new(n);

                assert_eq!(graph.number_of_edges(), 0);
                assert_eq!(graph.number_of_nodes(), n);

                assert_eq!(graph.vertices_range().len(), n as usize);
                assert_eq!(graph.vertices().collect_vec(), (0..n).collect_vec());
            }
        }
    };
    ($graph:ident<$undirected:literal>: AdjacencyList) => {
        #[test]
        fn test_adjacency_list() {
            let rng = &mut Pcg64Mcg::seed_from_u64(3);

            for n in [10 as NumNodes, 20, 50] {
                for m_ub in [n * 2, n * 5, n * 10] {
                    for _ in 0..10 {
                        let edges = random_edges(rng, n, m_ub as NumEdges).into_iter();

                        let mut adj_matrix: Vec<NodeBitSet> = vec![NodeBitSet::new(n); n as usize];
                        let mut edges: Vec<Edge> = edges.map(|e| {
                            let Edge(u, v) = e.into();
                            adj_matrix[u as usize].set_bit(v);

                            if $undirected {
                                adj_matrix[v as usize].set_bit(u);
                            }

                            Edge(u, v)
                        }).collect_vec();

                        let graph = <$graph>::from_edges(n, edges.clone().into_iter());

                        if $undirected {
                            edges.iter_mut().for_each(|e| {
                                *e = e.normalized();
                            });
                        }

                        edges.sort_unstable();
                        edges.dedup();

                        let m = edges.len() as NumEdges;

                        assert_eq!(graph.number_of_nodes(), n);
                        assert_eq!(graph.number_of_edges(), m);
                        assert_eq!(graph.vertices_range().len(), n as usize);
                        assert_eq!(graph.vertices().collect_vec(), (0..n).collect_vec());

                        assert_eq!(edges, graph.ordered_edges($undirected).collect_vec());

                        for u in 0..n {
                            assert_eq!(graph.neighbors_of_as_bitset(u), adj_matrix[u as usize]);
                            assert_eq!(graph.degree_of(u), adj_matrix[u as usize].cardinality());
                        }
                    }
                }
            }
        }
    };
    ($graph:ident<$undirected:literal>: DirectedAdjacencyList) => {
        #[test]
        fn test_directed_adjacency_list() {
            assert!(!$undirected);

            let rng = &mut Pcg64Mcg::seed_from_u64(3);

            for n in [10 as NumNodes, 20, 50] {
                for m_ub in [n * 2, n * 5, n * 10] {
                    for _ in 0..10 {
                        let edges = random_edges(rng, n, m_ub as NumEdges).into_iter();

                        let mut adj_matrix_in: Vec<NodeBitSet> = vec![NodeBitSet::new(n); n as usize];

                        let graph = <$graph>::from_edges(n, edges.map(|e| {
                            let Edge(u, v) = e.into();
                            adj_matrix_in[v as usize].set_bit(u);

                            Edge(u, v)
                        }));

                        println!("{adj_matrix_in:?}");
                        println!("{:?}", graph.ordered_edges(false).collect_vec());

                        for u in 0..n {
                            println!("AAA {u}");
                            assert_eq!(graph.in_neighbors_of_as_bitset(u), adj_matrix_in[u as usize]);
                            println!("BBB");
                            assert_eq!(graph.in_degree_of(u), adj_matrix_in[u as usize].cardinality());
                        }
                    }
                }
            }
        }
    };
    ($graph:ident<$undirected:literal>: GraphEdgeEditing) => {
        #[test]
        fn test_graph_edge_editing() {
            let rng = &mut Pcg64Mcg::seed_from_u64(3);

            for n in [10 as NumNodes, 20, 50] {
                for m_ub in [n * 2, n * 5, n * 10] {
                    for _ in 0..10 {
                        let edges = random_edges(rng, n, m_ub as NumEdges).into_iter();

                        let mut graph = <$graph>::new(n);

                        let mut adj_matrix: Vec<NodeBitSet> = vec![NodeBitSet::new(n); n as usize];

                        edges.for_each(|e| {
                            let Edge(u, v) = e.into();
                            adj_matrix[u as usize].set_bit(v);
                            graph.try_add_edge(u, v);

                            if $undirected {
                                adj_matrix[v as usize].set_bit(u);
                            }
                        });

                        let rng = &mut Pcg64Mcg::seed_from_u64(4);

                        let mut m = graph.number_of_edges();
                        for _ in 0..(m / 2) {
                            let u = rng.random_range(0..n);
                            let v = rng.random_range(0..n);

                            if adj_matrix[u as usize].clear_bit(v) {
                                assert!(graph.try_remove_edge(u, v));
                                m -= 1;

                                if $undirected && u != v {
                                    assert!(adj_matrix[v as usize].clear_bit(u));
                                }
                            }

                            assert_eq!(m, graph.number_of_edges());
                        }

                        graph.remove_edges_at_nodes(0..n);
                        assert!(graph.is_singleton_graph());
                    }
                }
            }
        }
    };
    ($graph:ident<$undirected:literal>: GraphDirectedEdgeEditing) => {
        #[test]
        fn test_graph_directed_edge_editing() {
            let rng = &mut Pcg64Mcg::seed_from_u64(3);

            for n in [10 as NumNodes, 20, 50] {
                for m_ub in [n * 2, n * 5, n * 10] {
                    for _ in 0..10 {
                        let edges = random_edges(rng, n, m_ub as NumEdges).into_iter();

                        let mut graph = <$graph>::new(n);

                        let mut adj_matrix_in: Vec<NodeBitSet> = vec![NodeBitSet::new(n); n as usize];

                        edges.for_each(|e| {
                            let Edge(u, v) = e.into();
                            adj_matrix_in[v as usize].set_bit(u);
                            graph.try_add_edge(u, v);
                        });

                        let mut graph_clone = graph.clone();
                        let mut m = graph.number_of_edges();
                        for u in 0..n {
                            m -= graph.out_degree_of(u);
                            graph.remove_edges_out_of_node(u);
                            assert_eq!(m, graph.number_of_edges());
                        }

                        assert!(graph.is_singleton_graph());

                        let mut m = graph_clone.number_of_edges();
                        for u in 0..n {
                            m -= graph_clone.in_degree_of(u);
                            graph_clone.remove_edges_into_node(u);
                            assert_eq!(m, graph_clone.number_of_edges());
                        }

                        assert!(graph_clone.is_singleton_graph());
                    }
                }
            }
        }
    };

}

pub(crate) use test_graph_ops;
