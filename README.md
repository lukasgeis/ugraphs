# Graphs

This library saves as a base implementation for (unweighted) graphs in Rust.
It was originally developed by [Manuel Penschuck](https://github.com/manpen) and [Johannes Meintrup](https://github.com/jmeintrup) as part of the [BreakingTheCycle](https://github.com/goethe-tcs/breaking-the-cycle) solver for the [Pace2022-Challenge](https://pacechallenge.org/2022/).
It was then further developed for the [Pace2025-Challenge](https://pacechallenge.org/2025/) as part of the [PaceYourself](https://github.com/manpen/pace25/tree/master) solver.

### Use-Cases

The libary serves two purposes:
* Providing efficient core functionality for unweighted graphs: including the crate via `Cargo.toml` suffices
* Starter for more complex solvers (e.g. for another PACE-Challenge etc.): fork this repo as a submodule to further refine the library for your purposes

### Disclaimers

* This library is currently under production.
* `stream-bitset` will be separated into its own crate in the (hopefully near) future
