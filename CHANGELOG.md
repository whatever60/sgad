# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [1.1.1] - 2026-03-20

### Changed

- Optimized Rust 2D Needleman-Wunsch hot path for short sequences, with major speedups in scaler-enabled mode.
- Added Criterion microbenchmark for short-sequence 2D runs (`rust/benches/nw2d_short.rs`) using `black_box`.
- Reduced traceback metadata overhead by removing per-cell step deltas and deriving movement from DP state.

### Verified

- Benchmark (release, scaler enabled) improved from ~173.34 us to ~6.53 us (len16) and ~550.04 us to ~17.85 us (len28).
- Python-vs-Rust consistency sweeps and `tests/test_rust_consistency.py` pass after optimization.

## [1.1.0] - 2026-03-17

### Added

- Symmetry-aware gap-close handling in Python and Rust 2D/3D aligners.
- New analysis and verification scripts for:
  - DP-vs-rescore consistency (2D/3D)
  - Symmetry sweeps (swap/reverse/complement)
  - Python-vs-Rust alignment/score parity
- High-level interfaces for external dimer assessment libraries:
  - Primer3 + `ntthal` batch analysis
  - IDT OligoAnalyzer batch integration

### Changed

- Needleman-Wunsch score parity behavior: DP-reported score now matches rescoring
  on returned alignments under the updated gap-close model.
- README updated with runnable examples and concrete outputs for core APIs.

[1.1.0]: https://github.com/whatever60/sgad/releases/tag/v1.1.0

[1.1.1]: https://github.com/whatever60/sgad/releases/tag/v1.1.1
