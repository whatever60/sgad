# SGAD

Semi-Global Alignment for Dimer calculation

This repository currently exposes two alignment entry points:

- `needleman_wunsch` in `src/sgad/pairwise.py` for 2-sequence alignment.
- `needleman_wunsch_3d` in `src/sgad/pairwise_3d.py` for exact 3-sequence alignment.

Both are global-style dynamic programming aligners with optional free ends (semiglobal behavior when enabled).

## Contents

- [Pairwise API: `needleman_wunsch`](#pairwise-api-needleman_wunsch)
    - [Example (dimer structure prediction)](#example-dimer-structure-prediction)
    - [User-specified values](#user-specified-values)
    - [Pairwise features](#pairwise-features)
- [3D API: `needleman_wunsch_3d`](#3d-api-needleman_wunsch_3d)
    - [Example (dimer + two primers)](#example-dimer--two-primers)
    - [User-specified values](#user-specified-values-1)
    - [3D features](#3d-features)
- [Rust Backend](#rust-backend)
    - [Rust 2D usage](#rust-2d-usage)
    - [Rust 2D multiprocessing caveat](#rust-2d-multiprocessing-caveat)
    - [Rust 3D usage](#rust-3d-usage)
    - [Benchmark Results](#benchmark-results)


## Pairwise API: `needleman_wunsch`

Signature (simplified):

```python
needleman_wunsch(
    seq1,
    seq2,
    score_matrix,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=False,
    score_scale_fn=score_scale_factor,
) -> tuple[str, str, float]
```

### Example (dimer structure prediction)

```python
from sgad import needleman_wunsch, score_scale_factor, to_ascii

primer1 = "GAGATATGAGGAGAGAGAGACAGAGG"  # right free only
primer2_rc = "GAACAGAGGGAGAGACTAACCTTG"  # left free only

seq1_left_free, seq1_right_free = False, True
seq2_left_free, seq2_right_free = True, False

mat = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

a1, a2, score = needleman_wunsch(
    primer1,
    primer2_rc,
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=seq1_left_free,
    seq1_right_free=seq1_right_free,
    seq2_left_free=seq2_left_free,
    seq2_right_free=seq2_right_free,
    score_scale_fn=score_scale_factor,
)

print(to_ascii(a1, a2))
print(score)
# Output:
# GAGATATGAGGAGAGAGAGACAGAGG               
#                 || |||||||               
#                 GA-ACAGAGGGAGAGACTAACCTTG
# 7.174206349206349
```

### User-specified values

- `score_matrix`: your substitution model.
- `gap_open` and `gap_extend`: affine gap penalties.
- Free-end flags: whether leading/trailing gaps are free per sequence.
- `score_scale_fn`:
    - Preimplemented options:
        - `score_scale_factor`: default behavior in this module.
        - `no_score_scale_factor`: disable scaling (always returns `1.0`).
    - User-defined option:
        - You can pass your own callable with the same signature as `score_scale_factor`.
        - The function must accept four indices plus the four `*_free` flags and return a float scale.

Example custom function:

```python
def my_score_scale_fn(
        seq1_left_idx: int,
        seq1_right_idx: int,
        seq2_left_idx: int,
        seq2_right_idx: int,
        seq1_left_free: bool,
        seq1_right_free: bool,
        seq2_left_free: bool,
        seq2_right_free: bool,
) -> float:
        # Replace with your own domain-specific scaling logic.
        return 1.0


a1, a2, score = needleman_wunsch(
        primer1,
        primer2_rc,
        score_matrix=mat,
        gap_open=-5,
        gap_extend=-1,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
        score_scale_fn=my_score_scale_fn,
)
```

### Pairwise features

Supported:

- Exact DP optimum for two sequences.
- Affine gap penalties.
- Per-sequence free left/right ends.
- Deterministic tie-breaking.
- Pluggable score scaling callback (`score_scale_fn`).

Not supported yet:

- Local alignment (Smith-Waterman).
- Automatic reverse-complement generation.
- Built-in ambiguous alphabet handling (for example `N`) unless you include it in `score_matrix`.
- Banding/heuristics for very long sequences.

## 3D API: `needleman_wunsch_3d`

Signature (simplified):

```python
needleman_wunsch_3d(
    seq1,
    seq2,
    seq3,
    score_matrix,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=False,
    seq3_left_free=False,
    seq3_right_free=False,
) -> tuple[str, str, str, float]
```

### Example (dimer + two primers)

```python
from sgad import needleman_wunsch_3d

dimer = "CCTGCTACTCTGTTCCCTCAATCTGATAGGTTCC"  # anchored
primer1 = "CCTGCTACTCTGTTCCTTCACATC"  # right free only
primer2_rc = "CTGTTCCCTCAATCTGATAGGTTCC"  # left free only

seq1_left_free = seq1_right_free = False
seq2_left_free, seq2_right_free = False, True
seq3_left_free, seq3_right_free = True, False

mat = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

a1, a2, a3, score = needleman_wunsch_3d(
    dimer,
    primer1,
    primer2_rc,
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=seq1_left_free,
    seq1_right_free=seq1_right_free,
    seq2_left_free=seq2_left_free,
    seq2_right_free=seq2_right_free,
    seq3_left_free=seq3_left_free,
    seq3_right_free=seq3_right_free,
)

print(a1)
print(a2)
print(a3)
print(score)
# Output:
# CCTGCTACTCTGTTCCCTCA-ATCTGATAGGTTCC
# CCTGCTACTCTGTTCCTTCACATC-----------
# ---------CTGTTCCCTCA-ATCTGATAGGTTCC
# 108.0
```

### User-specified values

- `score_matrix` for all letters you expect.
- Affine gap penalties.
- Left/right free-end settings for each of the 3 sequences.

### 3D features

Supported:

- Exact 3-sequence DP optimum (no heuristic shortcuts).
- Sum-of-pairs substitution scoring.
- Affine gaps per sequence.
- Per-sequence free terminal gaps (left/right).
- Deterministic tie-breaking.

Not supported yet:

- Pluggable `score_scale_fn` (3D currently has no scaling callback parameter).
- Local alignment mode.
- Banded or heuristic memory/time reduction for long inputs.
- Automatic sequence preprocessing (reverse-complementing, case-specific cleanup).

## Rust Backend

You can use Rust-accelerated implementations for both `needleman_wunsch` (2D) and
`needleman_wunsch_3d` (3D).

### Rust 2D usage

```python
from sgad.rust.pairwise import needleman_wunsch

a1, a2, score = needleman_wunsch(
    seq1,
    seq2,
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
)
```

Rust 2D score scaling options:

- No scaling by default (`score_scale_fn=None`).
- No scaling (`score_scale_fn=no_score_scale_factor`).
- Native Rust scaler objects created by `make_rust_score_scaler`.

Arbitrary Python scaling callables are not supported in the Rust 2D backend.

```python
from sgad.rust.pairwise import make_rust_score_scaler, needleman_wunsch

rust_scaler = make_rust_score_scaler(decay_exponent=1.3, temperature=0.9)

a1, a2, score = needleman_wunsch(
    seq1,
    seq2,
    score_matrix=mat,
    score_scale_fn=rust_scaler,
)
```

Equivalent direct-native 2D usage:

```python
from sgad.rust.sgad_rust_native import make_rust_score_scaler, needleman_wunsch

rust_scaler = make_rust_score_scaler(decay_exponent=1.3, temperature=0.9)

a1, a2, score = needleman_wunsch(
    seq1,
    seq2,
    score_matrix=mat,
    score_scaler_fn=rust_scaler,
)
```

### Rust 2D multiprocessing caveat

The scaler must be `None` or a `RustScoreScaler` object from `make_rust_score_scaler`.

`RustScoreScaler` is not picklable, so process-based parallel backends
(`multiprocessing`, joblib `loky`) cannot directly ship that object to workers.

Practical patterns:

- Construct the scaler inside each worker process (or lazily cache one per worker).
- Or use a thread-based backend when that fits your workload.

### Rust 3D usage

```python
from sgad.rust.pairwise_3d import needleman_wunsch_3d

a1, a2, a3, score = needleman_wunsch_3d(
    seq1,
    seq2,
    seq3,
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=False,
    seq3_left_free=False,
    seq3_right_free=False,
)
```

Equivalent direct-native 3D usage:

```python
from sgad.rust.sgad_rust_native import needleman_wunsch_3d

a1, a2, a3, score = needleman_wunsch_3d(
    seq1,
    seq2,
    seq3,
    score_matrix=mat,
)
```

### Benchmark Results

Based on `benchmarks/time_complexity.csv`, the Rust backend is consistently much faster
than the Python implementation for both 2D and 3D exact DP:

- 2D common-size comparison (`n=500..1500`) shows about `248x-252x` speedup
    (for example, `n=1500`: Python `34.84s` vs Rust `0.138s`).
- 3D common-size comparison (`n=20..100`) shows about `233x-282x` speedup
    (for example, `n=100`: Python `55.66s` vs Rust `0.198s`).
- Under the benchmark stopping rules (timeout/memory guard), Python stopped at smaller
    maximum sizes while Rust continued to larger sizes (`2D` up to `n=6500`, `3D` up to `n=260`
    in the recorded run).

Benchmarks were run on Ubuntu 22.04.5 LTS (`Linux 6.8.0-1044-aws`) on an `x86_64`
machine with an AMD EPYC 7R13 CPU (`16` vCPUs, `8` physical cores with SMT, `32 MiB` L3)
and `123 GiB` RAM (no swap), using `uv 0.7.15`, `rustc 1.87.0`, and `cargo 1.87.0`.
