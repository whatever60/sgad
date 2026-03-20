# SGAD

Semi-Global Alignment for Dimer calculation.

SGAD provides exact Needleman-Wunsch dynamic programming for 2-sequence and
3-sequence alignment, with symmetry-aware affine-gap scoring.

Core alignment controls in this package:

- Four free-end flags (semiglobal behavior):
  `seq1_left_free`, `seq1_right_free`, `seq2_left_free`, `seq2_right_free`
  (and seq3 variants in 3D).
- Position-biased / weighted scoring via `score_scale_fn` (2D API only).
- Gap-close penalty via `enable_gap_close_penalty` (default `True`), which
  splits affine open-vs-extend delta into open + close terms. This improves
  reverse/swap symmetry behavior and keeps DP score consistent with rescoring.
  Complement score invariance additionally requires a complement-symmetric
  substitution matrix.

Release history: see [CHANGELOG.md](CHANGELOG.md).

## 2D Needleman-Wunsch

```python
needleman_wunsch(
    seq1,
    seq2,
    score_matrix,
    gap_open=-5,
    gap_extend=-1,
    enable_gap_close_penalty=True,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=False,
    score_scale_fn=score_scale_factor,
) -> tuple[str, str, float]
```

### Example (Python 2D)

```python
from sgad.pairwise import needleman_wunsch, score_scale_factor, to_ascii

mat = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

a1, a2, score = needleman_wunsch(
    "GAGATATGAGGAGAGAGAGACAGAGG",
    "GAACAGAGGGAGAGACTAACCTTG",
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=True,
    seq2_left_free=True,
    seq2_right_free=False,
    score_scale_fn=score_scale_factor,
)

print(to_ascii(a1, a2, False, True, True, False))
print(score)
```

Output:

```text
GAGATATGAGGAGAGAGAGACAGAGG               
                || |||||||               
                GA-ACAGAGGGAGAGACTAACCTTG
8.97420634920635
```

### User-specified arguments

- `score_matrix`: substitution model.
- `gap_open`, `gap_extend`: affine-gap parameters.
- `enable_gap_close_penalty`: toggle split open/close gap accounting.
- Four free-end flags: semiglobal boundary behavior per sequence side.
- `score_scale_fn`: per-column weighting callback.
  - Use `no_score_scale_factor` to disable weighting.
  - Use `make_score_scaler(...)` for configurable inverse-distance weighting.

### Rust 2D usage

Rust wrapper API is under `sgad.rust`. `score_scale_fn` must be either `None`
or a `RustScoreScaler` object from `make_rust_score_scaler(...)`.

```python
from sgad.pairwise import to_ascii
from sgad.rust.pairwise import make_rust_score_scaler, needleman_wunsch

mat = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

rust_scaler = make_rust_score_scaler(decay_exponent=1.0, temperature=1.0)

a1, a2, score = needleman_wunsch(
    "GAGATATGAGGAGAGAGAGACAGAGG",
    "GAACAGAGGGAGAGACTAACCTTG",
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=True,
    seq2_left_free=True,
    seq2_right_free=False,
    enable_gap_close_penalty=True,
    score_scale_fn=rust_scaler,
)

print(to_ascii(a1, a2, False, True, True, False))
print(score)
```

Output:

```text
GAGATATGAGGAGAGAGAGACAGAGG               
                || |||||||               
                GA-ACAGAGGGAGAGACTAACCTTG
8.97420634920635
```

### Rust 2D multiprocessing caveat

`RustScoreScaler` objects are not picklable. In process-based parallelism
(`multiprocessing`, joblib `loky`), build the scaler inside each worker (or use
`score_scale_fn=None`). Thread-based execution avoids this serialization issue.

## 3D Needleman-Wunsch

Important: 3D currently does not expose score scaling (`score_scale_fn`).

```python
needleman_wunsch_3d(
    seq1,
    seq2,
    seq3,
    score_matrix,
    gap_open=-5,
    gap_extend=-1,
    enable_gap_close_penalty=True,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=False,
    seq3_left_free=False,
    seq3_right_free=False,
) -> tuple[str, str, str, float]
```

### Example (Python 3D)

```python
from sgad.pairwise_3d import needleman_wunsch_3d

mat = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

a1, a2, a3, score = needleman_wunsch_3d(
    "CCTGCTACTCTGTTCCCTCAATCTGATAGGTTCC",
    "CCTGCTACTCTGTTCCTTCACATC",
    "CTGTTCCCTCAATCTGATAGGTTCC",
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=True,
    seq3_left_free=True,
    seq3_right_free=False,
)

print(a1)
print(a2)
print(a3)
print(score)
```

Output:

```text
CCTGCTACTCTGTTCCCTCA-ATCTGATAGGTTCC
CCTGCTACTCTGTTCCTTCACATC-----------
---------CTGTTCCCTCA-ATCTGATAGGTTCC
108.0
```

### User-specified arguments

- `score_matrix`: substitution model (sum-of-pairs scoring in 3D).
- `gap_open`, `gap_extend`: affine-gap parameters.
- `enable_gap_close_penalty`: toggle split open/close gap accounting.
- Six free-end flags: semiglobal boundary behavior for all sequence sides.

### Rust 3D usage

```python
from sgad.rust.pairwise_3d import needleman_wunsch_3d

mat = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

a1, a2, a3, score = needleman_wunsch_3d(
    "CCTGCTACTCTGTTCCCTCAATCTGATAGGTTCC",
    "CCTGCTACTCTGTTCCTTCACATC",
    "CTGTTCCCTCAATCTGATAGGTTCC",
    score_matrix=mat,
    gap_open=-5,
    gap_extend=-1,
    seq1_left_free=False,
    seq1_right_free=False,
    seq2_left_free=False,
    seq2_right_free=True,
    seq3_left_free=True,
    seq3_right_free=False,
    enable_gap_close_penalty=True,
)

print(a1)
print(a2)
print(a3)
print(score)
```

Output:

```text
CCTGCTACTCTGTTCCCTCA-ATCTGATAGGTTCC
CCTGCTACTCTGTTCCTTCACATC-----------
---------CTGTTCCCTCA-ATCTGATAGGTTCC
108.0
```

### Rust 3D multiprocessing caveat

Unlike Rust 2D, there is no scaler object to serialize. For process pools,
ensure worker-call arguments remain picklable and import the Rust module in the
worker runtime as usual.

## Benchmarking Python vs Rust backends

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

## Interface to external dimer assessment libraries

### Primer3 interface

```python
from sgad.api.primer3 import heterodimer_batch_primer3

df = heterodimer_batch_primer3(
    primer1_seqs=["ACGTACGT"],
    primer2_seqs=["TGCATGCA"],
    primer1_names=["fwd_1"],
    primer2_names=["rev_1"],
    n_jobs=1,
)

print(df[["primer1_name", "primer2_name", "primer3_tm", "ntthal_t"]].to_string(index=False))
```

Output:

```text
primer1_name primer2_name  primer3_tm  ntthal_t
       fwd_1        rev_1  -70.205833  -70.2058
```

### IDT OligoAnalyzer interface

```python
from sgad.api.idt import heterodimer_batch_idt

res = heterodimer_batch_idt(
    primer1_seqs=["ACGTACGT"],
    primer2_seqs=["TGCATGCA"],
    primer1_names=["fwd_1"],
    primer2_names=["rev_1"],
    client_id="invalid",
    client_secret="invalid",
    idt_username="invalid",
    idt_password="invalid",
    timeout_s=5.0,
    max_retries=1,
    raise_on_error=False,
)

print(res[0])
```

This example intentionally uses invalid credentials to show the failure-record
shape returned when `raise_on_error=False`.

Output:

```text
{'primer1_name': 'fwd_1', 'primer2_name': 'rev_1', 'primer1': 'ACGTACGT', 'primer2': 'TGCATGCA', 'ok': False, 'response': None, 'status_code': 400, 'error': '400 Client Error: Bad Request for url: https://www.idtdna.com/Identityserver/connect/token'}
```
