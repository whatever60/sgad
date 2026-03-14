from __future__ import annotations

import itertools
import math
import random

import pytest

from sgad.pairwise import make_score_scaler
from sgad.pairwise import needleman_wunsch as py_nw
from sgad.rust.pairwise import make_rust_score_scaler
from sgad.rust.pairwise import needleman_wunsch as rs_nw

MAT = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

DNA_COMP = str.maketrans("ACGT", "TGCA")


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _rc(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1]


def _swap_flags(flags: tuple[bool, bool, bool, bool]) -> tuple[bool, bool, bool, bool]:
    # Under swap + reverse-complement dimer framing:
    # seq1_left <-> seq2_right, seq1_right <-> seq2_left.
    a, b, c, d = flags
    return (d, c, b, a)


def _score_py(
    primer1: str,
    primer2: str,
    *,
    flags: tuple[bool, bool, bool, bool],
    score_scale_fn,
) -> float:
    a, b, c, d = flags
    _aln1, _aln2, score = py_nw(
        primer1,
        _rc(primer2),
        score_matrix=MAT,
        gap_open=-5.0,
        gap_extend=-1.0,
        seq1_left_free=a,
        seq1_right_free=b,
        seq2_left_free=c,
        seq2_right_free=d,
        score_scale_fn=score_scale_fn,
    )
    return float(score)


def _score_rust(
    primer1: str,
    primer2: str,
    *,
    flags: tuple[bool, bool, bool, bool],
    score_scale_fn,
) -> float:
    a, b, c, d = flags
    _aln1, _aln2, score = rs_nw(
        primer1,
        _rc(primer2),
        score_matrix=MAT,
        gap_open=-5.0,
        gap_extend=-1.0,
        seq1_left_free=a,
        seq1_right_free=b,
        seq2_left_free=c,
        seq2_right_free=d,
        score_scale_fn=score_scale_fn,
    )
    return float(score)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=4)))
def test_pairwise_symmetry_python_no_scaling_with_flag_remap(
    flags: tuple[bool, bool, bool, bool],
) -> None:
    rng = random.Random(20260313)
    swapped_flags = _swap_flags(flags)
    for _ in range(8):
        n = rng.randint(8, 32)
        m = rng.randint(8, 32)
        p1 = _rand_seq(rng, n)
        p2 = _rand_seq(rng, m)
        s12 = _score_py(p1, p2, flags=flags, score_scale_fn=pw_no_scale)
        s21 = _score_py(p2, p1, flags=swapped_flags, score_scale_fn=pw_no_scale)
        assert math.isclose(s12, s21, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=4)))
def test_pairwise_symmetry_rust_no_scaling_with_flag_remap(
    flags: tuple[bool, bool, bool, bool],
) -> None:
    rng = random.Random(20260313)
    swapped_flags = _swap_flags(flags)
    for _ in range(8):
        n = rng.randint(8, 32)
        m = rng.randint(8, 32)
        p1 = _rand_seq(rng, n)
        p2 = _rand_seq(rng, m)
        s12 = _score_rust(p1, p2, flags=flags, score_scale_fn=None)
        s21 = _score_rust(p2, p1, flags=swapped_flags, score_scale_fn=None)
        assert math.isclose(s12, s21, rel_tol=0.0, abs_tol=1e-9)


def pw_no_scale(*args, **kwargs) -> float:
    _ = args, kwargs
    return 1.0


def test_pairwise_symmetry_python_scaled_when_flags_disabled() -> None:
    rng = random.Random(20260313)
    flags = (False, False, False, False)
    swapped_flags = _swap_flags(flags)
    scaler = make_score_scaler(decay_exponent=1.25, temperature=0.9)
    for _ in range(16):
        n = rng.randint(8, 32)
        m = rng.randint(8, 32)
        p1 = _rand_seq(rng, n)
        p2 = _rand_seq(rng, m)
        s12 = _score_py(p1, p2, flags=flags, score_scale_fn=scaler)
        s21 = _score_py(p2, p1, flags=swapped_flags, score_scale_fn=scaler)
        assert math.isclose(s12, s21, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_symmetry_rust_scaled_when_flags_disabled() -> None:
    rng = random.Random(20260313)
    flags = (False, False, False, False)
    swapped_flags = _swap_flags(flags)
    scaler = make_rust_score_scaler(decay_exponent=1.25, temperature=0.9)
    for _ in range(16):
        n = rng.randint(8, 32)
        m = rng.randint(8, 32)
        p1 = _rand_seq(rng, n)
        p2 = _rand_seq(rng, m)
        s12 = _score_rust(p1, p2, flags=flags, score_scale_fn=scaler)
        s21 = _score_rust(p2, p1, flags=swapped_flags, score_scale_fn=scaler)
        assert math.isclose(s12, s21, rel_tol=0.0, abs_tol=1e-9)
