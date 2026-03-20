from __future__ import annotations

import math
import random

from sgad.pairwise import needleman_wunsch as py_nw
from sgad.pairwise import make_score_scaler
from sgad.pairwise import score_scale_factor
from sgad.pairwise import no_score_scale_factor
from sgad.pairwise_3d import needleman_wunsch_3d as py_nw3
from sgad.rust.pairwise import make_rust_score_scaler
from sgad.rust.pairwise import needleman_wunsch as rs_nw
from sgad.rust.pairwise import needleman_wunsch_batch as rs_nw_batch
from sgad.rust.pairwise_3d import needleman_wunsch_3d as rs_nw3
from sgad.rust.sgad_rust_native import needleman_wunsch as rs_native_nw
from sgad.rust.sgad_rust_native import needleman_wunsch_batch as rs_native_nw_batch

MAT = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_pairwise_consistency_default_no_scaling() -> None:
    rng = random.Random(0)
    for _ in range(30):
        n = rng.randint(8, 40)
        m = rng.randint(8, 40)
        s1 = _rand_seq(rng, n)
        s2 = _rand_seq(rng, m)
        flags = dict(
            seq1_left_free=bool(rng.getrandbits(1)),
            seq1_right_free=bool(rng.getrandbits(1)),
            seq2_left_free=bool(rng.getrandbits(1)),
            seq2_right_free=bool(rng.getrandbits(1)),
        )

        py_a, py_b, py_score = py_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-5.25,
            gap_extend=-1.1,
            score_scale_fn=no_score_scale_factor,
            **flags,
        )
        rs_a, rs_b, rs_score = rs_nw(
            s1, s2, score_matrix=MAT, gap_open=-5.25, gap_extend=-1.1, **flags
        )

        assert py_a == rs_a
        assert py_b == rs_b
        assert math.isclose(py_score, rs_score, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_consistency_rust_scaler_matches_python_default() -> None:
    rng = random.Random(11)
    rust_scaler = make_rust_score_scaler(decay_exponent=1.0, temperature=1.0)
    for _ in range(20):
        n = rng.randint(8, 40)
        m = rng.randint(8, 40)
        s1 = _rand_seq(rng, n)
        s2 = _rand_seq(rng, m)
        flags = dict(
            seq1_left_free=bool(rng.getrandbits(1)),
            seq1_right_free=bool(rng.getrandbits(1)),
            seq2_left_free=bool(rng.getrandbits(1)),
            seq2_right_free=bool(rng.getrandbits(1)),
        )

        py_a, py_b, py_score = py_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-4.75,
            gap_extend=-0.9,
            score_scale_fn=score_scale_factor,
            **flags,
        )
        rs_a, rs_b, rs_score = rs_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-4.75,
            gap_extend=-0.9,
            score_scale_fn=rust_scaler,
            **flags,
        )

        assert py_a == rs_a
        assert py_b == rs_b
        assert math.isclose(py_score, rs_score, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_consistency_no_scaling() -> None:
    rng = random.Random(1)
    for _ in range(20):
        n = rng.randint(8, 30)
        m = rng.randint(8, 30)
        s1 = _rand_seq(rng, n)
        s2 = _rand_seq(rng, m)

        py_a, py_b, py_score = py_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-6.0,
            gap_extend=-1.5,
            score_scale_fn=no_score_scale_factor,
        )
        rs_a, rs_b, rs_score = rs_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-6.0,
            gap_extend=-1.5,
            score_scale_fn=None,
        )

        assert py_a == rs_a
        assert py_b == rs_b
        assert math.isclose(py_score, rs_score, rel_tol=0.0, abs_tol=1e-9)



def test_pairwise_batch_matches_single_wrapper_no_scaling() -> None:
    """Ensure batch wrapper output matches per-pair wrapper output."""
    rng = random.Random(5)
    flags = dict(
        seq1_left_free=bool(rng.getrandbits(1)),
        seq1_right_free=bool(rng.getrandbits(1)),
        seq2_left_free=bool(rng.getrandbits(1)),
        seq2_right_free=bool(rng.getrandbits(1)),
    )

    seq_pairs: list[tuple[str, str]] = []
    for _ in range(24):
        n = rng.randint(8, 40)
        m = rng.randint(8, 40)
        seq_pairs.append((_rand_seq(rng, n), _rand_seq(rng, m)))

    batch_out = rs_nw_batch(
        seq_pairs,
        score_matrix=MAT,
        gap_open=-5.6,
        gap_extend=-1.2,
        score_scale_fn=None,
        **flags,
    )

    assert len(batch_out) == len(seq_pairs)
    for (s1, s2), (batch_a, batch_b, batch_score) in zip(seq_pairs, batch_out):
        single_a, single_b, single_score = rs_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-5.6,
            gap_extend=-1.2,
            score_scale_fn=None,
            **flags,
        )
        assert batch_a == single_a
        assert batch_b == single_b
        assert math.isclose(batch_score, single_score, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_batch_custom_rust_scaler_direct_native() -> None:
    """Ensure native batch API matches Python reference with custom scaling."""
    rng = random.Random(6)
    decay_exponent = 1.1
    temperature = 0.85
    py_scaler = make_score_scaler(
        decay_exponent=decay_exponent,
        temperature=temperature,
    )
    rust_scaler = make_rust_score_scaler(
        decay_exponent=decay_exponent,
        temperature=temperature,
    )
    flags = dict(
        seq1_left_free=bool(rng.getrandbits(1)),
        seq1_right_free=bool(rng.getrandbits(1)),
        seq2_left_free=bool(rng.getrandbits(1)),
        seq2_right_free=bool(rng.getrandbits(1)),
    )

    seq_pairs: list[tuple[str, str]] = []
    for _ in range(20):
        n = rng.randint(8, 36)
        m = rng.randint(8, 36)
        seq_pairs.append((_rand_seq(rng, n), _rand_seq(rng, m)))

    batch_out = rs_native_nw_batch(
        seq_pairs,
        score_matrix=MAT,
        gap_open=-5.1,
        gap_extend=-1.05,
        score_scaler_fn=rust_scaler,
        **flags,
    )

    assert len(batch_out) == len(seq_pairs)
    for (s1, s2), (rs_a, rs_b, rs_score) in zip(seq_pairs, batch_out):
        py_a, py_b, py_score = py_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-5.1,
            gap_extend=-1.05,
            score_scale_fn=py_scaler,
            **flags,
        )
        assert rs_a == py_a
        assert rs_b == py_b
        assert math.isclose(rs_score, py_score, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_3d_consistency() -> None:
    rng = random.Random(2)
    for _ in range(12):
        n = rng.randint(5, 12)
        m = rng.randint(5, 12)
        l3 = rng.randint(5, 12)
        s1 = _rand_seq(rng, n)
        s2 = _rand_seq(rng, m)
        s3 = _rand_seq(rng, l3)

        flags = dict(
            seq1_left_free=bool(rng.getrandbits(1)),
            seq1_right_free=bool(rng.getrandbits(1)),
            seq2_left_free=bool(rng.getrandbits(1)),
            seq2_right_free=bool(rng.getrandbits(1)),
            seq3_left_free=bool(rng.getrandbits(1)),
            seq3_right_free=bool(rng.getrandbits(1)),
        )

        py_a, py_b, py_c, py_score = py_nw3(
            s1, s2, s3, score_matrix=MAT, gap_open=-5.3, gap_extend=-1.2, **flags
        )
        rs_a, rs_b, rs_c, rs_score = rs_nw3(
            s1, s2, s3, score_matrix=MAT, gap_open=-5.3, gap_extend=-1.2, **flags
        )

        assert py_a == rs_a
        assert py_b == rs_b
        assert py_c == rs_c
        assert math.isclose(py_score, rs_score, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_consistency_custom_rust_scaler_via_wrapper() -> None:
    rng = random.Random(3)
    decay_exponent = 1.3
    temperature = 0.9
    py_scaler = make_score_scaler(
        decay_exponent=decay_exponent,
        temperature=temperature,
    )
    rust_scaler = make_rust_score_scaler(
        decay_exponent=decay_exponent,
        temperature=temperature,
    )

    for _ in range(20):
        n = rng.randint(8, 35)
        m = rng.randint(8, 35)
        s1 = _rand_seq(rng, n)
        s2 = _rand_seq(rng, m)
        flags = dict(
            seq1_left_free=bool(rng.getrandbits(1)),
            seq1_right_free=bool(rng.getrandbits(1)),
            seq2_left_free=bool(rng.getrandbits(1)),
            seq2_right_free=bool(rng.getrandbits(1)),
        )

        py_a, py_b, py_score = py_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-5.8,
            gap_extend=-1.05,
            score_scale_fn=py_scaler,
            **flags,
        )
        rs_a, rs_b, rs_score = rs_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-5.8,
            gap_extend=-1.05,
            score_scale_fn=rust_scaler,
            **flags,
        )

        assert py_a == rs_a
        assert py_b == rs_b
        assert math.isclose(py_score, rs_score, rel_tol=0.0, abs_tol=1e-9)


def test_pairwise_consistency_custom_rust_scaler_direct_native() -> None:
    rng = random.Random(4)
    decay_exponent = 0.8
    temperature = 1.4
    py_scaler = make_score_scaler(
        decay_exponent=decay_exponent,
        temperature=temperature,
    )
    rust_scaler = make_rust_score_scaler(
        decay_exponent=decay_exponent,
        temperature=temperature,
    )

    for _ in range(15):
        n = rng.randint(8, 30)
        m = rng.randint(8, 30)
        s1 = _rand_seq(rng, n)
        s2 = _rand_seq(rng, m)
        flags = dict(
            seq1_left_free=bool(rng.getrandbits(1)),
            seq1_right_free=bool(rng.getrandbits(1)),
            seq2_left_free=bool(rng.getrandbits(1)),
            seq2_right_free=bool(rng.getrandbits(1)),
        )

        py_a, py_b, py_score = py_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-4.9,
            gap_extend=-0.95,
            score_scale_fn=py_scaler,
            **flags,
        )
        rs_a, rs_b, rs_score = rs_native_nw(
            s1,
            s2,
            score_matrix=MAT,
            gap_open=-4.9,
            gap_extend=-0.95,
            score_scaler_fn=rust_scaler,
            **flags,
        )

        assert py_a == rs_a
        assert py_b == rs_b
        assert math.isclose(py_score, rs_score, rel_tol=0.0, abs_tol=1e-9)
