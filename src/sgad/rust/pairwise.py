from __future__ import annotations

from . import sgad_rust_native as _native

RustScoreScaler = getattr(_native, "RustScoreScaler")
make_rust_score_scaler = getattr(_native, "make_rust_score_scaler")
_needleman_wunsch = getattr(_native, "needleman_wunsch")
_needleman_wunsch_batch = getattr(_native, "needleman_wunsch_batch")


def _validate_score_scale_fn(score_scale_fn) -> None:
    """Validate Rust score-scaler inputs for Rust pairwise APIs."""
    if score_scale_fn is not None and not isinstance(score_scale_fn, RustScoreScaler):
        raise NotImplementedError(
            "Rust backend supports None (no scaling), or RustScoreScaler from "
            f"make_rust_score_scaler, but got {type(score_scale_fn)}"
        )


def needleman_wunsch(
    seq1: str,
    seq2: str,
    *,
    score_matrix: dict[str, dict[str, int | float]],
    gap_open: float = -5.0,
    gap_extend: float = -1.0,
    seq1_left_free: bool = False,
    seq1_right_free: bool = False,
    seq2_left_free: bool = False,
    seq2_right_free: bool = False,
    enable_gap_close_penalty: bool = True,
    score_scale_fn=None,
) -> tuple[str, str, float]:
    """Rust-accelerated pairwise Needleman-Wunsch.

    Supports no-scaling mode by default and RustScoreScaler objects.
    Arbitrary Python callables for score scaling are intentionally unsupported.
    """
    _validate_score_scale_fn(score_scale_fn)

    return _needleman_wunsch(
        seq1,
        seq2,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
        enable_gap_close_penalty=enable_gap_close_penalty,
        score_scaler_fn=score_scale_fn,
    )


def needleman_wunsch_batch(
    seq_pairs: list[tuple[str, str]],
    *,
    score_matrix: dict[str, dict[str, int | float]],
    gap_open: float = -5.0,
    gap_extend: float = -1.0,
    seq1_left_free: bool = False,
    seq1_right_free: bool = False,
    seq2_left_free: bool = False,
    seq2_right_free: bool = False,
    enable_gap_close_penalty: bool = True,
    score_scale_fn=None,
) -> list[tuple[str, str, float]]:
    """Rust-accelerated batched pairwise Needleman-Wunsch for many sequence pairs."""
    _validate_score_scale_fn(score_scale_fn)

    return _needleman_wunsch_batch(
        seq_pairs,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
        enable_gap_close_penalty=enable_gap_close_penalty,
        score_scaler_fn=score_scale_fn,
    )
