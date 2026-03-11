from __future__ import annotations

import sgad_rust_native as _native

RustScoreScaler = getattr(_native, "RustScoreScaler")
make_rust_score_scaler = getattr(_native, "make_rust_score_scaler")
_needleman_wunsch = getattr(_native, "needleman_wunsch")


def needleman_wunsch(
    seq1: str,
    seq2: str,
    *,
    score_matrix: dict[str, dict[str, int | float]],
    gap_open: int = -5,
    gap_extend: int = -1,
    seq1_left_free: bool = False,
    seq1_right_free: bool = False,
    seq2_left_free: bool = False,
    seq2_right_free: bool = False,
    score_scale_fn=None,
) -> tuple[str, str, float]:
    """Rust-accelerated pairwise Needleman-Wunsch.

    Supports no-scaling mode by default and RustScoreScaler objects.
    Arbitrary Python callables for score scaling are intentionally unsupported.
    """
    if not isinstance(score_scale_fn, RustScoreScaler):
        raise NotImplementedError(
            "Rust backend supports no_score_scale_factor, None (no scaling), or RustScoreScaler from make_rust_score_scaler"
        )

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
        score_scaler_fn=score_scale_fn,
    )
