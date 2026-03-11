from __future__ import annotations

import math
import random
from typing import Protocol

import numpy as np


class ScoreScaleFn(Protocol):
    """Callable signature for per-column score scaling strategies."""

    def __call__(
        self,
        seq1_left_idx: int,
        seq1_right_idx: int,
        seq2_left_idx: int,
        seq2_right_idx: int,
        seq1_left_free: bool,
        seq1_right_free: bool,
        seq2_left_free: bool,
        seq2_right_free: bool,
    ) -> float: ...


def to_ascii(
    aligned_a: str,
    aligned_b: str,
    seq1_left_free: bool = False,
    seq1_right_free: bool = False,
    seq2_left_free: bool = False,
    seq2_right_free: bool = False,
    *,
    line_width: int | None = 120,
) -> str:
    """Render a pairwise alignment as a 3-line ASCII view (optionally wrapped).

    If a free-end flag is enabled, corresponding terminal gap runs are displayed
    as spaces in that sequence.
    """
    if len(aligned_a) != len(aligned_b):
        raise ValueError("Aligned strings must have the same length.")

    def _display_seq(seq: str, *, left_free: bool, right_free: bool) -> str:
        """Mask free terminal gap runs by replacing leading/trailing '-' with spaces."""
        chars = list(seq)
        if left_free:
            i = 0
            while i < len(chars) and chars[i] == "-":
                chars[i] = " "
                i += 1
        if right_free:
            i = len(chars) - 1
            while i >= 0 and chars[i] == "-":
                chars[i] = " "
                i -= 1
        return "".join(chars)

    disp_a = _display_seq(
        aligned_a, left_free=seq1_left_free, right_free=seq1_right_free
    )
    disp_b = _display_seq(
        aligned_b, left_free=seq2_left_free, right_free=seq2_right_free
    )

    mid = []
    for ca, cb in zip(disp_a, disp_b, strict=True):
        if ca == "-" or cb == "-":
            mid.append(" ")
        elif ca == " " or cb == " ":
            mid.append(" ")
        elif ca == cb:
            mid.append("|")
        else:
            mid.append(" ")
    mid_line = "".join(mid)

    if line_width is None or line_width <= 0:
        return f"{disp_a}\n{mid_line}\n{disp_b}\n"

    blocks: list[str] = []
    for k in range(0, len(disp_a), line_width):
        blocks.append(disp_a[k : k + line_width])
        blocks.append(mid_line[k : k + line_width])
        blocks.append(disp_b[k : k + line_width])
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def score_scale_factor(
    seq1_left_idx: int,
    seq1_right_idx: int,
    seq2_left_idx: int,
    seq2_right_idx: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
) -> float:
    """
    Compute the score scaling multiplier for a single alignment column.

    This implementation uses inverse-distance terms and only includes a term when
    the corresponding flag is enabled:

        seq1_left_free  -> 1 / (seq1_left_idx + 1)
        seq1_right_free -> 1 / (seq1_right_idx + 1)
        seq2_left_free  -> 1 / (seq2_left_idx + 1)
        seq2_right_free -> 1 / (seq2_right_idx + 1)

    The returned scale is the sum of all enabled terms. If no flags are enabled,
    this function returns 1.0.

    Args:
        seq1_left_idx: Distance from the seq1 left end or seq1 gap boundary.
        seq1_right_idx: Distance from the seq1 right end or seq1 gap boundary.
        seq2_left_idx: Distance from the seq2 left end or seq2 gap boundary.
        seq2_right_idx: Distance from the seq2 right end or seq2 gap boundary.
        seq1_left_free: Whether to include the seq1-left inverse-distance term.
        seq1_right_free: Whether to include the seq1-right inverse-distance term.
        seq2_left_free: Whether to include the seq2-left inverse-distance term.
        seq2_right_free: Whether to include the seq2-right inverse-distance term.

    Returns:
        Multiplicative score scale for the current alignment column.
    """
    penalties = []
    if seq1_left_free:
        penalties.append(1.0 / (seq1_left_idx + 1))
    if seq1_right_free:
        penalties.append(1.0 / (seq1_right_idx + 1))
    if seq2_left_free:
        penalties.append(1.0 / (seq2_left_idx + 1))
    if seq2_right_free:
        penalties.append(1.0 / (seq2_right_idx + 1))
    return sum(penalties) if penalties else 1.0


def make_score_scaler(
    decay_exponent: float = 1.0, temperature: float = 1.0
) -> ScoreScaleFn:
    """Build a configurable score-scaling callback.

    The returned callback mirrors `score_scale_factor` but each enabled term is:

        1.0 / (((idx + 1) / temperature) ** decay_exponent)

    where `idx` is one of seq1_left_idx, seq1_right_idx, seq2_left_idx,
    seq2_right_idx depending on the corresponding free-end flags.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    def _score_scale_fn(
        seq1_left_idx: int,
        seq1_right_idx: int,
        seq2_left_idx: int,
        seq2_right_idx: int,
        seq1_left_free: bool,
        seq1_right_free: bool,
        seq2_left_free: bool,
        seq2_right_free: bool,
    ) -> float:
        penalties = []
        if seq1_left_free:
            penalties.append(
                1.0 / (((seq1_left_idx + 1) / temperature) ** decay_exponent)
            )
        if seq1_right_free:
            penalties.append(
                1.0 / (((seq1_right_idx + 1) / temperature) ** decay_exponent)
            )
        if seq2_left_free:
            penalties.append(
                1.0 / (((seq2_left_idx + 1) / temperature) ** decay_exponent)
            )
        if seq2_right_free:
            penalties.append(
                1.0 / (((seq2_right_idx + 1) / temperature) ** decay_exponent)
            )
        return sum(penalties) if penalties else 1.0

    return _score_scale_fn


def no_score_scale_factor(
    seq1_left_idx: int,
    seq1_right_idx: int,
    seq2_left_idx: int,
    seq2_right_idx: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
) -> float:
    """Disable score scaling by returning 1.0 for every column."""
    _ = (
        seq1_left_idx,
        seq1_right_idx,
        seq2_left_idx,
        seq2_right_idx,
        seq1_left_free,
        seq1_right_free,
        seq2_left_free,
        seq2_right_free,
    )
    return 1.0


def column_score_scale_factor(
    mask: int,
    i_: int,
    j_: int,
    *,
    n: int,
    m: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    score_scale_fn: ScoreScaleFn = score_scale_factor,
) -> float:
    """
    Compute score scaling for a DP column from the selected scaling strategy.

    Mask semantics:
        0: (x, y)  -> letters from both sequences
        1: (-, y)  -> gap in seq1
        2: (x, -)  -> gap in seq2

    For substitution columns, nucleotide positions are used directly.
    For gap columns, the gapped sequence uses the gap boundary (Option A),
    while the non-gapped sequence uses the actual nucleotide position.

    Args:
        mask: Column mask (0, 1, or 2).
        i_: DP row after emitting the current column.
        j_: DP column after emitting the current column.
        n: Length of seq1.
        m: Length of seq2.
        seq1_left_free: Whether the seq1 left-end contribution is neutralized.
        seq1_right_free: Whether the seq1 right-end contribution is neutralized.
        seq2_left_free: Whether the seq2 left-end contribution is neutralized.
        seq2_right_free: Whether the seq2 right-end contribution is neutralized.

    Returns:
        Multiplicative score scale for the current alignment column.
    """
    if mask == 0:
        seq1_left_idx = i_ - 1
        seq1_right_idx = n - i_
        seq2_left_idx = j_ - 1
        seq2_right_idx = m - j_
    elif mask == 1:
        seq1_left_idx = i_
        seq1_right_idx = n - i_
        seq2_left_idx = j_ - 1
        seq2_right_idx = m - j_
    elif mask == 2:
        seq1_left_idx = i_ - 1
        seq1_right_idx = n - i_
        seq2_left_idx = j_
        seq2_right_idx = m - j_
    else:
        raise ValueError(f"Unsupported mask: {mask}")

    return score_scale_fn(
        seq1_left_idx,
        seq1_right_idx,
        seq2_left_idx,
        seq2_right_idx,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
    )


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
    score_scale_fn: ScoreScaleFn = score_scale_factor,
) -> tuple[str, str, float]:
    """
    Pairwise global alignment (Needleman-Wunsch) with affine gaps and free end-gaps,
    implemented using 2-bit mask states (the 2D counterpart of the 3D mask approach).

    DP lattice
    ----------
    A DP cell (i, j) means we have consumed:
        - i characters from seq1
        - j characters from seq2

    Scoring
    -------
        - Substitution: lookup in `score_matrix` (dict-of-dicts), multiplied by a
            score scale returned by `score_scale_fn`.
    - Gaps: affine (negative scores), also scaled per column by the same
            score scale:
          gap_run_score(length=k) = sum(base_gap_col_score * factor_for_that_column)

    Mask state model
    ----------------
    Each alignment column emits either a letter or '-' for each sequence. We encode the
    current column type with a 2-bit mask where a set bit means that sequence is a gap:

        bit0 (value 1): seq1 is a gap in this column
        bit1 (value 2): seq2 is a gap in this column

    Valid masks are 0..2 and we exclude mask 3 (binary 11) because '--' (all gaps)
    is an invalid column.

        mask 0 (00): (x, y)   letters from both sequences
        mask 1 (01): (-, y)   gap in seq1
        mask 2 (10): (x, -)   gap in seq2

    Each mask implies the DP move (di, dj):
        di = 0 if (mask & 1) else 1
        dj = 0 if (mask & 2) else 1

    Affine gaps via mask transitions
    --------------------------------
    Affine gaps are handled by comparing the current mask to the previous mask:

    - If seq1 is gapped in current mask:
        - extend if seq1 was also gapped in previous mask
        - else open
    - Likewise for seq2.

    End-free gaps (semiglobal behavior)
    -----------------------------------
    End-free gaps are implemented by zeroing the per-column gap penalty when the gap
    occurs on the corresponding boundary:

        - If seq1_left_free and i == 0 -> gaps in seq1 cost 0 in that column
        - If seq1_right_free and i == len(seq1) -> gaps in seq1 cost 0 in that column
        - Similarly for seq2 using j.

    This boundary-based approach keeps free gaps truly terminal with respect to the
    final alignment string.

    Tie-breaking (deterministic behavior)
    -------------------------------------
    Multiple alignments can share the same optimal score. This implementation is
    deterministic because it uses fixed iteration order and strict comparisons.

    During DP updates, predecessor states are scanned in `masks` order and the
    predecessor is updated only on a strict improvement (`>`). So exact-score ties
    keep the first predecessor encountered.

    With default `masks = [0, 1, 2]`, predecessor ties prefer:
        - mask 0 (x, y) over mask 1 (-, y) over mask 2 (x, -).

    In affine-gap terms (open vs extend), ties are resolved by that same predecessor
    ordering because open/extend is represented by `prev_mask -> cur_mask` transitions.

    At termination, the end state is selected with `np.argmax` over states at (n, m),
    so final-state ties also prefer earlier masks in the same order.

    Args:
        seq1: DNA sequence 1.
        seq2: DNA sequence 2.
        score_matrix: Substitution matrix as dict-of-dicts.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap by 1.
        seq1_left_free: If True, leading gaps in seq1 are free and the seq1-left
            positional contribution is neutralized to 1.
        seq1_right_free: If True, trailing gaps in seq1 are free and the seq1-right
            positional contribution is neutralized to 1.
        seq2_left_free: If True, leading gaps in seq2 are free and the seq2-left
            positional contribution is neutralized to 1.
        seq2_right_free: If True, trailing gaps in seq2 are free and the seq2-right
            positional contribution is neutralized to 1.
        score_scale_fn: Callable that returns a multiplicative score scale per
            alignment column. Use `no_score_scale_factor` to disable scaling.

    Returns:
        (aligned_seq1, aligned_seq2, best_score)
    """
    a = seq1.upper()
    b = seq2.upper()
    n = len(a)
    m = len(b)

    BIT_SEQ1 = 0
    BIT_SEQ2 = 1
    MASK_SEQ1 = 1 << BIT_SEQ1
    MASK_SEQ2 = 1 << BIT_SEQ2

    # ---- Empty-sequence edge cases ----
    if n == 0 and m == 0:
        return "", "", 0.0
    if n == 0:
        aligned_a = "-" * m
        aligned_b = b
        score = score_alignment(
            aligned_a,
            aligned_b,
            score_matrix=score_matrix,
            gap_open=gap_open,
            gap_extend=gap_extend,
            seq1_left_free=seq1_left_free,
            seq1_right_free=seq1_right_free,
            seq2_left_free=seq2_left_free,
            seq2_right_free=seq2_right_free,
            score_scale_fn=score_scale_fn,
        )
        return aligned_a, aligned_b, score
    if m == 0:
        aligned_a = a
        aligned_b = "-" * n
        score = score_alignment(
            aligned_a,
            aligned_b,
            score_matrix=score_matrix,
            gap_open=gap_open,
            gap_extend=gap_extend,
            seq1_left_free=seq1_left_free,
            seq1_right_free=seq1_right_free,
            seq2_left_free=seq2_left_free,
            seq2_right_free=seq2_right_free,
            score_scale_fn=score_scale_fn,
        )
        return aligned_a, aligned_b, score

    # Masks (exclude 3 == both gaps); Fixed order also defines tie preference.
    masks = np.array([0, 1, 2], dtype=np.uint8)
    num_states = masks.size
    # Compute per-mask DP step deltas in vectorized form.
    # Each row is [di, dj] where:
    #   - bit=0 (letter) -> step 1
    #   - bit=1 (gap) -> step 0.
    # With masks=[0, 1, 2], this yields [[1, 1], [0, 1], [1, 0]].
    step_deltas = 1 - ((masks[:, None] >> np.array([BIT_SEQ1, BIT_SEQ2])) & 1)

    neg_inf = -math.inf

    dp = np.full((num_states, n + 1, m + 1), neg_inf, dtype=np.float64)
    ptr_state = np.full((num_states, n + 1, m + 1), -1, dtype=np.int8)
    ptr_di = np.zeros((num_states, n + 1, m + 1), dtype=np.int8)
    ptr_dj = np.zeros((num_states, n + 1, m + 1), dtype=np.int8)

    # Start at (0,0) with mask 0 as benign prior state.
    dp[0, 0, 0] = 0.0
    ptr_state[0, 0, 0] = 0

    def gap_penalty(cur_mask: int, prev_mask: int, i_: int, j_: int) -> float:
        """Return affine gap cost for the current column mask at DP cell (i_, j_)."""
        pen = 0.0
        factor = column_score_scale_factor(
            cur_mask,
            i_,
            j_,
            n=n,
            m=m,
            seq1_left_free=seq1_left_free,
            seq1_right_free=seq1_right_free,
            seq2_left_free=seq2_left_free,
            seq2_right_free=seq2_right_free,
            score_scale_fn=score_scale_fn,
        )

        # seq1 is gapped in mask bit0
        if cur_mask & MASK_SEQ1:
            if not ((i_ == 0 and seq1_left_free) or (i_ == n and seq1_right_free)):
                base_gap = gap_extend if (prev_mask & MASK_SEQ1) else gap_open
                pen += float(base_gap) * factor

        # seq2 is gapped in mask bit1
        if cur_mask & MASK_SEQ2:
            if not ((j_ == 0 and seq2_left_free) or (j_ == m and seq2_right_free)):
                base_gap = gap_extend if (prev_mask & MASK_SEQ2) else gap_open
                pen += float(base_gap) * factor

        return pen

    def col_sub_score(mask: int, i_: int, j_: int) -> float:
        """Return substitution score for this column, or 0.0 for any gap-containing mask."""
        if mask != 0:
            return 0.0
        factor = column_score_scale_factor(
            mask,
            i_,
            j_,
            n=n,
            m=m,
            seq1_left_free=seq1_left_free,
            seq1_right_free=seq1_right_free,
            seq2_left_free=seq2_left_free,
            seq2_right_free=seq2_right_free,
            score_scale_fn=score_scale_fn,
        )
        return float(score_matrix[a[i_ - 1]][b[j_ - 1]]) * factor

    # ---- DP fill ----
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            if i == 0 and j == 0:
                continue

            for s, mask in enumerate(masks):
                di, dj = (int(step_deltas[s, 0]), int(step_deltas[s, 1]))
                if i < di or j < dj:
                    continue

                pi, pj = i - di, j - dj
                sub = col_sub_score(int(mask), i, j)

                best = neg_inf
                best_prev = -1

                for ps, prev_mask in enumerate(masks):
                    prev = dp[ps, pi, pj]
                    if prev == neg_inf:
                        continue

                    score = float(
                        prev + sub + gap_penalty(int(mask), int(prev_mask), i, j)
                    )
                    # Strict '>' means equal scores keep the earlier prev_mask (deterministic).
                    if score > best:
                        best = score
                        best_prev = ps

                if best_prev >= 0:
                    dp[s, i, j] = best
                    ptr_state[s, i, j] = best_prev
                    ptr_di[s, i, j] = di
                    ptr_dj[s, i, j] = dj

    # ---- Termination at (n,m) ----
    end_scores = dp[:, n, m]
    # np.argmax returns first max index, giving deterministic final-state tie-breaking.
    best_state = int(np.argmax(end_scores))
    best_score = float(end_scores[best_state])

    # ---- Backtrack ----
    i, j, state = n, m, best_state
    out_a: list[str] = []
    out_b: list[str] = []

    while i > 0 or j > 0:
        mask = int(masks[state])
        out_a.append("-" if (mask & MASK_SEQ1) else a[i - 1])
        out_b.append("-" if (mask & MASK_SEQ2) else b[j - 1])

        prev_state = int(ptr_state[state, i, j])
        di = int(ptr_di[state, i, j])
        dj = int(ptr_dj[state, i, j])

        if prev_state < 0:
            raise RuntimeError("Unset pointer encountered during traceback.")

        i -= di
        j -= dj
        state = prev_state

    aligned_a = "".join(reversed(out_a))
    aligned_b = "".join(reversed(out_b))

    return aligned_a, aligned_b, float(best_score)


def predict_dimer(
    primer1: str,
    primer2: str,
    score_matrix: dict[str, dict[str, int | float]],
    gap_open: int,
    gap_extend: int,
    score_scale_fn: ScoreScaleFn = score_scale_factor,
) -> tuple[str, str, float]:
    """
    Predict primer dimer structure via pairwise alignment.

    This wrapper reverse-complements `primer2` for alignment and calls
    `needleman_wunsch` with fixed dimer-oriented free-end settings:

    - seq1_left_free=False, seq1_right_free=True
    - seq2_left_free=True, seq2_right_free=False

    Args:
        primer1: Primer sequence 1.
        primer2: Primer sequence 2 (will be reverse-complemented internally).
        score_matrix: Substitution matrix as dict-of-dicts.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap by 1.
        score_scale_fn: Multiplicative score scaling callback.

    Returns:
        (aligned_primer1, aligned_primer2_complement, score)
    """
    primer2_rc = primer2.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

    seq1_left_free, seq1_right_free = False, True
    seq2_left_free, seq2_right_free = True, False

    a1, a2_rc, score = needleman_wunsch(
        primer1,
        primer2_rc,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
        score_scale_fn=score_scale_fn,
    )
    a2_comp = a2_rc.translate(str.maketrans("ACGTacgt", "TGCAtgca"))
    return a1, a2_comp, score


def dimer_to_ascii(
    aligned_primer1: str,
    aligned_primer2: str,
    *,
    line_width: int | None = 120,
) -> str:
    """Render dimer-alignment ASCII with fixed dimer free-end masking settings."""
    # Complement the second sequence
    aligned_primer2 = aligned_primer2.translate(str.maketrans("ACGTacgt", "TGCAtgca"))
    res = to_ascii(
        aligned_primer1,
        aligned_primer2,
        seq1_left_free=False,
        seq1_right_free=True,
        seq2_left_free=True,
        seq2_right_free=False,
        line_width=line_width,
    )
    # Restore complement on each displayed third-sequence line.
    lines = res.splitlines()
    if len(lines) < 3:
        raise ValueError("Expected at least 3 lines in the ASCII output.")
    for idx in range(2, len(lines), 4):
        lines[idx] = lines[idx].translate(str.maketrans("ACGTacgt", "TGCAtgca"))
    return "\n".join(lines) + "\n"


def score_alignment(
    aligned_seq1: str,
    aligned_seq2: str,
    *,
    score_matrix: dict[str, dict[str, int | float]],
    gap_open: int,
    gap_extend: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    score_scale_fn: ScoreScaleFn = score_scale_factor,
) -> float:
    """
    Score a fully specified alignment under:
      - substitution matrix
      - affine gaps (gap_open + gap_extend*(k-1), with negative scores)
      - optional free terminal gaps on each sequence end
    - configurable score scaling applied to every alignment column

    Args:
        aligned_seq1: Alignment string for seq1 (includes '-').
        aligned_seq2: Alignment string for seq2 (includes '-').
        score_matrix: Substitution matrix as dict-of-dicts.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap.
        seq1_left_free/seq1_right_free: Free-end flags for gaps in seq1 and neutralizers
            for the seq1 positional contributions.
        seq2_left_free/seq2_right_free: Free-end flags for gaps in seq2 and neutralizers
            for the seq2 positional contributions.
        score_scale_fn: Callable that returns a multiplicative score scale per column.

    Returns:
        Total alignment score as a float.
    """
    if len(aligned_seq1) != len(aligned_seq2):
        raise ValueError("Alignment strings must have same length.")

    ungapped_seq1 = aligned_seq1.replace("-", "")
    ungapped_seq2 = aligned_seq2.replace("-", "")
    n = len(ungapped_seq1)
    m = len(ungapped_seq2)

    def prev_is_gap_in_seq1(col_idx: int) -> bool:
        """Check whether the previous alignment column contains a gap in seq1 only."""
        return (
            col_idx > 0
            and aligned_seq1[col_idx - 1] == "-"
            and aligned_seq2[col_idx - 1] != "-"
        )

    def prev_is_gap_in_seq2(col_idx: int) -> bool:
        """Check whether the previous alignment column contains a gap in seq2 only."""
        return (
            col_idx > 0
            and aligned_seq2[col_idx - 1] == "-"
            and aligned_seq1[col_idx - 1] != "-"
        )

    total = 0.0
    seq1_pos = 0
    seq2_pos = 0

    for col_idx, (c1, c2) in enumerate(zip(aligned_seq1, aligned_seq2, strict=True)):
        if c1 != "-" and c2 != "-":
            factor = score_scale_fn(
                seq1_pos,
                n - seq1_pos - 1,
                seq2_pos,
                m - seq2_pos - 1,
                seq1_left_free=seq1_left_free,
                seq1_right_free=seq1_right_free,
                seq2_left_free=seq2_left_free,
                seq2_right_free=seq2_right_free,
            )
            total += float(score_matrix[c1][c2]) * factor
            seq1_pos += 1
            seq2_pos += 1
            continue

        if c1 == "-" and c2 == "-":
            raise ValueError("Invalid alignment column: both gaps.")

        if c1 == "-":
            free = (seq1_pos == 0 and seq1_left_free) or (
                seq1_pos == n and seq1_right_free
            )
            if not free:
                factor = score_scale_fn(
                    seq1_pos,
                    n - seq1_pos,
                    seq2_pos,
                    m - seq2_pos - 1,
                    seq1_left_free=seq1_left_free,
                    seq1_right_free=seq1_right_free,
                    seq2_left_free=seq2_left_free,
                    seq2_right_free=seq2_right_free,
                )
                base_gap = gap_extend if prev_is_gap_in_seq1(col_idx) else gap_open
                total += float(base_gap) * factor
            seq2_pos += 1
        else:
            free = (seq2_pos == 0 and seq2_left_free) or (
                seq2_pos == m and seq2_right_free
            )
            if not free:
                factor = score_scale_fn(
                    seq1_pos,
                    n - seq1_pos - 1,
                    seq2_pos,
                    m - seq2_pos,
                    seq1_left_free=seq1_left_free,
                    seq1_right_free=seq1_right_free,
                    seq2_left_free=seq2_left_free,
                    seq2_right_free=seq2_right_free,
                )
                base_gap = gap_extend if prev_is_gap_in_seq2(col_idx) else gap_open
                total += float(base_gap) * factor
            seq1_pos += 1

    return total


def brute_force_best_score(
    seq1: str,
    seq2: str,
    *,
    score_matrix: dict[str, dict[str, int | float]],
    gap_open: int,
    gap_extend: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    score_scale_fn: ScoreScaleFn = score_scale_factor,
) -> float:
    """
    Brute-force best score by enumerating all global alignments (paths to (n,m)).

    This is exponential; it is only practical for tiny sequences (e.g., <= 4).

    Args:
        seq1, seq2: Sequences to align.
        score_matrix: Substitution scoring as dict-of-dicts.
        gap_open, gap_extend: Affine gap scores (negative penalties).
        seq1_left_free/seq1_right_free/seq2_left_free/seq2_right_free: Free-end flags.
        score_scale_fn: Callable that returns a multiplicative score scale per column.

    Returns:
        Maximum achievable alignment score (float).
    """
    a = seq1.upper()
    b = seq2.upper()
    n = len(a)
    m = len(b)

    best = -math.inf

    def rec(i: int, j: int, out_a: list[str], out_b: list[str]) -> None:
        """Enumerate all alignment paths from DP position (i, j) and update best score."""
        nonlocal best

        if i == n and j == m:
            s = score_alignment(
                "".join(out_a),
                "".join(out_b),
                score_matrix=score_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend,
                seq1_left_free=seq1_left_free,
                seq1_right_free=seq1_right_free,
                seq2_left_free=seq2_left_free,
                seq2_right_free=seq2_right_free,
                score_scale_fn=score_scale_fn,
            )
            best = max(best, s)
            return

        # Diagonal: align a[i] with b[j]
        if i < n and j < m:
            out_a.append(a[i])
            out_b.append(b[j])
            rec(i + 1, j + 1, out_a, out_b)
            out_a.pop()
            out_b.pop()

        # Down: align a[i] with '-': gap in seq2
        if i < n:
            out_a.append(a[i])
            out_b.append("-")
            rec(i + 1, j, out_a, out_b)
            out_a.pop()
            out_b.pop()

        # Right: align '-' with b[j]: gap in seq1
        if j < m:
            out_a.append("-")
            out_b.append(b[j])
            rec(i, j + 1, out_a, out_b)
            out_a.pop()
            out_b.pop()

    rec(0, 0, [], [])
    return float(best)


if __name__ == "__main__":
    # ---- Demo runs ----
    s1 = "ACGTTGAC"
    s2 = "ACTTGACC"

    mat: dict[str, dict[str, int | float]] = {
        "A": {"A": 2, "C": -1, "G": -1, "T": -1},
        "C": {"A": -1, "C": 2, "G": -1, "T": -1},
        "G": {"A": -1, "C": -1, "G": 2, "T": -1},
        "T": {"A": -1, "C": -1, "G": -1, "T": 2},
    }

    aligned1, aligned2, score = needleman_wunsch(
        s1,
        s2,
        score_matrix=mat,
        gap_open=-5,
        gap_extend=-1,
    )
    print("=== Pure global (no free ends) ===")
    print(to_ascii(aligned1, aligned2))

    aligned1, aligned2, score = needleman_wunsch(
        s1,
        s2,
        score_matrix=mat,
        gap_open=-5,
        gap_extend=-1,
        seq1_left_free=True,
        seq1_right_free=True,
        seq2_left_free=True,
        seq2_right_free=True,
    )
    print("=== End-gap-free global (all ends free) ===")
    print(
        to_ascii(
            aligned1,
            aligned2,
            seq1_left_free=True,
            seq1_right_free=True,
            seq2_left_free=True,
            seq2_right_free=True,
        )
    )

    # ---- Correctness tests vs brute force on small sequences ----
    random.seed(0)
    alphabet = "ACGT"
    settings = [
        (
            "global",
            dict(
                seq1_left_free=False,
                seq1_right_free=False,
                seq2_left_free=False,
                seq2_right_free=False,
            ),
        ),
        (
            "endfree_all",
            dict(
                seq1_left_free=True,
                seq1_right_free=True,
                seq2_left_free=True,
                seq2_right_free=True,
            ),
        ),
        (
            "fit_seq1_to_seq2",
            dict(
                seq1_left_free=True,
                seq1_right_free=True,
                seq2_left_free=False,
                seq2_right_free=False,
            ),
        ),
        (
            "fit_seq2_to_seq1",
            dict(
                seq1_left_free=False,
                seq1_right_free=False,
                seq2_left_free=True,
                seq2_right_free=True,
            ),
        ),
        (
            "free_left_both_right_none",
            dict(
                seq1_left_free=True,
                seq1_right_free=False,
                seq2_left_free=True,
                seq2_right_free=False,
            ),
        ),
        (
            "free_right_both_left_none",
            dict(
                seq1_left_free=False,
                seq1_right_free=True,
                seq2_left_free=False,
                seq2_right_free=True,
            ),
        ),
    ]

    match = 2
    mismatch = -1
    gap_open = -5
    gap_extend = -1
    mat = {
        "A": {"A": match, "C": mismatch, "G": mismatch, "T": mismatch},
        "C": {"A": mismatch, "C": match, "G": mismatch, "T": mismatch},
        "G": {"A": mismatch, "C": mismatch, "G": match, "T": mismatch},
        "T": {"A": mismatch, "C": mismatch, "G": mismatch, "T": match},
    }

    print("=== Brute-force score checks (n,m <= 4) ===")
    all_ok = True
    tol = 1e-9
    for name, flags in settings:
        for _ in range(100):
            n = random.randint(0, 4)
            m = random.randint(0, 4)
            a = "".join(random.choice(alphabet) for _ in range(n))
            b = "".join(random.choice(alphabet) for _ in range(m))

            _, _, dp_score = needleman_wunsch(
                a,
                b,
                score_matrix=mat,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )
            brute = brute_force_best_score(
                a,
                b,
                score_matrix=mat,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )

            if abs(dp_score - brute) > tol:
                all_ok = False
                print(
                    f"[FAIL] setting={name} a={a!r} b={b!r} dp={dp_score} brute={brute} flags={flags}"
                )
                aligned_a, aligned_b, _ = needleman_wunsch(
                    a,
                    b,
                    score_matrix=mat,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    **flags,
                )
                print(to_ascii(aligned_a, aligned_b, line_width=None, **flags))
                break
        if not all_ok:
            break

    # Extra real-world style case from README
    primer1 = "GAGATATGAGGAGAGAGAGACAGAGG"  # right free only
    primer2_rc = "GAACAGAGGGAGAGACTAACCTTG"  # left free only
    seq1_left_free, seq1_right_free = False, True
    seq2_left_free, seq2_right_free = True, False

    a1, a2, dp_score = needleman_wunsch(
        primer1,
        primer2_rc,
        score_matrix=mat,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
        score_scale_fn=score_scale_factor,
    )

    rescored = score_alignment(
        a1,
        a2,
        score_matrix=mat,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=seq1_left_free,
        seq1_right_free=seq1_right_free,
        seq2_left_free=seq2_left_free,
        seq2_right_free=seq2_right_free,
        score_scale_fn=score_scale_factor,
    )

    if abs(dp_score - rescored) > tol:
        print(f"[FAIL] README pairwise case dp={dp_score} rescored={rescored}")
        print(a1)
        print(a2)
        raise SystemExit(1)

    print("=== README pairwise case ===")
    print(
        to_ascii(
            a1,
            a2,
            seq1_left_free=seq1_left_free,
            seq1_right_free=seq1_right_free,
            seq2_left_free=seq2_left_free,
            seq2_right_free=seq2_right_free,
        ),
        end="",
    )
    print(f"Score (DP): {dp_score}")
    print(f"Score (rescored): {rescored}")

    print("All tests passed!" if all_ok else "Some tests failed.")
