from __future__ import annotations

import math
import random

import numpy as np


def to_ascii(
    aligned_a: str,
    aligned_b: str,
    score: float,
    *,
    line_width: int | None = 120,
) -> str:
    """Render a pairwise alignment as a 3-line ASCII view (optionally wrapped)."""
    if len(aligned_a) != len(aligned_b):
        raise ValueError("Aligned strings must have the same length.")

    mid = []
    for ca, cb in zip(aligned_a, aligned_b, strict=True):
        if ca == "-" or cb == "-":
            mid.append(" ")
        elif ca == cb:
            mid.append("|")
        else:
            mid.append(".")
    mid_line = "".join(mid)

    header = f"Score: {score}"
    if line_width is None or line_width <= 0:
        return f"{header}\n{aligned_a}\n{mid_line}\n{aligned_b}\n"

    blocks = [header]
    for k in range(0, len(aligned_a), line_width):
        blocks.append(aligned_a[k : k + line_width])
        blocks.append(mid_line[k : k + line_width])
        blocks.append(aligned_b[k : k + line_width])
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


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
    - Substitution: lookup in `score_matrix` (dict-of-dicts).
    - Gaps: affine (negative scores):
          gap_run_score(length=k) = gap_open + gap_extend * (k - 1)

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

    Args:
        seq1: DNA sequence 1.
        seq2: DNA sequence 2.
        score_matrix: Substitution matrix as dict-of-dicts.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap by 1.
        seq1_left_free: If True, leading gaps in seq1 are free.
        seq1_right_free: If True, trailing gaps in seq1 are free.
        seq2_left_free: If True, leading gaps in seq2 are free.
        seq2_right_free: If True, trailing gaps in seq2 are free.

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
        # Entire alignment is gaps in seq1 vs letters in seq2
        gap_free = seq1_left_free or seq1_right_free
        score = 0.0 if gap_free else float(gap_open + gap_extend * (m - 1)) if m > 0 else 0.0
        return "-" * m, b, score
    if m == 0:
        # Entire alignment is letters in seq1 vs gaps in seq2
        gap_free = seq2_left_free or seq2_right_free
        score = 0.0 if gap_free else float(gap_open + gap_extend * (n - 1)) if n > 0 else 0.0
        return a, "-" * n, score

    # Masks (exclude 3 == both gaps)
    masks = np.array([0, 1, 2], dtype=np.int8)
    num_states = masks.size

    # Step deltas implied by mask bits
    step_i = np.array([0 if ((int(mask) >> BIT_SEQ1) & 1) else 1 for mask in masks], dtype=np.int8)
    step_j = np.array([0 if ((int(mask) >> BIT_SEQ2) & 1) else 1 for mask in masks], dtype=np.int8)

    neg_inf = -math.inf

    dp = np.full((num_states, n + 1, m + 1), neg_inf, dtype=np.float64)
    ptr_state = np.full((num_states, n + 1, m + 1), -1, dtype=np.int8)
    ptr_di = np.zeros((num_states, n + 1, m + 1), dtype=np.int8)
    ptr_dj = np.zeros((num_states, n + 1, m + 1), dtype=np.int8)

    # Start at (0,0) with mask 0 as benign prior state.
    dp[0, 0, 0] = 0.0
    ptr_state[0, 0, 0] = 0

    def gap_penalty(cur_mask: int, prev_mask: int, i_: int, j_: int) -> int:
        pen = 0

        # seq1 is gapped in mask bit0
        if cur_mask & MASK_SEQ1:
            if not ((i_ == 0 and seq1_left_free) or (i_ == n and seq1_right_free)):
                pen += gap_extend if (prev_mask & MASK_SEQ1) else gap_open

        # seq2 is gapped in mask bit1
        if cur_mask & MASK_SEQ2:
            if not ((j_ == 0 and seq2_left_free) or (j_ == m and seq2_right_free)):
                pen += gap_extend if (prev_mask & MASK_SEQ2) else gap_open

        return pen

    def col_sub_score(mask: int, i_: int, j_: int) -> float:
        if mask != 0:
            return 0.0
        return float(score_matrix[a[i_ - 1]][b[j_ - 1]])

    # ---- DP fill ----
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            if i == 0 and j == 0:
                continue

            for s, mask in enumerate(masks):
                di = int(step_i[s])
                dj = int(step_j[s])
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

                    score = float(prev + sub + gap_penalty(int(mask), int(prev_mask), i, j))
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
) -> int:
    """
    Score a fully specified alignment under:
      - substitution matrix
      - affine gaps (gap_open + gap_extend*(k-1), with negative scores)
      - optional free terminal gaps on each sequence end.

    Args:
        aligned_seq1: Alignment string for seq1 (includes '-').
        aligned_seq2: Alignment string for seq2 (includes '-').
        score_matrix: Substitution matrix as dict-of-dicts.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap.
        seq1_left_free/seq1_right_free: Free-end flags for gaps in seq1.
        seq2_left_free/seq2_right_free: Free-end flags for gaps in seq2.

    Returns:
        Total alignment score as an int.
    """
    if len(aligned_seq1) != len(aligned_seq2):
        raise ValueError("Alignment strings must have same length.")

    total = 0
    k = 0
    L = len(aligned_seq1)

    while k < L:
        c1 = aligned_seq1[k]
        c2 = aligned_seq2[k]

        if c1 != "-" and c2 != "-":
            total += int(score_matrix[c1][c2])
            k += 1
            continue

        if c1 == "-" and c2 == "-":
            raise ValueError("Invalid alignment column: both gaps.")

        if c1 == "-":
            # Gap run in seq1 (columns: '-' vs base)
            r = 1
            while k + r < L and aligned_seq1[k + r] == "-":
                r += 1
            is_leading = k == 0
            is_trailing = k + r == L
            free = (is_leading and seq1_left_free) or (is_trailing and seq1_right_free)
            if not free:
                total += gap_open + gap_extend * (r - 1)
            k += r
        else:
            # Gap run in seq2 (columns: base vs '-')
            r = 1
            while k + r < L and aligned_seq2[k + r] == "-":
                r += 1
            is_leading = k == 0
            is_trailing = k + r == L
            free = (is_leading and seq2_left_free) or (is_trailing and seq2_right_free)
            if not free:
                total += gap_open + gap_extend * (r - 1)
            k += r

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
) -> int:
    """
    Brute-force best score by enumerating all global alignments (paths to (n,m)).

    This is exponential; it is only practical for tiny sequences (e.g., <= 4).

    Args:
        seq1, seq2: Sequences to align.
        score_matrix: Substitution scoring as dict-of-dicts.
        gap_open, gap_extend: Affine gap scores (negative penalties).
        seq1_left_free/seq1_right_free/seq2_left_free/seq2_right_free: Free-end flags.

    Returns:
        Maximum achievable alignment score (int).
    """
    a = seq1.upper()
    b = seq2.upper()
    n = len(a)
    m = len(b)

    best = -10**18

    def rec(i: int, j: int, out_a: list[str], out_b: list[str]) -> None:
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
    return int(best)


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
    print(to_ascii(aligned1, aligned2, score))

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
    print(to_ascii(aligned1, aligned2, score))

    # ---- Correctness tests vs brute force on small sequences ----
    random.seed(0)
    alphabet = "ACGT"
    settings = [
        ("global", dict(seq1_left_free=False, seq1_right_free=False, seq2_left_free=False, seq2_right_free=False)),
        ("endfree_all", dict(seq1_left_free=True, seq1_right_free=True, seq2_left_free=True, seq2_right_free=True)),
        ("fit_seq1_to_seq2", dict(seq1_left_free=True, seq1_right_free=True, seq2_left_free=False, seq2_right_free=False)),
        ("fit_seq2_to_seq1", dict(seq1_left_free=False, seq1_right_free=False, seq2_left_free=True, seq2_right_free=True)),
        ("free_left_both_right_none", dict(seq1_left_free=True, seq1_right_free=False, seq2_left_free=True, seq2_right_free=False)),
        ("free_right_both_left_none", dict(seq1_left_free=False, seq1_right_free=True, seq2_left_free=False, seq2_right_free=True)),
    ]

    match = 2
    mismatch = -1
    gap_open = -5
    gap_extend = -1
    mat: dict[str, dict[str, int | float]] = {
        "A": {"A": match, "C": mismatch, "G": mismatch, "T": mismatch},
        "C": {"A": mismatch, "C": match, "G": mismatch, "T": mismatch},
        "G": {"A": mismatch, "C": mismatch, "G": match, "T": mismatch},
        "T": {"A": mismatch, "C": mismatch, "G": mismatch, "T": match},
    }

    print("=== Brute-force score checks (n,m <= 4) ===")
    all_ok = True
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

            if int(dp_score) != brute:
                all_ok = False
                print(f"[FAIL] setting={name} a={a!r} b={b!r} dp={dp_score} brute={brute} flags={flags}")
                aligned_a, aligned_b, _ = needleman_wunsch(
                    a,
                    b,
                    score_matrix=mat,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    **flags,
                )
                print(to_ascii(aligned_a, aligned_b, dp_score, line_width=None))
                break
        if not all_ok:
            break

    print("All tests passed!" if all_ok else "Some tests failed.")
