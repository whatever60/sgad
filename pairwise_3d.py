from __future__ import annotations

import math
import random
from itertools import combinations

import numpy as np


def score_alignment_3d(
    aligned_seq1: str,
    aligned_seq2: str,
    aligned_seq3: str,
    *,
    score_matrix: dict[str, dict[str, int]],
    gap_open: int,
    gap_extend: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    seq3_left_free: bool,
    seq3_right_free: bool,
) -> int:
    """
    Score a 3-sequence alignment under:
      - sum-of-pairs substitution scoring using `score_matrix`
      - affine gaps per sequence: gap_open + gap_extend*(k-1)  (negative scores)
      - optional free terminal gaps (leading/trailing) per sequence.

    Notes:
      - Columns with all three gaps are invalid.
      - Gap penalties are assessed independently for each sequence by scanning its '-' runs.

    Args:
        aligned_seq1/aligned_seq2/aligned_seq3: Alignment strings (same length, include '-').
        score_matrix: Substitution matrix as dict-of-dicts.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap.
        seq{1,2,3}_{left,right}_free: Whether leading/trailing gaps in that sequence are free.

    Returns:
        Total alignment score as int.
    """
    if not (len(aligned_seq1) == len(aligned_seq2) == len(aligned_seq3)):
        raise ValueError("All aligned strings must have the same length.")

    L = len(aligned_seq1)
    total = 0

    # --- Sum-of-pairs substitution score per column ---
    for pos in range(L):
        c1, c2, c3 = aligned_seq1[pos], aligned_seq2[pos], aligned_seq3[pos]
        if c1 == "-" and c2 == "-" and c3 == "-":
            raise ValueError("Invalid alignment column: all gaps.")

        letters: list[str] = []
        for c in (c1, c2, c3):
            if c != "-":
                letters.append(c)

        for x, y in combinations(letters, 2):
            total += int(score_matrix[x][y])

    # --- Affine gap penalties per sequence (independent scans) ---
    def add_gap_penalties(aligned: str, *, left_free: bool, right_free: bool) -> int:
        s = 0
        i = 0
        while i < L:
            if aligned[i] != "-":
                i += 1
                continue

            run_len = 1
            while i + run_len < L and aligned[i + run_len] == "-":
                run_len += 1

            is_leading = i == 0
            is_trailing = i + run_len == L
            free = (is_leading and left_free) or (is_trailing and right_free)
            if not free:
                s += gap_open + gap_extend * (run_len - 1)

            i += run_len

        return s

    total += add_gap_penalties(aligned_seq1, left_free=seq1_left_free, right_free=seq1_right_free)
    total += add_gap_penalties(aligned_seq2, left_free=seq2_left_free, right_free=seq2_right_free)
    total += add_gap_penalties(aligned_seq3, left_free=seq3_left_free, right_free=seq3_right_free)

    return total


def needleman_wunsch_3d(
    seq1: str,
    seq2: str,
    seq3: str,
    *,
    score_matrix: dict[str, dict[str, int]],
    gap_open: int = -5,
    gap_extend: int = -1,
    seq1_left_free: bool = False,
    seq1_right_free: bool = False,
    seq2_left_free: bool = False,
    seq2_right_free: bool = False,
    seq3_left_free: bool = False,
    seq3_right_free: bool = False,
) -> tuple[str, str, str, float]:
    """
    Exact 3-sequence alignment (3D DP) with affine gap penalties and free end-gaps.

    Overview
    --------
    This function performs an exact global alignment of three sequences using dynamic
    programming over a 3D lattice. A DP cell (i, j, k) represents having consumed:
        - i characters from seq1
        - j characters from seq2
        - k characters from seq3

    Scoring
    -------
    - Substitution: sum-of-pairs using `score_matrix` (one matrix for all pairs).
      For a column containing letters (ignoring gaps), the column substitution score is:
          S(x, y) + S(x, z) + S(y, z)    (when all three letters are present)
      and reduces to a single pair score when only two letters are present.
    - Gaps: affine, per-sequence (negative scores):
          gap_run_score(length=k) = gap_open + gap_extend * (k - 1)

    DP state model (bitmask states)
    -------------------------------
    Instead of M/IX/IY (pairwise case), we represent the current column type with a
    3-bit mask that encodes which sequences emit a gap '-' in the CURRENT column:

        bit0 (value 1): seq1 is a gap in this column
        bit1 (value 2): seq2 is a gap in this column
        bit2 (value 4): seq3 is a gap in this column

    If a bit is 1, that sequence emits '-' in the column; if 0, it emits a letter.
    Allowed masks are 0..6. Mask 7 (binary 111) would be an all-gap column '---' and is
    disallowed.

    Valid masks (examples, x/y/z are letters):
        mask 0 (000): (x, y, z)
        mask 1 (001): (-, y, z)
        mask 2 (010): (x, -, z)
        mask 3 (011): (-, -, z)
        mask 4 (100): (x, y, -)
        mask 5 (101): (-, y, -)
        mask 6 (110): (x, -, -)

    Each mask implies a move (di, dj, dk) through the 3D lattice:
        di = 0 if (mask & 1) else 1
        dj = 0 if (mask & 2) else 1
        dk = 0 if (mask & 4) else 1

    That is, if a sequence is gapped in the current column, we do not consume a character
    from that sequence; otherwise we consume exactly one.

    Affine gaps via mask transitions
    -------------------------------
    Affine gaps are computed by comparing the current mask to the previous mask:

    For each sequence that is gapped in the current mask:
        - if that sequence was also gapped in the previous mask  -> gap_extend
        - else                                                  -> gap_open

    If the current column gaps multiple sequences (e.g., mask 3 gaps seq1 and seq2),
    the per-column gap penalty is the sum of the contributions for each gapped sequence.

    End-free gaps (semiglobal behavior)
    -----------------------------------
    End-free gaps are implemented by zeroing out per-column gap penalties on boundary
    planes for that sequence:
        - if seqX_left_free  and indexX == 0       -> gap penalty for seqX is 0
        - if seqX_right_free and indexX == len(seqX) -> gap penalty for seqX is 0

    This boundary-based handling ensures gaps are only free when they are truly terminal
    with respect to the final alignment string, avoiding cases where a "padded" gap would
    become internal after additional columns.

    Args:
        seq1: DNA sequence 1.
        seq2: DNA sequence 2.
        seq3: DNA sequence 3.
        score_matrix: Substitution matrix (typically a numpy array) used for sum-of-pairs.
        gap_open: Negative gap score for opening a gap (per sequence).
        gap_extend: Negative gap score for extending a gap by 1 (per sequence).
        seq1_left_free: If True, leading gaps in seq1 are free.
        seq1_right_free: If True, trailing gaps in seq1 are free.
        seq2_left_free: If True, leading gaps in seq2 are free.
        seq2_right_free: If True, trailing gaps in seq2 are free.
        seq3_left_free: If True, leading gaps in seq3 are free.
        seq3_right_free: If True, trailing gaps in seq3 are free.

    Returns:
        (aligned_seq1, aligned_seq2, aligned_seq3, best_score)
    """
    a = seq1.upper()
    b = seq2.upper()
    c = seq3.upper()
    n, m, len3 = len(a), len(b), len(c)

    BIT_SEQ1 = 0
    BIT_SEQ2 = 1
    BIT_SEQ3 = 2
    MASK_SEQ1 = 1 << BIT_SEQ1
    MASK_SEQ2 = 1 << BIT_SEQ2
    MASK_SEQ3 = 1 << BIT_SEQ3

    # Masks exclude 7 (all gaps)
    masks = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int8)
    num_states = masks.size

    # Steps implied by mask bits
    step_i = np.array([0 if ((int(mask) >> BIT_SEQ1) & 1) else 1 for mask in masks], dtype=np.int8)
    step_j = np.array([0 if ((int(mask) >> BIT_SEQ2) & 1) else 1 for mask in masks], dtype=np.int8)
    step_k = np.array([0 if ((int(mask) >> BIT_SEQ3) & 1) else 1 for mask in masks], dtype=np.int8)

    neg_inf = -math.inf
    dp = np.full((num_states, n + 1, m + 1, len3 + 1), neg_inf, dtype=np.float64)

    # Pointers store (prev_state, di, dj, dk)
    ptr_state = np.full((num_states, n + 1, m + 1, len3 + 1), -1, dtype=np.int8)
    ptr_di = np.zeros((num_states, n + 1, m + 1, len3 + 1), dtype=np.int8)
    ptr_dj = np.zeros((num_states, n + 1, m + 1, len3 + 1), dtype=np.int8)
    ptr_dk = np.zeros((num_states, n + 1, m + 1, len3 + 1), dtype=np.int8)

    # Start: interpret state=mask 0 as "previous column had no gaps"
    dp[0, 0, 0, 0] = 0.0
    ptr_state[0, 0, 0, 0] = 0

    left_free = (seq1_left_free, seq2_left_free, seq3_left_free)
    right_free = (seq1_right_free, seq2_right_free, seq3_right_free)
    lens = (n, m, len3)

    def gap_penalty(cur_mask: int, prev_mask: int, i_: int, j_: int, k_: int) -> int:
        """Sum affine gap penalties for sequences gapped in cur_mask (with boundary-free logic)."""
        pen = 0

        # seq1 (bit0)
        if cur_mask & MASK_SEQ1:
            if not ((i_ == 0 and left_free[0]) or (i_ == lens[0] and right_free[0])):
                pen += gap_extend if (prev_mask & MASK_SEQ1) else gap_open

        # seq2 (bit1)
        if cur_mask & MASK_SEQ2:
            if not ((j_ == 0 and left_free[1]) or (j_ == lens[1] and right_free[1])):
                pen += gap_extend if (prev_mask & MASK_SEQ2) else gap_open

        # seq3 (bit2)
        if cur_mask & MASK_SEQ3:
            if not ((k_ == 0 and left_free[2]) or (k_ == lens[2] and right_free[2])):
                pen += gap_extend if (prev_mask & MASK_SEQ3) else gap_open

        return pen

    def col_sub_score(mask: int, i_: int, j_: int, k_: int) -> int:
        """Sum-of-pairs substitution score for the column ending at (i_,j_,k_) with this mask."""
        letters: list[str] = []
        if ((mask >> BIT_SEQ1) & 1) == 0:
            letters.append(a[i_ - 1])
        if ((mask >> BIT_SEQ2) & 1) == 0:
            letters.append(b[j_ - 1])
        if ((mask >> BIT_SEQ3) & 1) == 0:
            letters.append(c[k_ - 1])

        s = 0
        for x, y in combinations(letters, 2):
            s += int(score_matrix[x][y])
        return s

    # --- DP fill ---
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            for k in range(0, len3 + 1):
                if i == 0 and j == 0 and k == 0:
                    continue

                for s, mask in enumerate(masks):
                    di, dj, dk = int(step_i[s]), int(step_j[s]), int(step_k[s])
                    if i < di or j < dj or k < dk:
                        continue

                    pi, pj, pk = i - di, j - dj, k - dk
                    sub = col_sub_score(int(mask), i, j, k)

                    best = neg_inf
                    best_prev = -1

                    for ps, prev_mask in enumerate(masks):
                        prev = dp[ps, pi, pj, pk]
                        if prev == neg_inf:
                            continue
                        score = float(prev + sub + gap_penalty(int(mask), int(prev_mask), i, j, k))
                        if score > best:
                            best = score
                            best_prev = ps

                    if best_prev >= 0:
                        dp[s, i, j, k] = best
                        ptr_state[s, i, j, k] = best_prev
                        ptr_di[s, i, j, k] = di
                        ptr_dj[s, i, j, k] = dj
                        ptr_dk[s, i, j, k] = dk

    # --- Termination: best state at (n,m,len3) ---
    end_scores = dp[:, n, m, len3]
    best_state = int(np.argmax(end_scores))
    best_score = float(end_scores[best_state])

    # --- Backtrack ---
    i, j, k = n, m, len3
    state = best_state
    out1: list[str] = []
    out2: list[str] = []
    out3: list[str] = []

    while i > 0 or j > 0 or k > 0:
        mask = int(masks[state])

        out1.append("-" if (mask & MASK_SEQ1) else a[i - 1])
        out2.append("-" if (mask & MASK_SEQ2) else b[j - 1])
        out3.append("-" if (mask & MASK_SEQ3) else c[k - 1])

        prev_state = int(ptr_state[state, i, j, k])
        di = int(ptr_di[state, i, j, k])
        dj = int(ptr_dj[state, i, j, k])
        dk = int(ptr_dk[state, i, j, k])

        if prev_state < 0:
            raise RuntimeError("Unset pointer encountered during 3D traceback.")

        i -= di
        j -= dj
        k -= dk
        state = prev_state

    return (
        "".join(reversed(out1)),
        "".join(reversed(out2)),
        "".join(reversed(out3)),
        best_score,
    )


def brute_force_best_score_3d(
    seq1: str,
    seq2: str,
    seq3: str,
    *,
    score_matrix: dict[str, dict[str, int]],
    gap_open: int,
    gap_extend: int,
    seq1_left_free: bool,
    seq1_right_free: bool,
    seq2_left_free: bool,
    seq2_right_free: bool,
    seq3_left_free: bool,
    seq3_right_free: bool,
) -> int:
    """
    Brute-force best score by enumerating all 3-sequence global alignments.

    This is exponential; only feasible for tiny sequences (e.g., lengths <= 2).

    Move set per column (7 moves): consume letters from any non-empty subset of sequences.
    """
    a = seq1.upper()
    b = seq2.upper()
    c = seq3.upper()
    n, m, len3 = len(a), len(b), len(c)

    best = -10**18

    moves = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]

    def rec(i: int, j: int, k: int, out1: list[str], out2: list[str], out3: list[str]) -> None:
        nonlocal best
        if i == n and j == m and k == len3:
            s = score_alignment_3d(
                "".join(out1),
                "".join(out2),
                "".join(out3),
                score_matrix=score_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend,
                seq1_left_free=seq1_left_free,
                seq1_right_free=seq1_right_free,
                seq2_left_free=seq2_left_free,
                seq2_right_free=seq2_right_free,
                seq3_left_free=seq3_left_free,
                seq3_right_free=seq3_right_free,
            )
            best = max(best, s)
            return

        for di, dj, dk in moves:
            if i + di > n or j + dj > m or k + dk > len3:
                continue

            out1.append(a[i] if di else "-")
            out2.append(b[j] if dj else "-")
            out3.append(c[k] if dk else "-")

            rec(i + di, j + dj, k + dk, out1, out2, out3)

            out1.pop()
            out2.pop()
            out3.pop()

    rec(0, 0, 0, [], [], [])
    return int(best)


if __name__ == "__main__":
    # Randomized consistency tests vs brute force (keep lengths tiny!)
    random.seed(0)
    alphabet = "ACGT"
    mat = {
        "A": {"A": 2, "C": -1, "G": -1, "T": -1},
        "C": {"A": -1, "C": 2, "G": -1, "T": -1},
        "G": {"A": -1, "C": -1, "G": 2, "T": -1},
        "T": {"A": -1, "C": -1, "G": -1, "T": 2},
    }
    gap_open, gap_extend = -5, -1

    settings = [
        ("global", dict(
            seq1_left_free=False, seq1_right_free=False,
            seq2_left_free=False, seq2_right_free=False,
            seq3_left_free=False, seq3_right_free=False,
        )),
        ("endfree_all", dict(
            seq1_left_free=True, seq1_right_free=True,
            seq2_left_free=True, seq2_right_free=True,
            seq3_left_free=True, seq3_right_free=True,
        )),
        ("fit_seq1", dict(
            seq1_left_free=False, seq1_right_free=False,
            seq2_left_free=True, seq2_right_free=True,
            seq3_left_free=True, seq3_right_free=True,
        )),
        ("left_free_only_all", dict(
            seq1_left_free=True, seq1_right_free=False,
            seq2_left_free=True, seq2_right_free=False,
            seq3_left_free=True, seq3_right_free=False,
        )),
        ("right_free_only_all", dict(
            seq1_left_free=False, seq1_right_free=True,
            seq2_left_free=False, seq2_right_free=True,
            seq3_left_free=False, seq3_right_free=True,
        )),
    ]

    max_length = 4
    print(f"=== 3D brute-force score checks (lengths <= {max_length}) ===")
    for name, flags in settings:
        for _ in range(10):
            n = random.randint(0, max_length)
            m = random.randint(0, max_length)
            len3 = random.randint(0, max_length)

            s1 = "".join(random.choice(alphabet) for _ in range(n))
            s2 = "".join(random.choice(alphabet) for _ in range(m))
            s3 = "".join(random.choice(alphabet) for _ in range(len3))

            a1, a2, a3, dp_score = needleman_wunsch_3d(
                s1, s2, s3,
                score_matrix=mat,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )

            brute = brute_force_best_score_3d(
                s1, s2, s3,
                score_matrix=mat,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )

            scored = score_alignment_3d(
                a1, a2, a3,
                score_matrix=mat,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )

            if int(dp_score) != brute or int(dp_score) != scored:
                print(f"[FAIL] setting={name} s1={s1!r} s2={s2!r} s3={s3!r} dp={dp_score} brute={brute} scored={scored}")
                print(a1)
                print(a2)
                print(a3)
                print(flags)
                raise SystemExit(1)

    print("All tests passed!")