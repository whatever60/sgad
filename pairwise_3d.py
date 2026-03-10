from __future__ import annotations

import math
import random
from itertools import combinations

import numpy as np


def build_score_matrix(
    *,
    alphabet: str = "ACGT",
    match: int = 2,
    mismatch: int = -1,
) -> np.ndarray:
    """
    Build a simple substitution matrix.

    Default: diagonal = match, off-diagonal = mismatch.

    Args:
        alphabet: Symbols covered by the matrix (order matters).
        match: Score for identical symbols.
        mismatch: Score for non-identical symbols.

    Returns:
        A (K, K) numpy array of scores, where K=len(alphabet).
    """
    k = len(alphabet)
    mat = np.full((k, k), mismatch, dtype=np.int64)
    np.fill_diagonal(mat, match)
    return mat


def score_alignment_3d(
    aligned_seq1: str,
    aligned_seq2: str,
    aligned_seq3: str,
    *,
    score_matrix: np.ndarray,
    alphabet: str,
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
        score_matrix: Substitution matrix for symbols in `alphabet`.
        alphabet: Alphabet defining score_matrix indices.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap.
        seq{1,2,3}_{left,right}_free: Whether leading/trailing gaps in that sequence are free.

    Returns:
        Total alignment score as int.
    """
    if not (len(aligned_seq1) == len(aligned_seq2) == len(aligned_seq3)):
        raise ValueError("All aligned strings must have the same length.")
    if score_matrix.shape != (len(alphabet), len(alphabet)):
        raise ValueError("score_matrix shape must match len(alphabet).")

    sym_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    L = len(aligned_seq1)
    total = 0

    # --- Sum-of-pairs substitution score per column ---
    for pos in range(L):
        c1, c2, c3 = aligned_seq1[pos], aligned_seq2[pos], aligned_seq3[pos]
        if c1 == "-" and c2 == "-" and c3 == "-":
            raise ValueError("Invalid alignment column: all gaps.")

        letters: list[int] = []
        for c in (c1, c2, c3):
            if c != "-":
                if c not in sym_to_idx:
                    raise ValueError(f"Character {c!r} not in alphabet={alphabet!r}.")
                letters.append(sym_to_idx[c])

        for x, y in combinations(letters, 2):
            total += int(score_matrix[x, y])

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
    score_matrix: np.ndarray | None = None,
    alphabet: str = "ACGT",
    match: int = 2,
    mismatch: int = -1,
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

    Scoring:
      - Substitution: sum-of-pairs using `score_matrix` (one matrix for all pairs).
      - Gaps: affine, per-sequence (negative scores):
            gap_run_score(length=k) = gap_open + gap_extend * (k - 1)

    DP state model:
      - 7 states, each a 3-bit mask of which sequences are gaps in the CURRENT column.
        bit0 -> seq1 is gap
        bit1 -> seq2 is gap
        bit2 -> seq3 is gap
        Allowed masks: 0..6 (mask 7 = all gaps disallowed).

      - Each mask implies a move (di, dj, dk):
            di = 0 if seq1 gap else 1
            dj = 0 if seq2 gap else 1
            dk = 0 if seq3 gap else 1

    Affine gaps:
      - For each sequence that is gapped in current mask:
            if it was also gapped in prev mask => gap_extend
            else => gap_open
      - End-free gaps:
            if sequence index is at left boundary (0) and *_left_free => per-column penalty 0
            if sequence index is at right boundary (len) and *_right_free => per-column penalty 0
        This avoids the “pad both suffixes then one becomes internal” problem.

    Args:
        seq1/seq2/seq3: DNA sequences to align.
        score_matrix: Optional substitution matrix (KxK). If None, diagonal(match)/offdiag(mismatch).
        alphabet: Alphabet ordering for score_matrix indices.
        match/mismatch: Used only if score_matrix is None.
        gap_open/gap_extend: Negative gap scores.
        seq{1,2,3}_{left,right}_free: Free terminal gaps flags.

    Returns:
        (aligned_seq1, aligned_seq2, aligned_seq3, best_score)
    """
    a = seq1.upper()
    b = seq2.upper()
    c = seq3.upper()
    n, m, l = len(a), len(b), len(c)

    if score_matrix is None:
        score_matrix = build_score_matrix(alphabet=alphabet, match=match, mismatch=mismatch)
    else:
        score_matrix = np.asarray(score_matrix)
        if score_matrix.shape != (len(alphabet), len(alphabet)):
            raise ValueError("score_matrix must be square with size len(alphabet).")

    sym_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    try:
        a_idx = np.array([sym_to_idx[ch] for ch in a], dtype=np.int16)
        b_idx = np.array([sym_to_idx[ch] for ch in b], dtype=np.int16)
        c_idx = np.array([sym_to_idx[ch] for ch in c], dtype=np.int16)
    except KeyError as e:
        raise ValueError(f"Character {e.args[0]!r} not in alphabet={alphabet!r}.") from e

    # Masks exclude 7 (all gaps)
    masks = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int8)
    num_states = masks.size

    # Steps implied by mask bits
    step_i = np.array([0 if (mask & 1) else 1 for mask in masks], dtype=np.int8)
    step_j = np.array([0 if (mask & 2) else 1 for mask in masks], dtype=np.int8)
    step_k = np.array([0 if (mask & 4) else 1 for mask in masks], dtype=np.int8)

    neg_inf = -math.inf
    dp = np.full((num_states, n + 1, m + 1, l + 1), neg_inf, dtype=np.float64)

    # Pointers store (prev_state, di, dj, dk)
    ptr_state = np.full((num_states, n + 1, m + 1, l + 1), -1, dtype=np.int8)
    ptr_di = np.zeros((num_states, n + 1, m + 1, l + 1), dtype=np.int8)
    ptr_dj = np.zeros((num_states, n + 1, m + 1, l + 1), dtype=np.int8)
    ptr_dk = np.zeros((num_states, n + 1, m + 1, l + 1), dtype=np.int8)

    # Start: interpret state=mask 0 as "previous column had no gaps"
    dp[0, 0, 0, 0] = 0.0
    ptr_state[0, 0, 0, 0] = 0

    left_free = (seq1_left_free, seq2_left_free, seq3_left_free)
    right_free = (seq1_right_free, seq2_right_free, seq3_right_free)
    lens = (n, m, l)

    def gap_penalty(cur_mask: int, prev_mask: int, i_: int, j_: int, k_: int) -> int:
        """Sum affine gap penalties for sequences gapped in cur_mask (with boundary-free logic)."""
        pen = 0

        # seq1 (bit0)
        if cur_mask & 1:
            if not ((i_ == 0 and left_free[0]) or (i_ == lens[0] and right_free[0])):
                pen += gap_extend if (prev_mask & 1) else gap_open

        # seq2 (bit1)
        if cur_mask & 2:
            if not ((j_ == 0 and left_free[1]) or (j_ == lens[1] and right_free[1])):
                pen += gap_extend if (prev_mask & 2) else gap_open

        # seq3 (bit2)
        if cur_mask & 4:
            if not ((k_ == 0 and left_free[2]) or (k_ == lens[2] and right_free[2])):
                pen += gap_extend if (prev_mask & 4) else gap_open

        return pen

    def col_sub_score(mask: int, i_: int, j_: int, k_: int) -> int:
        """Sum-of-pairs substitution score for the column ending at (i_,j_,k_) with this mask."""
        letters: list[int] = []
        if not (mask & 1):
            letters.append(int(a_idx[i_ - 1]))
        if not (mask & 2):
            letters.append(int(b_idx[j_ - 1]))
        if not (mask & 4):
            letters.append(int(c_idx[k_ - 1]))

        s = 0
        for x, y in combinations(letters, 2):
            s += int(score_matrix[x, y])
        return s

    # --- DP fill ---
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            for k in range(0, l + 1):
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

    # --- Termination: best state at (n,m,l) ---
    end_scores = dp[:, n, m, l]
    best_state = int(np.argmax(end_scores))
    best_score = float(end_scores[best_state])

    # --- Backtrack ---
    i, j, k = n, m, l
    state = best_state
    out1: list[str] = []
    out2: list[str] = []
    out3: list[str] = []

    while i > 0 or j > 0 or k > 0:
        mask = int(masks[state])

        out1.append("-" if (mask & 1) else a[i - 1])
        out2.append("-" if (mask & 2) else b[j - 1])
        out3.append("-" if (mask & 4) else c[k - 1])

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
    score_matrix: np.ndarray,
    alphabet: str,
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
    n, m, l = len(a), len(b), len(c)

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
        if i == n and j == m and k == l:
            s = score_alignment_3d(
                "".join(out1),
                "".join(out2),
                "".join(out3),
                score_matrix=score_matrix,
                alphabet=alphabet,
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
            if i + di > n or j + dj > m or k + dk > l:
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
    mat = build_score_matrix(alphabet=alphabet, match=2, mismatch=-1)
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
            l = random.randint(0, max_length)

            s1 = "".join(random.choice(alphabet) for _ in range(n))
            s2 = "".join(random.choice(alphabet) for _ in range(m))
            s3 = "".join(random.choice(alphabet) for _ in range(l))

            a1, a2, a3, dp_score = needleman_wunsch_3d(
                s1, s2, s3,
                score_matrix=mat,
                alphabet=alphabet,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )

            brute = brute_force_best_score_3d(
                s1, s2, s3,
                score_matrix=mat,
                alphabet=alphabet,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )

            scored = score_alignment_3d(
                a1, a2, a3,
                score_matrix=mat,
                alphabet=alphabet,
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