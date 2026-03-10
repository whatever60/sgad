from __future__ import annotations

import math
import random
from typing import Iterable

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
) -> tuple[str, str, float]:
    """
    Needleman–Wunsch global alignment with affine gaps and configurable free end-gaps.

    This version uses:
      - a 3D tensor for DP scores: dp[state, i, j]
      - pointer tensors that store (prev_state, di, dj) so traceback is uniform:
            prev_state = ptr[state, i, j]
            di = ptr_di[state, i, j]
            dj = ptr_dj[state, i, j]
            i -= di; j -= dj; state = prev_state

    States (affine gap model):
      - M  : ends with a paired base (match/mismatch)
      - IX : ends with a gap in seq2  (seq1 base aligned to '-')
      - IY : ends with a gap in seq1  ('-' aligned to seq2 base)

    Gap scoring uses negative scores (penalties are negative values):
      gap_run_score(length=k) = gap_open + gap_extend * (k - 1)
    where gap_open <= 0 and gap_extend <= 0.

    End-gap flags semantics (free terminal gaps):
      - seq1_left_free  => leading gaps in seq1 are free  (top row IY[0,*])
      - seq1_right_free => trailing gaps in seq1 are free (may end early on last row)
      - seq2_left_free  => leading gaps in seq2 are free  (left col IX[*,0])
      - seq2_right_free => trailing gaps in seq2 are free (may end early on last col)

    Substitution scoring:
      - Provide `score_matrix` (shape KxK), with `alphabet` defining indices
      - If score_matrix is None, a diagonal matrix is built from match/mismatch.

    Args:
        seq1: DNA sequence 1.
        seq2: DNA sequence 2.
        score_matrix: Optional substitution matrix of shape (K,K).
        alphabet: Symbols covered by the substitution matrix.
        match: Used only if score_matrix is None (diagonal score).
        mismatch: Used only if score_matrix is None (off-diagonal score).
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap by 1.
        seq1_left_free: If True, gaps in seq1 at the left end are free (0 score).
        seq1_right_free: If True, gaps in seq1 at the right end are free (0 score).
        seq2_left_free: If True, gaps in seq2 at the left end are free (0 score).
        seq2_right_free: If True, gaps in seq2 at the right end are free (0 score).

    Returns:
        (aligned_seq1, aligned_seq2, best_score)
    """
    a = seq1.upper()
    b = seq2.upper()
    n = len(a)
    m = len(b)

    if score_matrix is None:
        score_matrix = build_score_matrix(alphabet=alphabet, match=match, mismatch=mismatch)
    else:
        score_matrix = np.asarray(score_matrix)
        if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
            raise ValueError("score_matrix must be a square 2D array.")
        if score_matrix.shape[0] != len(alphabet):
            raise ValueError("score_matrix size must match len(alphabet).")

    # Map symbols to indices for matrix lookup.
    sym_to_idx = {ch: i for i, ch in enumerate(alphabet)}

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

    # ---- DP storage (tensor form) ----
    M, IX, IY = 0, 1, 2
    neg_inf = -math.inf

    dp = np.full((3, n + 1, m + 1), neg_inf, dtype=np.float64)  # scores
    ptr = np.full((3, n + 1, m + 1), -1, dtype=np.int8)  # previous state
    ptr_di = np.zeros((3, n + 1, m + 1), dtype=np.int8)  # traceback delta i
    ptr_dj = np.zeros((3, n + 1, m + 1), dtype=np.int8)  # traceback delta j

    # Init origin
    dp[M, 0, 0] = 0.0
    ptr[M, 0, 0] = M
    ptr_di[M, 0, 0] = 0
    ptr_dj[M, 0, 0] = 0

    # ---- Boundary initialization ----
    # First column (j=0): only IX can be finite for i>0 (gap in seq2).
    # If seq2_left_free, leading gaps in seq2 are free => IX[i,0] = 0.
    if seq2_left_free:
        for i in range(1, n + 1):
            dp[IX, i, 0] = 0.0
            ptr[IX, i, 0] = M if i == 1 else IX
            ptr_di[IX, i, 0] = 1
            ptr_dj[IX, i, 0] = 0
    else:
        for i in range(1, n + 1):
            if i == 1:
                dp[IX, i, 0] = dp[M, i - 1, 0] + gap_open
                ptr[IX, i, 0] = M
            else:
                dp[IX, i, 0] = dp[IX, i - 1, 0] + gap_extend
                ptr[IX, i, 0] = IX
            ptr_di[IX, i, 0] = 1
            ptr_dj[IX, i, 0] = 0

    # First row (i=0): only IY can be finite for j>0 (gap in seq1).
    # If seq1_left_free, leading gaps in seq1 are free => IY[0,j] = 0.
    if seq1_left_free:
        for j in range(1, m + 1):
            dp[IY, 0, j] = 0.0
            ptr[IY, 0, j] = M if j == 1 else IY
            ptr_di[IY, 0, j] = 0
            ptr_dj[IY, 0, j] = 1
    else:
        for j in range(1, m + 1):
            if j == 1:
                dp[IY, 0, j] = dp[M, 0, j - 1] + gap_open
                ptr[IY, 0, j] = M
            else:
                dp[IY, 0, j] = dp[IY, 0, j - 1] + gap_extend
                ptr[IY, 0, j] = IY
            ptr_di[IY, 0, j] = 0
            ptr_dj[IY, 0, j] = 1

    # ---- DP fill ----
    for i in range(1, n + 1):
        ai = a[i - 1]
        try:
            ai_idx = sym_to_idx[ai]
        except KeyError as e:
            raise ValueError(f"Character {ai!r} not in alphabet={alphabet!r}.") from e

        for j in range(1, m + 1):
            bj = b[j - 1]
            try:
                bj_idx = sym_to_idx[bj]
            except KeyError as e:
                raise ValueError(f"Character {bj!r} not in alphabet={alphabet!r}.") from e

            sub = float(score_matrix[ai_idx, bj_idx])

            # M from diagonal of any state
            diag = dp[:, i - 1, j - 1]  # shape (3,)
            prev_state = int(np.argmax(diag))
            dp[M, i, j] = float(diag[prev_state] + sub)
            ptr[M, i, j] = prev_state
            ptr_di[M, i, j] = 1
            ptr_dj[M, i, j] = 1

            # IX from up: open from M or extend from IX
            open_from_M = dp[M, i - 1, j] + gap_open
            extend_from_IX = dp[IX, i - 1, j] + gap_extend
            if open_from_M >= extend_from_IX:
                dp[IX, i, j] = open_from_M
                ptr[IX, i, j] = M
            else:
                dp[IX, i, j] = extend_from_IX
                ptr[IX, i, j] = IX
            ptr_di[IX, i, j] = 1
            ptr_dj[IX, i, j] = 0

            # IY from left: open from M or extend from IY
            open_from_M = dp[M, i, j - 1] + gap_open
            extend_from_IY = dp[IY, i, j - 1] + gap_extend
            if open_from_M >= extend_from_IY:
                dp[IY, i, j] = open_from_M
                ptr[IY, i, j] = M
            else:
                dp[IY, i, j] = extend_from_IY
                ptr[IY, i, j] = IY
            ptr_di[IY, i, j] = 0
            ptr_dj[IY, i, j] = 1

    # ---- Termination (choose traceback start on allowed edges) ----
    def best_state_at(ii: int, jj: int) -> tuple[float, int]:
        states = dp[:, ii, jj]
        st = int(np.argmax(states))
        return float(states[st]), st

    best_i, best_j = n, m
    best_score, best_state = best_state_at(n, m)

    if seq1_right_free and seq2_right_free:
        # Can end on last row OR last col (but not arbitrary interior).
        best_score = -math.inf

        for jj in range(0, m + 1):
            sc, st = best_state_at(n, jj)
            if sc > best_score:
                best_score, best_state = sc, st
                best_i, best_j = n, jj

        for ii in range(0, n + 1):
            sc, st = best_state_at(ii, m)
            if sc > best_score:
                best_score, best_state = sc, st
                best_i, best_j = ii, m

    elif seq1_right_free and not seq2_right_free:
        best_score = -math.inf
        for jj in range(0, m + 1):
            sc, st = best_state_at(n, jj)
            if sc > best_score:
                best_score, best_state = sc, st
                best_i, best_j = n, jj

    elif seq2_right_free and not seq1_right_free:
        best_score = -math.inf
        for ii in range(0, n + 1):
            sc, st = best_state_at(ii, m)
            if sc > best_score:
                best_score, best_state = sc, st
                best_i, best_j = ii, m

    # ---- Backtrack ----
    i, j, state = best_i, best_j, best_state
    out_a: list[str] = []
    out_b: list[str] = []

    while i > 0 or j > 0:
        # Emit alignment column based on current state.
        if state == M:
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
        elif state == IX:
            out_a.append(a[i - 1])
            out_b.append("-")
        else:  # state == IY
            out_a.append("-")
            out_b.append(b[j - 1])

        prev_state = int(ptr[state, i, j])
        di = int(ptr_di[state, i, j])
        dj = int(ptr_dj[state, i, j])

        if prev_state < 0:
            raise RuntimeError("Unset pointer encountered during traceback.")

        i -= di
        j -= dj
        state = prev_state

    aligned_a = "".join(reversed(out_a))
    aligned_b = "".join(reversed(out_b))

    # ---- Pad suffix on the sequence that is allowed to have trailing gaps ----
    # If we ended on last row (i=n) with j<m, pad seq2's remaining suffix against gaps in seq1.
    if best_i == n and best_j < m and seq1_right_free:
        aligned_a += "-" * (m - best_j)
        aligned_b += b[best_j:]
    # If we ended on last col (j=m) with i<n, pad seq1's remaining suffix against gaps in seq2.
    if best_j == m and best_i < n and seq2_right_free:
        aligned_a += a[best_i:]
        aligned_b += "-" * (n - best_i)

    return aligned_a, aligned_b, float(best_score)


def score_alignment(
    aligned_seq1: str,
    aligned_seq2: str,
    *,
    score_matrix: np.ndarray,
    alphabet: str,
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
        score_matrix: Substitution matrix for symbols in `alphabet`.
        alphabet: Alphabet defining score_matrix indices.
        gap_open: Negative score for opening a gap.
        gap_extend: Negative score for extending a gap.
        seq1_left_free/seq1_right_free: Free-end flags for gaps in seq1.
        seq2_left_free/seq2_right_free: Free-end flags for gaps in seq2.

    Returns:
        Total alignment score as an int.
    """
    if len(aligned_seq1) != len(aligned_seq2):
        raise ValueError("Alignment strings must have same length.")
    if score_matrix.shape[0] != len(alphabet) or score_matrix.shape[1] != len(alphabet):
        raise ValueError("score_matrix shape must match len(alphabet).")

    sym_to_idx = {ch: i for i, ch in enumerate(alphabet)}

    total = 0
    k = 0
    L = len(aligned_seq1)

    while k < L:
        c1 = aligned_seq1[k]
        c2 = aligned_seq2[k]

        if c1 != "-" and c2 != "-":
            try:
                i1 = sym_to_idx[c1]
                i2 = sym_to_idx[c2]
            except KeyError as e:
                raise ValueError(f"Character {e.args[0]!r} not in alphabet={alphabet!r}.") from e
            total += int(score_matrix[i1, i2])
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
    score_matrix: np.ndarray,
    alphabet: str,
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
        score_matrix, alphabet: Substitution scoring.
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
                alphabet=alphabet,
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

    mat = build_score_matrix(alphabet="ACGT", match=2, mismatch=-1)

    aligned1, aligned2, score = needleman_wunsch(
        s1,
        s2,
        score_matrix=mat,
        alphabet="ACGT",
        gap_open=-5,
        gap_extend=-1,
    )
    print("=== Pure global (no free ends) ===")
    print(to_ascii(aligned1, aligned2, score))

    aligned1, aligned2, score = needleman_wunsch(
        s1,
        s2,
        score_matrix=mat,
        alphabet="ACGT",
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
    mat = build_score_matrix(alphabet=alphabet, match=match, mismatch=mismatch)

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
                alphabet=alphabet,
                gap_open=gap_open,
                gap_extend=gap_extend,
                **flags,
            )
            brute = brute_force_best_score(
                a,
                b,
                score_matrix=mat,
                alphabet=alphabet,
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
                    alphabet=alphabet,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    **flags,
                )
                print(to_ascii(aligned_a, aligned_b, dp_score, line_width=None))
                break
        if not all_ok:
            break

    print("All tests passed!" if all_ok else "Some tests failed.")
