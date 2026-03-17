from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from typing import Any

from sgad.pairwise import make_score_scaler, needleman_wunsch, no_score_scale_factor

DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
DNA_BASES = "ACGT"
SYMMETRIES = ("swap", "reverse", "complement")

# Symmetric and complement-invariant (Watson-Crick equivariant).
MAT_COMP_SYM = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 3, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 3, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

# Symmetric but NOT complement-invariant.
MAT_NON_COMP_SYM = {
    "A": {"A": 4, "C": -1, "G": -2, "T": -3},
    "C": {"A": -1, "C": 3, "G": -1, "T": -2},
    "G": {"A": -2, "C": -1, "G": 2, "T": -1},
    "T": {"A": -3, "C": -2, "G": -1, "T": 1},
}


def _comp_seq(seq: str) -> str:
    return seq.translate(DNA_COMP)


def _comp_alignment(aligned: str) -> str:
    out = []
    for ch in aligned:
        out.append("-" if ch == "-" else ch.translate(DNA_COMP))
    return "".join(out)


def _swap_flags(flags: tuple[bool, bool, bool, bool]) -> tuple[bool, bool, bool, bool]:
    s1_l, s1_r, s2_l, s2_r = flags
    return (s2_l, s2_r, s1_l, s1_r)


def _reverse_flags(
    flags: tuple[bool, bool, bool, bool],
) -> tuple[bool, bool, bool, bool]:
    s1_l, s1_r, s2_l, s2_r = flags
    return (s1_r, s1_l, s2_r, s2_l)


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(n))


def _is_transpose_symmetric(matrix: dict[str, dict[str, int | float]]) -> bool:
    for x in DNA_BASES:
        for y in DNA_BASES:
            if float(matrix[x][y]) != float(matrix[y][x]):
                return False
    return True


def _is_complement_symmetric(matrix: dict[str, dict[str, int | float]]) -> bool:
    for x in DNA_BASES:
        for y in DNA_BASES:
            cx = _comp_seq(x)
            cy = _comp_seq(y)
            if float(matrix[x][y]) != float(matrix[cx][cy]):
                return False
    return True


def _flags_str(flags: tuple[bool, bool, bool, bool]) -> str:
    return "".join("1" if x else "0" for x in flags)


def _expected_score_invariance(
    symmetry: str, *, matrix_transpose_symmetric: bool, matrix_complement_symmetric: bool
) -> bool | None:
    if symmetry == "swap":
        return matrix_transpose_symmetric
    if symmetry == "reverse":
        # Reverse symmetry is intentionally analyzed empirically here.
        return None
    if symmetry == "complement":
        return matrix_complement_symmetric
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _transform_input(
    symmetry: str,
    seq1: str,
    seq2: str,
    flags: tuple[bool, bool, bool, bool],
) -> tuple[str, str, tuple[bool, bool, bool, bool]]:
    if symmetry == "swap":
        return seq2, seq1, _swap_flags(flags)
    if symmetry == "reverse":
        return seq1[::-1], seq2[::-1], _reverse_flags(flags)
    if symmetry == "complement":
        return _comp_seq(seq1), _comp_seq(seq2), flags
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _transform_alignment_back(
    symmetry: str, aligned_a: str, aligned_b: str
) -> tuple[str, str]:
    if symmetry == "swap":
        return aligned_b, aligned_a
    if symmetry == "reverse":
        return aligned_a[::-1], aligned_b[::-1]
    if symmetry == "complement":
        return _comp_alignment(aligned_a), _comp_alignment(aligned_b)
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _run_nw(
    seq1: str,
    seq2: str,
    *,
    matrix: dict[str, dict[str, int | float]],
    gap_open: float,
    gap_extend: float,
    enable_gap_close_penalty: bool,
    flags: tuple[bool, bool, bool, bool],
    score_scale_fn,
) -> tuple[str, str, float]:
    s1_l, s1_r, s2_l, s2_r = flags
    return needleman_wunsch(
        seq1,
        seq2,
        score_matrix=matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        enable_gap_close_penalty=enable_gap_close_penalty,
        seq1_left_free=s1_l,
        seq1_right_free=s1_r,
        seq2_left_free=s2_l,
        seq2_right_free=s2_r,
        score_scale_fn=score_scale_fn,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Comprehensive 2D Needleman-Wunsch symmetry analysis across condition "
            "combinations: swap, reverse, complement."
        )
    )
    p.add_argument("--seq1", help="Optional fixed sequence 1")
    p.add_argument("--seq2", help="Optional fixed sequence 2")
    p.add_argument(
        "--num-pairs",
        type=int,
        default=40,
        help="Number of random pairs (ignored when --seq1/--seq2 are provided).",
    )
    p.add_argument("--min-len", type=int, default=8, help="Minimum random length.")
    p.add_argument("--max-len", type=int, default=32, help="Maximum random length.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument("--decay", type=float, default=1.0, help="Scaler decay exponent.")
    p.add_argument("--temp", type=float, default=1.0, help="Scaler temperature.")
    p.add_argument(
        "--abs-tol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for score comparisons.",
    )
    p.add_argument(
        "--max-failures-to-print",
        type=int,
        default=2,
        help="Maximum mismatch examples to print per condition row.",
    )
    p.add_argument(
        "--print-all-combos",
        action="store_true",
        help="Print one summary line for every condition row.",
    )
    p.add_argument(
        "--csv-out",
        help="Optional output CSV path for per-row results.",
    )
    p.add_argument(
        "--no-witness-pairs",
        action="store_true",
        help="Disable built-in deterministic witness pairs.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if (args.seq1 is None) ^ (args.seq2 is None):
        raise SystemExit(
            "Provide both --seq1 and --seq2, or neither (random simulation mode)."
        )
    if args.num_pairs <= 0:
        raise SystemExit("--num-pairs must be > 0.")
    if args.min_len <= 0:
        raise SystemExit("--min-len must be > 0.")
    if args.max_len < args.min_len:
        raise SystemExit("--max-len must be >= --min-len.")
    if args.max_failures_to_print < 0:
        raise SystemExit("--max-failures-to-print must be >= 0.")
    if args.seq1 is not None:
        s1 = args.seq1.upper()
        s2 = args.seq2.upper()
        if not set(s1).issubset(set(DNA_BASES)):
            raise SystemExit("--seq1 may only contain A/C/G/T.")
        if not set(s2).issubset(set(DNA_BASES)):
            raise SystemExit("--seq2 may only contain A/C/G/T.")

    if args.seq1 is not None and args.seq2 is not None:
        pairs = [(args.seq1.upper(), args.seq2.upper())]
        mode = "single-pair"
    else:
        rng = random.Random(args.seed)
        pairs = []
        if not args.no_witness_pairs:
            # Useful witnesses, especially to expose complement-symmetry breaks.
            pairs.extend(
                [
                    ("A", "A"),
                    ("A", "C"),
                    ("T", "G"),
                    ("AC", "GT"),
                    ("AGTC", "CTGA"),
                ]
            )
        pairs.extend(
            (
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
            )
            for _ in range(args.num_pairs)
        )
        mode = (
            f"simulated({len(pairs)} pairs incl. witnesses, "
            f"random={args.num_pairs}, len={args.min_len}..{args.max_len}, "
            f"seed={args.seed})"
        )

    matrices: dict[str, dict[str, dict[str, int | float]]] = {
        "comp_symmetric": MAT_COMP_SYM,
        "non_comp_symmetric": MAT_NON_COMP_SYM,
    }

    flags_combos = list(itertools.product([False, True], repeat=4))
    scaled_opts = [False, True]
    affine_opts = [False, True]
    gap_close_opts = [False, True]

    total_rows = 0
    expected_known_rows = 0
    expected_true_rows = 0
    expected_true_fail_rows = 0
    expected_false_rows = 0
    expected_false_pass_rows = 0
    expected_unknown_rows = 0
    expected_unknown_fail_rows = 0
    by_symmetry: dict[str, dict[str, int]] = {
        s: {
            "rows": 0,
            "score_full_pass_rows": 0,
            "alignment_full_pass_rows": 0,
            "expected_known_rows": 0,
            "expected_true_rows": 0,
            "expected_true_fail_rows": 0,
            "expected_unknown_rows": 0,
            "expected_unknown_fail_rows": 0,
        }
        for s in SYMMETRIES
    }
    rows: list[dict[str, Any]] = []

    for matrix_name, matrix in matrices.items():
        matrix_transpose_symmetric = _is_transpose_symmetric(matrix)
        matrix_complement_symmetric = _is_complement_symmetric(matrix)

        for scaled in scaled_opts:
            score_scale_fn = (
                make_score_scaler(decay_exponent=args.decay, temperature=args.temp)
                if scaled
                else no_score_scale_factor
            )
            for affine_on in affine_opts:
                gap_open = float(args.gap_open)
                gap_extend = float(args.gap_extend if affine_on else args.gap_open)
                for gap_close_on in gap_close_opts:
                    for flags in flags_combos:
                        for symmetry in SYMMETRIES:
                            total_rows += 1
                            by_symmetry[symmetry]["rows"] += 1
                            expected = _expected_score_invariance(
                                symmetry,
                                matrix_transpose_symmetric=matrix_transpose_symmetric,
                                matrix_complement_symmetric=matrix_complement_symmetric,
                            )
                            if expected is None:
                                expected_unknown_rows += 1
                                by_symmetry[symmetry]["expected_unknown_rows"] += 1
                            else:
                                expected_known_rows += 1
                                by_symmetry[symmetry]["expected_known_rows"] += 1
                                if expected:
                                    expected_true_rows += 1
                                    by_symmetry[symmetry]["expected_true_rows"] += 1
                                else:
                                    expected_false_rows += 1

                            score_failures = 0
                            alignment_failures = 0
                            first_score_failure: dict[str, Any] | None = None
                            first_alignment_failure: dict[str, Any] | None = None

                            for case_idx, (seq1, seq2) in enumerate(pairs):
                                aligned_a, aligned_b, score_base = _run_nw(
                                    seq1,
                                    seq2,
                                    matrix=matrix,
                                    gap_open=gap_open,
                                    gap_extend=gap_extend,
                                    enable_gap_close_penalty=gap_close_on,
                                    flags=flags,
                                    score_scale_fn=score_scale_fn,
                                )

                                t_seq1, t_seq2, t_flags = _transform_input(
                                    symmetry, seq1, seq2, flags
                                )
                                t_aligned_a, t_aligned_b, score_t = _run_nw(
                                    t_seq1,
                                    t_seq2,
                                    matrix=matrix,
                                    gap_open=gap_open,
                                    gap_extend=gap_extend,
                                    enable_gap_close_penalty=gap_close_on,
                                    flags=t_flags,
                                    score_scale_fn=score_scale_fn,
                                )

                                score_ok = math.isclose(
                                    float(score_base),
                                    float(score_t),
                                    rel_tol=0.0,
                                    abs_tol=args.abs_tol,
                                )
                                if not score_ok:
                                    score_failures += 1
                                    if first_score_failure is None:
                                        first_score_failure = {
                                            "case_idx": case_idx,
                                            "seq1": seq1,
                                            "seq2": seq2,
                                            "score_base": float(score_base),
                                            "score_transformed": float(score_t),
                                            "delta": float(score_base) - float(score_t),
                                            "aligned_base_a": aligned_a,
                                            "aligned_base_b": aligned_b,
                                            "aligned_transformed_a": t_aligned_a,
                                            "aligned_transformed_b": t_aligned_b,
                                        }

                                back_a, back_b = _transform_alignment_back(
                                    symmetry, t_aligned_a, t_aligned_b
                                )
                                align_ok = aligned_a == back_a and aligned_b == back_b
                                if not align_ok:
                                    alignment_failures += 1
                                    if first_alignment_failure is None:
                                        first_alignment_failure = {
                                            "case_idx": case_idx,
                                            "seq1": seq1,
                                            "seq2": seq2,
                                            "aligned_base_a": aligned_a,
                                            "aligned_base_b": aligned_b,
                                            "aligned_back_a": back_a,
                                            "aligned_back_b": back_b,
                                        }

                            score_pass = score_failures == 0
                            alignment_pass = alignment_failures == 0

                            if score_pass:
                                by_symmetry[symmetry]["score_full_pass_rows"] += 1
                            if alignment_pass:
                                by_symmetry[symmetry]["alignment_full_pass_rows"] += 1

                            if expected is None:
                                if not score_pass:
                                    expected_unknown_fail_rows += 1
                                    by_symmetry[symmetry]["expected_unknown_fail_rows"] += 1
                            else:
                                if expected and not score_pass:
                                    expected_true_fail_rows += 1
                                    by_symmetry[symmetry]["expected_true_fail_rows"] += 1
                                if (not expected) and score_pass:
                                    expected_false_pass_rows += 1

                            row = {
                                "matrix": matrix_name,
                                "scaled": int(scaled),
                                "affine_gap": int(affine_on),
                                "gap_close_penalty": int(gap_close_on),
                                "flags": _flags_str(flags),
                                "symmetry": symmetry,
                                "pairs_total": len(pairs),
                                "score_failures": score_failures,
                                "score_pass": int(score_pass),
                                "alignment_failures": alignment_failures,
                                "alignment_pass": int(alignment_pass),
                                "expected_score_invariant": (
                                    "unknown"
                                    if expected is None
                                    else ("true" if expected else "false")
                                ),
                            }
                            rows.append(row)

                            should_print = args.print_all_combos or (
                                (expected is True and not score_pass)
                                or (expected is None and not score_pass)
                            )
                            if should_print:
                                print(
                                    f"matrix={matrix_name} "
                                    f"scaled={int(scaled)} affine={int(affine_on)} "
                                    f"gap_close={int(gap_close_on)} "
                                    f"flags={_flags_str(flags)} symmetry={symmetry} "
                                    f"score_pass={len(pairs) - score_failures}/{len(pairs)} "
                                    f"align_exact={len(pairs) - alignment_failures}/{len(pairs)} "
                                    f"expected_score_invariant="
                                    f"{'unknown' if expected is None else int(expected)}"
                                )
                                if (
                                    first_score_failure is not None
                                    and args.max_failures_to_print > 0
                                ):
                                    print(
                                        f"  score_mismatch case={first_score_failure['case_idx']} "
                                        f"delta={first_score_failure['delta']:.12g}"
                                    )
                                    print(
                                        f"  seq1={first_score_failure['seq1']} "
                                        f"seq2={first_score_failure['seq2']}"
                                    )
                                if (
                                    first_alignment_failure is not None
                                    and args.max_failures_to_print > 0
                                ):
                                    print(
                                        f"  alignment_variant case={first_alignment_failure['case_idx']}"
                                    )

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "matrix",
                    "scaled",
                    "affine_gap",
                    "gap_close_penalty",
                    "flags",
                    "symmetry",
                    "pairs_total",
                    "score_failures",
                    "score_pass",
                    "alignment_failures",
                    "alignment_pass",
                    "expected_score_invariant",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    print()
    print("=== Matrix properties ===")
    for matrix_name, matrix in matrices.items():
        print(
            f"{matrix_name}: "
            f"transpose_symmetric={int(_is_transpose_symmetric(matrix))} "
            f"complement_symmetric={int(_is_complement_symmetric(matrix))}"
        )

    print()
    print("=== Global summary ===")
    print(f"mode={mode}")
    print(f"rows={total_rows}")
    print(f"expected_known_rows={expected_known_rows}")
    print(f"expected_true_rows={expected_true_rows}")
    print(f"expected_true_fail_rows={expected_true_fail_rows}")
    print(f"expected_false_rows={expected_false_rows}")
    print(f"expected_false_pass_rows={expected_false_pass_rows}")
    print(f"expected_unknown_rows={expected_unknown_rows}")
    print(f"expected_unknown_fail_rows={expected_unknown_fail_rows}")

    print()
    print("=== By symmetry ===")
    for symmetry in SYMMETRIES:
        stats = by_symmetry[symmetry]
        print(
            f"{symmetry}: rows={stats['rows']} "
            f"score_full_pass_rows={stats['score_full_pass_rows']} "
            f"alignment_full_pass_rows={stats['alignment_full_pass_rows']} "
            f"expected_known_rows={stats['expected_known_rows']} "
            f"expected_true_rows={stats['expected_true_rows']} "
            f"expected_true_fail_rows={stats['expected_true_fail_rows']} "
            f"expected_unknown_rows={stats['expected_unknown_rows']} "
            f"expected_unknown_fail_rows={stats['expected_unknown_fail_rows']}"
        )

    return 0 if expected_true_fail_rows == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
