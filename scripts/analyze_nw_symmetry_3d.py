from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from typing import Any

from sgad.pairwise_3d import needleman_wunsch_3d

DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
DNA_BASES = "ACGT"
SYMMETRIES = ("swap12", "swap13", "swap23", "reverse", "complement")

MAT_COMP_SYM = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 3, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 3, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}

MAT_NON_COMP_SYM = {
    "A": {"A": 4, "C": -1, "G": -2, "T": -3},
    "C": {"A": -1, "C": 3, "G": -1, "T": -2},
    "G": {"A": -2, "C": -1, "G": 2, "T": -1},
    "T": {"A": -3, "C": -2, "G": -1, "T": 1},
}


def _comp_seq(seq: str) -> str:
    return seq.translate(DNA_COMP)


def _comp_alignment(aligned: str) -> str:
    return "".join("-" if ch == "-" else ch.translate(DNA_COMP) for ch in aligned)


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(n))


def _flags_str(flags: tuple[bool, bool, bool, bool, bool, bool]) -> str:
    return "".join("1" if x else "0" for x in flags)


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


def _expected_score_invariance(
    symmetry: str, *, matrix_transpose_symmetric: bool, matrix_complement_symmetric: bool
) -> bool:
    if symmetry in ("swap12", "swap13", "swap23"):
        return matrix_transpose_symmetric
    if symmetry == "reverse":
        return True
    if symmetry == "complement":
        return matrix_complement_symmetric
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _transform_input(
    symmetry: str,
    seq1: str,
    seq2: str,
    seq3: str,
    flags: tuple[bool, bool, bool, bool, bool, bool],
) -> tuple[str, str, str, tuple[bool, bool, bool, bool, bool, bool]]:
    s1_l, s1_r, s2_l, s2_r, s3_l, s3_r = flags
    if symmetry == "swap12":
        return seq2, seq1, seq3, (s2_l, s2_r, s1_l, s1_r, s3_l, s3_r)
    if symmetry == "swap13":
        return seq3, seq2, seq1, (s3_l, s3_r, s2_l, s2_r, s1_l, s1_r)
    if symmetry == "swap23":
        return seq1, seq3, seq2, (s1_l, s1_r, s3_l, s3_r, s2_l, s2_r)
    if symmetry == "reverse":
        return (
            seq1[::-1],
            seq2[::-1],
            seq3[::-1],
            (s1_r, s1_l, s2_r, s2_l, s3_r, s3_l),
        )
    if symmetry == "complement":
        return _comp_seq(seq1), _comp_seq(seq2), _comp_seq(seq3), flags
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _transform_alignment_back(
    symmetry: str, aligned1: str, aligned2: str, aligned3: str
) -> tuple[str, str, str]:
    if symmetry == "swap12":
        return aligned2, aligned1, aligned3
    if symmetry == "swap13":
        return aligned3, aligned2, aligned1
    if symmetry == "swap23":
        return aligned1, aligned3, aligned2
    if symmetry == "reverse":
        return aligned1[::-1], aligned2[::-1], aligned3[::-1]
    if symmetry == "complement":
        return _comp_alignment(aligned1), _comp_alignment(aligned2), _comp_alignment(aligned3)
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _run_nw_3d(
    seq1: str,
    seq2: str,
    seq3: str,
    *,
    matrix: dict[str, dict[str, int | float]],
    gap_open: float,
    gap_extend: float,
    enable_gap_close_penalty: bool,
    flags: tuple[bool, bool, bool, bool, bool, bool],
) -> tuple[str, str, str, float]:
    s1_l, s1_r, s2_l, s2_r, s3_l, s3_r = flags
    return needleman_wunsch_3d(
        seq1,
        seq2,
        seq3,
        score_matrix=matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        enable_gap_close_penalty=enable_gap_close_penalty,
        seq1_left_free=s1_l,
        seq1_right_free=s1_r,
        seq2_left_free=s2_l,
        seq2_right_free=s2_r,
        seq3_left_free=s3_l,
        seq3_right_free=s3_r,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "3D NW symmetry analysis over swap/reverse/complement across condition "
            "combinations (no scaled-score dimension)."
        )
    )
    p.add_argument("--seq1", help="Optional fixed sequence 1")
    p.add_argument("--seq2", help="Optional fixed sequence 2")
    p.add_argument("--seq3", help="Optional fixed sequence 3")
    p.add_argument(
        "--num-triples",
        type=int,
        default=8,
        help="Number of random sequence triples (ignored with --seq1/--seq2/--seq3).",
    )
    p.add_argument("--min-len", type=int, default=5, help="Minimum random length.")
    p.add_argument("--max-len", type=int, default=14, help="Maximum random length.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument("--abs-tol", type=float, default=1e-9)
    p.add_argument(
        "--max-failures-to-print",
        type=int,
        default=2,
        help="Maximum mismatch examples to print per condition row.",
    )
    p.add_argument("--print-all-combos", action="store_true")
    p.add_argument("--csv-out", help="Optional output CSV path.")
    p.add_argument(
        "--no-witness-triples",
        action="store_true",
        help="Disable deterministic witness triples.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if (args.seq1 is None) ^ (args.seq2 is None) or (args.seq1 is None) ^ (args.seq3 is None):
        raise SystemExit(
            "Provide all of --seq1/--seq2/--seq3, or none (simulation mode)."
        )
    if args.num_triples <= 0:
        raise SystemExit("--num-triples must be > 0.")
    if args.min_len <= 0:
        raise SystemExit("--min-len must be > 0.")
    if args.max_len < args.min_len:
        raise SystemExit("--max-len must be >= --min-len.")
    if args.max_failures_to_print < 0:
        raise SystemExit("--max-failures-to-print must be >= 0.")
    if args.seq1 is not None:
        s1 = args.seq1.upper()
        s2 = args.seq2.upper()
        s3 = args.seq3.upper()
        if not set(s1).issubset(set(DNA_BASES)):
            raise SystemExit("--seq1 may only contain A/C/G/T.")
        if not set(s2).issubset(set(DNA_BASES)):
            raise SystemExit("--seq2 may only contain A/C/G/T.")
        if not set(s3).issubset(set(DNA_BASES)):
            raise SystemExit("--seq3 may only contain A/C/G/T.")

    if args.seq1 is not None and args.seq2 is not None and args.seq3 is not None:
        triples = [(args.seq1.upper(), args.seq2.upper(), args.seq3.upper())]
        mode = "single-triple"
    else:
        rng = random.Random(args.seed)
        triples: list[tuple[str, str, str]] = []
        if not args.no_witness_triples:
            triples.extend(
                [
                    ("A", "A", "A"),
                    ("A", "C", "T"),
                    ("AC", "GT", "TG"),
                    ("AGTC", "CTGA", "TCAA"),
                ]
            )
        triples.extend(
            (
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
            )
            for _ in range(args.num_triples)
        )
        mode = (
            f"simulated({len(triples)} triples incl. witnesses, "
            f"random={args.num_triples}, len={args.min_len}..{args.max_len}, "
            f"seed={args.seed})"
        )

    matrices: dict[str, dict[str, dict[str, int | float]]] = {
        "comp_symmetric": MAT_COMP_SYM,
        "non_comp_symmetric": MAT_NON_COMP_SYM,
    }
    affine_opts = [False, True]
    gap_close_opts = [False, True]
    flags_combos = list(itertools.product([False, True], repeat=6))

    total_rows = 0
    expected_true_rows = 0
    expected_true_fail_rows = 0
    expected_false_rows = 0
    expected_false_pass_rows = 0
    by_symmetry: dict[str, dict[str, int]] = {
        s: {
            "rows": 0,
            "score_full_pass_rows": 0,
            "alignment_full_pass_rows": 0,
            "expected_true_rows": 0,
            "expected_true_fail_rows": 0,
        }
        for s in SYMMETRIES
    }
    rows: list[dict[str, Any]] = []

    for matrix_name, matrix in matrices.items():
        matrix_transpose_symmetric = _is_transpose_symmetric(matrix)
        matrix_complement_symmetric = _is_complement_symmetric(matrix)

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
                        if expected:
                            expected_true_rows += 1
                            by_symmetry[symmetry]["expected_true_rows"] += 1
                        else:
                            expected_false_rows += 1

                        score_failures = 0
                        alignment_failures = 0
                        first_score_failure: dict[str, Any] | None = None
                        first_alignment_failure: dict[str, Any] | None = None

                        for case_idx, (seq1, seq2, seq3) in enumerate(triples):
                            base_a1, base_a2, base_a3, score_base = _run_nw_3d(
                                seq1,
                                seq2,
                                seq3,
                                matrix=matrix,
                                gap_open=gap_open,
                                gap_extend=gap_extend,
                                enable_gap_close_penalty=gap_close_on,
                                flags=flags,
                            )
                            t_seq1, t_seq2, t_seq3, t_flags = _transform_input(
                                symmetry, seq1, seq2, seq3, flags
                            )
                            t_a1, t_a2, t_a3, score_t = _run_nw_3d(
                                t_seq1,
                                t_seq2,
                                t_seq3,
                                matrix=matrix,
                                gap_open=gap_open,
                                gap_extend=gap_extend,
                                enable_gap_close_penalty=gap_close_on,
                                flags=t_flags,
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
                                        "seq3": seq3,
                                        "score_base": float(score_base),
                                        "score_transformed": float(score_t),
                                        "delta": float(score_base) - float(score_t),
                                    }

                            back1, back2, back3 = _transform_alignment_back(
                                symmetry, t_a1, t_a2, t_a3
                            )
                            align_ok = (
                                base_a1 == back1 and base_a2 == back2 and base_a3 == back3
                            )
                            if not align_ok:
                                alignment_failures += 1
                                if first_alignment_failure is None:
                                    first_alignment_failure = {"case_idx": case_idx}

                        score_pass = score_failures == 0
                        alignment_pass = alignment_failures == 0

                        if score_pass:
                            by_symmetry[symmetry]["score_full_pass_rows"] += 1
                        if alignment_pass:
                            by_symmetry[symmetry]["alignment_full_pass_rows"] += 1

                        if expected and not score_pass:
                            expected_true_fail_rows += 1
                            by_symmetry[symmetry]["expected_true_fail_rows"] += 1
                        if (not expected) and score_pass:
                            expected_false_pass_rows += 1

                        row = {
                            "matrix": matrix_name,
                            "affine_gap": int(affine_on),
                            "gap_close_penalty": int(gap_close_on),
                            "flags": _flags_str(flags),
                            "symmetry": symmetry,
                            "triples_total": len(triples),
                            "score_failures": score_failures,
                            "score_pass": int(score_pass),
                            "alignment_failures": alignment_failures,
                            "alignment_pass": int(alignment_pass),
                            "expected_score_invariant": int(expected),
                        }
                        rows.append(row)

                        should_print = args.print_all_combos or (expected and not score_pass)
                        if should_print:
                            print(
                                f"matrix={matrix_name} affine={int(affine_on)} "
                                f"gap_close={int(gap_close_on)} flags={_flags_str(flags)} "
                                f"symmetry={symmetry} "
                                f"score_pass={len(triples) - score_failures}/{len(triples)} "
                                f"align_exact={len(triples) - alignment_failures}/{len(triples)} "
                                f"expected_score_invariant={int(expected)}"
                            )
                            if (
                                first_score_failure is not None
                                and args.max_failures_to_print > 0
                            ):
                                print(
                                    f"  score_mismatch case={first_score_failure['case_idx']} "
                                    f"delta={first_score_failure['delta']:.12g}"
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
                    "affine_gap",
                    "gap_close_penalty",
                    "flags",
                    "symmetry",
                    "triples_total",
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
    print(f"expected_true_rows={expected_true_rows}")
    print(f"expected_true_fail_rows={expected_true_fail_rows}")
    print(f"expected_false_rows={expected_false_rows}")
    print(f"expected_false_pass_rows={expected_false_pass_rows}")

    print()
    print("=== By symmetry ===")
    for symmetry in SYMMETRIES:
        stats = by_symmetry[symmetry]
        print(
            f"{symmetry}: rows={stats['rows']} "
            f"score_full_pass_rows={stats['score_full_pass_rows']} "
            f"alignment_full_pass_rows={stats['alignment_full_pass_rows']} "
            f"expected_true_rows={stats['expected_true_rows']} "
            f"expected_true_fail_rows={stats['expected_true_fail_rows']}"
        )

    return 0 if expected_true_fail_rows == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
