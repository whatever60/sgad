from __future__ import annotations

import argparse
import itertools
import math
import random

from sgad.pairwise_3d import needleman_wunsch_3d as py_needleman_wunsch_3d
from sgad.rust.pairwise_3d import needleman_wunsch_3d as rust_needleman_wunsch_3d

MAT = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 3, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 3, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check consistency between Python and Rust 3D needleman_wunsch_3d "
            "(alignment strings + score)."
        )
    )
    p.add_argument("--seq1", help="Sequence 1 (optional; omit for simulation)")
    p.add_argument("--seq2", help="Sequence 2 (optional; omit for simulation)")
    p.add_argument("--seq3", help="Sequence 3 (optional; omit for simulation)")
    p.add_argument("--num-triples", type=int, default=80)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument("--disable-gap-close-penalty", action="store_true")
    p.add_argument("--abs-tol", type=float, default=1e-9)
    p.add_argument("--max-failures-to-print", type=int, default=3)
    p.add_argument(
        "--flags",
        nargs=6,
        type=int,
        metavar=(
            "SEQ1_LEFT",
            "SEQ1_RIGHT",
            "SEQ2_LEFT",
            "SEQ2_RIGHT",
            "SEQ3_LEFT",
            "SEQ3_RIGHT",
        ),
    )
    p.add_argument("--all-flags", action="store_true")
    args = p.parse_args()

    if (args.seq1 is None) ^ (args.seq2 is None) or (args.seq1 is None) ^ (args.seq3 is None):
        p.error("Provide all of --seq1/--seq2/--seq3, or none (simulation mode).")
    if args.flags is not None and any(v not in (0, 1) for v in args.flags):
        p.error("--flags values must be 0 or 1.")
    if not args.all_flags and args.flags is None:
        args.all_flags = True
    if args.num_triples <= 0:
        p.error("--num-triples must be > 0.")
    if args.min_len <= 0:
        p.error("--min-len must be > 0.")
    if args.max_len < args.min_len:
        p.error("--max-len must be >= --min-len.")
    if args.max_failures_to_print < 0:
        p.error("--max-failures-to-print must be >= 0.")
    if args.seq1 is not None:
        s1 = args.seq1.upper()
        s2 = args.seq2.upper()
        s3 = args.seq3.upper()
        if not set(s1).issubset(set("ACGT")):
            p.error("--seq1 may only contain A/C/G/T.")
        if not set(s2).issubset(set("ACGT")):
            p.error("--seq2 may only contain A/C/G/T.")
        if not set(s3).issubset(set("ACGT")):
            p.error("--seq3 may only contain A/C/G/T.")
    return args


def main() -> int:
    args = _parse_args()
    enable_gap_close_penalty = not args.disable_gap_close_penalty

    if args.all_flags:
        combos = list(itertools.product([False, True], repeat=6))
    else:
        combos = [tuple(bool(v) for v in args.flags)]

    if args.seq1 is not None and args.seq2 is not None and args.seq3 is not None:
        triples = [(args.seq1.upper(), args.seq2.upper(), args.seq3.upper())]
        mode = "single-triple"
    else:
        rng = random.Random(args.seed)
        triples = [
            (
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
            )
            for _ in range(args.num_triples)
        ]
        mode = (
            f"simulated({args.num_triples} triples, len={args.min_len}..{args.max_len}, "
            f"seed={args.seed})"
        )

    score_failures = 0
    alignment_failures = 0
    flag_score_passes = 0
    flag_alignment_passes = 0

    for flags in combos:
        s1_l, s1_r, s2_l, s2_r, s3_l, s3_r = flags
        flag_score_failures = 0
        flag_alignment_failures = 0
        printed = 0
        for case_idx, (seq1, seq2, seq3) in enumerate(triples):
            py_a1, py_a2, py_a3, py_score = py_needleman_wunsch_3d(
                seq1,
                seq2,
                seq3,
                score_matrix=MAT,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=enable_gap_close_penalty,
                seq1_left_free=s1_l,
                seq1_right_free=s1_r,
                seq2_left_free=s2_l,
                seq2_right_free=s2_r,
                seq3_left_free=s3_l,
                seq3_right_free=s3_r,
            )
            rs_a1, rs_a2, rs_a3, rs_score = rust_needleman_wunsch_3d(
                seq1,
                seq2,
                seq3,
                score_matrix=MAT,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=enable_gap_close_penalty,
                seq1_left_free=s1_l,
                seq1_right_free=s1_r,
                seq2_left_free=s2_l,
                seq2_right_free=s2_r,
                seq3_left_free=s3_l,
                seq3_right_free=s3_r,
            )

            score_ok = math.isclose(
                float(py_score),
                float(rs_score),
                rel_tol=0.0,
                abs_tol=args.abs_tol,
            )
            align_ok = py_a1 == rs_a1 and py_a2 == rs_a2 and py_a3 == rs_a3

            if not score_ok:
                flag_score_failures += 1
            if not align_ok:
                flag_alignment_failures += 1

            if (not score_ok or not align_ok) and printed < args.max_failures_to_print:
                printed += 1
                print(f"Mismatch case={case_idx} flags={tuple(int(x) for x in flags)}")
                if not score_ok:
                    delta = float(py_score) - float(rs_score)
                    print(
                        f"  score_py={float(py_score):.12g} "
                        f"score_rust={float(rs_score):.12g} delta={delta:.12g}"
                    )
                if not align_ok:
                    print("  alignment mismatch")
                    print(f"  py_a1={py_a1}")
                    print(f"  py_a2={py_a2}")
                    print(f"  py_a3={py_a3}")
                    print(f"  rs_a1={rs_a1}")
                    print(f"  rs_a2={rs_a2}")
                    print(f"  rs_a3={rs_a3}")
                print()

        score_failures += flag_score_failures
        alignment_failures += flag_alignment_failures
        if flag_score_failures == 0:
            flag_score_passes += 1
        if flag_alignment_failures == 0:
            flag_alignment_passes += 1
        print(
            f"flags={tuple(int(x) for x in flags)} "
            f"score_pass={len(triples) - flag_score_failures}/{len(triples)} "
            f"align_pass={len(triples) - flag_alignment_failures}/{len(triples)} "
            f"{'PASS' if (flag_score_failures == 0 and flag_alignment_failures == 0) else 'FAIL'}"
        )

    print(
        f"Summary: mode={mode}, gap_close_penalty={enable_gap_close_penalty}, "
        f"score_flag_combos_passed={flag_score_passes}/{len(combos)}, "
        f"alignment_flag_combos_passed={flag_alignment_passes}/{len(combos)}"
    )
    return 0 if score_failures == 0 and alignment_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
