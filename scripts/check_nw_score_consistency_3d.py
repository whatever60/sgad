from __future__ import annotations

import argparse
import itertools
import math
import random

from sgad.pairwise_3d import needleman_wunsch_3d, score_alignment_3d

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
            "Check score consistency: needleman_wunsch_3d score vs "
            "score_alignment_3d on the returned alignment."
        )
    )
    p.add_argument("--seq1", help="Sequence 1 (optional; omit for simulation)")
    p.add_argument("--seq2", help="Sequence 2 (optional; omit for simulation)")
    p.add_argument("--seq3", help="Sequence 3 (optional; omit for simulation)")
    p.add_argument(
        "--num-triples",
        type=int,
        default=120,
        help="Number of simulated triples when --seq1/--seq2/--seq3 are omitted",
    )
    p.add_argument("--min-len", type=int, default=6, help="Minimum simulated length")
    p.add_argument("--max-len", type=int, default=18, help="Maximum simulated length")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument(
        "--disable-gap-close-penalty",
        action="store_true",
        help="Disable gap-close penalty in both DP and rescoring.",
    )
    p.add_argument("--abs-tol", type=float, default=1e-9)
    p.add_argument(
        "--max-failures-to-print",
        type=int,
        default=3,
        help="Print up to this many mismatch examples per flag combo.",
    )
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
        help="Single free-end flag combo as 0/1 values (ignored by --all-flags).",
    )
    p.add_argument(
        "--all-flags",
        action="store_true",
        help="Check all 64 free-end flag combinations.",
    )
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
            f"simulated({args.num_triples} triples, "
            f"len={args.min_len}..{args.max_len}, seed={args.seed})"
        )

    failures = 0
    flag_passes = 0
    for flags in combos:
        s1_l, s1_r, s2_l, s2_r, s3_l, s3_r = flags
        flag_failures = 0
        printed = 0
        for case_idx, (seq1, seq2, seq3) in enumerate(triples):
            aligned_1, aligned_2, aligned_3, nw_score = needleman_wunsch_3d(
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
            rescored = score_alignment_3d(
                aligned_1,
                aligned_2,
                aligned_3,
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

            ok = math.isclose(
                float(nw_score),
                float(rescored),
                rel_tol=0.0,
                abs_tol=args.abs_tol,
            )
            if not ok:
                flag_failures += 1
                if printed < args.max_failures_to_print:
                    printed += 1
                    delta = float(nw_score) - float(rescored)
                    print(
                        f"Mismatch case={case_idx} "
                        f"flags={tuple(int(x) for x in flags)}"
                    )
                    print(
                        f"needleman_wunsch_3d={float(nw_score):.12g} "
                        f"score_alignment_3d={float(rescored):.12g} "
                        f"delta={delta:.12g}"
                    )
                    print(f"seq1={seq1}")
                    print(f"seq2={seq2}")
                    print(f"seq3={seq3}")
                    print(f"aligned_seq1={aligned_1}")
                    print(f"aligned_seq2={aligned_2}")
                    print(f"aligned_seq3={aligned_3}")
                    print()

        failures += flag_failures
        if flag_failures == 0:
            flag_passes += 1
        print(
            f"flags={tuple(int(x) for x in flags)} "
            f"pass={len(triples) - flag_failures}/{len(triples)} "
            f"{'PASS' if flag_failures == 0 else 'FAIL'}"
        )

    print(
        f"Summary: mode={mode}, gap_close_penalty={enable_gap_close_penalty}, "
        f"flag_combos_passed={flag_passes}/{len(combos)}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
