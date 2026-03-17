from __future__ import annotations

import argparse
import itertools
import math
import random

from sgad.pairwise import (
    make_score_scaler,
    needleman_wunsch,
    no_score_scale_factor,
)
from sgad.pairwise import score_alignment

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
            "Check score consistency: needleman_wunsch reported score vs "
            "score_alignment(aligned_seq1, aligned_seq2) on the returned alignment."
        )
    )
    p.add_argument("--seq1", help="Sequence 1 (optional; omit for simulation)")
    p.add_argument("--seq2", help="Sequence 2 (optional; omit for simulation)")
    p.add_argument(
        "--num-pairs",
        type=int,
        default=200,
        help="Number of simulated sequence pairs when --seq1/--seq2 are omitted",
    )
    p.add_argument("--min-len", type=int, default=8, help="Minimum simulated length")
    p.add_argument("--max-len", type=int, default=32, help="Maximum simulated length")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--scaled",
        action="store_true",
        help="Use make_score_scaler; default uses no_score_scale_factor.",
    )
    p.add_argument("--decay", type=float, default=1.0, help="Scaler decay exponent")
    p.add_argument("--temp", type=float, default=1.0, help="Scaler temperature")
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument(
        "--disable-gap-close-penalty",
        action="store_true",
        help="Disable gap close penalty in both needleman_wunsch and score_alignment.",
    )
    p.add_argument("--abs-tol", type=float, default=1e-9)
    p.add_argument(
        "--max-failures-to-print",
        type=int,
        default=5,
        help="Print up to this many mismatch examples per flag combo.",
    )
    p.add_argument(
        "--flags",
        nargs=4,
        type=int,
        metavar=("SEQ1_LEFT", "SEQ1_RIGHT", "SEQ2_LEFT", "SEQ2_RIGHT"),
        help="Single flag combo as 0/1 values (ignored if --all-flags is set)",
    )
    p.add_argument(
        "--all-flags",
        action="store_true",
        help="Check all 16 free-flag combinations",
    )

    args = p.parse_args()
    if (args.seq1 is None) ^ (args.seq2 is None):
        p.error("Provide both --seq1 and --seq2, or neither (simulation mode).")
    if args.flags is not None and any(v not in (0, 1) for v in args.flags):
        p.error("--flags values must be 0 or 1.")
    if not args.all_flags and args.flags is None:
        args.all_flags = True
    if args.num_pairs <= 0:
        p.error("--num-pairs must be > 0.")
    if args.min_len <= 0:
        p.error("--min-len must be > 0.")
    if args.max_len < args.min_len:
        p.error("--max-len must be >= --min-len.")
    if args.max_failures_to_print < 0:
        p.error("--max-failures-to-print must be >= 0.")
    if args.seq1 is not None:
        s1 = args.seq1.upper()
        s2 = args.seq2.upper()
        if not set(s1).issubset(set("ACGT")):
            p.error("--seq1 may only contain A/C/G/T characters.")
        if not set(s2).issubset(set("ACGT")):
            p.error("--seq2 may only contain A/C/G/T characters.")
    return args


def main() -> int:
    args = _parse_args()

    score_scale_fn = (
        make_score_scaler(decay_exponent=args.decay, temperature=args.temp)
        if args.scaled
        else no_score_scale_factor
    )
    enable_gap_close_penalty = not args.disable_gap_close_penalty

    if args.all_flags:
        combos = list(itertools.product([False, True], repeat=4))
    else:
        combos = [tuple(bool(v) for v in args.flags)]

    if args.seq1 is not None and args.seq2 is not None:
        pairs = [(args.seq1.upper(), args.seq2.upper())]
        mode = "single-pair"
    else:
        rng = random.Random(args.seed)
        pairs = [
            (
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
                _rand_seq(rng, rng.randint(args.min_len, args.max_len)),
            )
            for _ in range(args.num_pairs)
        ]
        mode = (
            f"simulated({args.num_pairs} pairs, "
            f"len={args.min_len}..{args.max_len}, seed={args.seed})"
        )

    failures = 0
    flag_passes = 0
    for flags in combos:
        a, b, c, d = flags
        flag_failures = 0
        printed = 0
        for case_idx, (seq1, seq2) in enumerate(pairs):
            aligned_1, aligned_2, nw_score = needleman_wunsch(
                seq1,
                seq2,
                score_matrix=MAT,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=enable_gap_close_penalty,
                seq1_left_free=a,
                seq1_right_free=b,
                seq2_left_free=c,
                seq2_right_free=d,
                score_scale_fn=score_scale_fn,
            )
            rescored = score_alignment(
                aligned_1,
                aligned_2,
                score_matrix=MAT,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=enable_gap_close_penalty,
                seq1_left_free=a,
                seq1_right_free=b,
                seq2_left_free=c,
                seq2_right_free=d,
                score_scale_fn=score_scale_fn,
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
                        f"Mismatch case={case_idx} flags={tuple(int(x) for x in flags)}"
                    )
                    print(
                        f"needleman_wunsch={float(nw_score):.12g} "
                        f"score_alignment={float(rescored):.12g} "
                        f"delta={delta:.12g}"
                    )
                    print(f"seq1={seq1}")
                    print(f"seq2={seq2}")
                    print(f"aligned_seq1={aligned_1}")
                    print(f"aligned_seq2={aligned_2}")
                    print()

        failures += flag_failures
        if flag_failures == 0:
            flag_passes += 1
        print(
            f"flags={tuple(int(x) for x in flags)} "
            f"pass={len(pairs) - flag_failures}/{len(pairs)} "
            f"{'PASS' if flag_failures == 0 else 'FAIL'}"
        )

    print(
        f"Summary: mode={mode}, scaled={args.scaled}, "
        f"gap_close_penalty={enable_gap_close_penalty}, "
        f"flag_combos_passed={flag_passes}/{len(combos)}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
