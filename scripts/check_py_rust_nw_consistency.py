from __future__ import annotations

import argparse
import itertools
import math
import random

from sgad.pairwise import make_score_scaler, no_score_scale_factor
from sgad.pairwise import needleman_wunsch as py_needleman_wunsch
from sgad.rust.pairwise import make_rust_score_scaler
from sgad.rust.pairwise import needleman_wunsch as rust_needleman_wunsch

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
            "Check consistency between Python and Rust 2D needleman_wunsch "
            "(alignment strings + score)."
        )
    )
    p.add_argument("--seq1", help="Sequence 1 (optional; omit for simulation)")
    p.add_argument("--seq2", help="Sequence 2 (optional; omit for simulation)")
    p.add_argument("--num-pairs", type=int, default=120)
    p.add_argument("--min-len", type=int, default=8)
    p.add_argument("--max-len", type=int, default=28)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scaled", action="store_true")
    p.add_argument("--decay", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument("--disable-gap-close-penalty", action="store_true")
    p.add_argument("--abs-tol", type=float, default=1e-9)
    p.add_argument("--max-failures-to-print", type=int, default=3)
    p.add_argument(
        "--flags",
        nargs=4,
        type=int,
        metavar=("SEQ1_LEFT", "SEQ1_RIGHT", "SEQ2_LEFT", "SEQ2_RIGHT"),
    )
    p.add_argument("--all-flags", action="store_true")

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
            p.error("--seq1 may only contain A/C/G/T.")
        if not set(s2).issubset(set("ACGT")):
            p.error("--seq2 may only contain A/C/G/T.")
    return args


def main() -> int:
    args = _parse_args()
    enable_gap_close_penalty = not args.disable_gap_close_penalty

    if args.scaled:
        py_scaler = make_score_scaler(decay_exponent=args.decay, temperature=args.temp)
        rust_scaler = make_rust_score_scaler(
            decay_exponent=args.decay, temperature=args.temp
        )
    else:
        py_scaler = no_score_scale_factor
        rust_scaler = None

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
            f"simulated({args.num_pairs} pairs, len={args.min_len}..{args.max_len}, "
            f"seed={args.seed})"
        )

    score_failures = 0
    alignment_failures = 0
    flag_score_passes = 0
    flag_alignment_passes = 0

    for flags in combos:
        s1_l, s1_r, s2_l, s2_r = flags
        flag_score_failures = 0
        flag_alignment_failures = 0
        printed = 0
        for case_idx, (seq1, seq2) in enumerate(pairs):
            py_a1, py_a2, py_score = py_needleman_wunsch(
                seq1,
                seq2,
                score_matrix=MAT,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=enable_gap_close_penalty,
                seq1_left_free=s1_l,
                seq1_right_free=s1_r,
                seq2_left_free=s2_l,
                seq2_right_free=s2_r,
                score_scale_fn=py_scaler,
            )
            rs_a1, rs_a2, rs_score = rust_needleman_wunsch(
                seq1,
                seq2,
                score_matrix=MAT,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=enable_gap_close_penalty,
                seq1_left_free=s1_l,
                seq1_right_free=s1_r,
                seq2_left_free=s2_l,
                seq2_right_free=s2_r,
                score_scale_fn=rust_scaler,
            )

            score_ok = math.isclose(
                float(py_score),
                float(rs_score),
                rel_tol=0.0,
                abs_tol=args.abs_tol,
            )
            align_ok = py_a1 == rs_a1 and py_a2 == rs_a2

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
                    print(f"  rs_a1={rs_a1}")
                    print(f"  rs_a2={rs_a2}")
                print()

        score_failures += flag_score_failures
        alignment_failures += flag_alignment_failures
        if flag_score_failures == 0:
            flag_score_passes += 1
        if flag_alignment_failures == 0:
            flag_alignment_passes += 1
        print(
            f"flags={tuple(int(x) for x in flags)} "
            f"score_pass={len(pairs) - flag_score_failures}/{len(pairs)} "
            f"align_pass={len(pairs) - flag_alignment_failures}/{len(pairs)} "
            f"{'PASS' if (flag_score_failures == 0 and flag_alignment_failures == 0) else 'FAIL'}"
        )

    print(
        f"Summary: mode={mode}, scaled={args.scaled}, "
        f"gap_close_penalty={enable_gap_close_penalty}, "
        f"score_flag_combos_passed={flag_score_passes}/{len(combos)}, "
        f"alignment_flag_combos_passed={flag_alignment_passes}/{len(combos)}"
    )
    return 0 if score_failures == 0 and alignment_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
