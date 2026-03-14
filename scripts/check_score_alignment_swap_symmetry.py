from __future__ import annotations

import argparse
import itertools
import math
import random

from sgad.pairwise import make_score_scaler, no_score_scale_factor, score_alignment

DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
MAT = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 3, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 3, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}


def _swap_flags(flags: tuple[bool, bool, bool, bool]) -> tuple[bool, bool, bool, bool]:
    # Dimer swap mapping with reverse-complemented seq2.
    # (seq1_left, seq1_right, seq2_left, seq2_right) -> (seq2_right, seq2_left, seq1_right, seq1_left)
    seq1_left, seq1_right, seq2_left, seq2_right = flags
    seq1_left, seq2_right = seq2_right, seq1_left
    seq1_right, seq2_left = seq2_left, seq1_right
    return (seq1_left, seq1_right, seq2_left, seq2_right)


def _revcomp_seq(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1]


def _revcomp_alignment(aligned: str) -> str:
    chars = []
    for ch in reversed(aligned):
        chars.append("-" if ch == "-" else ch.translate(DNA_COMP))
    return "".join(chars)


def _swap_alignment(
    aligned_seq1: str, aligned_seq2: str
) -> tuple[str, str]:
    # For dimer-style swap symmetry:
    #   (seq1, rc(seq2)) -> (seq2, rc(seq1))
    return _revcomp_alignment(aligned_seq2), _revcomp_alignment(aligned_seq1)


def _score(
    aligned_seq1: str,
    aligned_seq2: str,
    *,
    flags: tuple[bool, bool, bool, bool],
    score_scale_fn,
    gap_open: float,
    gap_extend: float,
    enable_gap_close_penalty: bool,
) -> float:
    a, b, c, d = flags
    return float(
        score_alignment(
            aligned_seq1,
            aligned_seq2,
            score_matrix=MAT,
            gap_open=gap_open,
            gap_extend=gap_extend,
            enable_gap_close_penalty=enable_gap_close_penalty,
            seq1_left_free=a,
            seq1_right_free=b,
            seq2_left_free=c,
            seq2_right_free=d,
            score_scale_fn=score_scale_fn,
        )
    )


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _rand_alignment(rng: random.Random, seq1: str, seq2_rc: str) -> tuple[str, str]:
    i = 0
    j = 0
    aligned_1: list[str] = []
    aligned_2: list[str] = []
    while i < len(seq1) or j < len(seq2_rc):
        moves = []
        if i < len(seq1) and j < len(seq2_rc):
            moves.append(0)  # match/mismatch column
        if i < len(seq1):
            moves.append(2)  # gap in seq2
        if j < len(seq2_rc):
            moves.append(1)  # gap in seq1
        move = rng.choice(moves)
        if move == 0:
            aligned_1.append(seq1[i])
            aligned_2.append(seq2_rc[j])
            i += 1
            j += 1
        elif move == 1:
            aligned_1.append("-")
            aligned_2.append(seq2_rc[j])
            j += 1
        else:
            aligned_1.append(seq1[i])
            aligned_2.append("-")
            i += 1
    return "".join(aligned_1), "".join(aligned_2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check swap symmetry for dimer-style scoring using score_alignment only.\n"
            "No DP search is used: random valid alignments are generated directly."
        )
    )
    p.add_argument("--seq1", help="Primer 1 sequence (optional; omit for simulation)")
    p.add_argument("--seq2", help="Primer 2 sequence (optional; omit for simulation)")
    p.add_argument(
        "--aligned-seq1",
        help=(
            "Direct aligned seq1 string (includes '-' columns). "
            "If set, --aligned-seq2 must also be set."
        ),
    )
    p.add_argument(
        "--aligned-seq2",
        help=(
            "Direct aligned seq2_rc string (includes '-' columns). "
            "If set, --aligned-seq1 must also be set."
        ),
    )
    p.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of simulated primer pairs when --seq1/--seq2 are omitted",
    )
    p.add_argument(
        "--alignments-per-pair",
        type=int,
        default=20,
        help="Number of random alignments generated per primer pair",
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
        help="Disable gap close penalty while scoring alignments.",
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
    if (args.aligned_seq1 is None) ^ (args.aligned_seq2 is None):
        p.error("Provide both --aligned-seq1 and --aligned-seq2, or neither.")
    if args.aligned_seq1 is not None and args.seq1 is not None:
        p.error("Use either aligned mode or raw sequence mode, not both.")
    if args.flags is not None and any(v not in (0, 1) for v in args.flags):
        p.error("--flags values must be 0 or 1.")
    if not args.all_flags and args.flags is None:
        args.all_flags = True
    if args.num_pairs <= 0:
        p.error("--num-pairs must be > 0.")
    if args.alignments_per_pair <= 0:
        p.error("--alignments-per-pair must be > 0.")
    if args.min_len <= 0:
        p.error("--min-len must be > 0.")
    if args.max_len < args.min_len:
        p.error("--max-len must be >= --min-len.")
    if args.max_failures_to_print < 0:
        p.error("--max-failures-to-print must be >= 0.")
    if args.aligned_seq1 is not None:
        a1 = args.aligned_seq1.upper()
        a2 = args.aligned_seq2.upper()
        if len(a1) != len(a2):
            p.error("--aligned-seq1 and --aligned-seq2 must have equal length.")
        if not set(a1).issubset(set("ACGT-")):
            p.error("--aligned-seq1 may only contain A/C/G/T/- characters.")
        if not set(a2).issubset(set("ACGT-")):
            p.error("--aligned-seq2 may only contain A/C/G/T/- characters.")
        for idx, (c1, c2) in enumerate(zip(a1, a2, strict=True)):
            if c1 == "-" and c2 == "-":
                p.error(f"Invalid aligned input at column {idx}: both gaps.")
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

    if args.all_flags:
        combos = list(itertools.product([False, True], repeat=4))
    else:
        combos = [tuple(bool(v) for v in args.flags)]

    alignments: list[tuple[str, str]] = []
    if args.aligned_seq1 is not None and args.aligned_seq2 is not None:
        alignments = [(args.aligned_seq1.upper(), args.aligned_seq2.upper())]
        mode = "single-alignment"
    else:
        if args.seq1 is not None and args.seq2 is not None:
            pairs = [(args.seq1.upper(), args.seq2.upper())]
            mode = f"single-pair({args.alignments_per_pair} random alignments)"
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
                "simulated("
                f"{args.num_pairs} pairs x {args.alignments_per_pair} alignments, "
                f"len={args.min_len}..{args.max_len}, seed={args.seed})"
            )

        rng = random.Random(args.seed)
        for seq1, seq2 in pairs:
            seq2_rc = _revcomp_seq(seq2)
            for _ in range(args.alignments_per_pair):
                alignments.append(_rand_alignment(rng, seq1, seq2_rc))

    failures = 0
    flag_passes = 0
    for flags in combos:
        swapped = _swap_flags(flags)
        flag_failures = 0
        printed = 0
        for case_idx, (aligned_12_a, aligned_12_b) in enumerate(alignments):
            aligned_21_a, aligned_21_b = _swap_alignment(aligned_12_a, aligned_12_b)
            s12 = _score(
                aligned_12_a,
                aligned_12_b,
                flags=flags,
                score_scale_fn=score_scale_fn,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=not args.disable_gap_close_penalty,
            )
            s21 = _score(
                aligned_21_a,
                aligned_21_b,
                flags=swapped,
                score_scale_fn=score_scale_fn,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                enable_gap_close_penalty=not args.disable_gap_close_penalty,
            )
            ok = math.isclose(s12, s21, rel_tol=0.0, abs_tol=args.abs_tol)
            if not ok:
                flag_failures += 1
                if printed < args.max_failures_to_print:
                    printed += 1
                    delta = s12 - s21
                    print(
                        f"Mismatch case={case_idx} "
                        f"flags={tuple(int(x) for x in flags)} "
                        f"swapped={tuple(int(x) for x in swapped)}"
                    )
                    print(
                        f"score_12={s12:.12g} score_21={s21:.12g} "
                        f"delta={delta:.12g}"
                    )
                    print(f"aligned_12_a={aligned_12_a}")
                    print(f"aligned_12_b={aligned_12_b}")
                    print(f"aligned_21_a={aligned_21_a}")
                    print(f"aligned_21_b={aligned_21_b}")
                    print()
        failures += flag_failures
        if flag_failures == 0:
            flag_passes += 1
        print(
            f"flags={tuple(int(x) for x in flags)} "
            f"swapped={tuple(int(x) for x in swapped)} "
            f"pass={len(alignments) - flag_failures}/{len(alignments)} "
            f"{'PASS' if flag_failures == 0 else 'FAIL'}"
        )

    print(
        f"Summary: mode={mode}, scaled={args.scaled}, "
        f"gap_close_penalty={not args.disable_gap_close_penalty}, "
        f"flag_combos_passed={flag_passes}/{len(combos)}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
