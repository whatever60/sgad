from __future__ import annotations

import argparse
import itertools
import math
import random

from sgad.pairwise import make_score_scaler, to_ascii, score_alignment
from sgad.pairwise import needleman_wunsch as py_needleman_wunsch
from sgad.pairwise import no_score_scale_factor
from sgad.rust.pairwise import make_rust_score_scaler
from sgad.rust.pairwise import needleman_wunsch as rust_needleman_wunsch

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
    # a, b, c, d = flags
    # return (d, c, b, a)
    seq1_left, seq1_right, seq2_left, seq2_right = flags
    seq1_left, seq2_right = seq2_right, seq1_left
    seq1_right, seq2_left = seq2_left, seq1_right
    return (seq1_left, seq1_right, seq2_left, seq2_right)


def _score(
    seq1: str,
    seq2: str,
    *,
    flags: tuple[bool, bool, bool, bool],
    backend: str,
    score_scale_fn,
    gap_open: float,
    gap_extend: float,
) -> float:
    seq2_rc = seq2.translate(DNA_COMP)[::-1]
    a, b, c, d = flags

    nw = py_needleman_wunsch if backend == "py" else rust_needleman_wunsch
    _aln1, _aln2, score = nw(
        seq1,
        seq2_rc,
        score_matrix=MAT,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=a,
        seq1_right_free=b,
        seq2_left_free=c,
        seq2_right_free=d,
        score_scale_fn=score_scale_fn,
    )
    # print(flags)
    print(to_ascii(_aln1, _aln2, a, b, c, d))
    # print(score)
    rescore = score_alignment(
        _aln1,
        _aln2,
        score_matrix=MAT,
        gap_open=gap_open,
        gap_extend=gap_extend,
        seq1_left_free=a,
        seq1_right_free=b,
        seq2_left_free=c,
        seq2_right_free=d,
        score_scale_fn=score_scale_fn,
    )
    print(rescore)
    return float(score)


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check swap symmetry for dimer-style scoring:\n"
            "score(seq1, seq2, flags) vs score(seq2, seq1, swapped_flags)."
        )
    )
    p.add_argument("--seq1", help="Primer 1 sequence (optional; omit for simulation)")
    p.add_argument("--seq2", help="Primer 2 sequence (optional; omit for simulation)")
    p.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of simulated primer pairs when --seq1/--seq2 are omitted",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=8,
        help="Minimum simulated primer length",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=32,
        help="Maximum simulated primer length",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for simulated primer pairs",
    )
    p.add_argument(
        "--backend", choices=["py", "rust"], default="rust", help="Backend to test"
    )
    p.add_argument(
        "--scaled",
        action="store_true",
        help="Use non-None scaling (make_score_scaler / make_rust_score_scaler)",
    )
    p.add_argument("--decay", type=float, default=1.0, help="Scaler decay exponent")
    p.add_argument("--temp", type=float, default=1.0, help="Scaler temperature")
    p.add_argument("--gap-open", type=float, default=-5.0)
    p.add_argument("--gap-extend", type=float, default=-1.0)
    p.add_argument("--abs-tol", type=float, default=1e-9)

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
    return args


def main() -> int:
    args = _parse_args()

    if args.backend == "py":
        score_scale_fn = (
            make_score_scaler(decay_exponent=args.decay, temperature=args.temp)
            if args.scaled
            else no_score_scale_factor
        )
    else:
        score_scale_fn = (
            make_rust_score_scaler(decay_exponent=args.decay, temperature=args.temp)
            if args.scaled
            else None
        )

    if args.all_flags:
        combos = list(itertools.product([False, True], repeat=4))
    else:
        combos = [tuple(bool(v) for v in args.flags)]

    if args.seq1 is not None and args.seq2 is not None:
        pairs = [(args.seq1, args.seq2)]
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
        mode = f"simulated({args.num_pairs} pairs, len={args.min_len}..{args.max_len}, seed={args.seed})"

    failures = 0
    flag_passes = 0
    for flags in combos:
        swapped = _swap_flags(flags)
        flag_failures = 0
        for idx, (seq1, seq2) in enumerate(pairs[96:]):
            print(idx)
            s12 = _score(
                seq1,
                seq2,
                flags=flags,
                backend=args.backend,
                score_scale_fn=score_scale_fn,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
            )
            s21 = _score(
                seq2,
                seq1,
                flags=swapped,
                backend=args.backend,
                score_scale_fn=score_scale_fn,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
            )
            ok = math.isclose(s12, s21, rel_tol=0.0, abs_tol=args.abs_tol)
            if not ok:
                flag_failures += 1
            # break
        failures += flag_failures
        if flag_failures == 0:
            flag_passes += 1
        print(
            f"flags={tuple(int(x) for x in flags)} "
            f"swapped={tuple(int(x) for x in swapped)} "
            f"pass={len(pairs) - flag_failures}/{len(pairs)} "
            f"{'PASS' if flag_failures == 0 else 'FAIL'}"
        )

    print(
        f"Summary: mode={mode}, backend={args.backend}, scaled={args.scaled}, "
        f"flag_combos_passed={flag_passes}/{len(combos)}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
