"""Compute primer3 + ntthal heterodimer thermodynamics for every pair in two primer sets."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

from joblib import Parallel, delayed
import pandas as pd
import primer3


_NTHAL_THERMO_RE = re.compile(
    r"dS\s*=\s*(?P<ds>[-+]?\d+(?:\.\d+)?)\s+"
    r"dH\s*=\s*(?P<dh>[-+]?\d+(?:\.\d+)?)\s+"
    r"dG\s*=\s*(?P<dg>[-+]?\d+(?:\.\d+)?)\s+"
    r"t\s*=\s*(?P<t>[-+]?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _normalize_primer_seq(seq: str) -> str:
    """Normalize primer strings before thermodynamic calculations."""
    return seq.strip().upper().replace(" ", "")


def _parse_ntthal_output(stdout: str, stderr: str) -> dict[str, float | str | None]:
    """Parse ntthal thermodynamics and ASCII structure from process output."""
    out = (stdout or "").strip()
    if not out and stderr:
        out = stderr.strip()

    if not out:
        raise RuntimeError("ntthal returned empty output.")

    m = _NTHAL_THERMO_RE.search(out)
    if not m:
        raise RuntimeError(f"Could not parse ntthal thermodynamics from output:\n{out}")

    ds = float(m.group("ds"))
    dh = float(m.group("dh"))
    dg = float(m.group("dg"))
    t = float(m.group("t"))

    lines = out.splitlines()
    struct_start = None
    for idx, line in enumerate(lines):
        if line.startswith("SEQ") or line.startswith("STR"):
            struct_start = idx
            break

    ascii_structure = ""
    if struct_start is not None:
        ascii_structure = "\n".join(lines[struct_start:]).rstrip()

    return {
        "ntthal_ds": ds,
        "ntthal_dh": dh,
        "ntthal_dg": dg,
        "ntthal_t": t,
        "ntthal_ascii_structure": ascii_structure,
    }


def _run_ntthal(seq1: str, seq2: str, *, ntthal_path: str, timeout_s: float) -> dict[str, float | str | None]:
    """Run ntthal for one pair and parse output fields used in the final DataFrame."""
    try:
        proc = subprocess.run(
            [ntthal_path, "-s1", seq1, "-s2", seq2],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ntthal executable not found: {ntthal_path!r}. "
            "Install ntthal or provide a valid ntthal_path."
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"ntthal failed (returncode={proc.returncode}) for pair ({seq1}, {seq2}). "
            f"stderr: {(proc.stderr or '').strip()}"
        )

    return _parse_ntthal_output(proc.stdout or "", proc.stderr or "")


def _compute_pair_row(
    s1: str,
    s2: str,
    n1: str,
    n2: str,
    *,
    ntthal_path: str,
    ntthal_timeout_s: float,
) -> dict[str, Any]:
    """Compute one (primer1, primer2) row for the long-form output."""
    p3 = primer3.bindings.calc_heterodimer(s1, s2, output_structure=True)
    p3_ascii = getattr(p3, "ascii_structure", "")
    if not p3_ascii:
        ascii_lines = getattr(p3, "ascii_structure_lines", None)
        p3_ascii = "\n".join(ascii_lines) if ascii_lines else ""

    nt = _run_ntthal(s1, s2, ntthal_path=ntthal_path, timeout_s=ntthal_timeout_s)

    return {
        "primer1_name": n1,
        "primer2_name": n2,
        "primer3_tm": float(p3.tm),
        "primer3_dg": float(p3.dg),
        "primer3_ds": float(p3.ds),
        "primer3_dh": float(p3.dh),
        "primer3_structure_found": bool(p3.structure_found),
        "primer3_ascii_structure": str(p3_ascii),
        "ntthal_ds": nt["ntthal_ds"],
        "ntthal_dh": nt["ntthal_dh"],
        "ntthal_dg": nt["ntthal_dg"],
        "ntthal_t": nt["ntthal_t"],
        "ntthal_ascii_structure": nt["ntthal_ascii_structure"],
    }


def heterodimer_batch_primer3(
    primer1_seqs: list[str],
    primer2_seqs: list[str],
    primer1_names: list[str],
    primer2_names: list[str],
    *,
    n_jobs: int = 1,
    ntthal_path: str = "ntthal",
    ntthal_timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Compute primer3 + ntthal metrics for every ``primer1 × primer2`` pair.

    Returns one long-form DataFrame with ``len(primer1_seqs) * len(primer2_seqs)`` rows
    and fixed columns:

    - ``primer1_name``
    - ``primer2_name``
    - ``primer3_tm``
    - ``primer3_dg``
    - ``primer3_ds``
    - ``primer3_dh``
    - ``primer3_structure_found``
    - ``primer3_ascii_structure``
    - ``ntthal_ds``
    - ``ntthal_dh``
    - ``ntthal_dg``
    - ``ntthal_t``
    - ``ntthal_ascii_structure``

    Args:
        primer1_seqs: Primer set 1 sequences (5′→3′).
        primer2_seqs: Primer set 2 sequences (5′→3′).
        primer1_names: Names for set 1 (same length as ``primer1_seqs``).
        primer2_names: Names for set 2 (same length as ``primer2_seqs``).
        n_jobs: Parallel workers for pairwise computations. ``1`` runs sequentially.
        ntthal_path: Path/name of the ``ntthal`` executable.
        ntthal_timeout_s: Timeout per ntthal subprocess call (seconds).

    Returns:
        Long-form :class:`pandas.DataFrame` with ``m×n`` rows.
    """
    if len(primer1_seqs) != len(primer1_names):
        raise ValueError("primer1_seqs and primer1_names must have the same length.")
    if len(primer2_seqs) != len(primer2_names):
        raise ValueError("primer2_seqs and primer2_names must have the same length.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if ntthal_timeout_s <= 0:
        raise ValueError("ntthal_timeout_s must be > 0.")

    s1 = [_normalize_primer_seq(x) for x in primer1_seqs]
    s2 = [_normalize_primer_seq(x) for x in primer2_seqs]

    tasks = [
        (s1[i], s2[j], primer1_names[i], primer2_names[j])
        for i in range(len(s1))
        for j in range(len(s2))
    ]

    if n_jobs == 1:
        rows = [
            _compute_pair_row(
                seq1,
                seq2,
                name1,
                name2,
                ntthal_path=ntthal_path,
                ntthal_timeout_s=ntthal_timeout_s,
            )
            for (seq1, seq2, name1, name2) in tasks
        ]
    else:
        rows = list(Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_pair_row)(
                seq1,
                seq2,
                name1,
                name2,
                ntthal_path=ntthal_path,
                ntthal_timeout_s=ntthal_timeout_s,
            )
            for (seq1, seq2, name1, name2) in tasks
        ))

    cols = [
        "primer1_name",
        "primer2_name",
        "primer3_tm",
        "primer3_dg",
        "primer3_ds",
        "primer3_dh",
        "primer3_structure_found",
        "primer3_ascii_structure",
        "ntthal_ds",
        "ntthal_dh",
        "ntthal_dg",
        "ntthal_t",
        "ntthal_ascii_structure",
    ]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    print("=== heterodimer_batch_primer3 self-check ===\n")

    seqs1 = ["ACGTACGT", "GCTAGCTA"]
    seqs2 = ["TGCATGCA", "CGATCGAT", "AAAATTTT"]
    names1 = ["fwd_1", "fwd_2"]
    names2 = ["rev_1", "rev_2", "rev_3"]

    if shutil.which("ntthal") is None:
        print("  SKIP  ntthal executable not found; install ntthal to run self-check.")
    else:
        df = heterodimer_batch_primer3(seqs1, seqs2, names1, names2, n_jobs=2)

        assert df.shape[0] == len(seqs1) * len(seqs2), f"row count: {df.shape[0]}"
        expected_cols = [
            "primer1_name",
            "primer2_name",
            "primer3_tm",
            "primer3_dg",
            "primer3_ds",
            "primer3_dh",
            "primer3_structure_found",
            "primer3_ascii_structure",
            "ntthal_ds",
            "ntthal_dh",
            "ntthal_dg",
            "ntthal_t",
            "ntthal_ascii_structure",
        ]
        assert list(df.columns) == expected_cols, f"columns: {list(df.columns)}"
        print("  PASS  shape and schema")
        print(f"        rows={df.shape[0]}, cols={df.shape[1]}")

        # Spot-check one known row and dtypes/contents.
        row = df[(df["primer1_name"] == "fwd_1") & (df["primer2_name"] == "rev_1")].iloc[0]
        assert isinstance(row["primer3_structure_found"], (bool,))
        assert isinstance(row["primer3_ascii_structure"], str)
        assert isinstance(row["ntthal_ascii_structure"], str)
        assert float(row["primer3_dg"]) == float(row["primer3_dg"])
        assert float(row["ntthal_dg"]) == float(row["ntthal_dg"])
        print("  PASS  row content types")
        print(
            "        sample:",
            {
                "primer1_name": row["primer1_name"],
                "primer2_name": row["primer2_name"],
                "primer3_dg": row["primer3_dg"],
                "ntthal_dg": row["ntthal_dg"],
            },
        )

        # Validate deterministic row ordering by pair names in nested-loop order.
        pairs = list(zip(df["primer1_name"], df["primer2_name"], strict=True))
        expected_pairs = [(a, b) for a in names1 for b in names2]
        assert pairs == expected_pairs, "unexpected row ordering"
        print("  PASS  deterministic m×n row ordering")

        print("\nAll heterodimer_batch_primer3 checks passed.")
