"""Parse primer3 heterodimer ASCII output into canonical to_ascii format."""

from __future__ import annotations

from ._common import (
    wrap_alignment,
    normalize_top_row,
    normalize_bottom_row,
)


_WC_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def build_middle_line(top: str, bot: str) -> str:
    """Return a middle line with ``|`` where top and bot form a WC complementary pair."""
    mid: list[str] = []
    for a, b in zip(top, bot):
        if a not in ("-", " ") and b not in ("-", " ") and a.translate(_WC_COMPLEMENT) == b:
            mid.append("|")
        else:
            mid.append(" ")
    return "".join(mid)


def parse_primer3(ascii_structure: str, *, line_width: int | None = 120) -> str:
    """Parse a primer3 heterodimer ASCII structure block into canonical to_ascii format.

    primer3 emits (optionally wrapped) ``SEQ``- and ``STR``-prefixed rows where:

    - ``SEQ`` is the top strand (5′→3′).
    - ``STR`` is the bottom strand written 3′→5′ (which is the Watson-Crick
      complement of the top strand at every paired column).

    This function overlays wrapped lines, applies per-strand gap conventions, and
    complements the bottom strand so that identical characters appear at every
    paired column — matching the convention used by :func:`sgad.pairwise.to_ascii`.

    Args:
        ascii_structure: Multiline string containing ``SEQ``- and ``STR``-prefixed
            rows as returned by
            ``primer3.bindings.calc_heterodimer(..., output_structure=True)``
            or by the ``ntthal`` command-line utility.
        line_width: Wrap each output block to this many columns.
            ``None`` or ``≤ 0`` disables wrapping (default: 120).

    Returns:
        A 3-line string (optionally wrapped) in to_ascii format::

            <top strand: - for internal/leading gaps, spaces for free-end overhangs>
            <middle: | at each Watson-Crick pair>
            <bottom strand: complement of 3′→5′, spaces for free left-end overhang>
    """
    raw_lines = [ln.rstrip("\n") for ln in ascii_structure.splitlines()]

    bodies: dict[str, list[str]] = {"SEQ": [], "STR": []}
    for line in raw_lines:
        if line.startswith("SEQ") or line.startswith("STR"):
            label = line[:3]
            body = line[3:].expandtabs(8)
            bodies[label].append(body)

    all_bodies = [b for lst in bodies.values() for b in lst]
    if not all_bodies:
        return "\n\n\n"

    width = max(len(b) for b in all_bodies)

    def _overlay(label: str) -> str:
        """Merge wrapped alignment rows for ``label`` into a single ``width``-wide string.

        Each row is left-padded to ``width`` with spaces. For each column, the
        output takes the first non-space character found when scanning the rows
        in order. If all rows contain a space at that column, the output contains
        a space there as well.

        This is intended for wrapped ``SEQ``/``STR``-style alignment rows where
        each wrapped row contributes characters to different column positions.
        If multiple rows contain non-space characters in the same column, the
        earliest row wins.

        Args:
            label: Key in ``bodies`` whose wrapped rows should be merged. In
                practice this is usually ``"SEQ"`` or ``"STR"``.

        Returns:
            A string of length ``width``. If ``bodies`` has no rows for ``label``,
            returns ``" " * width``.

        Example — two SEQ lines that together cover a 14-column alignment::

            bodies["SEQ"] == [
                "ACGT      ",   # columns 0-3 filled, 4-9 spaces
                "      ACGT",   # columns 0-5 spaces, 6-9 filled
            ]
            # width = 10
            # col 0: 'A' (from line 0)  col 4: ' '
            # col 1: 'C' (from line 0)  col 5: ' '
            # col 2: 'G' (from line 0)  col 6: 'A' (from line 1)
            # col 3: 'T' (from line 0)  col 7: 'C' (from line 1)
            #                           col 8: 'G' (from line 1)
            #                           col 9: 'T' (from line 1)
            _overlay("SEQ")  # => "ACGT  ACGT"

        Example — a gap in the middle represented as a space on both lines::

            bodies["SEQ"] == [
                "ACG   T",   # col 3 is a space (gap placeholder)
                "       ",   # second line is all spaces (no content here)
            ]
            _overlay("SEQ")  # => "ACG   T"
            # The space at col 3 survives because no line provides a non-space
            # there; downstream normalisation converts it to '-'.
        """
        rows = [b.ljust(width) for b in bodies.get(label, [])]
        if not rows:
            return " " * width
        out: list[str] = []
        for col in range(width):
            ch = " "
            for row in rows:
                if row[col] != " ":
                    ch = row[col]
                    break
            out.append(ch)
        return "".join(out)

    seq_raw = _overlay("SEQ")
    str_raw = _overlay("STR")

    # Crop to bounding box of non-space content in either row.
    content = [i for i in range(width) if seq_raw[i] != " " or str_raw[i] != " "]
    if not content:
        return "\n\n\n"
    lo, hi = min(content), max(content)
    seq_raw = seq_raw[lo : hi + 1]
    str_raw = str_raw[lo : hi + 1]

    # Drop columns where both rows are spaces (pure formatting artefacts).
    seq_cols: list[str] = []
    str_cols: list[str] = []
    for a, b in zip(seq_raw, str_raw):
        if a == " " and b == " ":
            continue
        seq_cols.append(a)
        str_cols.append(b)

    seq = "".join(seq_cols)
    st = "".join(str_cols)

    # Apply per-strand end conventions.
    top = normalize_top_row(seq)
    bot = normalize_bottom_row(st)

    # STR is 3′→5′; display without complement so primer2 bases appear in their
    # original orientation (3′→5′ on bottom, matching dimer_to_ascii).
    mid = build_middle_line(top, bot)
    return wrap_alignment(top, mid, bot, line_width)



def has_internal_gap(alignment_block: str) -> bool:
    """
    Decide whether an alignment contains an internal gap by first normalizing it.

    Args:
        alignment_block: Multiline alignment text containing wrapped "SEQ"/"STR" rows.

    Returns:
        True if either SEQ or STR contains a '-' between its first and last base; else False.
    """
    normalized = parse_primer3(alignment_block)
    seq_line, _, str_line = normalized.splitlines()

    def row_has_internal_dash(row: str) -> bool:
        base_positions = [i for i, ch in enumerate(row) if ch.isalpha()]
        if len(base_positions) < 2:
            return False
        left = base_positions[0]
        right = base_positions[-1]
        return "-" in row[left + 1 : right]

    return row_has_internal_dash(seq_line) or row_has_internal_dash(str_line)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _check(
        desc: str, result: str, expected_top: str, expected_mid: str, expected_bot: str
    ) -> None:
        lines = result.rstrip("\n").split("\n")
        assert len(lines) == 3, (
            f"[{desc}] expected 3 lines, got {len(lines)}: {lines!r}"
        )
        assert lines[0] == expected_top, (
            f"[{desc}] top:  got {lines[0]!r}, want {expected_top!r}"
        )
        assert lines[1] == expected_mid, (
            f"[{desc}] mid:  got {lines[1]!r}, want {expected_mid!r}"
        )
        assert lines[2] == expected_bot, (
            f"[{desc}] bot:  got {lines[2]!r}, want {expected_bot!r}"
        )
        print(f"  PASS  {desc}")
        for line in lines:
            print(f"        {line!r}")

    print("=== parse_primer3 self-check ===\n")

    # ------------------------------------------------------------------
    # 1. Simple space-separated body: perfect 4-mer WC duplex
    #    SEQ ACGT  5'→3'
    #    STR TGCA  3'→5' (WC complement of ACGT, read right-to-left)
    #    Bottom displayed as "TGCA" (no complement applied)
    #    WC pairs: A–T, C–G, G–C, T–A → all |
    # ------------------------------------------------------------------
    _check(
        "perfect 4-mer duplex (space body)",
        parse_primer3("SEQ ACGT\nSTR TGCA\n", line_width=None),
        "ACGT",
        "||||",
        "TGCA",
    )

    # ------------------------------------------------------------------
    # 2. Tab-separated body (as primer3.bindings emits)
    # ------------------------------------------------------------------
    _check(
        "tab-separated body",
        parse_primer3("SEQ\tACGTACGT\nSTR\tTGCATGCA\n", line_width=None),
        "ACGTACGT",
        "||||||||",
        "TGCATGCA",
    )

    # ------------------------------------------------------------------
    # 3. 5′ overhangs on both strands
    #    SEQ "AAACGT" has a 5′ overhang "AAA" on the left.
    #    STR "GCAAA" (3′→5′) is offset right by 3 columns, so it has a
    #    5′ overhang "AA" on the right. The central "GCA" gives 3 WC pairs.
    # ------------------------------------------------------------------
    block3 = "SEQ  AAACGT\nSTR     GCAAA\n"
    r3 = parse_primer3(block3, line_width=None)
    lines3 = r3.rstrip("\n").split("\n")
    assert lines3[0] == "AAACGT  ", f"top: {lines3[0]!r}"
    assert lines3[1] == "   |||  ", f"mid: {lines3[1]!r}"
    assert lines3[2] == "   GCAAA", f"bot: {lines3[2]!r}"
    print("  PASS  5′ overhangs on both strands")
    for line in lines3:
        print(f"        {line!r}")

    # ------------------------------------------------------------------
    # 4. Wrapped multi-line SEQ/STR (as ntthal emits)
    #    The two SEQ lines are overlaid column-by-column.
    # ------------------------------------------------------------------
    block4 = (
        "SEQ  CCTCTGCTACAA\n"
        "SEQ              CTTCT\n"
        "STR              GAAGA\n"
        "STR       GTATGCGCAA\n"
    )
    r4 = parse_primer3(block4, line_width=None)
    lines4 = r4.rstrip("\n").split("\n")
    assert len(lines4) == 3
    # The overlaid SEQ row has content from both lines; check no crash and 3 lines present
    print("  PASS  wrapped multi-line SEQ/STR")
    for line in lines4:
        print(f"        {line!r}")

    # ------------------------------------------------------------------
    # 5. Internal gap in STR
    #    SEQ  ACGT
    #    STR  TG T   (space between G and T becomes '-' internally)
    # ------------------------------------------------------------------
    block5 = "SEQ ACGT\nSTR TG T\n"
    r5 = parse_primer3(block5, line_width=None)
    lines5 = r5.rstrip("\n").split("\n")
    assert "-" in lines5[2], f"expected internal '-' in bot: {lines5[2]!r}"
    print("  PASS  internal gap in STR becomes '-'")
    for line in lines5:
        print(f"        {line!r}")

    # ------------------------------------------------------------------
    # 6. Cross-validate overhang-to-space convention against dimer_to_ascii
    #
    # Both parse_primer3 and dimer_to_ascii display:
    #   top: primer1 (SEQ) 5′→3′ (leading gaps as '-', trailing gaps as ' ')
    #   bot: primer2 (STR) 3′→5′, original bases, no complement
    #        (leading spaces for free left end, trailing '-' for fixed right end)
    #
    # normalize_top_row:    left+internal spaces → '-'; trailing '-' → ' '
    # normalize_bottom_row: leading '-' → ' ';  right+internal spaces → '-'
    # ------------------------------------------------------------------
    from sgad.pairwise import dimer_to_ascii as _dimer_to_ascii

    def _space_mask(row: str) -> list[bool]:
        return [c == " " for c in row]

    # Case A: free bottom-left overhang
    #   SEQ body starts 3 columns before STR → bottom has 3 leading spaces.
    #   STR="GCA" (3′→5′) aligns with the last 3 bases of SEQ="AAACGT"
    #   dimer equivalent: aligned_primer2_comp="---GCA"
    #   dimer_to_ascii shows bottom as "   GCA" (leading "---" free → spaces)
    p3_A = parse_primer3("SEQ  AAACGT\nSTR     GCA\n", line_width=None)
    dim_A = _dimer_to_ascii("AAACGT", "---GCA", line_width=None)
    A_top_p3, _, A_bot_p3 = p3_A.rstrip("\n").split("\n")
    A_top_dim, _, A_bot_dim = dim_A.rstrip("\n").split("\n")
    assert _space_mask(A_top_p3) == _space_mask(A_top_dim), (
        f"Case A top mismatch: p3={A_top_p3!r}  dim={A_top_dim!r}"
    )
    assert _space_mask(A_bot_p3) == _space_mask(A_bot_dim), (
        f"Case A bot mismatch: p3={A_bot_p3!r}  dim={A_bot_dim!r}"
    )
    assert A_bot_p3 == "   GCA", f"expected '   GCA', got {A_bot_p3!r}"
    assert A_bot_dim == "   GCA", f"expected dimer_to_ascii '   GCA', got {A_bot_dim!r}"
    print("  PASS  Case A — free bottom-left overhang spaces match dimer_to_ascii")
    print(f"        parse_primer3 top={A_top_p3!r}  bot={A_bot_p3!r}")
    print(f"        dimer_ascii   top={A_top_dim!r}  bot={A_bot_dim!r}")

    # Case B: free top-right overhang
    #   STR body is longer than SEQ on the right → top line has 2 trailing spaces.
    #   SEQ="ACGT" (4-mer), STR="TGCAGG" (6-mer; first 4 pair with ACGT 3′→5′).
    #   dimer equivalent: aligned_primer2_comp="TGCAGG"
    #   dimer_to_ascii shows top as "ACGT  " (trailing "--" free → spaces)
    p3_B = parse_primer3("SEQ ACGT\nSTR TGCAGG\n", line_width=None)
    dim_B = _dimer_to_ascii("ACGT--", "TGCAGG", line_width=None)
    B_top_p3, _, B_bot_p3 = p3_B.rstrip("\n").split("\n")
    B_top_dim, _, B_bot_dim = dim_B.rstrip("\n").split("\n")
    assert _space_mask(B_top_p3) == _space_mask(B_top_dim), (
        f"Case B top mismatch: p3={B_top_p3!r}  dim={B_top_dim!r}"
    )
    assert _space_mask(B_bot_p3) == _space_mask(B_bot_dim), (
        f"Case B bot mismatch: p3={B_bot_p3!r}  dim={B_bot_dim!r}"
    )
    assert B_top_p3[4:] == "  ", f"expected 2 trailing spaces in top, got {B_top_p3!r}"
    assert B_bot_p3 == "TGCAGG", f"expected bot 'TGCAGG', got {B_bot_p3!r}"
    assert B_bot_dim == "TGCAGG", f"expected dimer_to_ascii bot 'TGCAGG', got {B_bot_dim!r}"
    print("  PASS  Case B — free top-right overhang spaces match dimer_to_ascii")
    print(f"        parse_primer3 top={B_top_p3!r}  bot={B_bot_p3!r}")
    print(f"        dimer_ascii   top={B_top_dim!r}  bot={B_bot_dim!r}")

    print("\nAll parse_primer3 checks passed.")
