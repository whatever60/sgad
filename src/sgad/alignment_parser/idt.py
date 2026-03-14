"""Parse IDT OligoAnalyzer heterodimer response dicts into canonical to_ascii format."""

from __future__ import annotations

from ._common import (
    wrap_alignment,
    normalize_top_row,
    normalize_bottom_row,
)


def parse_idt(
    primer1: str,
    primer2: str,
    client_resp: dict,
    *,
    line_width: int | None = 120,
) -> str:
    """Parse one IDT OligoAnalyzer heterodimer structure dict into canonical to_ascii format.

    IDT returns a **list** of structure dicts for each heterodimer query; pass
    **one element** of that list as *client_resp*.  The function reconstructs
    the 3-line ASCII diagram from the padding and bond arrays, normalises gap
    conventions, and complements the bottom strand so that identical characters
    appear at every Watson-Crick paired column — matching
    :func:`sgad.pairwise.to_ascii`.

    .. note::
        IDT uses two non-zero bond values: ``2`` (primary Watson-Crick pair)
        and ``1`` (secondary Watson-Crick pair).  IDT defines secondary pairs
        as WC base pairs that lie outside the longest consecutive WC run — on
        the IDT website these are rendered as ``:`` to visually distinguish
        them from the main duplex core.  Both values represent genuine WC
        hydrogen bonds and are rendered as ``|`` in this output.  The middle
        line is taken directly from the IDT ``Bonds`` array rather than
        inferred from base identity.

    Args:
        primer1: Primary sequence (5′→3′) — the same sequence sent to the API.
        primer2: Secondary sequence (5′→3′) — the same sequence sent to the API.
        client_resp: One structure dict from the IDT heterodimer JSON response.
            Expected keys: ``TopLinePadding``, ``BondLinePadding``,
            ``BottomLinePadding``, ``Bonds`` (list of ``0 | 1 | 2``).
        line_width: Wrap each output block to this many columns (default: 120).

    Returns:
        A 3-line string (optionally wrapped) in to_ascii format::

            <top strand: - for internal/leading gaps, spaces for free-end overhangs>
            <middle: | at each Watson-Crick pair>
            <bottom strand: complement of 3′→5′, spaces for free left-end overhang>
    """
    top_pad = int(client_resp["TopLinePadding"])
    bond_pad = int(client_resp["BondLinePadding"])
    bot_pad = int(client_resp["BottomLinePadding"])
    bonds = client_resp["Bonds"]

    bond_str = "".join({0: " ", 1: "|", 2: "|"}[int(b)] for b in bonds)

    # IDT encodes the bottom strand as primer2 reversed (read 3′→5′).
    top_raw = " " * top_pad + primer1
    mid_raw = " " * bond_pad + bond_str
    bot_raw_seq = " " * bot_pad + primer2[::-1]

    # Pad all three rows to the same width.
    w = max(len(top_raw), len(mid_raw), len(bot_raw_seq))
    top_raw = top_raw.ljust(w)
    mid_raw = mid_raw.ljust(w)
    bot_raw_seq = bot_raw_seq.ljust(w)

    # Apply gap-normalisation rules.
    top_n = normalize_top_row(top_raw)
    bot_n = normalize_bottom_row(bot_raw_seq)

    # Crop outer all-space columns (use all three rows to decide).
    non_space = [
        i
        for i in range(w)
        if not (top_n[i] == " " and mid_raw[i] == " " and bot_n[i] == " ")
    ]
    if not non_space:
        return "\n\n\n"
    lo, hi = min(non_space), max(non_space)
    top_n = top_n[lo : hi + 1]
    mid_raw = mid_raw[lo : hi + 1]
    bot_n = bot_n[lo : hi + 1]

    # Bottom strand is primer2 read 3′→5′ (already reversed); display without complement.
    return wrap_alignment(top_n, mid_raw, bot_n, line_width)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _check(desc: str, result: str, expected_top: str, expected_mid: str, expected_bot: str) -> None:
        lines = result.rstrip("\n").split("\n")
        assert len(lines) == 3, f"[{desc}] expected 3 lines, got {len(lines)}: {lines!r}"
        assert lines[0] == expected_top, f"[{desc}] top:  got {lines[0]!r}, want {expected_top!r}"
        assert lines[1] == expected_mid, f"[{desc}] mid:  got {lines[1]!r}, want {expected_mid!r}"
        assert lines[2] == expected_bot, f"[{desc}] bot:  got {lines[2]!r}, want {expected_bot!r}"
        print(f"  PASS  {desc}")
        for line in lines:
            print(f"        {line!r}")

    print("=== parse_idt self-check ===\n")

    # ------------------------------------------------------------------
    # 1. Perfect 4-mer WC duplex, no padding
    #    primer1=ACGT  (5′→3′ on top)
    #    primer2=ACGT  (5′→3′); primer2[::-1]="TGCA" displayed 3′→5′ on bottom
    #    WC pairs: A–T, C–G, G–C, T–A → all |
    # ------------------------------------------------------------------
    _check(
        "perfect 4-mer duplex, no padding",
        parse_idt(
            "ACGT", "ACGT",
            {"TopLinePadding": 0, "BondLinePadding": 0, "BottomLinePadding": 0, "Bonds": [2, 2, 2, 2]},
            line_width=None,
        ),
        "ACGT", "||||", "TGCA",
    )

    # ------------------------------------------------------------------
    # 2. 5′ overhangs on both strands
    #    primer1=AAACGT (top 5′ overhang = "AAA" on the left)
    #    primer2=AAACG  (bottom 5′ overhang = "AA" on the right, since primer2[::-1])
    #    IDT places primer2[::-1]="GCAAA" at BottomLinePadding=3
    #    Bonds=[2,2,2] at BondLinePadding=3 for the central 3 WC pairs
    # ------------------------------------------------------------------
    _check(
        "5′ overhangs on both strands",
        parse_idt(
            "AAACGT", "AAACG",
            {"TopLinePadding": 0, "BondLinePadding": 3, "BottomLinePadding": 3, "Bonds": [2, 2, 2]},
            line_width=None,
        ),
        "AAACGT  ", "   |||  ", "   GCAAA",
    )

    # ------------------------------------------------------------------
    # 3. IDT Bonds=1 (secondary pairs) render as |
    #    primer1="GT", primer2="GT" → primer2[::-1]="TG" displayed 3′→5′
    #    G·T and T·G are non-canonical but IDT marks them as secondary bonds
    #    (Bonds=1).  Both 1 and 2 must produce '|' in the middle line.
    # ------------------------------------------------------------------
    _check(
        "IDT Bonds=1 (secondary pairs) render as |",
        parse_idt(
            "GT", "GT",
            {"TopLinePadding": 0, "BondLinePadding": 0, "BottomLinePadding": 0, "Bonds": [1, 1]},
            line_width=None,
        ),
        "GT", "||", "TG",
    )

    # ------------------------------------------------------------------
    # 4. Middle line follows IDT Bonds array directly: 0→' ', 1 and 2→'|'
    #    primer1=ACGT, primer2=ATTT → primer2[::-1]="TTTA" displayed 3′→5′
    #    Bonds=[2,0,0,2]: positions 0 and 3 bonded → "|  |"
    #    (Bonds=0 at positions 1 and 2 forces spaces regardless of base identity)
    # ------------------------------------------------------------------
    _check(
        "middle line follows IDT Bonds array (0→space, 1|2→|)",
        parse_idt(
            "ACGT", "ATTT",
            {"TopLinePadding": 0, "BondLinePadding": 0, "BottomLinePadding": 0, "Bonds": [2, 0, 0, 2]},
            line_width=None,
        ),
        "ACGT", "|  |", "TTTA",
    )

    # ------------------------------------------------------------------
    # 5. Cross-validate overhang-to-space convention against dimer_to_ascii
    #
    # Both parse_idt and dimer_to_ascii display:
    #   top: primer1 5′→3′ (leading gaps as '-', trailing gaps as ' ')
    #   bot: primer2 3′→5′ = primer2[::-1] (leading spaces for free left end,
    #        trailing gaps as '-' for fixed right end)
    #
    # normalize_top_row:    left+internal spaces → '-'; trailing '-' → ' '
    # normalize_bottom_row: leading '-' → ' ';  right+internal spaces → '-'
    # ------------------------------------------------------------------
    from sgad.pairwise import dimer_to_ascii as _dimer_to_ascii

    def _space_mask(row: str) -> list[bool]:
        return [c == " " for c in row]

    # Case A: free bottom-left overhang
    #   primer1="AAACGT", primer2="ACG" (3-mer aligns to the 3' end of primer1)
    #   primer2[::-1]="GCA" at BottomLinePadding=3 → bottom line: "   GCA"
    #   dimer equivalent: aligned_primer1="AAACGT", aligned_primer2_comp="---GCA"
    #   dimer_to_ascii shows bottom as "   GCA" (leading "---" free → spaces)
    idt_A = parse_idt(
        "AAACGT", "ACG",
        {"TopLinePadding": 0, "BondLinePadding": 3, "BottomLinePadding": 3, "Bonds": [2, 2, 2]},
        line_width=None,
    )
    dim_A = _dimer_to_ascii("AAACGT", "---GCA", line_width=None)
    A_top_idt, _, A_bot_idt = idt_A.rstrip("\n").split("\n")
    A_top_dim, _, A_bot_dim = dim_A.rstrip("\n").split("\n")
    assert _space_mask(A_top_idt) == _space_mask(A_top_dim), (
        f"Case A top mismatch: idt={A_top_idt!r}  dim={A_top_dim!r}"
    )
    assert _space_mask(A_bot_idt) == _space_mask(A_bot_dim), (
        f"Case A bot mismatch: idt={A_bot_idt!r}  dim={A_bot_dim!r}"
    )
    assert A_bot_idt == "   GCA", f"expected '   GCA' in bot, got {A_bot_idt!r}"
    assert A_bot_dim == "   GCA", f"expected dimer_to_ascii bot '   GCA', got {A_bot_dim!r}"
    print("  PASS  Case A — free bottom-left overhang spaces match dimer_to_ascii")
    print(f"        parse_idt  top={A_top_idt!r}  bot={A_bot_idt!r}")
    print(f"        dimer_ascii top={A_top_dim!r}  bot={A_bot_dim!r}")

    # Case B: free top-right overhang
    #   primer1="ACGT", primer2="GGACGT" → primer2[::-1]="TGCAGG"
    #   IDT Bonds=[2,2,2,2] at BondLinePadding=0, BottomLinePadding=0
    #   Top line: "ACGT  " (trailing spaces for free right end)
    #   Bottom line: "TGCAGG"
    #   dimer equivalent: aligned_primer1="ACGT--", aligned_primer2_comp="TGCAGG"
    #   dimer_to_ascii shows top as "ACGT  " (trailing "--" free → spaces)
    idt_B = parse_idt(
        "ACGT", "GGACGT",
        {"TopLinePadding": 0, "BondLinePadding": 0, "BottomLinePadding": 0, "Bonds": [2, 2, 2, 2]},
        line_width=None,
    )
    dim_B = _dimer_to_ascii("ACGT--", "TGCAGG", line_width=None)
    B_top_idt, _, B_bot_idt = idt_B.rstrip("\n").split("\n")
    B_top_dim, _, B_bot_dim = dim_B.rstrip("\n").split("\n")
    assert _space_mask(B_top_idt) == _space_mask(B_top_dim), (
        f"Case B top mismatch: idt={B_top_idt!r}  dim={B_top_dim!r}"
    )
    assert _space_mask(B_bot_idt) == _space_mask(B_bot_dim), (
        f"Case B bot mismatch: idt={B_bot_idt!r}  dim={B_bot_dim!r}"
    )
    assert B_top_idt[4:] == "  ", f"expected 2 trailing spaces in top, got {B_top_idt!r}"
    assert B_bot_idt == "TGCAGG", f"expected bot 'TGCAGG', got {B_bot_idt!r}"
    assert B_bot_dim == "TGCAGG", f"expected dimer_to_ascii bot 'TGCAGG', got {B_bot_dim!r}"
    print("  PASS  Case B — free top-right overhang spaces match dimer_to_ascii")
    print(dim_B)
    print(f"        parse_idt  top={B_top_idt!r}  bot={B_bot_idt!r}")
    print(f"        dimer_ascii top={B_top_dim!r}  bot={B_bot_dim!r}")

    print("\nAll parse_idt checks passed.")
