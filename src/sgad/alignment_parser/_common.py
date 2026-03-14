"""Shared low-level utilities for alignment normalisation and rendering.

These functions are internal to the ``sgad.alignment_parser`` package and
are not part of the public API.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Gap-normalisation helpers
# ---------------------------------------------------------------------------


def _first_last_letter(row: str) -> tuple[int | None, int | None]:
    """Return indices of the first and last alphabetic character in *row*."""
    idxs = [i for i, ch in enumerate(row) if ch.isalpha()]
    if not idxs:
        return None, None
    return idxs[0], idxs[-1]


def normalize_top_row(row: str) -> str:
    """Apply SEQ / top-strand end convention.

    - Left of first base and internal span: spaces → ``-``; existing ``-`` kept.
    - Right of last base: ``-`` → space (free 3′ overhang shown as spaces).
    """
    first, last = _first_last_letter(row)
    if first is None:
        return row.replace(" ", "-")
    assert last is not None
    chars = list(row)
    for i in range(0, last + 1):
        if chars[i] == " ":
            chars[i] = "-"
    for i in range(last + 1, len(chars)):
        if chars[i] == "-":
            chars[i] = " "
    return "".join(chars)


def normalize_bottom_row(row: str) -> str:
    """Apply STR / bottom-strand end convention.

    - Left of first base: ``-`` → space (free 5′ overhang of primer2 shown as spaces).
    - Internal span: spaces → ``-``; existing ``-`` kept.
    - Right of last base: spaces → ``-``; existing ``-`` kept.
    """
    first, last = _first_last_letter(row)
    if first is None:
        return row.replace(" ", "-")
    assert last is not None
    chars = list(row)
    for i in range(0, first):
        if chars[i] == "-":
            chars[i] = " "
    for i in range(first, last + 1):
        if chars[i] == " ":
            chars[i] = "-"
    for i in range(last + 1, len(chars)):
        if chars[i] == " ":
            chars[i] = "-"
    return "".join(chars)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def wrap_alignment(top: str, mid: str, bot: str, line_width: int | None) -> str:
    """Wrap a 3-line alignment to *line_width* columns (same semantics as to_ascii).

    Args:
        top: Top strand string.
        mid: Middle (pairing) string.
        bot: Bottom strand string.
        line_width: Column limit; ``None`` or ``≤ 0`` disables wrapping.

    Returns:
        Formatted alignment string ending with ``'\\n'``.
    """
    if line_width is None or line_width <= 0:
        return f"{top}\n{mid}\n{bot}\n"
    blocks: list[str] = []
    for k in range(0, len(top), line_width):
        blocks.append(top[k : k + line_width])
        blocks.append(mid[k : k + line_width])
        blocks.append(bot[k : k + line_width])
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"
