"""Parse heterodimer ASCII alignments from primer3 or IDT into the canonical
format produced by :func:`sgad.pairwise.to_ascii`.

In that format:

- **Line 1** (top): top strand with ``-`` for internal/leading gaps and spaces
  for free-end overhangs.
- **Line 2** (middle): ``|`` where both strands carry an identical character
  (Watson-Crick paired positions, because the bottom strand is stored as
  the complement of the 3′→5′ representation).
- **Line 3** (bottom): bottom strand expressed as the complement of the
  3′→5′ sequence, so that identical characters line up with their WC partners
  on the top strand.  Leading spaces mark the free left-end overhang.

Submodules
----------
_primer3
    :func:`parse_primer3` — handles ``SEQ``/``STR``-prefixed primer3 output.
_idt
    :func:`parse_idt` — handles IDT OligoAnalyzer JSON response dicts.
_common
    Shared low-level normalisation and rendering helpers.
"""

from .primer3 import parse_primer3
from .idt import parse_idt
