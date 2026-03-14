"""High-level heterodimer batch APIs.

Functions
---------
heterodimer_batch_primer3
    Compute primer3 heterodimer ΔG / Tm for every pair in two primer sets.
    Requires ``primer3-py`` and ``pandas``.
heterodimer_batch_idt
    Run IDT OligoAnalyzer heterodimer analysis for every pair in two primer sets.
    Requires ``requests``.

Submodules
----------
_primer3
    :func:`heterodimer_batch_primer3` implementation.
_idt
    :func:`heterodimer_batch_idt` implementation, plus :class:`_IdtClient`
    and :class:`_RollingRateLimiter` internals.
"""

from .primer3 import heterodimer_batch_primer3
from .idt import heterodimer_batch_idt
