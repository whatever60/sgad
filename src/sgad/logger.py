"""Logging helpers for alignment score event tracing."""

from types import SimpleNamespace
from typing import Callable, Iterable


class GapPenaltyLogger:
    """Filter, build, format, and dispatch score-alignment events."""

    def __init__(
        self,
        *,
        gap_event_logger: Callable[[object], None] | None,
        gap_event_types: Iterable[str] | None,
    ) -> None:
        """Store output callback and optional event-type whitelist."""
        self._gap_event_logger = gap_event_logger
        self._allowed_gap_events = (
            set(gap_event_types) if gap_event_types is not None else None
        )

    @staticmethod
    def make_event(
        *,
        event: str,
        col_idx: int,
        seq1_pos_: int,
        seq2_pos_: int,
        mask: int,
        prev_mask: int,
        raw_penalty: float,
        factor: float,
        scaled_penalty: float,
        seq1_char_: str | None = None,
        seq2_char_: str | None = None,
    ) -> object:
        """Build one event object with attribute-style access."""
        return SimpleNamespace(
            event=event,
            col_idx=col_idx,
            seq1_pos=seq1_pos_,
            seq2_pos=seq2_pos_,
            mask=mask,
            prev_mask=prev_mask,
            raw_penalty=raw_penalty,
            factor=factor,
            scaled_penalty=scaled_penalty,
            seq1_char=seq1_char_,
            seq2_char=seq2_char_,
        )

    @staticmethod
    def format_event(event: object) -> str:
        """Format one score event as a compact single-line debug record."""
        chars = ""
        seq1_char = getattr(event, "seq1_char")
        seq2_char = getattr(event, "seq2_char")
        if seq1_char is not None and seq2_char is not None:
            chars = f" seq1_char={seq1_char} seq2_char={seq2_char}"
        return (
            "[score_alignment] "
            f"event={getattr(event, 'event')} "
            f"col_idx={getattr(event, 'col_idx')} "
            f"seq1_pos={getattr(event, 'seq1_pos')} "
            f"seq2_pos={getattr(event, 'seq2_pos')} "
            f"mask={getattr(event, 'mask')} "
            f"prev_mask={getattr(event, 'prev_mask')} "
            f"raw_penalty={getattr(event, 'raw_penalty'):.6g} "
            f"factor={getattr(event, 'factor'):.6g} "
            f"scaled_penalty={getattr(event, 'scaled_penalty'):.6g}"
            f"{chars}"
        )

    @classmethod
    def stdout(cls, event: object) -> None:
        """Print one score event to stdout."""
        print(cls.format_event(event))

    def emit(
        self,
        *,
        event: str,
        col_idx: int,
        seq1_pos_: int,
        seq2_pos_: int,
        mask: int,
        prev_mask: int,
        raw_penalty: float,
        factor: float,
        scaled_penalty: float,
        seq1_char_: str | None = None,
        seq2_char_: str | None = None,
    ) -> None:
        """Emit one score event if callback and event type are enabled."""
        if self._gap_event_logger is None:
            return
        if (
            self._allowed_gap_events is not None
            and event not in self._allowed_gap_events
        ):
            return
        self._gap_event_logger(
            self.make_event(
                event=event,
                col_idx=col_idx,
                seq1_pos_=seq1_pos_,
                seq2_pos_=seq2_pos_,
                mask=mask,
                prev_mask=prev_mask,
                raw_penalty=raw_penalty,
                factor=factor,
                scaled_penalty=scaled_penalty,
                seq1_char_=seq1_char_,
                seq2_char_=seq2_char_,
            )
        )
