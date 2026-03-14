"""Run IDT OligoAnalyzer heterodimer analysis for every pair in two primer sets."""

import time
from collections import deque

import requests


class _RollingRateLimiter:
    """Enforce at most *max_calls* calls within any rolling *period_s*-second window."""

    def __init__(
        self,
        max_calls: int,
        period_s: float,
        safety_margin_s: float = 0.05,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        if period_s <= 0:
            raise ValueError("period_s must be > 0")
        self._max_calls = max_calls
        self._period_s = period_s
        self._safety_margin_s = safety_margin_s
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        """Block until a call is permitted, then record the call timestamp."""
        now = time.monotonic()
        while self._timestamps and (now - self._timestamps[0]) >= self._period_s:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_calls:
            oldest = self._timestamps[0]
            sleep_s = (oldest + self._period_s) - now + self._safety_margin_s
            if sleep_s > 0:
                time.sleep(sleep_s)
            now = time.monotonic()
            while self._timestamps and (now - self._timestamps[0]) >= self._period_s:
                self._timestamps.popleft()
        self._timestamps.append(time.monotonic())

    def reset(self) -> None:
        """Clear rate-limiter history (call after sleeping through a backoff)."""
        self._timestamps.clear()


class _IdtClient:
    """Minimal IDT SciTools OligoAnalyzer REST client with automatic token refresh."""

    _TOKEN_URL = "https://www.idtdna.com/Identityserver/connect/token"
    _HETERODIMER_URL = "https://www.idtdna.com/restapi/v1/OligoAnalyzer/HeteroDimer"
    _TOKEN_REFRESH_SKEW_S = 30.0

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        idt_username: str,
        idt_password: str,
        scope: str = "test",
        timeout_s: float = 30.0,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._idt_username = idt_username
        self._idt_password = idt_password
        self._scope = scope
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._access_token: str | None = None
        self._token_expires_at_mono: float = 0.0

    def close(self) -> None:
        self._session.close()

    def _fetch_token(self) -> tuple[str, float]:
        resp = self._session.post(
            self._TOKEN_URL,
            data={
                "grant_type": "password",
                "username": self._idt_username,
                "password": self._idt_password,
                "scope": self._scope,
            },
            auth=(self._client_id, self._client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not token or not expires_in:
            raise RuntimeError(f"Unexpected token response: {payload}")
        return str(token), float(expires_in)

    def _ensure_token(self) -> str:
        now = time.monotonic()
        if (
            self._access_token is None
            or (now + self._TOKEN_REFRESH_SKEW_S) >= self._token_expires_at_mono
        ):
            token, expires_in = self._fetch_token()
            self._access_token = token
            self._token_expires_at_mono = time.monotonic() + expires_in
        return self._access_token  # type: ignore[return-value]

    def invalidate_token(self) -> None:
        self._access_token = None
        self._token_expires_at_mono = 0.0

    def heterodimer(self, primer1: str, primer2: str) -> list[dict]:
        """Call the IDT HeteroDimer endpoint and return the parsed JSON list."""
        token = self._ensure_token()
        resp = self._session.post(
            self._HETERODIMER_URL,
            params={"primary": primer1, "secondary": primer2},
            headers={"Authorization": f"Bearer {token}"},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return resp.json()


def heterodimer_batch_idt(
    primer1_seqs: list[str],
    primer2_seqs: list[str],
    primer1_names: list[str],
    primer2_names: list[str],
    *,
    client_id: str,
    client_secret: str,
    idt_username: str,
    idt_password: str,
    scope: str = "test",
    timeout_s: float = 30.0,
    max_calls_per_minute: int = 500,
    max_retries: int = 6,
    raise_on_error: bool = False,
) -> list[dict]:
    """Run IDT OligoAnalyzer heterodimer analysis for every (primer1 × primer2) pair.

    Calls are rate-limited to *max_calls_per_minute* (the SciTools Plus limit
    is 500 calls/min) and retried on transient failures (HTTP 429, 401/403,
    and network errors).

    Args:
        primer1_seqs: DNA sequences for the first primer set (5′→3′).
        primer2_seqs: DNA sequences for the second primer set (5′→3′).
        primer1_names: Labels for *primer1_seqs*.
        primer2_names: Labels for *primer2_seqs*.
        client_id: IDT API client id.
        client_secret: IDT API client secret.
        idt_username: IDT account username.
        idt_password: IDT account password.
        scope: OAuth scope (default: ``"test"``).
        timeout_s: Per-request HTTP timeout in seconds.
        max_calls_per_minute: API rate limit ceiling (default: 500).
        max_retries: Maximum retries per pair on transient errors.
        raise_on_error: If ``True``, raise immediately on the first error
            instead of recording a failure record and continuing.

    Returns:
        A list of dicts, one per (primer1, primer2) pair in row-major order
        (primer1 index varies slowest).  Each dict contains:

        - ``primer1_name``, ``primer2_name`` — labels from the input lists.
        - ``primer1``, ``primer2`` — sequences.
        - ``ok`` — ``True`` if the API call succeeded.
        - ``response`` — IDT JSON result (list of structure dicts), or
          ``None`` on failure.
        - ``status_code`` — HTTP status code, or ``None`` on network error.
        - ``error`` — error string, or ``None`` on success.

    Raises:
        ValueError: If sequence/name lists have mismatched lengths.
    """
    if len(primer1_seqs) != len(primer1_names):
        raise ValueError("primer1_seqs and primer1_names must have the same length.")
    if len(primer2_seqs) != len(primer2_names):
        raise ValueError("primer2_seqs and primer2_names must have the same length.")

    # Build the full cross-product of pairs (row-major: primer1 index varies slowest).
    pairs = [
        (primer1_seqs[i], primer2_seqs[j], primer1_names[i], primer2_names[j])
        for i in range(len(primer1_seqs))
        for j in range(len(primer2_seqs))
    ]

    limiter = _RollingRateLimiter(max_calls=max_calls_per_minute, period_s=60.0)
    client = _IdtClient(
        client_id=client_id,
        client_secret=client_secret,
        idt_username=idt_username,
        idt_password=idt_password,
        scope=scope,
        timeout_s=timeout_s,
    )

    results: list[dict] = []

    # ## `except requests.HTTPError as exc:`
    # This is for cases where the server **did respond**, but with an error HTTP status. In this function, that means things like:
    # - `429` rate limited
    # - `401` or `403` auth/token problems
    # - other non-2xx API failures
    # This block exists separately because you can inspect `exc.response`, including:
    # - `status_code`
    # - headers like `Retry-After`
    # That lets the code do smarter recovery:
    # - wait and retry on `429`
    # - refresh token and retry on `401/403`
    # - otherwise record the failure
    # ## `except requests.RequestException as exc:`
    # This is the broader "requests failed" bucket for cases where the problem is more about the request/connection itself than an HTTP status response, such as:
    # - timeout
    # - connection error
    # - DNS failure
    # - SSL error
    # - too many redirects
    # In these cases there may be **no usable HTTP response**, so the code stores `status_code: None` and retries with exponential backoff.
    # One important detail: `HTTPError` is a subclass of `RequestException`, so the order matters. If you caught `RequestException` first, the `HTTPError` block would never run.
    # So the structure is basically:
    # - `HTTPError` → API returned an error response
    # - `RequestException` → network/request machinery failed
    try:
        for p1, p2, name1, name2 in pairs:
            attempt = 0
            while True:
                attempt += 1
                limiter.acquire()
                try:
                    resp_json = client.heterodimer(p1, p2)
                    results.append(
                        {
                            "primer1_name": name1,
                            "primer2_name": name2,
                            "primer1": p1,
                            "primer2": p2,
                            "ok": True,
                            "response": resp_json,
                            "status_code": 200,
                            "error": None,
                        }
                    )
                    break

                except requests.HTTPError as exc:
                    status = (
                        exc.response.status_code if exc.response is not None else None
                    )

                    if status == 429 and attempt <= max_retries:
                        retry_after: float | None = None
                        if exc.response is not None:
                            ra_hdr = exc.response.headers.get("Retry-After")
                            if ra_hdr is not None:
                                try:
                                    retry_after = float(ra_hdr)
                                except ValueError:
                                    pass
                        backoff_s = (
                            retry_after
                            if retry_after is not None
                            else float(2 ** (attempt - 1))
                        )
                        time.sleep(min(backoff_s, 120.0))
                        limiter.reset()
                        continue

                    # Refresh token if auth fails, then retry
                    if status in (401, 403) and attempt <= max_retries:
                        client._access_token = None
                        client._token_expires_at_mono = 0.0
                        time.sleep(1.0)
                        limiter.reset()
                        continue

                    results.append(
                        {
                            "primer1_name": name1,
                            "primer2_name": name2,
                            "primer1": p1,
                            "primer2": p2,
                            "ok": False,
                            "response": None,
                            "status_code": status,
                            "error": str(exc),
                        }
                    )
                    if raise_on_error:
                        raise
                    break

                except requests.RequestException as exc:
                    if attempt <= max_retries:
                        time.sleep(min(float(2 ** (attempt - 1)), 120.0))
                        limiter.reset()
                        continue

                    results.append(
                        {
                            "primer1_name": name1,
                            "primer2_name": name2,
                            "primer1": p1,
                            "primer2": p2,
                            "ok": False,
                            "response": None,
                            "status_code": None,
                            "error": str(exc),
                        }
                    )
                    if raise_on_error:
                        raise
                    break

    finally:
        client.close()

    return results


if __name__ == "__main__":
    from unittest.mock import MagicMock, patch

    print("=== heterodimer_batch_idt self-check ===\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _FAKE_RESPONSE = [
        {
            "StartPosition": 0,
            "TopLinePadding": 0,
            "BondLinePadding": 0,
            "BottomLinePadding": 0,
            "Bonds": [2, 2, 2, 2],
            "DeltaG": -3.5,
            "BasePairs": 4,
            "Dimer": None,
        }
    ]

    def _make_client_mock(response=None, side_effect=None):
        """Return a mock _IdtClient whose heterodimer() returns *response*."""
        mock = MagicMock(spec=_IdtClient)
        if side_effect is not None:
            mock.heterodimer.side_effect = side_effect
        else:
            mock.heterodimer.return_value = response or _FAKE_RESPONSE
        return mock

    # ------------------------------------------------------------------
    # 1. Happy path: 2×2 cross-product → 4 result records
    # ------------------------------------------------------------------
    with patch(
        f"{__name__}._IdtClient", return_value=_make_client_mock(_FAKE_RESPONSE)
    ):
        results = heterodimer_batch_idt(
            ["ACGT", "GCTA"],
            ["TGCA", "CGAT"],
            ["p1", "p2"],
            ["q1", "q2"],
            client_id="x",
            client_secret="x",
            idt_username="x",
            idt_password="x",
        )

    assert len(results) == 4, f"expected 4 records, got {len(results)}"
    assert all(r["ok"] for r in results), "all results should be ok"
    pairs_expected = [("p1", "q1"), ("p1", "q2"), ("p2", "q1"), ("p2", "q2")]
    pairs_got = [(r["primer1_name"], r["primer2_name"]) for r in results]
    assert pairs_got == pairs_expected, f"pair order wrong: {pairs_got}"
    print("  PASS  happy path: 4 records in row-major order, all ok=True")

    # ------------------------------------------------------------------
    # 2. Non-retryable HTTP error (e.g. 404) → ok=False, no retry
    # ------------------------------------------------------------------
    err_response = MagicMock()
    err_response.status_code = 404
    http_err = requests.HTTPError(response=err_response)

    mock_client = _make_client_mock(side_effect=http_err)
    with patch(f"{__name__}._IdtClient", return_value=mock_client):
        results_err = heterodimer_batch_idt(
            ["ACGT"],
            ["TGCA"],
            ["p1"],
            ["q1"],
            client_id="x",
            client_secret="x",
            idt_username="x",
            idt_password="x",
            max_retries=3,
        )

    assert len(results_err) == 1
    assert not results_err[0]["ok"]
    assert results_err[0]["status_code"] == 404
    # heterodimer() should be called exactly once (no retry for 404)
    assert mock_client.heterodimer.call_count == 1, (
        f"expected 1 call, got {mock_client.heterodimer.call_count}"
    )
    print("  PASS  non-retryable HTTP 404 → ok=False, no retry")

    # ------------------------------------------------------------------
    # 3. raise_on_error=True propagates the exception
    # ------------------------------------------------------------------
    err_response2 = MagicMock()
    err_response2.status_code = 500
    http_err2 = requests.HTTPError(response=err_response2)

    with patch(
        f"{__name__}._IdtClient", return_value=_make_client_mock(side_effect=http_err2)
    ):
        try:
            heterodimer_batch_idt(
                ["ACGT"],
                ["TGCA"],
                ["p1"],
                ["q1"],
                client_id="x",
                client_secret="x",
                idt_username="x",
                idt_password="x",
                raise_on_error=True,
            )
        except requests.HTTPError:
            print("  PASS  raise_on_error=True propagates HTTPError")
        else:
            raise AssertionError("expected HTTPError not raised")

    # ------------------------------------------------------------------
    # 4. ValueError on mismatched list lengths
    # ------------------------------------------------------------------
    try:
        heterodimer_batch_idt(
            ["ACGT"],
            ["TGCA"],
            ["p1", "p2"],
            ["q1"],  # primer1_seqs/names mismatch
            client_id="x",
            client_secret="x",
            idt_username="x",
            idt_password="x",
        )
    except ValueError as e:
        print(f"  PASS  ValueError on mismatched lengths: {e}")
    else:
        raise AssertionError("expected ValueError not raised")

    # ------------------------------------------------------------------
    # 5. _RollingRateLimiter: bursts up to max_calls then throttles
    # ------------------------------------------------------------------
    limiter = _RollingRateLimiter(max_calls=3, period_s=60.0)
    # First 3 calls should go through instantly.
    t0 = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"first 3 calls took too long: {elapsed:.3f}s"
    print(
        f"  PASS  _RollingRateLimiter: first 3 calls instant ({elapsed * 1000:.1f}ms)"
    )

    # After reset(), the 4th call should also be instant.
    limiter.reset()
    t1 = time.monotonic()
    limiter.acquire()
    elapsed2 = time.monotonic() - t1
    assert elapsed2 < 0.5, f"call after reset() took too long: {elapsed2:.3f}s"
    print(
        f"  PASS  _RollingRateLimiter: call after reset() is instant ({elapsed2 * 1000:.1f}ms)"
    )

    # ------------------------------------------------------------------
    # 6. _RollingRateLimiter: invalid construction args
    # ------------------------------------------------------------------
    try:
        _RollingRateLimiter(max_calls=0, period_s=10.0)
    except ValueError:
        print("  PASS  _RollingRateLimiter raises ValueError for max_calls=0")

    try:
        _RollingRateLimiter(max_calls=5, period_s=0.0)
    except ValueError:
        print("  PASS  _RollingRateLimiter raises ValueError for period_s=0.0")

    print("\nAll heterodimer_batch_idt checks passed.")
