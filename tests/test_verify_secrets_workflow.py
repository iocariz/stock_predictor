"""A secret that exists is not a secret that works.

``TIINGO_API_KEY`` has been configured since 2026-09-01, correctly named and
correctly referenced by the training workflow. None of that establishes that
the *value* is a working key: GitHub never exposes a secret to anyone, so only
a run that actually calls the vendor can tell you. Nothing had.

The only thing that would have exercised it is the monthly retrain, which takes
up to twelve hours and burns real Tiingo quota — so a wrong value would be
discovered by losing a scheduled run, which is how the 2026-09-01 cron failed in
the first place. This workflow is the cheap version: one authentication call,
one small price call, no training, no dependency sync.

The tests below pin the two properties that make it safe to run and worth
running: it must never print a secret, and it must actually call the vendor
rather than checking that the variable is non-empty — which is all the training
workflow's preflight can do.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "verify-secrets.yml"


@pytest.fixture(scope="module")
def text() -> str:
    assert WF.exists(), f"{WF} is missing"
    return WF.read_text()


def test_it_parses(text: str) -> None:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    assert doc["jobs"], "no jobs"


def test_it_is_manual_only(text: str) -> None:
    """No schedule. This proves a secret on demand; it is not a monitor, and a
    cron calling a vendor on a timer is quota nobody asked to spend."""
    yaml = pytest.importorskip("yaml")
    # PyYAML reads a bare `on:` key as the boolean True.
    doc = yaml.safe_load(text)
    triggers = doc.get("on", doc.get(True))
    assert set(triggers) == {"workflow_dispatch"}, triggers


def test_it_reads_both_keys(text: str) -> None:
    assert "secrets.TIINGO_API_KEY" in text
    assert "secrets.FRED_API_KEY" in text


def test_it_never_echoes_a_secret(text: str) -> None:
    """The whole point is to report a verdict, not a value. Any interpolation
    of a secret into a shell word that gets printed would put it in the log --
    GitHub masks known secrets, but relying on that is not a design."""
    for line in text.splitlines():
        if "${{ secrets." not in line:
            continue
        assert re.match(r"\s*[A-Z_]+:\s*\$\{\{\s*secrets\.[A-Z_]+\s*\}\}\s*$", line), (
            f"a secret is interpolated somewhere other than an env binding: {line!r}")
    lowered = text.lower()
    for bad in ("echo \"$tiingo", "echo $tiingo", "echo \"$fred", "echo $fred"):
        assert bad not in lowered, bad


def test_it_actually_calls_the_vendor(text: str) -> None:
    """Presence is what the training preflight already checks, and it is not
    the question. This has to prove the value works."""
    assert "api.tiingo.com" in text


def test_it_fails_when_the_key_is_rejected(text: str) -> None:
    """A check that reports success on a 401 is worse than no check."""
    assert "::error::" in text
    assert re.search(r"exit\s+1", text), "no failing exit path"


def test_a_rejected_key_is_not_read_as_a_pass(text: str) -> None:
    """Tiingo answers ``/api/test/`` with **HTTP 200 even for a wrong token**,
    putting the rejection in the body: a bad key returns
    ``200 "Auth Token was not correct"``. Verified against a deliberately wrong
    key while writing this. A check reading only the status would print a pass
    at that step and leave the real failure to the price call -- which happens
    to catch it, but by accident rather than by design."""
    assert "success" in text, "the auth step does not inspect the response body"
    body_check = text[text.index("api/test/"):]
    assert re.search(r'if\s+"success"\s+not\s+in', body_check), (
        "the auth step trusts the HTTP status, which Tiingo returns as 200 "
        "for a rejected token")


def test_it_does_not_train_or_sync(text: str) -> None:
    """Thirty seconds, not twelve hours. Installing the project would make this
    slower than the thing it exists to avoid."""
    for heavy in ("run_pipeline.sh", "train-sp500", "uv sync"):
        assert heavy not in text, heavy


def test_it_is_bounded(text: str) -> None:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    for name, job in doc["jobs"].items():
        assert job.get("timeout-minutes", 999) <= 10, name


def test_a_missing_optional_key_is_reported_not_fatal(text: str) -> None:
    """FRED is referenced by the training workflow and has never been set. That
    degrades macro-merge quality; it does not stop a run, and this must say so
    rather than failing and teaching people to ignore it."""
    assert "FRED" in text
    fred_block = text[text.index("FRED_API_KEY", text.index("jobs:")):]
    assert "::warning::" in fred_block or "::notice::" in fred_block
