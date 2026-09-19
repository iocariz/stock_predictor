"""The scheduled run has to survive a throttled vendor.

The first night the cron fired it died on the coverage guard:

    DownloadCoverageError: equity download: only 73/470 current index members
    returned (15.5%), below the 98% threshold.

The guard was right -- a partial download silently changes the traded universe
and every cross-sectional feature computed from it -- but the run was then
simply lost, and an unattended job that gives up on the first throttle is not
automation. Yahoo throttles large batches, and the default batch is 100.

Two things this pins:

* ``run_pipeline.sh`` can pass ``--batch-size`` through to predict-sp500 at all.
  It could not: the flag existed on the CLI and had no environment variable, so
  the documented remedy for a throttle was unreachable from the automation.
* ``scripts/predict_cron.sh`` retries, and retries *only* when a retry can help.
  A bad model path is not going to fix itself on the third attempt.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_pipeline.sh"
CRON = ROOT / "scripts" / "predict_cron.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


def _pipeline_cmd(**env: str) -> str:
    out = subprocess.run(
        ["bash", str(PIPELINE), "predict"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "DRY_RUN": "1", **env},
    )
    return out.stdout.strip().splitlines()[-1]


# ---------------------------------------------------------------------------
# The flag has to be reachable from the automation
# ---------------------------------------------------------------------------


def test_batch_size_reaches_predict() -> None:
    assert "--batch-size 50" in _pipeline_cmd(PREDICT_BATCH_SIZE="50")


def test_batch_size_is_absent_when_unset() -> None:
    """Unset must keep the provider's own default, not hardcode one here."""
    assert "--batch-size" not in _pipeline_cmd(PREDICT_BATCH_SIZE="")


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def _run_cron(tmp_path: Path, script: str, **env: str):
    """Drive predict_cron.sh with a stub pipeline that we control."""
    stub = tmp_path / "run_pipeline.sh"
    stub.write_text(script)
    stub.chmod(0o755)
    return subprocess.run(
        ["bash", str(CRON)],
        capture_output=True, text=True,
        env={**os.environ, "PIPELINE_CMD": str(stub), "RETRY_WAIT": "0",
             "LOG_FILE": str(tmp_path / "out.log"), **env},
    )


def test_a_transient_failure_is_retried(tmp_path: Path) -> None:
    """Fails once, succeeds on the second attempt: the cron should recover."""
    counter = tmp_path / "n"
    res = _run_cron(tmp_path, f"""#!/bin/bash
n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
if [[ $n -lt 2 ]]; then
  echo "DownloadCoverageError: only 73/470 current index members returned" >&2
  exit 1
fi
echo "Portfolio updated: portfolio_state.json"
""")
    assert res.returncode == 0, res.stderr
    assert counter.read_text().strip() == "2", "did not retry exactly once"


def test_retries_are_bounded(tmp_path: Path) -> None:
    """A vendor that is down all night must not spawn runs forever."""
    counter = tmp_path / "n"
    res = _run_cron(tmp_path, f"""#!/bin/bash
n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
echo "DownloadCoverageError: only 73/470 returned" >&2
exit 1
""", MAX_ATTEMPTS="3")
    assert res.returncode != 0, "a run that never succeeded reported success"
    assert counter.read_text().strip() == "3"


def test_a_permanent_failure_is_not_retried(tmp_path: Path) -> None:
    """Retrying a missing model three times just delays the same error."""
    counter = tmp_path / "n"
    res = _run_cron(tmp_path, f"""#!/bin/bash
n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
echo "FileNotFoundError: artifacts/model.pkl" >&2
exit 1
""")
    assert res.returncode != 0
    assert counter.read_text().strip() == "1", "retried a permanent failure"


def test_success_runs_once(tmp_path: Path) -> None:
    counter = tmp_path / "n"
    res = _run_cron(tmp_path, f"""#!/bin/bash
n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}
echo "Portfolio updated: portfolio_state.json"
""")
    assert res.returncode == 0
    assert counter.read_text().strip() == "1", "retried a successful run"


def test_the_retry_lowers_the_batch_size(tmp_path: Path) -> None:
    """The remedy the error message itself names. Retrying an identical
    request against a throttling vendor is a slower way to fail."""
    args = tmp_path / "args"
    res = _run_cron(tmp_path, f"""#!/bin/bash
echo "PREDICT_BATCH_SIZE=$PREDICT_BATCH_SIZE" >> {args}
n=$(wc -l < {args} | tr -d ' ')
if [[ $n -lt 2 ]]; then
  echo "DownloadCoverageError: only 73/470 returned" >&2
  exit 1
fi
echo "Portfolio updated"
""")
    assert res.returncode == 0, res.stderr
    seen = [int(ln.split("=")[1]) for ln in args.read_text().split() if "=" in ln]
    assert seen[1] < seen[0], f"batch size not lowered on retry: {seen}"


# ---------------------------------------------------------------------------
# Idle sleep
# ---------------------------------------------------------------------------


def test_the_run_holds_off_idle_sleep(tmp_path: Path) -> None:
    """The failure nine scheduled attempts actually had.

    This Mac suspends after ~45 seconds idle and wakes for ~3. A 470-ticker
    download cannot complete through that: connections drop and the result
    looks exactly like a vendor that served a fraction of the universe. Every
    scheduled attempt across 2026-09-16..19 came back 15-26% complete, at three
    batch sizes and two times of day, while every run started by hand -- with
    someone at the keyboard keeping the machine awake -- returned 470/470.
    Batch size and hour were both read as causes and both were wrong; the
    confound was the operator's presence.

    caffeinate re-parents the command rather than wrapping it, so process
    ancestry cannot be asserted on portably. What is checked here is the
    decision the wrapper records; that it holds a real PreventUserIdleSystem
    assertion was verified out-of-band with `pmset -g assertions`.
    """
    log = tmp_path / "out.log"
    stub = tmp_path / "run_pipeline.sh"
    stub.write_text('#!/bin/bash\necho "Portfolio updated"\n')
    stub.chmod(0o755)
    res = subprocess.run(
        ["bash", str(CRON)],
        capture_output=True, text=True,
        env={**os.environ, "PIPELINE_CMD": str(stub), "RETRY_WAIT": "0",
             "LOG_FILE": str(log)},
    )
    assert res.returncode == 0, res.stderr
    text = log.read_text()
    if shutil.which("caffeinate"):
        assert "caffeinate -i holding off idle sleep" in text, text
    else:
        assert "no caffeinate" in text, text


def test_a_missing_caffeinate_does_not_break_the_run(tmp_path: Path) -> None:
    """caffeinate is macOS-only. On a machine without it the run must still
    happen -- degraded, not dead. CI is Linux."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    stub = tmp_path / "run_pipeline.sh"
    stub.write_text('#!/bin/bash\necho "Portfolio updated"\n')
    stub.chmod(0o755)
    res = subprocess.run(
        ["bash", str(CRON)],
        capture_output=True, text=True,
        # A PATH with no caffeinate on it, but keep the basics the script needs.
        env={**os.environ, "PIPELINE_CMD": str(stub), "RETRY_WAIT": "0",
             "LOG_FILE": str(tmp_path / "out.log"),
             "PATH": f"{fake_bin}:/usr/bin:/bin"},
    )
    assert res.returncode == 0, f"no caffeinate broke the run: {res.stderr}"
