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


def test_the_whole_run_is_held_awake_not_just_the_download(tmp_path: Path) -> None:
    """Three wrong causes preceded this one; the fourth is read off the power
    log rather than inferred from correlation.

    launchd wakes a sleeping Mac into *DarkWake*, which carries a 45-second
    "wake linger" budget. When it expires the system sleeps again:

      06:51:09  DarkWake        <- launchd woke the machine to run the job
      06:51:54  powerd TimedOut InternalPreventSleep
                "com.apple.powermanagement.acwakelinger" 00:00:45
                Summary- [System: PrevIdle] Using AC
      07:09:20  attempt 1 fails -> 113/470

    caffeinate -i asserts PreventUserIdleSystemSleep, which stops sleep caused
    by the *idle timeout*. This is not an idle timeout, so PrevIdle was held
    and the machine slept through it. -s is the assertion that prevents system
    sleep outright, and is valid on AC.

    The first attempt also wrapped only the download, leaving the backoff
    exposed: a 300s wait stretched to 40 minutes because the machine suspended
    inside it. So the assertion has to cover the whole script, which is what
    re-exec gives -- not one subprocess of it.
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
        assert "caffeinate -s" in text, text
        assert "PrevIdle" not in text, "still describing the idle-only assertion"
    else:
        assert "no caffeinate" in text, text


def test_the_reexec_does_not_loop(tmp_path: Path) -> None:
    """Re-exec guards on an env var. If the guard fails the script calls itself
    forever, which on a scheduled job is far worse than the bug it fixes."""
    log = tmp_path / "out.log"
    stub = tmp_path / "run_pipeline.sh"
    stub.write_text('#!/bin/bash\necho "Portfolio updated"\n')
    stub.chmod(0o755)
    res = subprocess.run(
        ["bash", str(CRON)],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PIPELINE_CMD": str(stub), "RETRY_WAIT": "0",
             "LOG_FILE": str(log)},
    )
    assert res.returncode == 0
    # Exactly one run's worth of attempt lines, not one per re-exec.
    assert log.read_text().count("attempt 1/") == 1, log.read_text()


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


# ---------------------------------------------------------------------------
# File descriptors
# ---------------------------------------------------------------------------


def test_the_run_raises_the_file_descriptor_limit(tmp_path: Path) -> None:
    """The actual cause of every failed scheduled run.

    launchd starts jobs with `maxfiles 256`; an interactive shell here has
    1048576. yfinance opens many concurrent connections to fetch 470 tickers,
    and under 256 descriptors most of them fail. The partial frame that comes
    back is indistinguishable from a throttled vendor, which is why this was
    misdiagnosed four times -- as batch size, as time of day, as idle sleep,
    and as DarkWake. Every manual run succeeded because a shell's limit is
    four orders of magnitude higher.

    Proven by controlled experiment rather than inference: same shell, same
    minute, same network, only `ulimit -n` changed.

        ulimit -n 1048576  ->  470/470  (100.0%)
        ulimit -n 256      ->  115/470  ( 24.5%)

    and launchd kickstart at 08:24 returned 97/470 four minutes after a manual
    run returned 470/470.
    """
    seen = tmp_path / "fds"
    stub = tmp_path / "run_pipeline.sh"
    stub.write_text(f'#!/bin/bash\nulimit -n > {seen}\necho "Portfolio updated"\n')
    stub.chmod(0o755)
    # Start from launchd's soft limit, not the test runner's. Inheriting a
    # shell's 1048576 would make this pass without the fix.
    res = subprocess.run(
        ["bash", "-c", f"ulimit -Sn 256; exec bash {CRON}"],
        capture_output=True, text=True,
        env={**os.environ, "PIPELINE_CMD": str(stub), "RETRY_WAIT": "0",
             "LOG_FILE": str(tmp_path / "out.log")},
    )
    assert res.returncode == 0, res.stderr
    got = seen.read_text().strip()
    assert got == "unlimited" or int(got) >= 4096, (
        f"download ran with only {got} file descriptors; 256 (the launchd "
        "default) truncates the universe to ~25%")


def test_an_unraisable_limit_does_not_kill_the_run(tmp_path: Path) -> None:
    """A hard limit below the target must degrade, not abort. The run is still
    worth attempting, and the guard will reject a short download anyway."""
    stub = tmp_path / "run_pipeline.sh"
    stub.write_text('#!/bin/bash\necho "Portfolio updated"\n')
    stub.chmod(0o755)
    res = subprocess.run(
        ["bash", "-c", f"ulimit -Hn 128 2>/dev/null; exec bash {CRON}"],
        capture_output=True, text=True,
        env={**os.environ, "PIPELINE_CMD": str(stub), "RETRY_WAIT": "0",
             "LOG_FILE": str(tmp_path / "out.log")},
    )
    assert res.returncode == 0, f"a low hard limit aborted the run: {res.stderr}"
