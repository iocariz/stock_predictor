"""The automation must measure the configuration it trades.

`run_pipeline.sh` used to pass the strategy settings to `predict-sp500` only.
`backtest-sp500` received nothing and fell back to its own defaults, so
`TOP_N=25` traded twenty-five names against a simulation of fifteen. The
defaults happened to agree, which is why nothing looked wrong.

These tests drive the real script with `DRY_RUN=1` and compare the two commands
it would run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.sh"

# Anything that changes *what is held* must reach both sides. Paths, providers
# and the kill-switch are legitimately one-sided.
SELECTION_FLAGS = (
    "--top-n", "--holding-days", "--max-cohorts", "--slippage-bps",
    "--weighting", "--exit-rank", "--rank-offset",
    "--commission-per-share", "--commission-per-order",
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


def _cmd(mode: str, **env: str) -> list[str]:
    out = subprocess.run(
        ["bash", str(SCRIPT), mode],
        capture_output=True, text=True, check=True,
        env={**os.environ, "DRY_RUN": "1", **env},
    )
    return out.stdout.strip().split()


def _flag(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_the_script_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize("flag", SELECTION_FLAGS)
def test_every_selection_flag_reaches_both_sides(flag: str) -> None:
    assert _flag(_cmd("backtest"), flag) == _flag(_cmd("predict"), flag) is not None


def test_an_override_moves_both_sides_together() -> None:
    """The actual failure: TOP_N reached the account and not the simulation."""
    env = {"TOP_N": "25"}
    assert _flag(_cmd("backtest", **env), "--top-n") == "25"
    assert _flag(_cmd("predict", **env), "--top-n") == "25"


def test_a_score_floor_reaches_both_sides() -> None:
    """Newly reachable live. Passing it to only one side is what the shared
    execution core was built to prevent."""
    env = {"MIN_PROB": "0.4"}
    assert _flag(_cmd("backtest", **env), "--min-prob") == "0.4"
    assert _flag(_cmd("predict", **env), "--min-prob") == "0.4"


def test_optional_rules_are_omitted_when_unset() -> None:
    """An empty MIN_PROB must not become `--min-prob ''`."""
    for mode in ("backtest", "predict"):
        cmd = _cmd(mode)
        assert "--min-prob" not in cmd
        assert "--min-cross-section" not in cmd


def test_the_rebalance_schedule_matches_by_default() -> None:
    """backtest-sp500 defaults to Friday and predict-sp500 to any day, so a
    daily cron opened cohorts on a schedule the simulation never modelled."""
    for mode in ("backtest", "predict"):
        assert _flag(_cmd(mode), "--rebalance-day") == "Friday"


def test_the_schedule_can_be_opened_up_on_both_sides_together() -> None:
    for mode in ("backtest", "predict"):
        assert "--rebalance-day" not in _cmd(mode, REBALANCE_DAY="any")


def test_the_hold_mode_selects_the_matching_engine() -> None:
    """`--mode rank-hold` and `--hold-mode rank` are the same choice under two
    names; one env var has to drive both."""
    bt, pr = _cmd("backtest", HOLD_MODE="rank"), _cmd("predict", HOLD_MODE="rank")
    assert _flag(bt, "--mode") == "rank-hold"
    assert _flag(pr, "--hold-mode") == "rank"

    bt_fixed = _cmd("backtest", HOLD_MODE="fixed")
    assert "--mode" not in bt_fixed, "the cohort engine is the default"
    assert _flag(_cmd("predict", HOLD_MODE="fixed"), "--hold-mode") == "fixed"


def test_the_live_side_keeps_its_own_kill_switch() -> None:
    """Not every flag is shared — the drawdown halt has no simulation twin."""
    assert _flag(_cmd("predict"), "--max-drawdown") is not None
    assert "--max-drawdown" not in _cmd("backtest")


def test_a_dry_run_does_not_execute_anything() -> None:
    cmd = _cmd("predict")
    assert cmd[0] == "uv", "DRY_RUN must print the command, not run it"


def test_the_training_sanity_backtest_declares_its_config() -> None:
    """train-sp500 has no strategy flags, so its --run-backtest measures
    defaults. It must say so rather than let the numbers be read as the
    strategy's."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src" / "stock_predictor" / "cli.py").read_text()
    block = src[src.index("bt_config = BacktestConfig()") - 800:]
    assert "sanity backtest at defaults" in block
    assert re.search(r"run_pipeline\.sh", block), "point the reader at the real one"
