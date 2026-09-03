"""Two strategies compared under different execution rules is not a comparison.

``--compare-with`` ran the second strategy without the execution panel::

    result   = backtest_fn(scored,   config, provider=bt_provider, **kwargs)
    result_b = backtest_fn(scored_b, config, provider=bt_provider)

So side A priced its fills from the independent execution panel and side B fell
back to forward-filled quotes out of the point-in-time scored panel. That is the
exact substitution this project already measured as worth several points of
return -- ``+17.28%`` against ``+22.95%`` on rank-hold -- which means a
comparison could report a gap that is entirely an artifact of how each side was
priced. Side B also skipped the panel-coverage check that side A must pass, so
a mismatch that halts the primary run passed silently in the comparison.

An absent panel is a different situation from an asymmetric one: with no panel
at all *both* sides fall back, and the comparison is still like-for-like even
though both numbers are wrong. What could not be allowed is one of each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import BacktestConfig, _check_execution_panel, run_backtest

DATES = pd.bdate_range("2024-01-01", periods=180)
TICKERS = [f"T{i:02d}" for i in range(12)]


def _scored(*, noise: float = 0.0) -> pd.DataFrame:
    """A scored panel whose own ``adj_close`` is deliberately wrong.

    A panel that agrees with the execution panel cannot show the bug, and
    neither can one that is uniformly rescaled: a constant factor cancels
    between entry and exit and leaves the return identical. The two sources have
    to disagree *per cell* -- which is what a differently-adjusted panel
    actually looks like -- before it matters which one prices the fill.
    """
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, (len(DATES), len(TICKERS))),
                                axis=0))
    if noise:
        px = px * (1 + np.random.default_rng(11).normal(0, noise, px.shape))
    rows = []
    for di, d in enumerate(DATES):
        order = np.random.default_rng(100 + di).permutation(len(TICKERS))
        for i, t in enumerate(TICKERS):
            rows.append({"date": d, "ticker": t, "prob": float(order[i]),
                         "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


def _exec(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.pivot_table(index="date", columns="ticker",
                              values="adj_close", aggfunc="first").reindex(DATES)


def _cfg() -> BacktestConfig:
    return BacktestConfig(top_n=4, holding_days=21, max_overlapping_cohorts=2,
                          slippage_bps=5.0, benchmark_ticker=None,
                          rebalance_day="Friday", reject_stale_fills=False)


# ---------------------------------------------------------------------------
# Why it matters
# ---------------------------------------------------------------------------


def test_the_two_price_sources_give_different_answers() -> None:
    """The premise. If these agreed, the asymmetry would be harmless."""
    truth = _scored()
    panel = _exec(truth)
    skewed = _scored(noise=0.05)     # scored panel disagrees per cell

    with_panel = run_backtest(skewed, _cfg(), execution_prices=panel)
    without = run_backtest(skewed, _cfg())
    assert float(with_panel.daily_nav.iloc[-1]) != pytest.approx(
        float(without.daily_nav.iloc[-1]), rel=1e-9)


def test_the_same_panel_gives_the_same_answer() -> None:
    """Control: the difference above comes from the price source, not noise."""
    skewed = _scored(noise=0.05)
    panel = _exec(_scored())
    a = run_backtest(skewed, _cfg(), execution_prices=panel)
    b = run_backtest(skewed, _cfg(), execution_prices=panel)
    assert float(a.daily_nav.iloc[-1]) == pytest.approx(
        float(b.daily_nav.iloc[-1]), rel=0)


# ---------------------------------------------------------------------------
# The coverage check both sides must pass
# ---------------------------------------------------------------------------


def test_a_panel_that_covers_the_scores_passes() -> None:
    truth = _scored()
    _check_execution_panel(truth, _exec(truth), label="comparison",
                           allow_mismatch=False)


def test_an_absent_panel_halts_rather_than_falling_back(capsys) -> None:
    """Side B used to skip this entirely, so a run the primary path refuses
    went ahead silently in the comparison."""
    with pytest.raises(SystemExit):
        _check_execution_panel(_scored(), None, label="comparison",
                               allow_mismatch=False)


def test_a_panel_missing_the_comparisons_tickers_is_caught() -> None:
    """Two runs can be scored on different universes. The panel has to cover
    whichever one it is about to price."""
    truth = _scored()
    panel = _exec(truth).drop(columns=TICKERS[:4])
    with pytest.raises(SystemExit):
        _check_execution_panel(truth, panel, label="comparison",
                               allow_mismatch=False)


def test_the_override_still_works(capsys) -> None:
    _check_execution_panel(_scored(), None, label="comparison",
                           allow_mismatch=True)
    assert "allow-price-mismatch" in capsys.readouterr().out


def test_the_label_names_the_side_that_failed(capsys) -> None:
    """'Refusing to backtest' is unhelpful when two panels are in play."""
    _check_execution_panel(_scored(), None, label="comparison",
                           allow_mismatch=True)
    assert "comparison" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------


def test_the_comparison_leg_is_given_the_execution_panel() -> None:
    """The fix itself, read off the source: both legs must receive kwargs."""
    import inspect

    from stock_predictor import backtest

    src = inspect.getsource(backtest.main)
    # The long-short branch has its own compare_with block, so take the last
    # one -- the shared-engine path -- rather than the first.
    body = src.split("if args.compare_with is not None:")[-1]
    call = next(ln for ln in body.splitlines() if "backtest_fn(scored_b" in ln)
    assert "**kwargs" in call, f"comparison leg still drops kwargs: {call.strip()}"


def test_both_workflows_pass_execution_prices() -> None:
    """The sweep ran on scores alone, so every variant in the grid was priced
    from the scored panel."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    for name in ("backtest-sweep.yml", "train-sp500.yml"):
        # Join continuations first: the invocation spans several lines.
        text = (root / name).read_text().replace("\\\n", " ")
        invocations = [ln for ln in text.splitlines()
                       if "backtest_sweep.py" in ln and "uv run" in ln]
        assert invocations, f"{name} does not run the sweep"
        for line in invocations:
            assert "--execution-prices" in line, f"{name}: {line.strip()}"


def test_the_sweep_downloads_the_artifact_the_trainer_uploads() -> None:
    """They had drifted apart: the trainer uploads ``model-bundle-*`` and both
    download sites asked for ``model-and-scores-*``, so the sweep could not
    have run at all."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    uploaded = set()
    wanted = set()
    for name in ("backtest-sweep.yml", "train-sp500.yml"):
        lines = (root / name).read_text().splitlines()
        for i, line in enumerate(lines):
            if "name: model-" not in line:
                continue
            stem = line.split("name:")[1].strip().split("-${{")[0]
            context = "\n".join(lines[max(0, i - 6):i])
            (uploaded if "upload-artifact" in context else wanted).add(stem)
    assert uploaded, "no upload found"
    assert wanted <= uploaded, f"downloads {wanted} do not match uploads {uploaded}"
