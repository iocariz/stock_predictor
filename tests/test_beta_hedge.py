"""Dollar-neutral is not market-neutral, and the reports should say so.

Equalising notional equalises dollars, not exposure. Measured on the real
panel, the dollar-neutral book carries **beta +0.292 (t +4.76)** — because this
model ranks volatility positively, so the long leg holds beta-1.27 names and
the short leg beta-0.66 ones. Roughly a quarter of the "market-neutral" return
was unhedged market exposure.

Two things follow: every run should report its beta, and there should be a way
to hedge it that does not distort the stock selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.long_short import LongShortConfig, run_long_short_backtest
from stock_predictor.stats import market_exposure

DATES = pd.bdate_range("2024-01-01", periods=400)
N = 60
HEDGE = "SPY"


def _market() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0004, 0.01, len(DATES)), index=DATES)


def _panel(mkt: pd.Series) -> pd.DataFrame:
    """Ticker i has beta rising with rank, and the score ranks it long-first,
    so the long book is systematically higher beta — the real book's shape."""
    rng = np.random.default_rng(11)
    betas = np.linspace(1.4, 0.6, N)
    idio = rng.normal(0, 0.008, (len(DATES), N))
    rets = mkt.to_numpy()[:, None] * betas[None, :] + idio
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            rows.append({"date": d, "ticker": f"T{i:02d}",
                         "prob": float(N - i), "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


class _BenchProvider:
    """Serves the same market series the panel was built from."""

    def __init__(self, mkt: pd.Series):
        self._nav = 100.0 * (1 + mkt).cumprod()

    def download_benchmark(self, ticker, start, end):
        return self._nav.rename(ticker)


def _cfg(**kw) -> LongShortConfig:
    base = dict(rebalance_every=21, slippage_bps=0.0, risk_free_rate=0.0,
                short_borrow_annual=0.0, benchmark_ticker=HEDGE)
    base.update(kw)
    return LongShortConfig(**base)


def _run(cfg, mkt):
    return run_long_short_backtest(_panel(mkt), cfg, provider=_BenchProvider(mkt))


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def test_market_exposure_recovers_a_known_beta() -> None:
    rng = np.random.default_rng(3)
    mkt = rng.normal(0, 0.01, 800)
    port = 0.4 * mkt + rng.normal(0, 0.002, 800)
    out = market_exposure(pd.Series(port), pd.Series(mkt), overlap=1)
    assert out["beta"] == pytest.approx(0.4, abs=0.05)
    assert abs(out["beta_t"]) > 5


def test_a_flat_portfolio_has_no_beta() -> None:
    rng = np.random.default_rng(4)
    mkt = rng.normal(0, 0.01, 500)
    out = market_exposure(pd.Series(rng.normal(0, 0.002, 500)),
                          pd.Series(mkt), overlap=1)
    assert abs(out["beta"]) < 0.1


def test_alpha_is_annualised_and_signed() -> None:
    rng = np.random.default_rng(5)
    mkt = rng.normal(0, 0.01, 800)
    out = market_exposure(pd.Series(0.0004 + 0.0 * mkt), pd.Series(mkt), overlap=1)
    assert out["alpha_ann"] == pytest.approx(0.0004 * 252, rel=0.1)


def test_too_little_overlap_yields_nan_not_a_crash() -> None:
    out = market_exposure(pd.Series([0.01]), pd.Series([0.01]), overlap=1)
    assert np.isnan(out["beta"])


# ---------------------------------------------------------------------------
# The book reports its own exposure
# ---------------------------------------------------------------------------


def test_an_unhedged_dollar_neutral_book_reports_positive_beta() -> None:
    """The finding this module exists for: equal notional, unequal exposure."""
    mkt = _market()
    res = _run(_cfg(), mkt)
    assert res.metrics["beta"] > 0.15
    assert res.metrics["beta_t"] > 2


def test_beta_is_absent_rather_than_faked_without_a_benchmark() -> None:
    res = _run(_cfg(benchmark_ticker=None), _market())
    assert "beta" not in res.metrics


# ---------------------------------------------------------------------------
# The hedge
# ---------------------------------------------------------------------------


def test_hedging_reduces_the_realised_beta() -> None:
    mkt = _market()
    unhedged = _run(_cfg(), mkt)
    hedged = _run(_cfg(hedge_beta=unhedged.metrics["beta"]), mkt)
    assert abs(hedged.metrics["beta"]) < abs(unhedged.metrics["beta"]) / 2


def test_no_hedge_is_the_default_and_changes_nothing() -> None:
    """Regression guard: the feature must be inert until asked for."""
    mkt = _market()
    a = _run(_cfg(), mkt)
    b = _run(_cfg(hedge_beta=None), mkt)
    assert a.daily_nav.iloc[-1] == pytest.approx(b.daily_nav.iloc[-1])
    assert a.metrics["hedge_beta"] == 0.0


def test_a_zero_hedge_is_also_inert() -> None:
    mkt = _market()
    assert _run(_cfg(hedge_beta=0.0), mkt).daily_nav.iloc[-1] == pytest.approx(
        _run(_cfg(), mkt).daily_nav.iloc[-1]
    )


def test_the_hedge_is_a_short_position_in_the_benchmark() -> None:
    res = _run(_cfg(hedge_beta=0.3), _market())
    assert res.metrics["hedge_beta"] == pytest.approx(0.3)
    assert res.costs["hedge_notional"] > 0


def test_the_hedge_does_not_join_the_stock_selection() -> None:
    """It is an overlay. It must not consume a decile slot or be ranked."""
    mkt = _market()
    plain = _run(_cfg(), mkt)
    hedged = _run(_cfg(hedge_beta=0.3), mkt)
    assert hedged.n_rebalances == plain.n_rebalances


def test_a_negative_hedge_is_rejected() -> None:
    """Shorting a negative beta is a long index position, which is not what
    this overlay is for; ask for it explicitly some other way."""
    with pytest.raises(ValueError, match="hedge_beta"):
        LongShortConfig(hedge_beta=-0.3)


def test_hedging_without_a_benchmark_is_rejected() -> None:
    """Silently ignoring the request would be worse than refusing it."""
    with pytest.raises(ValueError, match="benchmark"):
        LongShortConfig(hedge_beta=0.3, benchmark_ticker=None)
