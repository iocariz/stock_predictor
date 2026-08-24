"""A missing bar is not a delisting, and a 2-session return is not a label.

`_terminal_forward_returns` treated *any* ticker whose last quote preceded the
panel end as delisted, and replaced every missing forward return before it with
the return to that final quote. Two things follow, both measured on the real
panel:

* **False delistings.** 155 tickers were labelled delisted, of which **AVB, EA
  and EQR are current S&P 500 members** — the same three whose recent vendor
  gaps the coverage guard flags. specs.md:587 is explicit that missing terminal
  vendor data is not automatically a delisting.
* **Variable horizons.** The filled labels measured **1 to 2059 sessions**
  (median 33) and were all used as 63-session targets. 146 of them measured a
  single session. The 2059-session cases are not terminal at all — they are
  interior gaps, where a row years earlier is handed a return to the final
  quote.

Evidence resolves both: a verified corporate action gives a date and a value,
and the fill is bounded to rows within one horizon of it. Without evidence the
label stays censored, and the resulting survivorship bias is stated rather than
papered over with a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import build_labeled_panel

DATES = pd.bdate_range("2024-01-01", periods=120)
H = 10


def _prices(dies_at: int | None = 60, gap: tuple[int, int] | None = None):
    """ALIVE trades throughout; DYING stops at *dies_at*; optional interior gap."""
    df = pd.DataFrame(
        {"ALIVE": np.linspace(100, 130, len(DATES)),
         "DYING": np.linspace(100, 40, len(DATES))},
        index=DATES,
    )
    if dies_at is not None:
        df.loc[DATES[dies_at]:, "DYING"] = np.nan
    if gap is not None:
        df.iloc[gap[0]:gap[1], df.columns.get_loc("ALIVE")] = np.nan
    return df


def _evidence(date="2024-03-25", proceeds=41.0) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["DYING"], "date": [pd.Timestamp(date)],
                         "proceeds": [proceeds]})


def _labels(panel, ticker):
    sub = panel[panel.ticker == ticker].sort_values("date")
    return sub[sub.has_label]


# ---------------------------------------------------------------------------
# No evidence means censored
# ---------------------------------------------------------------------------


def test_without_evidence_the_tail_stays_censored() -> None:
    """A gap is not proof. specs.md:587."""
    panel = build_labeled_panel(_prices(), None, H, 0.05)
    dying = _labels(panel, "DYING")
    assert dying["date"].max() < DATES[59], "labelled past its last usable target"


def test_an_interior_gap_is_never_treated_as_a_delisting() -> None:
    """The 2059-session cases: a row years before the final quote handed a
    return measured to it."""
    panel = build_labeled_panel(_prices(dies_at=None, gap=(30, 45)), None, H, 0.05)
    alive = panel[(panel.ticker == "ALIVE") & panel.has_label]
    # rows whose +H target falls inside the gap must have no label
    for i in range(20, 30):
        row = alive[alive.date == DATES[i]]
        assert row.empty, f"row {i} labelled across an interior gap"


def test_a_still_trading_name_is_untouched() -> None:
    panel = build_labeled_panel(_prices(), None, H, 0.05)
    alive = _labels(panel, "ALIVE")
    assert alive["date"].max() == DATES[-H - 1]


# ---------------------------------------------------------------------------
# Evidence produces a bounded, real label
# ---------------------------------------------------------------------------


def test_evidence_labels_the_terminal_window() -> None:
    panel = build_labeled_panel(_prices(), None, H, 0.05,
                                delisting_evidence=_evidence())
    dying = _labels(panel, "DYING")
    assert dying["date"].max() >= DATES[50], "evidence should extend the labels"


def test_the_label_uses_the_verified_proceeds() -> None:
    panel = build_labeled_panel(_prices(), None, H, 0.05,
                                delisting_evidence=_evidence(proceeds=41.0))
    row = panel[(panel.ticker == "DYING") & (panel.date == DATES[55])].iloc[0]
    px = float(row["adj_close"])
    assert row["fwd_ret"] == pytest.approx(41.0 / px - 1.0)


def test_the_filled_window_is_bounded_by_one_horizon() -> None:
    """Otherwise a row far from the event carries a much longer return while
    claiming to be a horizon-H target."""
    proceeds = 41.0
    panel = build_labeled_panel(_prices(), None, H, 0.05,
                                delisting_evidence=_evidence(proceeds=proceeds))
    dying = _labels(panel, "DYING").copy()
    # A *filled* row is one whose return is measured to the proceeds; a
    # normally-labelled row is measured to its own +H price. Only the filled
    # ones are in question here.
    expected = proceeds / dying["adj_close"].astype(float) - 1.0
    filled = dying[np.isclose(dying["fwd_ret"].astype(float), expected)]
    assert len(filled), "evidence produced no filled labels"
    event_pos = list(DATES).index(pd.Timestamp("2024-03-25"))
    earliest = min(list(DATES).index(d) for d in filled["date"])
    assert earliest >= event_pos - H, (
        f"filled a row {event_pos - earliest} sessions before the event, "
        f"beyond the {H}-session horizon"
    )


def test_a_total_loss_is_representable() -> None:
    panel = build_labeled_panel(_prices(), None, H, 0.05,
                                delisting_evidence=_evidence(proceeds=0.0))
    row = panel[(panel.ticker == "DYING") & (panel.date == DATES[55])].iloc[0]
    assert row["fwd_ret"] == pytest.approx(-1.0)


def test_evidence_for_an_unknown_ticker_changes_nothing() -> None:
    ev = pd.DataFrame({"ticker": ["NOPE"], "date": [DATES[50]], "proceeds": [1.0]})
    a = build_labeled_panel(_prices(), None, H, 0.05)
    b = build_labeled_panel(_prices(), None, H, 0.05, delisting_evidence=ev)
    assert int(a.has_label.sum()) == int(b.has_label.sum())


# ---------------------------------------------------------------------------
# The old heuristic is still reachable, and named
# ---------------------------------------------------------------------------


def test_the_unsound_heuristic_requires_asking_for_it() -> None:
    """Kept for reproducing older panels; it infers delisting from absence."""
    panel = build_labeled_panel(_prices(), None, H, 0.05,
                                terminal_fill="assume_delisted")
    assert _labels(panel, "DYING")["date"].max() >= DATES[50]


def test_the_default_does_not_assume(monkeypatch) -> None:
    a = build_labeled_panel(_prices(), None, H, 0.05)
    b = build_labeled_panel(_prices(), None, H, 0.05, terminal_fill="assume_delisted")
    assert int(b.has_label.sum()) > int(a.has_label.sum())


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="terminal_fill"):
        build_labeled_panel(_prices(), None, H, 0.05, terminal_fill="magic")
