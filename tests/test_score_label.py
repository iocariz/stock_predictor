"""What the number next to a ticker actually is.

The daily report printed `P(+5%)=29.000` for a lambdarank model. That is a raw
ranking score, not a probability, and it is unbounded — 29.000 is not a 2900%
chance of anything. The deployed model has been a ranker since 2026-08-21, so
this was the line an operator would read every morning and misinterpret.

The label is derived the same way the score is: a classifier exposes
predict_proba, a ranker only predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.training import model_scores, score_label


class _Ranker:
    def predict(self, X):
        return np.arange(len(X), dtype=float)


class _Classifier(_Ranker):
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


X = pd.DataFrame({"f": [1.0, 2.0]})


def test_a_classifier_reports_a_probability() -> None:
    assert score_label(_Classifier()) == "P(+5%)"


def test_a_ranker_reports_a_score() -> None:
    assert score_label(_Ranker()) == "score"


def test_the_label_matches_how_the_number_was_produced() -> None:
    """The two must agree: whatever branch model_scores takes, the label
    describes it."""
    clf, rk = _Classifier(), _Ranker()
    assert (score_label(clf) == "P(+5%)") == hasattr(clf, "predict_proba")
    assert (score_label(rk) == "score") == (not hasattr(rk, "predict_proba"))
    assert model_scores(clf, X).tolist() == [0.6, 0.6]
    assert model_scores(rk, X).tolist() == [0.0, 1.0]


def test_the_label_is_short_enough_to_align_a_table() -> None:
    for m in (_Classifier(), _Ranker()):
        assert len(score_label(m)) <= 8


def test_the_misleading_literal_is_gone_from_the_report() -> None:
    """Regression: a hardcoded P(+5%) reintroduces the lie for rank models."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src" / "stock_predictor" / "predict.py").read_text()
    assert 'P(+5%)=' not in src, "the label must be derived, not hardcoded"
