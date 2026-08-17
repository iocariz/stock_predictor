"""Training CLI argument contract."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stock_predictor.cli import parse_args, resolve_objective


def _args(*argv: str):
    with patch("sys.argv", ["train-sp500", *argv]):
        return parse_args()


def test_rank_is_the_default_objective() -> None:
    """The binary +5%-in-10-days label is satisfied mechanically by
    volatility (score-vs-vol IC +0.75), so the ranker is the default."""
    assert resolve_objective(_args()) == "rank"


def test_binary_objective_is_still_reachable() -> None:
    assert resolve_objective(_args("--objective", "binary")) == "binary"


def test_explicit_rank_objective_selects_rank() -> None:
    assert resolve_objective(_args("--objective", "rank")) == "rank"


def test_legacy_rank_objective_flag_still_accepted() -> None:
    """--rank-objective predates the default flip; it must not error."""
    assert resolve_objective(_args("--rank-objective")) == "rank"


def test_legacy_flag_does_not_override_an_explicit_binary_choice() -> None:
    args = _args("--objective", "binary", "--rank-objective")
    with pytest.raises(SystemExit, match="conflict"):
        resolve_objective(args)


def test_label_target_defaults_to_raw() -> None:
    assert _args().label_target == "raw"


def test_unknown_objective_rejected() -> None:
    with pytest.raises(SystemExit):
        _args("--objective", "lambdamart")
