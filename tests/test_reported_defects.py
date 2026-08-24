"""Six defects reported together, each small and each load-bearing somewhere.

They have nothing in common except that all six were found by reading rather
than by running, and none of them would have shown up as a failure — a model
one tree too large, a cache that answers the wrong question, a shell builtin
that does not exist on the machine the script targets.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. best_iteration_ is already the effective tree count
# ---------------------------------------------------------------------------


def test_best_iteration_needs_no_off_by_one_correction() -> None:
    """The premise behind ``+ 1`` — pin it, so a version bump is visible.

    On lightgbm 4.6 ``best_iteration_``, ``n_estimators_`` and
    ``booster_.num_trees()`` all agree. Adding one asked the production refit
    for a tree the early-stopping split never justified.
    """
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + rng.normal(0, 0.5, 2000) > 0).astype(int)
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1)
    m.fit(X[:1500], y[:1500], eval_set=[(X[1500:], y[1500:])],
          callbacks=[lgb.early_stopping(20, verbose=False)])

    assert m.best_iteration_ == m.n_estimators_ == m.booster_.num_trees()


def test_the_refit_uses_exactly_the_stopped_tree_count() -> None:
    from stock_predictor.training import train_final_model

    rng = np.random.default_rng(1)
    n = 1200
    df = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2022-01-03", periods=n // 20), 20),
        "f0": rng.normal(size=n),
        "f1": rng.normal(size=n),
    })
    df["target_5pct"] = (df["f0"] + rng.normal(0, 0.5, n) > 0).astype(int)

    model, n_trees = train_final_model(
        df, ["f0", "f1"], {"n_estimators": 200, "learning_rate": 0.05},
        seed=0, purge_days=1,
    )
    # The refit was asked for n_trees; it must have built exactly that many.
    assert model.booster_.num_trees() == n_trees


# ---------------------------------------------------------------------------
# 2. Promotion validates the artifact and swaps it atomically
# ---------------------------------------------------------------------------


def _write_model(path: Path, obj: object, meta: dict) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump({"model": obj, "meta": meta}, f)
    path.with_suffix(".meta.json").write_text(json.dumps(meta))


def _good_meta() -> dict:
    return {"horizon": 63, "fitted_through": "2026-06-30", "feature_cols": ["a"]}


class _NotAModel:
    """Loads fine, pickles fine, cannot score anything."""


def test_an_object_that_cannot_score_is_refused(tmp_path: Path) -> None:
    from stock_predictor.deploy import PromotionError, promote_model

    cand = tmp_path / "model_candidate.pkl"
    _write_model(cand, _NotAModel(), _good_meta())
    with pytest.raises(PromotionError, match="score|predict"):
        promote_model(cand, tmp_path / "model.pkl", archive_dir=tmp_path / "arch")


def test_a_real_model_is_promoted(tmp_path: Path) -> None:
    import lightgbm as lgb

    from stock_predictor.deploy import promote_model

    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(200, 2)), rng.integers(0, 2, 200)
    m = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)

    cand = tmp_path / "model_candidate.pkl"
    _write_model(cand, m, _good_meta())
    out = promote_model(cand, tmp_path / "model.pkl", archive_dir=tmp_path / "arch")
    assert out.deployed.exists()
    assert out.deployed.with_suffix(".meta.json").exists()


def test_a_failed_promotion_leaves_the_live_pair_untouched(tmp_path: Path) -> None:
    """The point of atomicity: never a live model with someone else's metadata."""
    import lightgbm as lgb

    from stock_predictor.deploy import PromotionError, promote_model

    rng = np.random.default_rng(3)
    good = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(
        rng.normal(size=(200, 2)), rng.integers(0, 2, 200))

    live = tmp_path / "model.pkl"
    _write_model(live, good, {**_good_meta(), "tag": "live"})
    before = live.read_bytes()
    before_meta = live.with_suffix(".meta.json").read_text()

    cand = tmp_path / "model_candidate.pkl"
    _write_model(cand, _NotAModel(), {**_good_meta(), "tag": "candidate"})
    with pytest.raises(PromotionError):
        promote_model(cand, live, archive_dir=tmp_path / "arch")

    assert live.read_bytes() == before
    assert live.with_suffix(".meta.json").read_text() == before_meta


def test_metadata_and_model_are_never_left_disagreeing(tmp_path: Path) -> None:
    """A candidate with no metadata file must not inherit the old one."""
    import lightgbm as lgb

    from stock_predictor.deploy import PromotionError, promote_model

    rng = np.random.default_rng(4)
    m = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(
        rng.normal(size=(200, 2)), rng.integers(0, 2, 200))

    live = tmp_path / "model.pkl"
    _write_model(live, m, {**_good_meta(), "tag": "live"})

    cand = tmp_path / "model_candidate.pkl"
    _write_model(cand, m, _good_meta())
    cand.with_suffix(".meta.json").unlink()      # candidate metadata missing

    with pytest.raises(PromotionError, match="metadata"):
        promote_model(cand, live, archive_dir=tmp_path / "arch")
    assert json.loads(live.with_suffix(".meta.json").read_text())["tag"] == "live"


# ---------------------------------------------------------------------------
# 3. The vendor cache answers the question it was asked
# ---------------------------------------------------------------------------


def _provider(tmp_path: Path, responses: dict[str, pd.DataFrame]):
    from stock_predictor.providers.hybrid_provider import HybridProvider

    p = HybridProvider(tiingo_api_key="x", cache_dir=tmp_path)
    calls: list[tuple[str, str, str | None]] = []

    def fake(ticker, start, end):
        calls.append((ticker, start, end))
        return responses.get(ticker, pd.DataFrame())

    p._fetch_one = fake                                   # type: ignore[method-assign]
    return p, calls


def _bars(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"date": idx, "close": 1.0, "volume": 100.0})


def test_a_wider_range_is_not_served_from_a_narrower_cache(tmp_path: Path) -> None:
    """The reported defect: the key was the ticker alone, so the second call
    silently returned the first call's shorter history."""
    p, calls = _provider(tmp_path, {"AAA": _bars("2024-01-01", "2024-06-28")})
    p.fetch_missing(["AAA"], "2024-01-01", "2024-06-28")
    assert len(calls) == 1

    p2, calls2 = _provider(tmp_path, {"AAA": _bars("2020-01-01", "2024-06-28")})
    out = p2.fetch_missing(["AAA"], "2020-01-01", "2024-06-28")
    assert len(calls2) == 1, "a range the cache cannot cover must be re-fetched"
    assert out["AAA"]["date"].min() <= pd.Timestamp("2020-01-02")


def test_a_covered_range_is_served_from_cache(tmp_path: Path) -> None:
    """The cache still has to work, or this is just a slow downloader."""
    p, calls = _provider(tmp_path, {"AAA": _bars("2020-01-01", "2024-06-28")})
    p.fetch_missing(["AAA"], "2020-01-01", "2024-06-28")
    p2, calls2 = _provider(tmp_path, {})
    out = p2.fetch_missing(["AAA"], "2021-01-01", "2024-01-01")
    assert calls2 == [], "a fully covered sub-range must not re-fetch"
    assert not out["AAA"].empty


def test_an_empty_result_is_not_cached_forever(tmp_path: Path) -> None:
    """Caching misses avoids re-asking a rate-limited vendor. Caching them
    *permanently* turns a bad afternoon into a permanently absent ticker."""
    p, _ = _provider(tmp_path, {})
    p.fetch_missing(["ZZZ"], "2024-01-01", "2024-06-28")

    p2, calls2 = _provider(tmp_path, {})
    p2.fetch_missing(["ZZZ"], "2024-01-01", "2024-06-28")
    assert calls2 == [], "inside the TTL, a known miss is not re-asked"

    p3, calls3 = _provider(tmp_path, {"ZZZ": _bars("2024-01-01", "2024-06-28")})
    p3.empty_ttl_days = 0                                 # pretend the TTL lapsed
    out = p3.fetch_missing(["ZZZ"], "2024-01-01", "2024-06-28")
    assert len(calls3) == 1, "once the TTL lapses the miss is retried"
    assert not out["ZZZ"].empty


# ---------------------------------------------------------------------------
# 4. The reporting module imports on its own
# ---------------------------------------------------------------------------


def test_backtest_reporting_imports_in_a_fresh_interpreter() -> None:
    """Import order should not be part of the public API."""
    r = subprocess.run(
        [sys.executable, "-c",
         "import stock_predictor.backtest_reporting as m; print(m.print_report.__name__)"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "print_report" in r.stdout


def test_the_backtest_re_export_still_works() -> None:
    r = subprocess.run(
        [sys.executable, "-c",
         "from stock_predictor.backtest import print_report, plot_backtest; "
         "print(print_report.__name__, plot_backtest.__name__)"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "print_report plot_backtest" in r.stdout


# ---------------------------------------------------------------------------
# 5. The pipeline script runs on the shell macOS actually ships
# ---------------------------------------------------------------------------


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.sh"


def test_run_pipeline_uses_no_bash_4_only_builtins() -> None:
    """macOS ships bash 3.2. ``mapfile``/``readarray`` are bash 4+, and the
    failure is a silent empty array, not an error."""
    code = [ln.split("#", 1)[0] for ln in SCRIPT.read_text().splitlines()]
    for builtin in ("mapfile", "readarray"):
        hits = [ln for ln in code if re.search(rf"(^|[;|&(]\s*){builtin}\b", ln)]
        assert not hits, f"{builtin} is bash 4+; macOS ships 3.2: {hits}"


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="no system bash")
def test_run_pipeline_parses_under_system_bash() -> None:
    r = subprocess.run(["/bin/bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="no system bash")
@pytest.mark.parametrize("cmd", ["train-full", "evaluate", "refit", "backtest", "predict"])
def test_run_pipeline_executes_under_system_bash(cmd: str) -> None:
    """Parsing is not enough. Two of this script's bash-4 dependencies are
    runtime-only: ``mapfile`` is an unknown *command*, and under ``set -u``
    bash 3.2 treats an empty ``${arr[@]}`` as unbound. Both parse fine.
    """
    r = subprocess.run(
        ["/bin/bash", str(SCRIPT), cmd],
        capture_output=True, text=True,
        env={**os.environ, "DRY_RUN": "1"},
        cwd=SCRIPT.resolve().parents[1],
    )
    assert r.returncode == 0, r.stderr
    # DRY_RUN echoes the command it would have run. backtest and predict are
    # the two that expand strategy_flags -- the array mapfile used to leave
    # silently empty -- so that is where the flags must actually appear.
    if cmd in ("backtest", "predict"):
        assert "--top-n" in r.stdout, r.stdout
        assert "--holding-days" in r.stdout, r.stdout
