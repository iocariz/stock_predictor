"""Fill in the companies that left the index, a quota window at a time.

Yahoo stops serving most companies once they are acquired, renamed or taken
private. Tiingo keeps them, but its free tier allows only ~50-70 new tickers per
window, and a training panel needs a few hundred. A single run therefore cannot
recover them all, and a run that stops early looks exactly like a run that
succeeded -- which is how a rebuild produced a panel with 148 departed names as
empty columns and reported success.

So this resumes. Each pass fetches what the quota allows, caches it, and stops
cleanly; the next pass picks up where it left off. Run it until it reports
nothing left to fetch, then rebuild the baseline.

    uv run python scripts/recover_delisted.py --passes 6 --wait 3600

Names Tiingo genuinely does not serve are recorded as empty and not retried
inside the TTL, so repeated passes converge instead of looping forever on the
same failures.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from stock_predictor.data_provider import _load_dotenv
from stock_predictor.pit import load_sp500_stints
from stock_predictor.providers.hybrid_provider import DEFAULT_CACHE, HybridProvider

MIN_ROWS = 20
"""A handful of prints is not a recovered ticker."""


def departed_names(start: str, end: str) -> list[str]:
    """Index members whose stint ended inside the window."""
    st = load_sp500_stints()
    st["ticker"] = st["ticker"].astype(str)
    st["end_date"] = pd.to_datetime(st["end_date"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    gone = st[(st["end_date"].notna()) & (st["end_date"] >= lo) & (st["end_date"] <= hi)]
    return sorted(set(gone["ticker"]))


def _manifest(cache_dir: Path) -> dict:
    path = cache_dir / "_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def outstanding(names: list[str], cache_dir: Path) -> tuple[list[str], int, int]:
    """(still to fetch, already recovered, known-absent at the vendor)."""
    man = _manifest(cache_dir)
    todo, have, absent = [], 0, 0
    for t in names:
        entry = man.get(t)
        if entry is None:
            todo.append(t)
            continue
        if entry.get("empty"):
            absent += 1
            continue
        cached = cache_dir / f"{t}.parquet"
        if cached.exists():
            try:
                if len(pd.read_parquet(cached)) >= MIN_ROWS:
                    have += 1
                    continue
            except Exception:  # noqa: BLE001 - an unreadable cache file is a refetch
                pass
        todo.append(t)
    return todo, have, absent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="defaults to today")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--wait", type=int, default=3600,
                    help="seconds between passes; Tiingo's window is hourly")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    _load_dotenv()
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        raise SystemExit("TIINGO_API_KEY is not set; nothing to recover with.")

    end = args.end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    names = departed_names(args.start, end)
    print(f"{len(names)} companies left the index between {args.start} and {end}")

    for i in range(1, args.passes + 1):
        todo, have, absent = outstanding(names, args.cache_dir)
        print(f"\npass {i}/{args.passes}: {have} cached, {absent} absent at vendor, "
              f"{len(todo)} to fetch")
        if not todo:
            print("Nothing left to fetch.")
            break
        provider = HybridProvider(tiingo_api_key=key, cache_dir=args.cache_dir)
        got = provider.fetch_missing(todo, args.start, end)
        print(f"  recovered {len(got)} this pass")
        if i < args.passes:
            still, _, _ = outstanding(names, args.cache_dir)
            if not still:
                print("Nothing left to fetch.")
                break
            print(f"  waiting {args.wait}s for the quota window…")
            time.sleep(args.wait)

    todo, have, absent = outstanding(names, args.cache_dir)
    total = len(names)
    print(f"\n{have}/{total} recovered ({have / total:.1%}), "
          f"{absent} unavailable at the vendor, {len(todo)} still outstanding")
    if todo:
        print("Re-run to continue; the cache resumes where it stopped.")


if __name__ == "__main__":
    main()
