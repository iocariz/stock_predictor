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

START_DEFAULT = "2010-01-01"
"""The window the training panel asks for, and so the window an all-clear is
about. ``outstanding`` without a start used to answer about whatever was on
disk, which is how it reported nothing outstanding for a panel missing 174
names."""

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


def outstanding(
    names: list[str], cache_dir: Path, start: str | None = None,
    end: str | None = None,
) -> tuple[list[str], int, int]:
    """(still to fetch, already recovered, unusable at the vendor).

    *start* is the window being asked for, defaulting to this script's own
    START. It is not optional in effect: an answer about "the cache" rather
    than about a window is what produced a false all-clear on a survivorship
    audit. This reported ``0 still outstanding`` while a 2010-2026 training
    panel was missing 174 departed members, because the cache had been filled
    for a narrower window and MIN_ROWS alone could not see that. The run that
    followed completed on a degraded panel and reported inflated alpha.

    So the usability question is delegated to the provider -- the same
    ``_cache_is_usable`` the training run will apply, for the same (start,
    end) -- rather than reimplemented here. Two definitions of "recovered" is
    how this script and the provider disagreed in the first place.

    MIN_ROWS is still applied on top, for the one case the provider does not
    cover: a ticker the vendor serves with 1-19 prints is cached ``empty:
    False``, so the provider counts it usable while a panel cannot use it. BK
    and DF sat at 5 rows and WRK at 1, and each pass reported "recovered 3"
    and "3 still outstanding" in the same breath -- so those are counted
    unavailable rather than handed back forever.
    """
    from stock_predictor.providers.hybrid_provider import HybridProvider

    start = start or START_DEFAULT
    end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    probe = HybridProvider(tiingo_api_key="probe", cache_dir=cache_dir)
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
        rows = None
        df = pd.DataFrame()
        if cached.exists():
            try:
                df = pd.read_parquet(cached)
                rows = len(df)
            except Exception:  # noqa: BLE001 - an unreadable cache file is a refetch
                rows = None
        # The provider's own verdict for the window the panel will request.
        if rows is not None and not probe._cache_is_usable(
            t, cached, df, str(start), str(end), man,
        ):
            todo.append(t)
            continue
        if rows is None:
            todo.append(t)
        elif rows >= MIN_ROWS:
            have += 1
        elif str(entry.get("start", "9999")) > str(start):
            todo.append(t)   # short because the request was, not the vendor
        else:
            absent += 1
    return todo, have, absent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default=START_DEFAULT)
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
        todo, have, absent = outstanding(names, args.cache_dir, args.start, end)
        print(f"\npass {i}/{args.passes}: {have} cached, {absent} absent at vendor, "
              f"{len(todo)} to fetch")
        if not todo:
            print("Nothing left to fetch.")
            break
        provider = HybridProvider(tiingo_api_key=key, cache_dir=args.cache_dir)
        got = provider.fetch_missing(todo, args.start, end)
        print(f"  recovered {len(got)} this pass")
        if i < args.passes:
            still, _, _ = outstanding(names, args.cache_dir, args.start, end)
            if not still:
                print("Nothing left to fetch.")
                break
            print(f"  waiting {args.wait}s for the quota window…")
            time.sleep(args.wait)

    todo, have, absent = outstanding(names, args.cache_dir, args.start, end)
    total = len(names)
    print(f"\n{have}/{total} recovered ({have / total:.1%}), "
          f"{absent} unavailable at the vendor, {len(todo)} still outstanding")
    if todo:
        print("Re-run to continue; the cache resumes where it stopped.")


if __name__ == "__main__":
    main()
