#!/usr/bin/env python3
"""Backward-compatible launcher. Prefer: `uv run backtest-sp500`."""

from stock_predictor.backtest import main

if __name__ == "__main__":
    main()
