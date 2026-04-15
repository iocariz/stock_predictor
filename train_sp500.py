#!/usr/bin/env python3
"""Backward-compatible launcher. Prefer: `uv run train-sp500` (editable install)."""

from stock_predictor.cli import main

if __name__ == "__main__":
    main()
