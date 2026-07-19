"""Shared helpers for india_news_digest.py and eod_recap.py."""
import json
import os
import sys
import yfinance as yf


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fetch_price_snapshot(ticker):
    """Returns {'last': float, 'prev_close': float} for an NSE ticker,
    or None if the lookup fails. Never raises."""
    try:
        info = yf.Ticker(f"{ticker}.NS").fast_info
        last = info.get("last_price")
        prev_close = info.get("previous_close")
        if last is not None and prev_close:
            return {"last": float(last), "prev_close": float(prev_close)}
    except Exception as e:
        print(f"[warn] price snapshot failed for '{ticker}': {e}", file=sys.stderr)
    return None
