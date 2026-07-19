#!/usr/bin/env python3
"""
India Markets Impact Digest (end-of-day recap)
- Reads today's predictions (saved by india_news_digest.py that morning)
- Fetches each stock's actual closing price via Yahoo Finance
- Compares predicted vs actual move, sends a recap notification
- Appends today's accuracy stats to history.json, which the next morning's
  run reads back to calibrate its predictions

Env vars required:
  NTFY_TOPIC       - your ntfy.sh topic name
Optional:
  NTFY_SERVER      - default https://ntfy.sh
"""

import os
import sys
import requests
from datetime import datetime, timezone

from utils import load_json, save_json, fetch_price_snapshot

HISTORY_PATH = "history.json"
PREDICTIONS_DIR = "predictions"
MAX_HISTORY_RUNS = 30


def collect_predicted_stocks(digest):
    """Flattens the gainer/loser predictions across all items into one list,
    tagged with which story and horizon they came from."""
    out = []
    for bucket in ("short_term", "long_term"):
        horizon = "short_term" if bucket == "short_term" else "long_term"
        for item in digest.get(bucket, []):
            for key in ("gainer", "loser"):
                stock = item.get(key)
                if stock:
                    out.append({**stock, "headline": item.get("headline", ""), "horizon": horizon})
    return out


def grade_predictions(stocks):
    graded = []
    for s in stocks:
        snap = fetch_price_snapshot(s["ticker"])
        if snap is None or not s.get("baseline_price"):
            print(f"[warn] skipping grading for '{s['ticker']}' - no price data", file=sys.stderr)
            continue
        baseline = s["baseline_price"]
        actual_close = snap["last"]
        actual_pct = ((actual_close - baseline) / baseline) * 100
        predicted_pct = s["predicted_pct"]
        direction_correct = (predicted_pct >= 0) == (actual_pct >= 0)
        graded.append({
            **s,
            "actual_pct": actual_pct,
            "direction_correct": direction_correct,
            "abs_error": abs(predicted_pct - actual_pct),
        })
    return graded


def format_ntfy_message(graded, no_data_count):
    if not graded:
        return "No predictions could be graded today (price data unavailable)."

    hits = sum(1 for g in graded if g["direction_correct"])
    avg_err = sum(g["abs_error"] for g in graded) / len(graded)

    lines = [f"📋 RECAP — {hits}/{len(graded)} calls right, avg error {avg_err:.2f}pp", ""]

    for horizon, label in (("short_term", "⚡ SHORT-TERM"), ("long_term", "🧭 LONG-TERM")):
        bucket = [g for g in graded if g["horizon"] == horizon]
        if not bucket:
            continue
        lines.append(label)
        # group the gainer+loser predictions back under their shared headline
        by_headline = {}
        for g in bucket:
            by_headline.setdefault(g["headline"], []).append(g)
        for i, (headline, preds) in enumerate(by_headline.items(), 1):
            lines.append(f"{i}. {headline}")
            for g in preds:
                mark = "✅" if g["direction_correct"] else "❌"
                lines.append(
                    f"   {mark} {g['ticker']}: predicted {g['predicted_pct']:+.1f}%, "
                    f"actual {g['actual_pct']:+.1f}%"
                )
        lines.append("")

    if no_data_count:
        lines.append(f"({no_data_count} prediction(s) couldn't be graded - no price data)")
        lines.append("")
    lines.append("Feeds into tomorrow's calibration note automatically.")
    return "\n".join(lines)


def send_to_ntfy(message, topic, server):
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    resp = requests.post(
        f"{server}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Title": f"India Markets Digest — {today} (EOD recap)".encode("utf-8"),
            "Tags": "bar_chart,white_check_mark",
            "Priority": "default",
        },
        timeout=15,
    )
    resp.raise_for_status()


def main():
    ntfy_topic = os.environ.get("NTFY_TOPIC")
    ntfy_server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
    if not ntfy_topic:
        print("ERROR: NTFY_TOPIC not set", file=sys.stderr)
        sys.exit(1)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pred_path = f"{PREDICTIONS_DIR}/{today_str}.json"
    record = load_json(pred_path, None)
    if record is None:
        print(f"No predictions file found at {pred_path} - nothing to recap today.", file=sys.stderr)
        sys.exit(0)  # not an error - e.g. weekend/holiday with no morning run

    stocks = collect_predicted_stocks(record["digest"])
    print(f"Grading {len(stocks)} predictions...")
    graded = grade_predictions(stocks)
    no_data_count = len(stocks) - len(graded)

    message = format_ntfy_message(graded, no_data_count)
    print("\n--- RECAP ---")
    print(message)
    print("-------------\n")

    print(f"Posting to ntfy.sh/{ntfy_topic} ...")
    send_to_ntfy(message, ntfy_topic, ntfy_server)

    if graded:
        hits = sum(1 for g in graded if g["direction_correct"])
        history = load_json(HISTORY_PATH, {"runs": []})
        history["runs"].append({
            "date": today_str,
            "n_predictions": len(graded),
            "direction_accuracy_pct": round(100 * hits / len(graded), 1),
            "avg_abs_error_pts": round(sum(g["abs_error"] for g in graded) / len(graded), 2),
        })
        history["runs"] = history["runs"][-MAX_HISTORY_RUNS:]
        save_json(HISTORY_PATH, history)
        print(f"Logged to {HISTORY_PATH}: {hits}/{len(graded)} correct.")

    print("Done.")


if __name__ == "__main__":
    main()
