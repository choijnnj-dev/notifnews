#!/usr/bin/env python3
"""
India Markets Impact Digest (morning run)
- Pulls headlines from RSS feeds (macro, legislative, geopolitical)
- Gemini picks 5 short-term + 5 long-term stories, and for each: 2 stocks
  predicted to rise and 2 predicted to fall, with a predicted % move
- Predictions are saved to predictions/<date>.json so the end-of-day script
  can compare them against what actually happened
- Recent prediction accuracy (from history.json) is fed back into the prompt
  so Gemini has visibility into its own recent track record
- Posts the digest to ntfy.sh

Env vars required:
  GEMINI_API_KEY   - Google AI Studio key (free tier)
  NTFY_TOPIC       - your ntfy.sh topic name
Optional:
  NTFY_SERVER      - default https://ntfy.sh
  GEMINI_MODEL     - default gemini-flash-latest
"""

import os
import sys
import json
import time
import requests
import feedparser
from datetime import datetime, timezone

from utils import load_json, save_json, fetch_price_snapshot

FEEDS = [
    ("Economic Times - Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("Livemint - Economy", "https://www.livemint.com/rss/economy"),
    ("Moneycontrol - Economy", "https://www.moneycontrol.com/rss/economy.xml"),
    ("PIB India - Releases", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
    ("RBI - Press Releases", "https://www.rbi.org.in/pressreleases_rss.xml"),
    ("The Hindu - National", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("The Hindu - International", "https://www.thehindu.com/news/international/feeder/default.rss"),
    ("Economic Times - World", "https://economictimes.indiatimes.com/news/international/world-news/rssfeeds/1898055973.cms"),
    ("Indian Express - India", "https://indianexpress.com/section/india/feed/"),
    ("Indian Express - World", "https://indianexpress.com/section/world/feed/"),
]

MAX_ITEMS_PER_FEED = 12
FEED_TIMEOUT_SECS = 15
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
CATEGORY_TAG = {"macro": "M", "legislative": "L", "geopolitical": "G"}
HISTORY_PATH = "history.json"
PREDICTIONS_DIR = "predictions"


def fetch_feed(name, url):
    items = []
    try:
        resp = requests.get(
            url,
            timeout=FEED_TIMEOUT_SECS,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IndiaNewsDigest/1.0)"},
        )
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
            title = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "").strip()
            link = getattr(entry, "link", "").strip()
            if title:
                items.append({"source": name, "title": title, "summary": summary[:300], "link": link})
    except Exception as e:
        print(f"[warn] failed to fetch '{name}': {e}", file=sys.stderr)
    return items


def collect_headlines(feed_list):
    all_items = []
    for name, url in feed_list:
        all_items.extend(fetch_feed(name, url))
    return all_items


def summarize_track_record(history, n=7):
    """Turns recent EOD accuracy logs into a short calibration note for the
    prompt. This is NOT real learning - it just gives the model visibility
    into how far off its recent predictions have been."""
    runs = history.get("runs", [])[-n:]
    if not runs:
        return ("No prediction accuracy history yet (this is an early run). "
                "Be conservative with predicted percentage magnitudes until "
                "there's a track record to calibrate against.")
    avg_acc = sum(r["direction_accuracy_pct"] for r in runs) / len(runs)
    avg_err = sum(r["avg_abs_error_pts"] for r in runs) / len(runs)
    note = (f"Over the last {len(runs)} trading day(s), directional accuracy was "
            f"{avg_acc:.0f}% and average magnitude error was {avg_err:.2f} percentage "
            f"points. ")
    if avg_err > 1.5:
        note += "Recent magnitude predictions have run too large - lean toward smaller, more realistic % moves (most single-day moves from news are under 2%). "
    if avg_acc < 55:
        note += "Recent directional calls have been close to a coin flip - only pick a clear gainer/loser if the story genuinely implies a directional bias, otherwise favor smaller magnitudes. "
    return note


def build_gemini_prompt(items, track_record_note):
    lines = [f"{i+1}. [{it['source']}] {it['title']} — {it['summary']}" for i, it in enumerate(items)]
    candidates = "\n".join(lines) if lines else "(no items fetched)"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = f"""You are a financial news editor preparing a same-day briefing for an Indian
equities trader. Today's date: {today}.

CALIBRATION NOTE FROM YOUR RECENT TRACK RECORD:
{track_record_note}

Below are raw candidate headlines from RSS feeds covering macroeconomics,
legislative/policy, and geopolitics. Some are duplicates, irrelevant (sports,
entertainment, local crime), or low-signal — ignore those.

=== CANDIDATE HEADLINES ===
{candidates}

Task:
Select stories that matter for Indian stock markets — mix macroeconomic,
legislative/policy, and geopolitical stories freely across both lists below,
whichever fits the time horizon better.

1. SHORT-TERM (5 items): stories likely to move markets in the next 1-5
   trading days.
2. LONG-TERM (5 items): stories that shape market direction over months/years.
3. For each: a SHORT headline (max ~10 words, your own phrasing) and ONE line
   on market impact (max ~12 words). Terse, no filler, no repeating the
   headline in the impact line.
4. Tag category as one of: "macro", "legislative", "geopolitical".
5. Every single item, in BOTH lists, MUST name exactly 1 real, currently
   NSE-listed stock most likely to RISE and 1 most likely to FALL because of
   this story (exact NSE ticker symbols like "RELIANCE", "TCS", "HDFCBANK" -
   never invent one, never leave this blank - always make your best-judgment
   call). Give each a predicted TODAY'S percentage move as a positive number
   (magnitude only - direction is implied by which field it's in). Keep
   magnitudes realistic per the calibration note above.
6. For LONG-TERM items only, ALSO give each of the 2 stocks an overall
   multi-period forecast: "timeframe" (a plain duration like "3 months" or
   "6 weeks", OR if the story's effect hasn't started yet, the point it
   kicks in, e.g. "starts Sep 2026"), "overall_direction" ("up" or "down" -
   this can differ from today's short-term move if you expect a reversal),
   and "overall_pct" (positive magnitude over that timeframe).
7. If fewer than 5 genuinely relevant stories exist for a horizon, return fewer.

Respond ONLY with JSON matching this exact shape, nothing else:
{{
  "short_term": [{{"headline": "...", "impact": "...", "category": "macro|legislative|geopolitical", "source": "...",
      "gainer": {{"ticker": "...", "predicted_pct_today": 1.2}},
      "loser": {{"ticker": "...", "predicted_pct_today": 0.9}}}}],
  "long_term": [{{"headline": "...", "impact": "...", "category": "macro|legislative|geopolitical", "source": "...",
      "gainer": {{"ticker": "...", "predicted_pct_today": 1.2, "timeframe": "3 months", "overall_direction": "up", "overall_pct": 8.5}},
      "loser": {{"ticker": "...", "predicted_pct_today": 0.9, "timeframe": "3 months", "overall_direction": "down", "overall_pct": 5.0}}}}]
}}
"""
    return prompt


def call_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "response_mime_type": "application/json"},
    }
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def attach_baselines(digest):
    """For the predicted gainer/loser on every story, fetch and attach
    yesterday's close as the baseline the EOD script will measure the actual
    move against. If the price lookup fails, that field is dropped (can't
    grade a prediction with no baseline)."""
    cache = {}
    for bucket in ("short_term", "long_term"):
        for item in digest.get(bucket, []):
            for key, sign in (("gainer", 1), ("loser", -1)):
                stock = item.get(key)
                if not stock:
                    continue
                ticker = stock.get("ticker", "").strip().upper()
                if not ticker:
                    item[key] = None
                    continue
                if ticker not in cache:
                    cache[ticker] = fetch_price_snapshot(ticker)
                snap = cache[ticker]
                if snap is None:
                    print(f"[warn] dropping '{ticker}' - no price data", file=sys.stderr)
                    item[key] = None
                    continue
                stock["ticker"] = ticker
                stock["predicted_pct_today"] = sign * abs(float(stock.get("predicted_pct_today", 0)))
                stock["baseline_price"] = snap["prev_close"]
    return digest


def format_ntfy_message(digest):
    def short_term_line(item):
        g, l = item.get("gainer"), item.get("loser")
        if not g or not l:
            return None
        return (f"{g['ticker']} 📈{g['predicted_pct_today']:+.1f}%   "
                f"{l['ticker']} 📉{l['predicted_pct_today']:+.1f}%")

    def long_term_lines(item):
        g, l = item.get("gainer"), item.get("loser")
        if not g or not l:
            return []
        out = []
        for s, arrow_today in ((g, "📈"), (l, "📉")):
            overall_arrow = "↑" if s.get("overall_direction") == "up" else "↓"
            out.append(
                f"   {s['ticker']} {arrow_today}{s['predicted_pct_today']:+.1f}% today"
                f"  ·  {s.get('timeframe', '?')}: {overall_arrow}{s.get('overall_pct', 0):.1f}%"
            )
        return out

    def block_short(title, items):
        lines = [title]
        for i, item in enumerate(items[:5], 1):
            tag = CATEGORY_TAG.get(item.get("category", "").lower(), "?")
            lines.append(f"{i}. [{tag}] {item['headline']} — {item['impact']}")
            line = short_term_line(item)
            if line:
                lines.append(f"   {line}")
        return lines

    def block_long(title, items):
        lines = [title]
        for i, item in enumerate(items[:5], 1):
            tag = CATEGORY_TAG.get(item.get("category", "").lower(), "?")
            lines.append(f"{i}. [{tag}] {item['headline']} — {item['impact']}")
            lines.extend(long_term_lines(item))
        return lines

    lines = block_short("⚡ SHORT-TERM (1-5 days)", digest.get("short_term", []))
    lines.append("")
    lines += block_long("🧭 LONG-TERM (months+)", digest.get("long_term", []))
    lines.append("")
    lines.append("M=macro · L=legislative · G=geopolitical")
    lines.append("Predictions are AI-inferred estimates, not verified forecasts. EOD recap follows at market close.")
    return "\n".join(lines)


def send_to_ntfy(message, topic, server, title_suffix):
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    resp = requests.post(
        f"{server}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Title": f"India Markets Digest — {today} {title_suffix}".encode("utf-8"),
            "Tags": "india,chart_with_upwards_trend,earth_asia",
            "Priority": "default",
        },
        timeout=15,
    )
    resp.raise_for_status()


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    ntfy_topic = os.environ.get("NTFY_TOPIC")
    ntfy_server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not ntfy_topic:
        print("ERROR: NTFY_TOPIC not set", file=sys.stderr)
        sys.exit(1)

    print("Fetching feeds...")
    items = collect_headlines(FEEDS)
    print(f"  -> {len(items)} items")
    if not items:
        print("ERROR: all feeds failed, nothing to summarize", file=sys.stderr)
        sys.exit(1)

    history = load_json(HISTORY_PATH, {"runs": []})
    track_record_note = summarize_track_record(history)
    prompt = build_gemini_prompt(items, track_record_note)

    print(f"Calling Gemini ({GEMINI_MODEL})...")
    max_attempts = 5
    digest = None
    for attempt in range(max_attempts):
        try:
            digest = call_gemini(prompt, api_key)
            break
        except Exception as e:
            wait = min(60, (2 ** attempt) * 5)
            print(f"[warn] Gemini call failed (attempt {attempt+1}/{max_attempts}): {e}", file=sys.stderr)
            if attempt < max_attempts - 1:
                print(f"[info] retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
    if digest is None:
        print(f"ERROR: Gemini call failed after {max_attempts} attempts", file=sys.stderr)
        sys.exit(1)

    print("Fetching baseline prices for predicted stocks...")
    digest = attach_baselines(digest)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_json(f"{PREDICTIONS_DIR}/{today_str}.json", {"date": today_str, "digest": digest})

    message = format_ntfy_message(digest)
    print("\n--- DIGEST ---")
    print(message)
    print("--------------\n")

    print(f"Posting to ntfy.sh/{ntfy_topic} ...")
    send_to_ntfy(message, ntfy_topic, ntfy_server, "(morning)")
    print("Done.")


if __name__ == "__main__":
    main()
