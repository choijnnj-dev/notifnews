#!/usr/bin/env python3
"""
India Markets Impact Digest
- Pulls headlines from a set of RSS feeds (macro, legislative, geopolitical)
- Sends the raw headlines to Gemini (Flash, free tier) to pick, summarize,
  and infer top affected NSE stocks per story
- Looks up real live price moves for those stocks via Yahoo Finance (free,
  no API key) to attach actual confirmed price action
- Posts a compact 5 short-term + 5 long-term digest to ntfy.sh

Env vars required:
  GEMINI_API_KEY   - Google AI Studio key (free tier)
  NTFY_TOPIC       - your ntfy.sh topic name (e.g. "johnadams-india-digest")
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
import yfinance as yf
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Feed sources. One pool, mixing macro/legislative/geopolitical sources -
# Gemini does the categorizing and time-horizon sorting, not the feed choice.
# ---------------------------------------------------------------------------
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


def fetch_feed(name, url):
    """Fetch a single RSS feed. Never raises - returns [] on any failure
    so one dead feed doesn't kill the whole run."""
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


def build_gemini_prompt(items):
    lines = [f"{i+1}. [{it['source']}] {it['title']} — {it['summary']}" for i, it in enumerate(items)]
    candidates = "\n".join(lines) if lines else "(no items fetched)"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = f"""You are a financial news editor preparing a same-day briefing for an Indian
equities trader. Today's date: {today}.

Below are raw candidate headlines pulled from RSS feeds covering macroeconomics,
legislative/policy, and geopolitics. Some are duplicates, irrelevant (sports,
entertainment, local crime), or low-signal — ignore those.

=== CANDIDATE HEADLINES ===
{candidates}

Task:
Select stories that matter for Indian stock markets — mix macroeconomic
(inflation, RBI policy, fiscal data, trade), legislative/policy (new bills,
regulations, budget/tax changes, corporate law), and geopolitical (border
tensions, global conflicts, sanctions, diplomacy affecting trade/oil) stories
freely across both lists below, whichever fits the time horizon better.

1. SHORT-TERM (5 items): stories likely to move markets in the next 1-5
   trading days — data releases, rate decisions, sudden geopolitical flare-ups,
   immediate policy announcements.
2. LONG-TERM (5 items): stories that shape market direction over months/years —
   structural reforms, multi-year trade/diplomatic shifts, long-run fiscal or
   regulatory changes.
3. For each: a SHORT headline (max ~10 words, your own concise phrasing, not
   copied verbatim) and ONE line on market impact (max ~12 words). Be terse,
   no filler words, no repeating the headline in the impact line.
4. Tag each item's category as one of: "macro", "legislative", "geopolitical".
5. For each item, name the TOP 3 Indian stocks most likely affected (best
   judgment based on sector/company exposure — this is an inference, not a
   guarantee). Use the exact NSE ticker symbol (e.g. "RELIANCE", "TCS",
   "HDFCBANK") as listed on the NSE, not the full company name, and only
   real, currently-listed NSE tickers — never invent one. Give fewer than 3
   if you aren't confident in more.
6. If fewer than 5 genuinely relevant stories exist for a horizon, return fewer
   rather than padding with filler.

Respond ONLY with JSON matching this exact shape, nothing else:
{{
  "short_term": [{{"headline": "...", "impact": "...", "category": "macro|legislative|geopolitical", "source": "...", "stocks": ["TICKER1", "TICKER2", "TICKER3"]}}],
  "long_term": [{{"headline": "...", "impact": "...", "category": "macro|legislative|geopolitical", "source": "...", "stocks": ["TICKER1", "TICKER2", "TICKER3"]}}]
}}
"""
    return prompt


def call_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "response_mime_type": "application/json",
        },
    }
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def collect_tickers(digest):
    tickers = set()
    for bucket in ("short_term", "long_term"):
        for item in digest.get(bucket, []):
            for t in item.get("stocks", []) or []:
                if t:
                    tickers.add(t.strip().upper())
    return tickers


def fetch_price_moves(tickers):
    """Look up today's % move for each NSE ticker via Yahoo Finance.
    Never raises - any ticker that fails to resolve is just omitted,
    since a bad/delisted ticker shouldn't break the whole digest."""
    moves = {}
    for ticker in tickers:
        try:
            yf_symbol = f"{ticker}.NS"
            info = yf.Ticker(yf_symbol).fast_info
            last = info.get("last_price")
            prev_close = info.get("previous_close")
            if last is not None and prev_close:
                pct = ((last - prev_close) / prev_close) * 100
                arrow = "▲" if pct >= 0 else "▼"
                moves[ticker] = f"{arrow}{abs(pct):.1f}%"
        except Exception as e:
            print(f"[warn] price lookup failed for '{ticker}': {e}", file=sys.stderr)
    return moves


def format_stock_tags(stocks, price_moves):
    if not stocks:
        return ""
    tags = []
    for t in stocks[:3]:
        t = t.strip().upper()
        move = price_moves.get(t)
        tags.append(f"{t} {move}" if move else t)
    return " | ".join(tags)


def format_ntfy_message(digest, price_moves):
    def block(title, items):
        lines = [title]
        for i, item in enumerate(items[:5], 1):
            tag = CATEGORY_TAG.get(item.get("category", "").lower(), "?")
            lines.append(f"{i}. [{tag}] {item['headline']} — {item['impact']}")
            stock_line = format_stock_tags(item.get("stocks", []), price_moves)
            if stock_line:
                lines.append(f"   🏷 {stock_line}")
        return lines

    lines = block("⚡ SHORT-TERM (1-5 days)", digest.get("short_term", []))
    lines.append("")
    lines += block("🧭 LONG-TERM (months+)", digest.get("long_term", []))
    lines.append("")
    lines.append("M=macro · L=legislative · G=geopolitical")
    lines.append("Stocks = AI-inferred exposure, not verified causation")
    return "\n".join(lines)


def send_to_ntfy(message, topic, server):
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    resp = requests.post(
        f"{server}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Title": f"India Markets Digest — {today}".encode("utf-8"),
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

    prompt = build_gemini_prompt(items)

    print(f"Calling Gemini ({GEMINI_MODEL})...")
    for attempt in range(3):
        try:
            digest = call_gemini(prompt, api_key)
            break
        except Exception as e:
            print(f"[warn] Gemini call failed (attempt {attempt+1}/3): {e}", file=sys.stderr)
            time.sleep(5)
    else:
        print("ERROR: Gemini call failed after 3 attempts", file=sys.stderr)
        sys.exit(1)

    tickers = collect_tickers(digest)
    print(f"Looking up live price moves for {len(tickers)} tickers...")
    price_moves = fetch_price_moves(tickers)
    print(f"  -> resolved {len(price_moves)}/{len(tickers)}")

    message = format_ntfy_message(digest, price_moves)
    print("\n--- DIGEST ---")
    print(message)
    print("--------------\n")

    print(f"Posting to ntfy.sh/{ntfy_topic} ...")
    send_to_ntfy(message, ntfy_topic, ntfy_server)
    print("Done.")


if __name__ == "__main__":
    main()
