#!/usr/bin/env python3
"""
India Macro/Geopolitical News Digest
- Pulls headlines from a set of RSS feeds (macro/legislative + geopolitical)
- Sends the raw headlines to Gemini (Flash, free tier) to rank/summarize
- Posts a compact top-5 + top-5 digest to ntfy.sh

Env vars required:
  GEMINI_API_KEY   - Google AI Studio key (free tier)
  NTFY_TOPIC       - your ntfy.sh topic name (e.g. "johnadams-india-digest")
Optional:
  NTFY_SERVER      - default https://ntfy.sh
  GEMINI_MODEL     - default gemini-2.5-flash
"""

import os
import sys
import json
import time
import requests
import feedparser
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Feed sources. Each feed is tagged with an intended bucket so we can still
# make a sane fallback split even if Gemini's classification is fuzzy.
# ---------------------------------------------------------------------------
MACRO_LEGISLATIVE_FEEDS = [
    ("Economic Times - Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("Business Standard - Economy", "https://www.business-standard.com/rss/economy-102.rss"),
    ("Business Standard - Finance", "https://www.business-standard.com/rss/finance-103.rss"),
    ("Livemint - Economy", "https://www.livemint.com/rss/economy"),
    ("Moneycontrol - Economy", "https://www.moneycontrol.com/rss/economy.xml"),
    ("PIB India - Releases", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
    ("RBI - Press Releases", "https://www.rbi.org.in/pressreleases_rss.xml"),
]

GEOPOLITICAL_FEEDS = [
    ("The Hindu - National", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("The Hindu - International", "https://www.thehindu.com/news/international/feeder/default.rss"),
    ("Economic Times - World", "https://economictimes.indiatimes.com/news/international/world-news/rssfeeds/1898055973.cms"),
    ("Indian Express - India", "https://indianexpress.com/section/india/feed/"),
    ("Indian Express - World", "https://indianexpress.com/section/world/feed/"),
]

MAX_ITEMS_PER_FEED = 12
FEED_TIMEOUT_SECS = 15
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


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


def build_gemini_prompt(macro_items, geo_items):
    def fmt(items):
        lines = []
        for i, it in enumerate(items):
            lines.append(f"{i+1}. [{it['source']}] {it['title']} — {it['summary']}")
        return "\n".join(lines) if lines else "(no items fetched)"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = f"""You are a financial news editor preparing a same-day briefing for an Indian equities trader.
Today's date: {today}.

Below are raw headlines pulled from RSS feeds in two buckets. Some may be duplicates,
irrelevant (sports, entertainment, local crime), or low-signal — ignore those.

=== MACRO / LEGISLATIVE / POLICY CANDIDATE HEADLINES ===
{fmt(macro_items)}

=== GEOPOLITICAL CANDIDATE HEADLINES ===
{fmt(geo_items)}

Task:
1. Select the TOP 5 macroeconomic and legislative/policy stories most relevant to
   someone trading Indian stocks today (RBI policy, inflation data, budget/tax
   changes, new bills/regulations, corporate law, trade policy, fiscal data, etc).
2. Select the TOP 5 geopolitical stories most relevant to Indian markets
   (border tensions, diplomatic shifts, global conflicts affecting oil/trade,
   major foreign policy moves, sanctions, elections in major trade partners, etc).
3. For each selected story write ONE short punchy line (max ~18 words) explaining
   why it matters for markets/traders. No fluff, no repeating the headline verbatim.
4. If fewer than 5 genuinely relevant stories exist in a bucket, return fewer —
   do not pad with irrelevant filler.

Respond ONLY with JSON matching this exact shape, nothing else:
{{
  "macro_legislative": [{{"headline": "...", "why_it_matters": "...", "source": "..."}}],
  "geopolitical": [{{"headline": "...", "why_it_matters": "...", "source": "..."}}]
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


def format_ntfy_message(digest):
    lines = []
    lines.append("📊 MACRO / LEGISLATIVE")
    for i, item in enumerate(digest.get("macro_legislative", [])[:5], 1):
        lines.append(f"{i}. {item['headline']}")
        lines.append(f"   → {item['why_it_matters']}")
    lines.append("")
    lines.append("🌍 GEOPOLITICAL")
    for i, item in enumerate(digest.get("geopolitical", [])[:5], 1):
        lines.append(f"{i}. {item['headline']}")
        lines.append(f"   → {item['why_it_matters']}")
    return "\n".join(lines)


def send_to_ntfy(message, topic, server):
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    resp = requests.post(
        f"{server}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Title": f"India Macro & Geopolitics — {today}".encode("utf-8"),
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

    print("Fetching macro/legislative feeds...")
    macro_items = collect_headlines(MACRO_LEGISLATIVE_FEEDS)
    print(f"  -> {len(macro_items)} items")

    print("Fetching geopolitical feeds...")
    geo_items = collect_headlines(GEOPOLITICAL_FEEDS)
    print(f"  -> {len(geo_items)} items")

    if not macro_items and not geo_items:
        print("ERROR: all feeds failed, nothing to summarize", file=sys.stderr)
        sys.exit(1)

    prompt = build_gemini_prompt(macro_items, geo_items)

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

    message = format_ntfy_message(digest)
    print("\n--- DIGEST ---")
    print(message)
    print("--------------\n")

    print(f"Posting to ntfy.sh/{ntfy_topic} ...")
    send_to_ntfy(message, ntfy_topic, ntfy_server)
    print("Done.")


if __name__ == "__main__":
    main()
