#!/usr/bin/env python3
"""Daily Article Digest - fetches curated articles from HN API + RSS feeds."""

import sys
import json
import time
import random
import hashlib
import os
import re
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser

try:
    import requests
    import feedparser
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
    print("Install with: pip install requests feedparser", file=sys.stderr)
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".digest-state.json")
MAX_AGE_DAYS = 90
NUM_ARTICLES = 20
HN_TOP_N = 50  # fetch top 50 from HN API, then filter

CATEGORY_EMOJIS = {
    "tech": "💻",
    "ai": "🤖",
    "programming": "👨‍💻",
    "finance": "💰",
    "growth": "🌱",
}

CATEGORY_TARGETS_MORNING = {"tech": 6, "ai": 4, "programming": 4, "finance": 3, "growth": 3}
CATEGORY_TARGETS_EVENING = {"tech": 3, "ai": 3, "programming": 3, "finance": 6, "growth": 5}

SOURCE_WEIGHTS = {
    # Weight 3 (premium)
    "Meta Engineering": 3,
    "Netflix TechBlog": 3,
    "MIT Tech Review": 3,
    "ByteByteGo": 3,
    "The Engineering Manager": 3,
    "First Round Review": 3,
    "Wait But Why": 3,
    "HBR": 3,
    # Weight 2 (standard)
    "McKinsey": 2,
    "InfoQ": 2,
    "TechCrunch": 2,
    "The Verge": 2,
    "Bloomberg Markets": 2,
    "Ars Technica": 2,
    "Farnam Street": 2,
    # Weight 1 (bulk)
    "Hacker News": 1,
    "Dev.to": 1,
}

RSS_FEEDS_MORNING = [
    ("https://techcrunch.com/feed/", "TechCrunch", "tech"),
    ("https://www.technologyreview.com/feed/", "MIT Tech Review", "ai"),
    ("https://www.theverge.com/rss/index.xml", "The Verge", "tech"),
    ("https://engineering.fb.com/feed/", "Meta Engineering", "programming"),
    ("https://medium.com/feed/netflixtechblog", "Netflix TechBlog", "programming"),
    ("https://blog.bytebytego.com/feed", "ByteByteGo", "tech"),
    ("https://feeds.feedburner.com/InfoQ", "InfoQ", "programming"),
    ("https://www.theengineeringmanager.com/feed", "The Engineering Manager", "growth"),
    ("https://dev.to/feed", "Dev.to", "programming"),
    ("https://waitbutwhy.com/feed", "Wait But Why", "growth"),
]

RSS_FEEDS_EVENING = [
    ("https://feeds.bloomberg.com/markets/news.rss", "Bloomberg Markets", "finance"),
    ("https://fs.blog/feed/", "Farnam Street", "growth"),
    ("https://firstround.libsyn.com/rss", "First Round Review", "growth"),
    ("http://feeds.harvardbusiness.org/harvardbusiness/", "HBR", "growth"),
    ("https://www.mckinsey.com/rss", "McKinsey", "finance"),
    ("https://waitbutwhy.com/feed", "Wait But Why", "growth"),
    ("https://arstechnica.com/feed/", "Ars Technica", "tech"),
    ("https://www.theengineeringmanager.com/feed", "The Engineering Manager", "growth"),
]

# Fallback feeds if we don't get enough articles
RSS_FEEDS_FALLBACK = [
    ("https://techcrunch.com/feed/", "TechCrunch", "tech"),
    ("https://www.theverge.com/rss/index.xml", "The Verge", "tech"),
    ("https://fs.blog/feed/", "Farnam Street", "growth"),
]


# ── Helpers ────────────────────────────────────────────────────────────────

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self):
        return " ".join(self.parts).strip()


def strip_html(html_str):
    s = HTMLStripper()
    s.feed(html_str or "")
    return s.get_text()


def extract_keywords(title):
    """Extract meaningful keywords from title for fuzzy dedup."""
    stop_words = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "how", "what",
        "which", "who", "whom", "when", "where", "why", "this", "that",
        "these", "those", "it", "its", "not", "no", "if", "your", "you",
        "we", "our", "they", "their", "he", "she", "his", "her", "my",
        "new", "old", "more", "most", "all", "some", "any", "other", "into",
    }
    words = re.findall(r"[a-zA-Z0-9]+", title.lower())
    return set(w for w in words if w not in stop_words and len(w) > 2)


def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


def parse_date(entry):
    """Try to extract a publish date from a feedparser entry."""
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        tv = getattr(entry, key, None)
        if tv:
            try:
                return datetime(*tv[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    return None


def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        # Handle both old "sent" and new "sent_titles" keys
        if "sent_titles" not in data and "sent" in data:
            data["sent_titles"] = data["sent"]
        if "sent_titles" not in data:
            data["sent_titles"] = []
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sent_titles": []}


def save_state(state):
    # Keep only last 100 titles
    state["sent_titles"] = state["sent_titles"][-100:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ── Fetchers ───────────────────────────────────────────────────────────────

def fetch_hn_topstories(limit=HN_TOP_N):
    """Fetch top stories from HN Firebase API."""
    articles = []
    try:
        resp = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        story_ids = resp.json()[:limit]

        for i, sid in enumerate(story_ids):
            try:
                if i > 0 and i % 5 == 0:
                    time.sleep(0.3)
                item = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    headers={"User-Agent": USER_AGENT},
                    timeout=10,
                )
                item.raise_for_status()
                data = item.json()
                if not data or data.get("type") != "story" or not data.get("title"):
                    continue
                url = data.get("url") or f"https://news.ycombinator.com/item?id={sid}"
                articles.append({
                    "title": data["title"],
                    "url": url,
                    "source": "Hacker News",
                    "score": (data.get("score") or 0),
                    "comments": data.get("descendants") or 0,
                    "category": classify_hn_title(data["title"]),
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: HN API fetch failed: {e}", file=sys.stderr)
    return articles


def classify_hn_title(title):
    """Simple keyword-based category classification for HN titles."""
    t = title.lower()
    ai_kw = ["ai", "gpt", "llm", "openai", "anthropic", "machine learning", "deep learning",
             "neural", "transformer", "model", "chatbot", "claude", "gemini", "diffusion",
             "copilot", "agent", "alignment", "training"]
    fin_kw = ["startup", "funding", "ipo", "stock", "market", "bitcoin", "crypto",
              "revenue", "profit", "bank", "invest", "valuation", "economy"]
    prog_kw = ["rust", "python", "javascript", "typescript", "golang", "cpp", "database",
               "sql", "api", "open source", "framework", "library", "compiler", "linux",
               "kubernetes", "docker", "devops", "git", "code", "debug", "release",
               "programming", "developer", "software", "algorithm", "performance"]
    growth_kw = ["productivity", "habit", "leadership", "management", "career",
                 "communication", "writing", "book", "podcast", "course", "learn"]

    for kw in ai_kw:
        if kw in t:
            return "ai"
    for kw in fin_kw:
        if kw in t:
            return "finance"
    for kw in prog_kw:
        if kw in t:
            return "programming"
    for kw in growth_kw:
        if kw in t:
            return "growth"
    return "tech"


def fetch_rss(feed_url, source_name, category, max_items=15):
    """Fetch articles from an RSS/Atom feed."""
    articles = []
    try:
        d = feedparser.parse(feed_url, request_headers={"User-Agent": USER_AGENT})
        for entry in d.entries[:max_items]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            if not title or not link:
                continue

            # 90-day freshness filter
            pub_date = parse_date(entry)
            if pub_date:
                age = datetime.now(timezone.utc) - pub_date
                if age.days > MAX_AGE_DAYS:
                    continue

            articles.append({
                "title": title,
                "url": link,
                "source": source_name,
                "score": 0,
                "comments": 0,
                "category": category,
            })
    except Exception as e:
        print(f"Warning: Failed to fetch {source_name}: {e}", file=sys.stderr)
    return articles


# ── Selection ──────────────────────────────────────────────────────────────

def dedup(articles, threshold=0.55):
    """Phase 1: Remove near-duplicates within the batch using fuzzy title matching."""
    seen = []
    for a in articles:
        kw = extract_keywords(a["title"])
        is_dup = False
        for s in seen:
            if jaccard_similarity(kw, s["kw"]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen.append({**a, "kw": kw})
    for a in seen:
        del a["kw"]
    return seen


def cross_edition_dedup(articles, state, threshold=0.55):
    """Phase 2: Remove articles already sent in a previous edition."""
    previous_titles = [t.lower() for t in state.get("sent_titles", [])]
    filtered = []
    for a in articles:
        is_dup = False
        for prev in previous_titles:
            if jaccard_similarity(extract_keywords(a["title"]), extract_keywords(prev)) >= threshold:
                is_dup = True
                break
        if not is_dup:
            filtered.append(a)
    # Fallback: if filtering removed too many, skip cross-edition dedup
    if len(filtered) < NUM_ARTICLES // 2 and len(articles) >= NUM_ARTICLES:
        return articles
    return filtered


def select_articles(articles, category_targets):
    """Phase 3: Category-balanced weighted selection."""
    # Apply source weight
    for a in articles:
        weight = SOURCE_WEIGHTS.get(a["source"], 1)
        a["final_score"] = a["score"] * weight + weight * 10  # base score from weight

    # Group by category
    by_cat = {}
    for a in articles:
        by_cat.setdefault(a["category"], []).append(a)

    # Sort: tech and programming by score (highest first), others shuffled
    for cat, items in by_cat.items():
        if cat in ("tech", "programming"):
            items.sort(key=lambda x: x["final_score"], reverse=True)
        else:
            random.shuffle(items)

    # Fill slots by category target
    selected = []
    for cat in category_targets:
        target = category_targets[cat]
        pool = by_cat.get(cat, [])
        count = min(target, len(pool))
        selected.extend(pool[:count])

    # Fill remaining slots from any category
    if len(selected) < NUM_ARTICLES:
        used = set(id(a) for a in selected)
        remaining = [a for a in articles if id(a) not in used]
        remaining.sort(key=lambda x: x["final_score"], reverse=True)
        selected.extend(remaining[:NUM_ARTICLES - len(selected)])

    # Shuffle final selection
    random.shuffle(selected)
    return selected[:NUM_ARTICLES]


# ── Output ─────────────────────────────────────────────────────────────────

def format_output(articles, edition):
    """Format articles as plain text digest."""
    now = datetime.now()
    date_str = f"{now.year}年{now.month:02d}月{now.day:02d}日 {now.strftime('%H:%M')}"
    edition_label = "早报" if edition == "morning" else "晚报"

    sources = sorted(set(a["source"] for a in articles))
    source_str = ", ".join(sources)

    lines = [
        f"📰 每日精选 | {edition_label}",
        f"📅 {date_str}",
        "━━━━━━━━━━━━━━━━━━",
        "",
    ]

    for i, a in enumerate(articles, 1):
        emoji = CATEGORY_EMOJIS.get(a["category"], "📄")
        lines.append(f"{i}. {emoji} {a['title']}")
        lines.append(f"   {a['url']}")
        lines.append("")

    lines.extend([
        "━━━━━━━━━━━━━━━━━━",
        f"来源: {source_str}",
        f"共精选 {len(articles)} 篇",
    ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    edition = sys.argv[1] if len(sys.argv) > 1 else "morning"
    if edition not in ("morning", "evening"):
        print(f"Usage: {sys.argv[0]} [morning|evening]", file=sys.stderr)
        sys.exit(1)

    is_morning = edition == "morning"
    feeds = RSS_FEEDS_MORNING if is_morning else RSS_FEEDS_EVENING
    targets = CATEGORY_TARGETS_MORNING if is_morning else CATEGORY_TARGETS_EVENING

    print(f"Fetching {edition} edition...", file=sys.stderr)

    # Fetch from all sources
    all_articles = []

    # HN API
    hn_articles = fetch_hn_topstories()
    all_articles.extend(hn_articles)
    print(f"  HN API: {len(hn_articles)} articles", file=sys.stderr)

    # RSS feeds
    total_fetched = 0
    for feed_url, source_name, category in feeds:
        time.sleep(0.3)
        items = fetch_rss(feed_url, source_name, category)
        all_articles.extend(items)
        total_fetched += len(items)
        print(f"  {source_name}: {len(items)} articles", file=sys.stderr)

    print(f"  Total fetched: {len(all_articles)}", file=sys.stderr)

    # Fallback if not enough
    if len(all_articles) < 15:
        print("  Fetching fallback feeds...", file=sys.stderr)
        for feed_url, source_name, category in RSS_FEEDS_FALLBACK:
            time.sleep(0.3)
            items = fetch_rss(feed_url, source_name, category)
            all_articles.extend(items)
        print(f"  Total after fallback: {len(all_articles)}", file=sys.stderr)

    # Selection pipeline
    articles = dedup(all_articles)
    print(f"  After dedup: {len(articles)}", file=sys.stderr)

    state = load_state()
    articles = cross_edition_dedup(articles, state)
    print(f"  After cross-edition dedup: {len(articles)}", file=sys.stderr)

    selected = select_articles(articles, targets)

    # Update state with selected titles
    state["sent_titles"].extend([a["title"] for a in selected])
    save_state(state)

    # Format and output
    output = format_output(selected, edition)
    print(output, file=sys.stdout)

    # Byte count for WeChat limit check
    byte_count = len(output.encode("utf-8"))
    print(f"\n[DEBUG] Output: {byte_count} bytes ({len(output)} chars), {len(selected)} articles", file=sys.stderr)
    if byte_count > 4000:
        print(f"  ⚠️ WARNING: Output exceeds 4000-byte WeChat limit!", file=sys.stderr)


if __name__ == "__main__":
    main()
