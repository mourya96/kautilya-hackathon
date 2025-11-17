"""
Data Loader - Efficient JSON parsing and filtering for large datasets
"""

import json
import ijson
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and filtering of large JSON news datasets"""

    def __init__(self, file_path: str = "data.json"):
        self.file_path = file_path
        self.articles: List[Dict[str, Any]] = []

    def load_and_filter(self, min_rating: float = 0.8) -> int:
        """
        Load dataset and filter by source rating using streaming parser.
        Supports the following structures:

        [
          {...},
          {...}
        ]

        { "articles": [ ... ] }
        { "items": [ ... ] }
        { "data": [ ... ] }
        { "news": [ ... ] }
        """
        try:
            with open(self.file_path, "rb") as f:
                key = self._detect_structure(f)

                if key == "array":
                    logger.info("Detected array JSON → using 'item'")
                    f.seek(0)
                    parser = ijson.items(f, "item")
                else:
                    logger.info(f"Detected object JSON with '{key}' → using '{key}.item'")
                    f.seek(0)
                    parser = ijson.items(f, f"{key}.item")

                count = self._process_stream(parser, min_rating)

                # if nothing was detected → fallback loader
                if count == 0:
                    logger.warning("Streaming parser returned 0 articles. Falling back to standard loader.")
                    return self._load_standard(min_rating)

                return count

        except Exception as e:
            logger.error(f"Streaming parser failed: {e}")
            return self._load_standard(min_rating)

    # ---------------------------------------------------------
    # Structure detection
    # ---------------------------------------------------------
    def _detect_structure(self, f):
        """Detect dataset format: array or object with known keys."""
        f.seek(0)
        first_char = self._peek_first_char(f)

        if first_char == b'[':
            return "array"

        # It's an object → inspect keys
        f.seek(0)
        try:
            obj = json.loads(f.read().decode("utf-8"))
            for key in ["items", "articles", "data", "news"]:
                if key in obj:
                    return key
        except Exception:
            pass

        # Fallback
        return "items"

    def _peek_first_char(self, f):
        while True:
            c = f.read(1)
            if not c or not c.strip():
                continue
            f.seek(0)
            return c

    # ---------------------------------------------------------
    # Streaming processor
    # ---------------------------------------------------------
    def _process_stream(self, parser, min_rating):
        total = 0
        filtered = 0

        for article in parser:
            total += 1

            # Extract rating
            rating = article.get("source_rating", 0)
            if rating == 0:
                rating = article.get("sourceRating", 0)
            if rating == 0:
                rating = article.get("rating", 0)

            if rating >= min_rating:
                cleaned = self._clean_article(article)
                if cleaned:
                    self.articles.append(cleaned)
                    filtered += 1

        logger.info(f"Processed {total} total articles")
        logger.info(f"Loaded {len(self.articles)} articles with rating >= {min_rating}")

        return filtered

    # ---------------------------------------------------------
    # Fallback loader
    # ---------------------------------------------------------
    def _load_standard(self, min_rating):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        else:
            for key in ["items", "articles", "data", "news"]:
                if key in data:
                    items = data[key]
                    break
            else:
                raise ValueError("Unsupported JSON structure")

        total = len(items)

        for article in items:
            rating = article.get("source_rating", 0)
            if rating == 0:
                rating = article.get("sourceRating", 0)
            if rating == 0:
                rating = article.get("rating", 0)

            if rating >= min_rating:
                cleaned = self._clean_article(article)
                if cleaned:
                    self.articles.append(cleaned)

        logger.info(f"Processed {total} total articles")
        logger.info(f"Loaded {len(self.articles)} articles with rating >= {min_rating}")

        return len(self.articles)

    # ---------------------------------------------------------
    # Article cleaning
    # ---------------------------------------------------------
    def _clean_article(self, article):
        headline = (article.get("title")
                    or article.get("headline")
                    or article.get("text")
                    or article.get("summary", ""))

        if not headline or len(headline.strip()) < 10:
            return None

        date_str = (article.get("published_at")
                    or article.get("publishedAt")
                    or article.get("date")
                    or article.get("timestamp"))
        date = self._parse_date(date_str)

        content = (article.get("story")
                   or article.get("content")
                   or article.get("description")
                   or article.get("text")
                   or article.get("body", ""))

        full_text = f"{headline}. {content}".strip()

        # Normalize rating
        rating = article.get("source_rating", 0)
        if rating == 0:
            rating = article.get("sourceRating", 0)
        if rating == 0:
            rating = article.get("rating", 0)

        try:
            rating = float(rating)
        except:
            rating = 0.0

        source = article.get("source", "")
        if not source and article.get("url"):
            from urllib.parse import urlparse
            try:
                domain = urlparse(article["url"]).netloc
                source = domain.replace("www.", "").split(".")[0].title()
            except:
                source = "Unknown"
        if not source:
            source = "Unknown"

        return {
            "id": article.get("id") or hash(headline),
            "headline": headline.strip(),
            "content": content.strip(),
            "full_text": full_text,
            "date": date,
            "url": article.get("url", ""),
            "source": source,
            "source_rating": rating,
            "author": article.get("author", ""),
            "category": article.get("contentType", "General"),
            "tags": article.get("tags", []),
            "raw": article
        }

    # ---------------------------------------------------------
    # Date parser
    # ---------------------------------------------------------
    def _parse_date(self, s):
        if not s:
            return ""
        try:
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%Y/%m/%d"
            ]
            for fmt in formats:
                try:
                    cleaned = str(s).split("+")[0].split(".")[0]
                    return datetime.strptime(cleaned, fmt).isoformat()
                except:
                    continue
            return str(s)
        except:
            return ""

    # ---------------------------------------------------------
    def get_articles(self):
        return self.articles

    def get_article_count(self):
        return len(self.articles)

    def get_date_range(self):
        dates = [a["date"] for a in self.articles if a["date"]]
        if not dates:
            return None, None
        return min(dates), max(dates)
