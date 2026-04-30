import logging
from typing import List

from GoogleNews import GoogleNews

# Strict Imports
from config import NEWS_COUNT, NEWS_DATE_RANGE, NEWS_MAX_RETRIES

logger = logging.getLogger(__name__)


class SentimentAnalysis:
    """Perform sentiment analysis on news articles with no fallbacks."""

    def __init__(self):
        self.google_news = None
        self.news_date_range = NEWS_DATE_RANGE

    def fetch_news(self, company_name: str, num_news: int = NEWS_COUNT) -> List[str]:
        """Fetch news articles strictly."""
        try:
            if self.google_news is None:
                self.google_news = GoogleNews(lang="en", region="US")
                self.google_news.set_period(self.news_date_range)

            self.google_news.clear()
            search_query = f"{company_name} stock"

            for attempt in range(NEWS_MAX_RETRIES):
                try:
                    self.google_news.search(search_query)
                    break
                except Exception as e:
                    if attempt == NEWS_MAX_RETRIES - 1:
                        raise e
                    import time

                    time.sleep(1)

            results = self.google_news.results()
            if not results:
                raise ValueError(f"No news results found for {company_name}")

            news_texts = []
            for entry in results[:num_news]:
                title = entry["title"]
                desc = entry["desc"]
                news_texts.append(f"{title} {desc}")

            return news_texts
        except Exception as e:
            logger.error(f"News fetch failure for {company_name}: {e}")
            raise e

    def analyze_sentiment(self, texts: List[str]) -> float:
        """Analyze sentiment using TextBlob with no neutral bias or fallbacks."""
        if not texts:
            raise ValueError("No texts provided for sentiment analysis")

        try:
            from textblob import TextBlob

            scores = []
            for text in texts:
                blob = TextBlob(text)
                scores.append(blob.sentiment.polarity)

            if not scores:
                return 0.0

            return sum(scores) / len(scores)
        except Exception as e:
            logger.error(f"Sentiment calculation failure: {e}")
            raise e

    def perform_sentiment_analysis(self, company_name: str) -> float:
        """Perform complete sentiment analysis strictly."""
        try:
            news_texts = self.fetch_news(company_name)
            return self.analyze_sentiment(news_texts)
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis failure for {company_name}: {e}")
            raise e
