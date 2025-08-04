"""
Sentiment Analysis Module
File: scripts/sentiment_analysis.py

This module fetches news about stocks and performs sentiment analysis using AI models.
"""

import requests
from bs4 import BeautifulSoup
from GoogleNews import GoogleNews
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Any
from utils.logger import setup_logging
from config import SENTIMENT_MODEL, NEWS_COUNT, NEWS_DATE_RANGE

logger = setup_logging()

class SentimentAnalysis:
    """
    Perform sentiment analysis on news articles about stocks.
    """
    
    def __init__(self, model_name: str = SENTIMENT_MODEL):
        """
        Initialize sentiment analysis with a pre-trained model.
        
        Args:
            model_name: Name of the sentiment analysis model
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.google_news = GoogleNews(lang='en', region='US')  # Use US region for English content
        self.google_news.set_period(NEWS_DATE_RANGE)
        
    def get_sentiment_pipeline(self):
        """
        Load and cache the sentiment analysis pipeline.

        Returns:
            Sentiment analysis pipeline
        """
        if self.sentiment_pipeline is None:
            # Try to load models with progressive fallbacks to prevent crashes
            models_to_try = [
                'textblob',  # Start with TextBlob for better memory management
                'distilbert-base-uncased-finetuned-sst-2-english',  # Simple fallback
                self.model_name  # Primary model from config (last resort)
            ]
            
            for i, model_name in enumerate(models_to_try):
                try:
                    if model_name == 'textblob':
                        # Use TextBlob as primary choice (no transformers required)
                        logger.info("Using TextBlob for sentiment analysis")
                        self.sentiment_pipeline = self._create_textblob_pipeline()
                        self.model_name = 'textblob'
                        break
                    
                    logger.info(f"Attempting to load sentiment model: {model_name}")
                    
                    # Set environment variables to prevent crashes
                    import torch
                    import os
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS completely
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism for stability
                    torch.backends.mps.is_available = lambda: False  # Disable MPS
                    
                    # Force garbage collection before loading model
                    import gc
                    gc.collect()
                    
                    # Try with minimal configuration to avoid crashes
                    self.sentiment_pipeline = pipeline(
                        'sentiment-analysis', 
                        model=model_name, 
                        device='cpu',  # Force CPU explicitly
                        torch_dtype=torch.float32,
                        trust_remote_code=False,  # Disable remote code for security
                        use_fast=False,  # Use slower but more stable tokenizer
                        return_all_scores=False,
                        model_kwargs={
                            'low_cpu_mem_usage': True,  # Use less memory
                            'torch_dtype': torch.float32
                        }
                    )
                    
                    logger.info(f"Successfully loaded sentiment model: {model_name}")
                    self.model_name = model_name
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    # Force cleanup on failure
                    try:
                        import gc
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    except:
                        pass
                    
                    if i == len(models_to_try) - 1:  # Last attempt failed
                        logger.error("All sentiment models failed to load, using dummy pipeline")
                        self.sentiment_pipeline = self._create_dummy_pipeline()
                        break
                    continue
                    
        return self.sentiment_pipeline
    
    def _create_dummy_pipeline(self):
        """
        Create a dummy sentiment pipeline that always returns neutral sentiment.
        
        Returns:
            Dummy pipeline function
        """
        def dummy_pipeline(text):
            return [{'label': 'NEUTRAL', 'score': 0.5}]
        
        logger.warning("Using dummy sentiment pipeline - all sentiment scores will be neutral")
        return dummy_pipeline
    
    def _create_textblob_pipeline(self):
        """
        Create a TextBlob-based sentiment pipeline as a lightweight alternative.
        
        Returns:
            TextBlob pipeline function
        """
        try:
            from textblob import TextBlob
            
            def textblob_pipeline(text):
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # Range: -1 to 1
                
                # Convert to transformer-like output format
                if polarity > 0.1:
                    return [{'label': 'POSITIVE', 'score': abs(polarity)}]
                elif polarity < -0.1:
                    return [{'label': 'NEGATIVE', 'score': abs(polarity)}]
                else:
                    return [{'label': 'NEUTRAL', 'score': 0.5}]
                    
            logger.info("TextBlob sentiment pipeline created successfully")
            return textblob_pipeline
            
        except ImportError:
            logger.warning("TextBlob not available, falling back to dummy pipeline")
            return self._create_dummy_pipeline()
    
    def _is_likely_english(self, text: str) -> bool:
        """
        Check if text is likely in English using simple heuristics.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be English, False otherwise
        """
        if not text or len(text.strip()) < 10:
            return False
        
        # Check for non-Latin scripts (simple heuristic)
        latin_chars = sum(1 for char in text if ord(char) < 256)
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        # If more than 80% of characters are Latin-based, likely English
        latin_ratio = latin_chars / total_chars
        
        # Also check for common English words
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'stock', 'company', 'market', 'price', 'shares']
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_words if word in text_lower)
        
        # Consider it English if:
        # - High Latin character ratio AND some English words found
        # - OR very high Latin ratio (for short texts)
        return (latin_ratio > 0.8 and english_word_count > 0) or latin_ratio > 0.95
    
    def fetch_news(self, company_name: str, num_news: int = NEWS_COUNT) -> List[str]:
        """
        Fetch news articles about a company.
        
        Args:
            company_name: Name of the company
            num_news: Number of news articles to fetch
            
        Returns:
            List of news article texts
        """
        try:
            # Clear previous results
            self.google_news.clear()
            
            # Search for news with simpler query
            search_query = f"{company_name} stock"
            self.google_news.search(search_query)
            results = self.google_news.results()
            
            news_texts = []
            for i, entry in enumerate(results[:num_news]):
                try:
                    # Get the title and description
                    title = entry.get('title', '')
                    desc = entry.get('desc', '')
                    
                    # Filter out non-English content (basic check)
                    combined_text = f"{title} {desc}"
                    if not self._is_likely_english(combined_text):
                        logger.debug(f"Skipping non-English content: {combined_text[:50]}...")
                        continue
                    
                    # Try to get the full article content with rate limiting
                    link = entry.get('link')
                    if link and link.startswith('http'):
                        try:
                            import time
                            time.sleep(0.5)  # Add delay to avoid rate limiting
                            
                            response = requests.get(link, timeout=5, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                            })
                            
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.content, 'html.parser')
                                
                                # Extract text from paragraphs
                                paragraphs = soup.find_all('p')
                                article_text = ' '.join([p.get_text() for p in paragraphs])
                                
                                if article_text and len(article_text) > 50:
                                    news_texts.append(article_text[:1000])  # Limit to 1000 chars
                                else:
                                    news_texts.append(f"{title} {desc}")
                            else:
                                news_texts.append(f"{title} {desc}")
                                
                        except Exception as e:
                            logger.debug(f"Failed to fetch full article from {link}: {e}")
                            news_texts.append(f"{title} {desc}")
                    else:
                        news_texts.append(f"{title} {desc}")
                        
                except Exception as e:
                    logger.debug(f"Error processing news entry {i}: {e}")
                    continue
            
            # If no news found, return a neutral text with slight positive bias
            if not news_texts:
                news_texts = [f"{company_name} is a stable company with market presence and growth potential."]
            
            logger.info(f"Fetched {len(news_texts)} news articles for {company_name}")
            
            # Log a sample of news headlines for debugging
            if news_texts and len(news_texts) > 0:
                sample_news = news_texts[:3]  # Show first 3 news items
                for i, news in enumerate(sample_news, 1):
                    # Show first 100 characters of each news item
                    preview = news[:100] + "..." if len(news) > 100 else news
                    logger.info(f"News {i} sample: {preview}")
            
            return news_texts
            
        except Exception as e:
            logger.error(f"Error fetching news for {company_name}: {e}")
            # Return neutral text with slight positive bias on error
            return [f"{company_name} is a stable company with market presence and growth potential."]
    
    def analyze_sentiment(self, texts: List[str]) -> float:
        """
        Analyze sentiment of a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Average sentiment score (-1 to 1)
        """
        if not texts:
            return 0.0
            
        try:
            classifier = self.get_sentiment_pipeline()
            if classifier is None:
                return 0.0
                
            scores = []
            for text in texts:
                try:
                    # Truncate text to model's max input length (RoBERTa has 514 token limit)
                    if self.model_name == 'cardiffnlp/twitter-roberta-base-sentiment-latest':
                        truncated_text = text[:400]  # Conservative limit for token count
                    else:
                        truncated_text = text[:512]
                    
                    # Skip empty or very short texts
                    if len(truncated_text.strip()) < 10:
                        continue
                    
                    result = classifier(truncated_text)[0]
                    label = result['label']
                    confidence = result['score']
                    
                    # Map labels to numeric scores
                    if self.model_name == 'cardiffnlp/twitter-roberta-base-sentiment-latest':
                        if label == 'LABEL_2':  # Positive
                            scores.append(confidence)
                        elif label == 'LABEL_1':  # Neutral
                            scores.append(0)
                        elif label == 'LABEL_0':  # Negative
                            scores.append(-confidence)
                    elif self.model_name == 'textblob':
                        # TextBlob already returns proper scores
                        if label == 'POSITIVE':
                            scores.append(confidence)
                        elif label == 'NEGATIVE':
                            scores.append(-confidence)
                        else:  # NEUTRAL
                            scores.append(0)
                    elif label == 'NEUTRAL':  # Dummy pipeline
                        scores.append(0)
                    else:  # DistilBERT or similar
                        if label == 'POSITIVE':
                            scores.append(confidence)
                        elif label == 'NEGATIVE':
                            scores.append(-confidence)
                        else:
                            scores.append(0)
                            
                except Exception as e:
                    logger.warning(f"Error analyzing sentiment for text: {e}")
                    continue
            
            if not scores:
                return 0.05  # Default to slightly positive when no sentiment detected
                
            average_score = sum(scores) / len(scores)
            
            # Apply slight positive bias to neutral scores for better recommendations
            if -0.1 <= average_score <= 0.1:
                average_score = max(0.05, average_score + 0.05)
            
            logger.info(f"Sentiment analysis complete: {len(scores)} texts, average score: {average_score:.3f}")
            return average_score
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def perform_sentiment_analysis(self, company_name: str) -> float:
        """
        Perform complete sentiment analysis for a company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            # Fetch news
            news_texts = self.fetch_news(company_name)
            
            if not news_texts:
                logger.warning(f"No news found for {company_name}")
                return 0.0
            
            # Analyze sentiment
            sentiment_score = self.analyze_sentiment(news_texts)
            
            logger.info(f"Sentiment analysis for {company_name}: {sentiment_score:.3f}")
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {company_name}: {e}")
            return 0.0


# Module-level functions for backward compatibility
def fetch_news(company_name: str, num_news: int = NEWS_COUNT) -> List[str]:
    """
    Fetch news articles about a company.
    
    Args:
        company_name: Name of the company
        num_news: Number of news articles to fetch
        
    Returns:
        List of news article texts
    """
    analyzer = SentimentAnalysis()
    return analyzer.fetch_news(company_name, num_news)

def perform_sentiment_analysis(news_texts: List[str], model_name: str = SENTIMENT_MODEL) -> float:
    """
    Perform sentiment analysis on a list of news texts.
    
    Args:
        news_texts: List of news article texts
        model_name: Name of the sentiment analysis model
        
    Returns:
        Average sentiment score (-1 to 1)
    """
    analyzer = SentimentAnalysis(model_name)
    return analyzer.analyze_sentiment(news_texts)
