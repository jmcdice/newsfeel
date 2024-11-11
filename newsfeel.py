#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from newspaper import Article
from openai import OpenAI  # Updated import
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Check for OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set.")
    exit(1)

# Instantiate the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)  # Updated client instantiation

EXPIRATION_LENGTH = datetime.timedelta(days=2)

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    logging.error("NEWS_API_KEY environment variable not set.")
    exit(1)

NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'


@dataclass
class SentimentCacheEntry:
    cached_time: datetime.datetime
    sentiment: str
    confidence: int
    response: str
    content_hash: str
    content: str
    key_insights: str  # New field for key insights


def send_query(input_text: str, context: str) -> str:
    """
    Send a query to OpenAI's ChatCompletion API and return the response text.
    """
    try:
        completion = client.chat.completions.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7
        )
        response_text = completion.choices[0].message.content.strip()
        logging.debug(f"OpenAI response: {response_text}")
        return response_text
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "Error: Token limit exceeded"


def get_article_content(url: str, title: str) -> str:
    """
    Fetches and returns the content of an article from a given URL.
    """
    article = Article(url, language='en')
    try:
        article.download()
        article.parse()
        content = article.text
        if not content:
            logging.warning(f"Empty content retrieved for URL: {url}. Using title as content.")
            return title
        logging.debug(f"Successfully fetched content for URL: {url}")
        return content
    except Exception as e:
        logging.error(f"Error fetching article content from {url}: {e}")
        return title


def get_news_articles(topic: str, num_articles: int) -> list:
    """
    Fetch news articles using NewsAPI.
    """
    params = {
        'q': topic,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': num_articles,
        'apiKey': NEWS_API_KEY
    }
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        logging.info(f"Fetched {len(articles)} articles for topic '{topic}'.")
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news articles: {e}")
        return []


def get_cached_sentiment_analysis(
    url: str,
    title: str,
    content: str,
    cache_file: str,
    sentiment_cache: Dict[str, 'SentimentCacheEntry']
) -> Tuple[str, int, Optional[str], str]:
    """
    Analyze the sentiment of the article content, using cache if available.
    """
    if not content:
        logging.warning(f"No content for URL: {url}. Skipping sentiment analysis.")
        return "Unknown", 0, None, ""

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now = datetime.datetime.now()

    if url in sentiment_cache:
        cache_entry = sentiment_cache[url]
        time_diff = now - cache_entry.cached_time

        if time_diff <= EXPIRATION_LENGTH:
            logging.debug(f"Article found in cache: {url}")
            return cache_entry.sentiment, cache_entry.confidence, cache_entry.response, cache_entry.key_insights
        else:
            logging.info(f"Article cache expired for URL: {url}")

    context_text = (
        "Analyze the sentiment of this article and rate it as 'bullish', 'very bullish', "
        "'neutral', 'bearish', or 'very bearish' based on the content. Then print on a line by "
        "itself: 'Sentiment: <sentiment>' where <sentiment> is the sentiment you chose. "
        "Print on a line by itself: 'Confidence: <confidence>' where <confidence> is "
        "a number between 0 and 10 that represents how confident you are in your sentiment choice. "
        "Then, please provide a bullet point list of key insights from the article that explain the reason for your sentiment choice."
    )

    logging.debug(f"Analyzing sentiment for URL: {url}")
    start_time = time.monotonic()

    # Truncate content if it's too long
    max_tokens = 2048  # Adjust based on OpenAI's token limit
    content = content[:max_tokens]

    response = send_query(content, context_text)

    elapsed_time = time.monotonic() - start_time
    logging.debug(f"GPT query time for URL {url}: {elapsed_time:.3f} seconds")

    if "Error: Token limit exceeded" in response:
        logging.error(f"Token limit exceeded for URL {url}. Ignoring the article.")
        return "Unknown", 0, None, ""

    sentiment_map = {
        'very bullish': 'Very Bullish',
        'bullish': 'Bullish',
        'neutral': 'Neutral',
        'unknown': 'Unknown',
        'bearish': 'Bearish',
        'very bearish': 'Very Bearish'
    }

    match = re.search(
        r'Sentiment:\s*([a-zA-Z\s]+)\s*Confidence:\s*(\d+)',
        response,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        sentiment = match.group(1).lower().strip()
        confidence = int(match.group(2))
        confidence_end = match.end()
        key_insights = response[confidence_end:].strip()
        logging.debug(f"Parsed sentiment for URL {url}: {sentiment}, Confidence: {confidence}")
    else:
        sentiment = "Unknown"
        confidence = 0
        key_insights = ""
        logging.warning(f"Failed to parse sentiment for URL {url}. Response: {response}")

    fsentiment = sentiment_map.get(sentiment.lower(), "Unknown")

    cache_entry = SentimentCacheEntry(
        cached_time=now,
        sentiment=fsentiment,
        confidence=confidence,
        response=response,
        content_hash=content_hash,
        content=content,
        key_insights=key_insights  # Store key insights
    )
    sentiment_cache[url] = cache_entry

    # Save cache to disk
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(sentiment_cache, f)
        logging.debug(f"Cache updated and saved for URL: {url}")
    except Exception as e:
        logging.error(f"Failed to save cache for URL {url}: {e}")

    return fsentiment, confidence, response, key_insights


def analyze_cache_sentiments(cache_file: str, topic: str):
    """
    Analyze sentiments in the cache and print a summary.
    """
    try:
        with open(cache_file, "rb") as f:
            sentiment_cache = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"No cache file found at {cache_file}")
        return
    except Exception as e:
        logging.error(f"Failed to load cache file {cache_file}: {e}")
        return

    sentiments = {
        'Very Bullish': 0,
        'Bullish': 0,
        'Neutral': 0,
        'Bearish': 0,
        'Very Bearish': 0,
        'Unknown': 0
    }
    total_confidence = 0
    total_articles = 0

    now = datetime.datetime.now()

    for url, entry in sentiment_cache.items():
        time_diff = now - entry.cached_time
        if time_diff > EXPIRATION_LENGTH:
            continue

        sentiments[entry.sentiment] += 1
        total_confidence += entry.confidence
        total_articles += 1

    if total_articles == 0:
        logging.info("No non-expired articles in cache to analyze.")
        return

    general_sentiment = max(sentiments, key=sentiments.get)
    average_confidence = total_confidence / total_articles

    sentiment_weights = {
        'Very Bullish': 2,
        'Bullish': 1,
        'Neutral': 0,
        'Bearish': -1,
        'Very Bearish': -2
    }
    weighted_sentiment = sum(
        sentiments[sentiment] * sentiment_weights.get(sentiment, 0) for sentiment in sentiments
    ) / total_articles

    print(f"\nSentiment Analysis for: {topic}\n")
    print(f"General Sentiment: {general_sentiment}")
    print(f"Total Articles: {total_articles}")
    print(f"Average Confidence: {average_confidence:.2f}\n")
    print("Sentiment Counts:")
    for sentiment, count in sentiments.items():
        percentage = (count / total_articles) * 100 if total_articles > 0 else 0
        print(f" {count} ({percentage:.2f}%) Sentiment: {sentiment}")

    print(f"\nWeighted Sentiment: {weighted_sentiment:.2f}")


def print_cache_info(cache_file: str, print_entries: bool = False):
    """
    Print information about the cache and optionally print all entries.
    """
    try:
        with open(cache_file, "rb") as f:
            sentiment_cache = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"No cache file found at {cache_file}")
        return
    except Exception as e:
        logging.error(f"Failed to load cache file {cache_file}: {e}")
        return

    now = datetime.datetime.now()
    expired_entries = 0
    non_expired_entries = 0

    for entry in sentiment_cache.values():
        time_diff = now - entry.cached_time
        if time_diff > EXPIRATION_LENGTH:
            expired_entries += 1
        else:
            non_expired_entries += 1

    logging.info(f"Total articles in cache: {len(sentiment_cache)}")
    logging.info(f"Expired articles: {expired_entries}")
    logging.info(f"Non-expired articles: {non_expired_entries}")

    if print_entries:
        for url, entry in sentiment_cache.items():
            print(f"URL: {url}")
            print(f"Cached Time: {entry.cached_time}")
            print(f"Sentiment: {entry.sentiment}")
            print(f"Confidence: {entry.confidence}")
            print(f"Key Insights:\n{entry.key_insights}")
            print(f"Response: {entry.response}\n")


def generate_final_summary(results, topic):
    """
    Generate a summary analysis of the results using OpenAI API.
    """
    summary_input = f"Provide a concise summary and analysis of the following articles related to '{topic}'. Highlight common themes, sentiments, and any significant information.\n\n"
    for idx, res in enumerate(results):
        summary_input += f"Article {idx + 1}: {res['title']}\n"
        summary_input += f"Sentiment: {res['sentiment']}\n"
        summary_input += f"Key Insights:\n{res['key_insights']}\n\n"

    context_text = (
        "Based on the provided article summaries, generate a concise summary that reflects the overall sentiment and key points. Highlight common themes and significant information."
    )

    logging.debug("Generating final summary using OpenAI API.")
    summary = send_query(summary_input, context_text)

    return summary


def analyze_summaries(cache_file: str, topic: str) -> Optional[str]:
    """
    Generate a summary analysis of the cached articles.
    """
    try:
        with open(cache_file, "rb") as f:
            sentiment_cache = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"No cache file found at {cache_file}")
        return None
    except Exception as e:
        logging.error(f"Failed to load cache file {cache_file}: {e}")
        return None

    now = datetime.datetime.now()
    contents = []

    for entry in sentiment_cache.values():
        time_diff = now - entry.cached_time
        if time_diff <= EXPIRATION_LENGTH:
            contents.append(entry.content)

    if not contents:
        logging.info("No non-expired articles in cache to analyze.")
        return None

    combined_content = "\n\n".join(contents)

    context_text = (
        f"Provide a concise summary and analysis of the following articles related to '{topic}'. "
        f"Highlight common themes, sentiments, and any significant information."
    )

    logging.debug("Generating summary analysis using OpenAI API.")
    summary = send_query(combined_content, context_text)

    return summary


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process a specified number of articles.')
    parser.add_argument('-n', '--num_articles', type=int, default=5, help='Number of articles to process')
    parser.add_argument('--print_cache', action='store_true', help='Print everything in the cache')
    parser.add_argument('--analyze_cache', action='store_true', help='Analyze cache sentiments and exit')
    parser.add_argument('--analyze_summaries', action='store_true', help='Analyze summaries of cached articles and exit')
    parser.add_argument('-t', '--topic', type=str, default='Financial News', help='Topic to fetch news for')
    parser.add_argument('--loglevel', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('-o', '--output_file', type=str, help='File to write output results to')

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.debug(f"Logging level set to {args.loglevel.upper()}")

    main_topic = args.topic.lower().replace(' ', '-')

    # Create cache directory if it doesn't exist
    cache_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    if not os.path.exists(cache_directory):
        try:
            os.makedirs(cache_directory)
            logging.debug(f"Created cache directory at {cache_directory}")
        except Exception as e:
            logging.error(f"Failed to create cache directory at {cache_directory}: {e}")
            exit(1)
    else:
        logging.debug(f"Cache directory exists at {cache_directory}")

    # Update cache_file based on the given topic
    topic_lower = args.topic.lower().replace(' ', '-')
    cache_file = os.path.join(cache_directory, f"article-cache-{topic_lower}.pkl")
    logging.debug(f"Cache file will be located at {cache_file}")

    # Load the cache
    try:
        with open(cache_file, "rb") as f:
            sentiment_cache = pickle.load(f)
        logging.debug(f"Loaded existing cache from {cache_file} with {len(sentiment_cache)} entries.")
    except FileNotFoundError:
        sentiment_cache = {}
        logging.info(f"No existing cache found. Starting fresh for topic: {args.topic}")
    except Exception as e:
        logging.error(f"Failed to load cache file {cache_file}: {e}")
        sentiment_cache = {}

    if args.print_cache:
        print_cache_info(cache_file, print_entries=True)
    elif args.analyze_cache:
        analyze_cache_sentiments(cache_file, main_topic)
    elif args.analyze_summaries:
        summary_analysis = analyze_summaries(cache_file, main_topic)
        if summary_analysis:
            print(summary_analysis)
    else:
        articles = get_news_articles(args.topic, args.num_articles)

        if not articles:
            logging.warning(f"No articles found for topic: {args.topic}")
            return

        results = []

        for idx, article in enumerate(tqdm(articles, desc="Processing articles")):
            title = article.get('title', 'No Title')
            url = article.get('url', '')
            if not url:
                logging.warning(f"No URL found for article titled '{title}'. Skipping.")
                continue

            logging.debug(f"Fetching content for URL: {url}")
            content = get_article_content(url, title)
            sentiment, confidence, response, key_insights = get_cached_sentiment_analysis(
                url, title, content, cache_file, sentiment_cache
            )

            results.append({
                'index': idx + 1,
                'title': title,
                'url': url,
                'sentiment': sentiment,
                'confidence': confidence,
                'key_insights': key_insights,  # Include key insights
                'response': response
            })

            logging.debug(f"Article {idx + 1}:")
            logging.debug(f"Title: {title}")
            logging.debug(f"Sentiment: {sentiment}\n")

        # Analyze cache sentiments
        analyze_cache_sentiments(cache_file, main_topic)
        print_cache_info(cache_file)

        # Generate final summary
        summary = generate_final_summary(results, main_topic)
        print("\nFinal Summary:\n")
        print(summary)

        # Output results to CSV if specified
        if args.output_file:
            import csv
            fieldnames = ['index', 'title', 'url', 'sentiment', 'confidence', 'key_insights', 'response']
            try:
                with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
                logging.info(f"Results written to {args.output_file}")
            except Exception as e:
                logging.error(f"Failed to write results to CSV: {e}")


if __name__ == "__main__":
    main()

