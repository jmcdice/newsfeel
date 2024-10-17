#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import logging
import os
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

from GoogleNews import GoogleNews
from newspaper import Article
from tqdm import tqdm

# Import the new OpenAI client
from openai import OpenAI

# Instantiate the OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

EXPIRATION_LENGTH = datetime.timedelta(days=2)


@dataclass
class SentimentCacheEntry:
    cached_time: datetime.datetime
    sentiment: str
    confidence: int
    response: str
    content_hash: str
    content: str


def send_query(input_text: str, context: str) -> str:
    """
    Send a query to OpenAI's ChatCompletion API and return the response text.

    Parameters:
        input_text (str): The user input to send to the model.
        context (str): The system prompt or context for the model.

    Returns:
        str: The response text from the model.
    """
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        response_text = "Error: Token limit exceeded"
    return response_text

def get_article_content(url: str, title: str) -> str:
    """
    Fetches and returns the content of an article from a given URL.

    Parameters:
        url (str): The fully qualified URL of the article to fetch.
        title (str): The title of the article.

    Returns:
        str: The content of the article, or the title if content could not be fetched.
    """
    article = Article(url, language='en')
    try:
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Error fetching article content from {url}: {e}")
        return title

def get_cached_sentiment_analysis(
    url: str,
    title: str,
    content: str,
    args: argparse.Namespace,
    cache_file: str,
    sentiment_cache: Dict[str, 'SentimentCacheEntry']
) -> Tuple[str, int, Optional[str]]:
    """
    Analyze the sentiment of the article content, using cache if available.

    Parameters:
        url (str): The URL of the article.
        title (str): The title of the article.
        content (str): The content of the article.
        args (argparse.Namespace): Parsed command-line arguments.
        cache_file (str): Path to the cache file.
        sentiment_cache (Dict[str, SentimentCacheEntry]): The sentiment cache.

    Returns:
        Tuple[str, int, Optional[str]]: Sentiment label, confidence score, and optional response.
    """
    if not content:
        return "Unknown", 0, None

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now = datetime.datetime.now()

    if url in sentiment_cache:
        cache_entry = sentiment_cache[url]
        time_diff = now - cache_entry.cached_time

        if time_diff <= EXPIRATION_LENGTH:
            logging.debug("Article found in cache.")
            return cache_entry.sentiment, cache_entry.confidence, cache_entry.response
        else:
            logging.info("Article found in cache, but expired.")

    context_text = (
        "Analyze the sentiment of this article and rate it as 'bullish', 'very bullish', "
        "'neutral', 'bearish', or 'very bearish' based on the content. Then print on a line by "
        "itself: 'Sentiment: <sentiment>' where <sentiment> is the sentiment you chose. "
        "Print on a line by itself: 'Confidence: <confidence>' where <confidence> is "
        "a number between 0 and 10 that represents how confident you are in your sentiment choice. "
        "Then, please provide an explanation for your sentiment choice."
    )

    logging.debug("Analyzing article...")
    start_time = time.monotonic()

    try:
        response = send_query(content, context_text)
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        response = send_query(content, title)

    elapsed_time = time.monotonic() - start_time
    logging.debug(f"GPT query time: {elapsed_time:.3f} seconds")

    if "Error: Token limit exceeded" in response:
        logging.error("Token limit exceeded. Ignoring the article.")
        return "Unknown", 0, None

    sentiment_map = {
        'very bullish': 'Very Bullish',
        'bullish': 'Bullish',
        'neutral': 'Neutral',
        'unknown': 'Unknown',
        'bearish': 'Bearish',
        'very bearish': 'Very Bearish'
    }

    match = re.search(
        r'Sentiment:\s*([a-zA-Z\s]+)[^0-9]*\s*Confidence:\s*(\d+)',
        response,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        sentiment = match.group(1).lower().strip()
        confidence = int(match.group(2))
    else:
        sentiment = "Unknown"
        confidence = 0

    fsentiment = sentiment_map.get(sentiment.lower(), "Unknown")

    cache_entry = SentimentCacheEntry(
        cached_time=now,
        sentiment=fsentiment,
        confidence=confidence,
        response=response,
        content_hash=content_hash,
        content=content
    )
    sentiment_cache[url] = cache_entry

    # Save cache to disk
    with open(cache_file, "wb") as f:
        pickle.dump(sentiment_cache, f)

    return fsentiment, confidence, response


def analyze_cache_sentiments(
    cache_file: str,
    topic: str = '',
    print_results: bool = True
) -> Dict[str, Any]:
    """
    Analyze sentiments from the cache and optionally print results.

    Parameters:
        cache_file (str): Path to the cache file.
        topic (str): Topic of the articles.
        print_results (bool): Whether to print the analysis results.

    Returns:
        Dict[str, Any]: A dictionary containing analysis information.
    """
    analysis_info = {}
    if not os.path.exists(cache_file):
        if print_results:
            logging.info("Cache file does not exist. No sentiments to analyze.")
        return analysis_info

    sentiment_mapping = {
        'very bullish': 2,
        'bullish': 1,
        'neutral': 0,
        'bearish': -1,
        'very bearish': -2
    }
    sentiment_sum = 0
    confidence_sum = 0
    total_articles = 0

    with open(cache_file, 'rb') as f:
        sentiment_cache = pickle.load(f)

    sentiment_counts = {sentiment: 0 for sentiment in sentiment_mapping.keys()}

    for cache_entry in sentiment_cache.values():
        sentiment_key = cache_entry.sentiment.lower()
        if sentiment_key not in sentiment_counts:
            sentiment_counts[sentiment_key] = 0
        sentiment_counts[sentiment_key] += 1
        sentiment_sum += sentiment_mapping.get(sentiment_key, 0)
        confidence_sum += cache_entry.confidence
        total_articles += 1

    if total_articles > 0:
        weighted_sentiment = sentiment_sum / total_articles
        sentiment_result = 'Neutral'

        if weighted_sentiment <= -1.5:
            sentiment_result = 'Very Bearish'
        elif weighted_sentiment < 0:
            sentiment_result = 'Bearish'
        elif weighted_sentiment >= 1.5:
            sentiment_result = 'Very Bullish'
        elif weighted_sentiment > 0:
            sentiment_result = 'Bullish'

        if print_results:
            print(f'Sentiment Analysis for: {topic}\n')
            print(f'General Sentiment: {sentiment_result}')
            print(f'Total Articles: {total_articles}')
            print(f'Average Confidence: {confidence_sum / total_articles:.2f}\n')

            print('Sentiment Counts:')
            for sentiment, count in sentiment_counts.items():
                percentage = count / total_articles * 100
                print(f' {count} ({percentage:.2f}%) Sentiment: {sentiment.capitalize()}')

            print(f'\nWeighted Sentiment: {weighted_sentiment:.2f}')

        analysis_info = {
            'sentiment_result': sentiment_result,
            'total_articles': total_articles,
            'average_confidence': confidence_sum / total_articles,
            'sentiment_counts': sentiment_counts,
            'weighted_sentiment': weighted_sentiment
        }
    else:
        if print_results:
            logging.info('No articles found in cache for sentiment analysis.')

    return analysis_info


def analyze_summaries(cache_file: str, topic: str = '') -> Optional[str]:
    """
    Analyze summaries of cached articles and generate a summary analysis.

    Parameters:
        cache_file (str): Path to the cache file.
        topic (str): Topic of the articles.

    Returns:
        Optional[str]: The summary analysis or None if cache is empty.
    """
    if not os.path.exists(cache_file):
        logging.info("Cache file does not exist. No summaries to analyze.")
        return None

    with open(cache_file, 'rb') as f:
        sentiment_cache = pickle.load(f)

    summaries = []

    for cache_entry in sentiment_cache.values():
        summary_entry = f"Sentiment: {cache_entry.sentiment}, Response: {cache_entry.response}"
        summaries.append(summary_entry)

    all_summaries = '\n'.join(summaries)

    sentiment_analysis = analyze_cache_sentiments(cache_file, topic, print_results=False)

    sentiment_analysis_text = (
        f"Based on the analysis of {sentiment_analysis['total_articles']} articles, "
        f"the general sentiment for {topic} is {sentiment_analysis['sentiment_result']} "
        f"with an average confidence of {sentiment_analysis['average_confidence']:.2f}. "
        f"The weighted sentiment is {sentiment_analysis['weighted_sentiment']:.2f}. "
        f"Sentiment counts are as follows:\n"
    )
    for sentiment, count in sentiment_analysis['sentiment_counts'].items():
        sentiment_analysis_text += f"{sentiment.capitalize()}: {count}\n"

    context_text = (
        f"Please provide a summary in financial analysis style, including future-looking predictions for the "
        f"topic '{topic}', based on the following summaries of articles and sentiment analysis:\n\n"
        f"{sentiment_analysis_text}\n"
        f"Article Sentiments and Responses:\n{all_summaries}"
    )

    summary_analysis = send_query(all_summaries, context_text)

    return summary_analysis


def print_cache_info(cache_file: str, print_entries: bool = False) -> None:
    """
    Print information about the cache.

    Parameters:
        cache_file (str): Path to the cache file.
        print_entries (bool): Whether to print all cache entries.
    """
    now = datetime.datetime.now()
    try:
        with open(cache_file, 'rb') as f:
            sentiment_cache = pickle.load(f)
    except FileNotFoundError:
        logging.info("Cache file does not exist.")
        return

    total_articles = len(sentiment_cache)
    expired_count = 0

    for url, cache_entry in sentiment_cache.items():
        time_diff = now - cache_entry.cached_time
        if time_diff > EXPIRATION_LENGTH:
            expired_count += 1
        elif print_entries:
            print(f"URL: {url}")
            print(f"Content hash: {cache_entry.content_hash}")
            print(f"Cached time: {cache_entry.cached_time}")
            print(f"Sentiment: {cache_entry.sentiment}")
            print(f"Confidence: {cache_entry.confidence}")
            print(f"Response: {cache_entry.response}")
            print(f"Content: {cache_entry.content}")
            print("")

    if not print_entries:
        print(f"Total articles in cache: {total_articles}")
        print(f"Expired articles: {expired_count}")
        print(f"Non-expired articles: {total_articles - expired_count}")


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
    logging.getLogger().setLevel(getattr(logging, args.loglevel.upper(), None))

    main_topic = args.topic.lower().replace(' ', '-')

    # Create cache directory if it doesn't exist
    cache_directory = 'cache'
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    # Update cache_file based on the given topic
    topic_lower = args.topic.lower().replace(' ', '-')
    cache_file = os.path.join(cache_directory, f"article-cache-{topic_lower}.pkl")

    # Load the cache
    try:
        with open(cache_file, "rb") as f:
            sentiment_cache = pickle.load(f)
    except FileNotFoundError:
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
        googlenews = GoogleNews()
        googlenews.get_news(topic_lower)
        result = googlenews.result()

        num_articles = min(args.num_articles, len(result))
        logging.info(f"Processing {num_articles} articles...\n")

        results = []

        for idx, article in enumerate(tqdm(result[:num_articles], desc="Processing articles")):
            title = article['title']
            url = urljoin('https://news.google.com', article['link'])  # Correct URL construction
            content = get_article_content(url, title)
            sentiment, confidence, response = get_cached_sentiment_analysis(
                url, title, content, args, cache_file, sentiment_cache
            )
        
            results.append({
                'index': idx + 1,
                'title': title,
                'url': url,
                'sentiment': sentiment,
                'confidence': confidence,
                'response': response
            })

            logging.debug(f"Article {idx + 1}:")
            logging.debug(f"Title: {title}")
            logging.debug(f"Sentiment: {sentiment}\n")

        # Analyze cache sentiments
        analyze_cache_sentiments(cache_file, main_topic)
        print_cache_info(cache_file)

        # Output results to CSV if specified
        if args.output_file:
            import csv
            fieldnames = ['index', 'title', 'url', 'sentiment', 'confidence', 'response']
            with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
