#!/usr/bin/env python3

import os
import re
import sys
import time
import requests
import logging
import hashlib
import pickle
import argparse
from urllib.parse import urljoin
from collections import Counter
from GoogleNews import GoogleNews
import openai
from newspaper import Article
from urllib.parse import urlparse, urlunparse
import datetime

EXPIRATION_LENGTH = datetime.timedelta(days=2)
sentiment_cache = {}


logging.getLogger('GoogleNews').setLevel(logging.ERROR)

openai.api_key = os.environ["OPENAI_API_KEY"]

def send_query(input, context):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                { "role": "system", "content": context },
                { "role": "user", "content": input }
            ],
            temperature=0.7,
        )
        response_text = response['choices'][0]['message']['content'].strip()
    except openai.error.InvalidRequestError as e:
        print(f"Error: {e}")
        response_text = "Error: Token limit exceeded"
    return response_text

def get_article_content(url, title):
    url = urljoin("https://news.google.com", url)
    article = Article(url, language='en')
    try:
        article.download()
        article.parse()
    except Exception as e:
        # probably got a subscription error, so just return the title
        return title
    return article.text

def get_cached_sentiment_analysis(url, title, content, args, cache_file):
    if content is None or len(content) == 0:
        return "Unknown", 0, None

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now = datetime.datetime.now()

    if url in sentiment_cache:
        cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash = sentiment_cache[url]
        time_diff = now - cached_time

        if time_diff <= EXPIRATION_LENGTH:
            if args.debug:
                print("Article found in cache.")
            return cached_sentiment, cached_confidence, cached_response
        else:
            print("Article found in cache, but expired.")

    context_text = "Analyze the sentiment of this article and rate it as 'bullish', 'very bullish', \
                   'neutral', 'bearish', or 'very bearish' based on the content. Then print on a line by \
                   itself: 'Sentiment: <sentiment>' where <sentiment> is the sentiment you chose. \
                   Print on a line by itself: 'Confidence: <confidence>' where <confidence> is \
                   a number between 0 and 10 that represents how confident you are in your sentiment choice. \
                   Then, please provide an explanation for your sentiment choice."

    if args.debug:
        print("Analyzing article...")
        start_time = time.monotonic()

    try:
        response = send_query(content, context_text)
    except Exception as e:
        response = send_query(content, title)

    if args.debug:
        end_time = time.monotonic()
        elapsed_time = end_time - start_time
        print(f"GPT query time: {elapsed_time:.3f} seconds")

    # Check if response indicates an error
    if "Error: Token limit exceeded" in response:
        print("Error: Token limit exceeded. Ignoring the article.")
        return "Unknown", 0, None

    sentiment_map = {
        'very bullish': 'Very Bullish',
        'bullish': 'Bullish',
        'neutral': 'Neutral',
        'unknown': 'Unknown',
        'bearish': 'Bearish',
        'very bearish': 'Very Bearish'
    }

    match = re.search(r'Sentiment:\s*([a-zA-Z\s]+)[^0-9]*\s*Confidence:\s*(\d+)', response, re.IGNORECASE | re.DOTALL)

    if match:
        sentiment = match.group(1).lower().strip()
        confidence = int(match.group(2))
    else:
        sentiment = "Unknown"
        confidence = 0

    fsentiment = sentiment_map.get(sentiment.lower(), "Unknown")
    sentiment_cache[url] = (now, fsentiment, confidence, response, content_hash)

    # Save cache to disk
    with open(cache_file, "wb") as f:
        pickle.dump(sentiment_cache, f)

    return fsentiment, confidence, None


def analyze_cache_sentiments(cache_file, topic=''):
    if not os.path.exists(cache_file):
        print("Cache file does not exist. No sentiments to analyze.")
        return

    sentiment_mapping = {'bullish': 1, 'neutral': 0, 'bearish': -1}
    sentiment_sum = 0
    confidence_sum = 0
    total_articles = 0

    with open(cache_file, 'rb') as f:
        sentiment_cache = pickle.load(f)

    #print(f"Loaded sentiment cache: {sentiment_cache}")  # Add this line to print the sentiment_cache content

    sentiment_counts = {'bullish': 0, 'neutral': 0, 'bearish': 0, 'unknown': 0, 'very bullish': 0}

    for content_hash, (cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash) in sentiment_cache.items():
        sentiment_key = cached_sentiment.lower()
        if sentiment_key not in sentiment_counts:
            sentiment_counts[sentiment_key] = 0
        sentiment_counts[sentiment_key] += 1
        confidence_sum += cached_confidence  
        total_articles += 1  # Add this line to increment the total_articles variable

    if total_articles > 0:
        bullish_percentage = (sentiment_counts['bullish'] + sentiment_counts['very bullish']) / total_articles * 100
        bearish_percentage = sentiment_counts['bearish'] / total_articles * 100

        sentiment_result = 'Neutral'
        if bearish_percentage > 75:
            sentiment_result = 'Very Bearish'
        elif bearish_percentage > 50:
            sentiment_result = 'Bearish'
        elif bullish_percentage > 75:
            sentiment_result = 'Very Bullish'
        elif bullish_percentage > 50:
            sentiment_result = 'Bullish'

        print(f'Sentiment Analysis for: {topic}\n')
        print(f'General Sentiment: {sentiment_result}')
        print(f'Total Articles: {total_articles}')
        print(f'Average Confidence: {confidence_sum / total_articles:.2f}\n')

        print('Sentiment Counts:')
        for sentiment, count in sentiment_counts.items():
            percentage = count / total_articles * 100
            print(f' {count} ({percentage:.2f}%) Sentiment: {sentiment.capitalize()}')

    else:
        print('No articles found in cache for sentiment analysis.')


def print_cache_info(args):
    now = datetime.datetime.now()
    total_articles = len(sentiment_cache)
    expired_count = 0

    print(f"Total articles in cache: {total_articles}")

    for url, (cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash) in sentiment_cache.items():
        time_diff = now - cached_time
        if time_diff > EXPIRATION_LENGTH:
            expired_count += 1
        elif args.print_cache:  # Check if --print_cache argument is given
            print(f"URL: {url}")
            print(f"Content hash: {cached_content_hash}")
            print(f"Cached time: {cached_time}")
            print(f"Sentiment: {cached_sentiment}")
            print(f"Confidence: {cached_confidence}")
            print(f"Response: {cached_response}")
            print("")

    if not args.print_cache:  # Print cache info if --print_cache argument is not given
        print(f"Total articles in cache: {total_articles}")
        print(f"Expired articles: {expired_count}")
        print(f"Non-expired articles: {total_articles - expired_count}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process a specified number of articles.')
    parser.add_argument('-n', '--num_articles', type=int, default=5, help='Number of articles to process')
    parser.add_argument('--print_cache', action='store_true', help='Print everything in the cache')
    parser.add_argument('--analyze_cache', action='store_true', help='Analyze cache sentiments and exit')
    parser.add_argument('-t', '--topic', type=str, default='Financial News', help='Topic to fetch news for')  # New argument for topic
    parser.add_argument('--debug', action='store_true', help='Print debug info')

    args = parser.parse_args()

    main_topic = args.topic.lower().replace(' ', '-')

    # Create cache directory if it doesn't exist
    cache_directory = 'cache'
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    # Update cache_file based on the given topic
    topic_lower = args.topic.lower().replace(' ', '-')
    cache_file = os.path.join(cache_directory, f"article-cache-{topic_lower}.pkl")


    # Update the cache loading logic
    global sentiment_cache
    try:
        with open(cache_file, "rb") as f:
            sentiment_cache = pickle.load(f)
    except FileNotFoundError:
        sentiment_cache = {}

    if args.print_cache:  
        print_cache_info(args)
    elif args.analyze_cache:  # Analyze cache sentiments and exit
        analyze_cache_sentiments(cache_file, main_topic)
        exit()
    else:  
        googlenews = GoogleNews()
        googlenews.get_news(topic_lower)
        result = googlenews.result()

        num_articles = args.num_articles if args.num_articles <= len(result) else len(result)
        print(f"Processing {num_articles} articles...\n")

        for idx, article in enumerate(result[:num_articles]):
            title = article['title']
            url = "https://" + article['link']
            content = get_article_content(url, title)
            sentiment, confidence, _ = get_cached_sentiment_analysis(url, title, content, args, cache_file)

            # Process sentiment analysis results
            if args.debug:
                print(f"Article {idx + 1}:")
                print(f"Title: {title}")
                print(f"Sentiment: {sentiment}")
                print("\n")

        # Analyze cache sentiments
        analyze_cache_sentiments(cache_file, main_topic)
        print_cache_info(args)

if __name__ == "__main__":
    main()
