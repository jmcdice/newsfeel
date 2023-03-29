#!/usr/bin/env python3

import os
import re
import sys
import requests
import logging
import hashlib
import pickle
import argparse
from urllib.parse import urljoin
from termcolor import colored
from GoogleNews import GoogleNews
import openai
from newspaper import Article
from urllib.parse import urlparse, urlunparse
import datetime

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

def get_cached_sentiment_analysis(url, title, content, args):
    if content is None or len(content) == 0:
        return "Unknown", 0, None

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now = datetime.datetime.now()

    if url in sentiment_cache:
        cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash = sentiment_cache[url]
        time_diff = now - cached_time

        if time_diff <= expiration_length:
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
    try:
        response = send_query(content, context_text)
    except Exception as e:
        response = send_query(content, title)

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

def analyze_cache_sentiments():
    sentiment_mapping = {'bullish': 1, 'neutral': 0, 'bearish': -1}
    sentiment_sum = 0
    confidence_sum = 0
    total_articles = 0

    with open(cache_file, 'rb') as f:
        sentiment_cache = pickle.load(f)

    for content_hash, (cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash) in sentiment_cache.items():
        sentiment_sum += sentiment_mapping.get(cached_sentiment.lower(), 0)
        confidence_sum += float(cached_confidence)
        total_articles += 1

    if total_articles > 0:
        avg_sentiment = sentiment_sum / total_articles
        avg_confidence = confidence_sum / total_articles

        sentiment_result = 'Neutral'
        if avg_sentiment < -1.5:
            sentiment_result = 'Very Bearish'
        elif avg_sentiment < -0.5:
            sentiment_result = 'Bearish'
        elif avg_sentiment > 0.5:
            sentiment_result = 'Bullish'
        elif avg_sentiment > 1.5:
            sentiment_result = 'Very Bullish'

        print(f'Cache Sentiment Analysis:')
        print(f'Sentiment: {sentiment_result}')
        print(f'Total Articles: {total_articles}')
        print(f'Average Sentiment: {avg_sentiment}')
        print(f'Average Confidence: {avg_confidence:.2f}')
    else:
        print('No articles found in cache for sentiment analysis.')

def print_cache_info(args):
    now = datetime.datetime.now()
    total_articles = len(sentiment_cache)
    expired_count = 0

    print(f"Total articles in cache: {total_articles}")

    for url, (cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash) in sentiment_cache.items():
        time_diff = now - cached_time
        if time_diff > expiration_length:
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

cache_file = "article_cache.pkl"
expiration_length = datetime.timedelta(days=2)

try:
    with open(cache_file, "rb") as f:
        sentiment_cache = pickle.load(f)
except FileNotFoundError:
    sentiment_cache = {}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process a specified number of articles.')
    parser.add_argument('-n', '--num_articles', type=int, default=5, help='Number of articles to process')
    parser.add_argument('--print_cache', action='store_true', help='Print everything in the cache')
    parser.add_argument('--analyze_cache', action='store_true', help='Analyze cache sentiments and exit')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    args = parser.parse_args()

    # Update cache_file based on the given topic
    topic_lower = args.topic.lower()
    cache_file = f"article-{topic_lower}.pkl"

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
        analyze_cache_sentiments()
        exit()
    else:  
        googlenews = GoogleNews(period='1d')
        googlenews.get_news('Financial News')
        result = googlenews.result()

        num_articles = args.num_articles if args.num_articles <= len(result) else len(result)
        print(f"Processing {num_articles} articles...\n")

        for idx, article in enumerate(result[:num_articles]):
            title = article['title']
            url = "https://" + article['link']
            content = get_article_content(url, title)
            sentiment, confidence, _ = get_cached_sentiment_analysis(url, title, content, args)

            # Process sentiment analysis results
            if args.debug:
                print(f"Article {idx + 1}:")
                print(f"Title: {title}")
                print(f"Sentiment: {sentiment}")
                print("\n")

        # Analyze cache sentiments
        analyze_cache_sentiments()

        print_cache_info(args)

if __name__ == "__main__":
    main()
