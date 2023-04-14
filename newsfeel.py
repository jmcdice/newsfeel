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
        cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash, cached_content = sentiment_cache[url]
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
    sentiment_cache[url] = (now, fsentiment, confidence, response, content_hash, content)


    # Save cache to disk
    with open(cache_file, "wb") as f:
        pickle.dump(sentiment_cache, f)

    return fsentiment, confidence, None

def analyze_cache_sentiments(cache_file, topic='', print_results=True):
    analysis_info = {}
    if not os.path.exists(cache_file):
        if print_results:
            print("Cache file does not exist. No sentiments to analyze.")
        return analysis_info

    sentiment_mapping = {'very bullish': 2, 'bullish': 1, 'neutral': 0, 'bearish': -1, 'very bearish': -2}
    sentiment_sum = 0
    confidence_sum = 0
    total_articles = 0

    with open(cache_file, 'rb') as f:
        sentiment_cache = pickle.load(f)

    sentiment_counts = {sentiment: 0 for sentiment in sentiment_mapping.keys()}

    for content_hash, (cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash, cached_content) in sentiment_cache.items():

        sentiment_key = cached_sentiment.lower()
        if sentiment_key not in sentiment_counts:
            sentiment_counts[sentiment_key] = 0
        sentiment_counts[sentiment_key] += 1
        sentiment_sum += sentiment_mapping.get(sentiment_key, 0)
        confidence_sum += cached_confidence
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
            print('No articles found in cache for sentiment analysis.')

    return analysis_info

def analyze_summaries(cache_file, topic=''):
    if not os.path.exists(cache_file):
        print("Cache file does not exist. No summaries to analyze.")
        return

    with open(cache_file, 'rb') as f:
        sentiment_cache = pickle.load(f)

    summaries = []

    for _, (_, cached_sentiment, _, cached_response, _, _) in sentiment_cache.items():
        summary_entry = f"Sentiment: {cached_sentiment}, Response: {cached_response}"
        summaries.append(summary_entry)

    all_summaries = '\n'.join(summaries)

    sentiment_analysis = analyze_cache_sentiments(cache_file, topic, print_results=False)

    sentiment_analysis_text = (f"Based on the analysis of {sentiment_analysis['total_articles']} articles, "
                               f"the general sentiment for {topic} is {sentiment_analysis['sentiment_result']} "
                               f"with an average confidence of {sentiment_analysis['average_confidence']:.2f}. "
                               f"The weighted sentiment is {sentiment_analysis['weighted_sentiment']:.2f}. "
                               f"Sentiment counts are as follows:\n")
    for sentiment, count in sentiment_analysis['sentiment_counts'].items():
        sentiment_analysis_text += f"{sentiment.capitalize()}: {count}\n"

    context_text = (f"Please provide a summary in financial analysis style, including future-looking predictions for the "
                    f"topic '{topic}', based on the following summaries of articles and sentiment analysis:\n\n"
                    f"{sentiment_analysis_text}\n"
                    f"Article Sentiments and Responses:\n{all_summaries}")

    summary_analysis = send_query(all_summaries, context_text)

    return summary_analysis


def print_cache_info(args):
    now = datetime.datetime.now()
    total_articles = len(sentiment_cache)
    expired_count = 0

    print(f"Total articles in cache: {total_articles}")

    for url, (cached_time, cached_sentiment, cached_confidence, cached_response, cached_content_hash, cached_content) in sentiment_cache.items():
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
            print(f"Content: {cached_content}")  # Add this line to print the cached content
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
    parser.add_argument('--analyze_summaries', action='store_true', help='Analyze summaries of cached articles and exit')  # New argument
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
    elif args.analyze_summaries:  # Analyze summaries of cached articles and exit
        summary_analysis = analyze_summaries(cache_file, main_topic)
        print(summary_analysis)
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
