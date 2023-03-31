# NewsFeel
## NewsFeel: A Sentiment Analysis Tool for Financial News

This repo contains a Python script for analyzing the sentiment of financial news articles. The script uses OpenAI's GPT-3 API to generate a sentiment analysis of each article, and caches the results to improve performance. It now supports custom topics for fetching news articles.

## Installation
Clone the repo:
```console
  git clone https://github.com/username/repo.git
```

Install the required packages:
```console
  pip install -r requirements.txt
```

Set up your OpenAI API key as an environment variable:
```console
  export OPENAI_API_KEY=your_api_key
```

## Usage
The script can be run using the following command:

```console
  python3 newsfeel.py [-h] [-n NUM_ARTICLES] [--print_cache] [--analyze_cache] [--debug] [-t TOPIC]
```

The optional arguments are:

- -n NUM_ARTICLES: the number of articles to process (default is 5)
- -t TOPIC: the topic to fetch news for (default is 'Financial News')
- --print_cache: print all cached sentiment analysis results
- --analyze_cache: analyze sentiment analysis results in the cache and exit
- --debug: print debug info

When the script is run, it fetches news articles based on the provided topic from Google News and processes each one using the GPT-3 API. The sentiment analysis results are printed to the console. Results are cached to improve performance, and the cache can be printed or analyzed using the optional arguments.

Note: before running the script, be sure to set up your OpenAI API key as an environment variable.

## Example Usage

```console
  ./newsfeel.py -n 100 --topic "Tesla" --debug
```

When running the command ./newsfeel.py -n 100 --topic "Tesla" --debug, the newsfeel.py script processes news articles related to the topic "Tesla" and performs sentiment analysis on them. The command-line arguments provided are:

`-n 100`: This argument specifies the number of articles to process. In this case, the script will process up to 100 articles.

`--topic "Tesla"`: This argument sets the topic for which the script fetches news articles. Here, the script will fetch articles related to "Tesla".

`--debug`: This argument enables the debug mode, which makes the script print additional information during its execution, such as the title of each article, its sentiment, and any other relevant debug information.

So, the command fetches up to 100 news articles related to "Tesla" and performs sentiment analysis on each article. It then caches the sentiment analysis results and provides an overall sentiment summary based on the individual article sentiments. In debug mode, the script will also print more information about each article being processed.

```console
  # First, build pickle cache

  ./newsfeel.py -n 100 --topic "Tesla" --debug
  https://news.google.com/search?q=Tesla%2Bwhen%3A1d&hl=en
  Processing 100 articles...

  ...

```

Then, analyze the cache:

```console

./newsfeel.py --analyze_cache -t Tesla
Sentiment Analysis for: Tesla

General Sentiment: Bullish
Total Articles: 103
Average Confidence: 7.49

Sentiment Counts:
 55 (53.40%) Sentiment: Bullish
 23 (22.33%) Sentiment: Neutral
 17 (16.50%) Sentiment: Bearish
  6 (5.83%) Sentiment: Unknown
  2 (1.94%) Sentiment: Very Bullish
```
