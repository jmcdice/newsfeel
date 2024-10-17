# Financial News Sentiment Analyzer

## Overview

This Python script analyzes the sentiment of financial news articles using the Google News API and OpenAI's GPT-3.5 model. It fetches articles based on a specified topic, analyzes their sentiment, and provides a summary of the overall sentiment trends.

## Features

- Fetch news articles from Google News based on a specified topic
- Analyze sentiment of articles using OpenAI's GPT-3.5 model
- Cache sentiment analysis results to avoid redundant API calls
- Provide summary analysis of sentiment trends
- Output results to CSV file (optional)
- Command-line interface for easy usage and customization

## Requirements

- Python 3.6+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Required Python packages (install via `pip install -r requirements.txt`):
  - GoogleNews
  - newspaper3k
  - tqdm
  - openai

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jmcdice/newsfeel.git
   cd newsfeel
   ```

2. Install the required packages:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the script using the following command:

```
python newsfeel.py [options]
```

### Options:

- `-n`, `--num_articles`: Number of articles to process (default: 5)
- `--print_cache`: Print everything in the cache
- `--analyze_cache`: Analyze cache sentiments and exit
- `--analyze_summaries`: Analyze summaries of cached articles and exit
- `-t`, `--topic`: Topic to fetch news for (default: "Financial News")
- `--loglevel`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `-o`, `--output_file`: File to write output results to (CSV format)

### Examples:

1. Analyze 10 articles about "Bitcoin":
   ```
   python newsfeel.py -n 10 -t "Bitcoin"
   ```

2. Analyze cache sentiments for previously fetched "Stock Market" articles:
   ```
   python newsfeel.py --analyze_cache -t "Stock Market"
   ```

3. Generate a summary analysis of cached "Cryptocurrency" articles:
   ```
   python newsfeel.py --analyze_summaries -t "Cryptocurrency"
   ```

4. Output results to a CSV file:
   ```
   python newsfeel.py -n 20 -t "Gold Price" -o results.csv
   ```

## Output

The script provides the following output:

1. Sentiment analysis for each processed article
2. Overall sentiment analysis summary, including:
   - General sentiment (Very Bearish, Bearish, Neutral, Bullish, Very Bullish)
   - Total number of articles analyzed
   - Average confidence score
   - Sentiment distribution
   - Weighted sentiment score

## Caching

The script uses a caching mechanism to store sentiment analysis results. This helps to:
- Reduce API calls to OpenAI
- Speed up subsequent runs on the same topic
- Allow for offline analysis of previously fetched articles

Cache files are stored in the `cache` directory with filenames based on the analyzed topic.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

