# YouTube Comment Sentiment Analysis

This project analyzes sentiment of YouTube comments using:
- YouTube API
- MongoDB Atlas
- Apache Spark
- Logistic Regression

## Team Members

| Name | Role | Code File |
|------|------|------------|
| Tan Guangxi | Data Collection | `YouTube.py` |
| Qi Yinxuan | Classification & Visualization | `classification_person4.py` |
| Wang Qixu | Data Processing (Spark + MongoDB) | `youtube_sentiment_analysis.ipynb` |

## Project Overview

1. **Data Collection** - Collect comments from YouTube videos using YouTube API v3
2. **Data Processing** - Clean text, tokenize, remove stopwords, extract features using Spark
3. **Classification** - Train Logistic Regression model for sentiment analysis (Positive/Negative)
4. **Visualization** - Confusion matrix, sentiment distribution, word cloud

## Data Source

- Video 1: Black Myth: Wukong Trailer (500 comments)
- Video 2: Deadpool & Wolverine Official Trailer (500 comments)

## Results

| Video | Accuracy | Positive | Negative |
|-------|----------|----------|----------|
| Black Myth: Wukong | 77.00% | 192 | 308 |
| Deadpool & Wolverine | 79.00% | 156 | 344 |

## How to Run

1. Clone this repository
2. Install required packages: `pip install pymongo pyspark sklearn wordcloud textblob`
3. Run the notebooks/scripts in order
