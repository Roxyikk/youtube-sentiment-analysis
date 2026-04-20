# YouTube Comment Sentiment Analysis

This project analyzes sentiment of YouTube comments using YouTube API, MongoDB Atlas, Apache Spark, and Logistic Regression.

## Team Members

| Name | Role | Responsibilities | Code File |
|------|------|------------------|-----------|
| Tan Guangxi | Data Collection | YouTube API comment crawling | `YouTube.py` |
| Wang Qixu | Data Processing | Spark data cleaning, tokenization, feature extraction | `youtube_sentiment_analysis.ipynb` |
| Qi Yinxuan | Classification & Visualization | Model training, confusion matrix, word cloud | `classification_person4.py` |
| Huang Ziyuan | System Design & Quotation | System architecture diagram, hardware/software quotation | (no code) |

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
2. Install required packages: `pip install pymongo pyspark scikit-learn wordcloud textblob google-api-python-client matplotlib seaborn pandas`
3. Set up MongoDB Atlas and update CONNECTION_STRING in the code
4. Get YouTube API key and update API_KEY in YouTube.py
5. Run the scripts in order: YouTube.py -> youtube_sentiment_analysis.ipynb -> classification_person4.py
