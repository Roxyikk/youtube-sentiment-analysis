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

1. **Clone this repository** - Run `git clone https://github.com/Roxyikk/youtube-sentiment-analysis.git` and `cd youtube-sentiment-analysis`

2. **Install required packages** - Run `pip install pymongo pyspark scikit-learn wordcloud textblob google-api-python-client matplotlib seaborn pandas`

3. **Set up MongoDB Atlas** - Create a free cluster at mongodb.com/atlas, get your connection string, and update `CONNECTION_STRING` in the code

4. **Get YouTube API key** - Enable YouTube Data API v3 in Google Cloud Console, create an API key, and update `YOUR_API_KEY` in `YouTube.py`

5. **Run the scripts in order** - First `python YouTube.py` to collect comments, then run `youtube_sentiment_analysis.ipynb` in Colab to process data with Spark, finally `python classification_person4.py` to train model and generate visualizations

## Repository Structure

```
youtube-sentiment-analysis/
├── YouTube.py                          # Data collection (Person 2)
├── youtube_sentiment_analysis.ipynb    # Spark processing (Person 3)
├── classification_person4.py           # Model training (Person 4)
└── README.md                           # This file
```
