# ==================================================
# STAT2630SEF Group Project - Sentiment Classification
# Person 4: Classification & Visualization
# This script processes TWO videos separately.
# ==================================================

# Step 1: Install required packages
!pip install pandas numpy matplotlib seaborn scikit-learn wordcloud textblob -q

# Step 2: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from wordcloud import WordCloud
from textblob import TextBlob
import io
from google.colab import files

print("All libraries loaded successfully.")

# ==================================================
# Function to process one video file
# ==================================================
def process_video(video_name, text_column='text_clean', label_column=''):
    """
    Process a single video CSV file:
    - Generate sentiment labels if missing
    - Train Logistic Regression model
    - Generate and save three visualizations
    - Return summary statistics
    """
    print(f"\n{'='*50}")
    print(f"Processing: {video_name}")
    print('='*50)
    
    # Upload file
    print(f"\nPlease upload the CSV file for {video_name}:")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"Loaded: {len(df)} rows")
    
    # Generate labels if needed
    if label_column == '' or label_column not in df.columns:
        print("No label column. Generating sentiment labels using TextBlob...")
        def get_sentiment(text):
            analysis = TextBlob(str(text))
            return 1 if analysis.sentiment.polarity > 0 else 0
        df['label'] = df[text_column].apply(get_sentiment)
        label_col = 'label'
    else:
        label_col = label_column
        print(f"Using existing label column: {label_col}")
    
    # Drop missing values
    df = df.dropna(subset=[text_column])
    print(f"Valid rows after cleaning: {len(df)}")
    
    # Label distribution
    pos_count = df[df[label_col] == 1].shape[0]
    neg_count = df[df[label_col] == 0].shape[0]
    print(f"Positive comments: {pos_count}")
    print(f"Negative comments: {neg_count}")
    
    # TF-IDF feature extraction
    print("\nExtracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X = tfidf.fit_transform(df[text_column].astype(str))
    y = df[label_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")
    
    # Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.2%}")
    
    # Confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Values:")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    # Visualization 1: Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {video_name}', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{video_name}.png'
    plt.savefig(cm_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {cm_filename}")
    
    # Visualization 2: Sentiment Distribution
    plt.figure(figsize=(6,4))
    counts = [neg_count, pos_count]
    labels = ['Negative', 'Positive']
    colors = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(labels, counts, color=colors)
    plt.title(f'Sentiment Distribution - {video_name}', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Comments', fontsize=12)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(count), ha='center', fontsize=12)
    plt.tight_layout()
    dist_filename = f'sentiment_distribution_{video_name}.png'
    plt.savefig(dist_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {dist_filename}")
    
    # Visualization 3: Word Cloud (Positive Comments)
    pos_comments = df[df[label_col] == 1][text_column].astype(str)
    if len(pos_comments) > 0:
        pos_text = " ".join(pos_comments)
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='viridis', max_words=100).generate(pos_text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - Positive Comments - {video_name}', fontsize=14)
        plt.tight_layout()
        wc_filename = f'wordcloud_positive_{video_name}.png'
        plt.savefig(wc_filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {wc_filename}")
    else:
        wc_filename = None
        print("No positive comments for word cloud.")
    
    # Download images
    files.download(cm_filename)
    files.download(dist_filename)
    if wc_filename:
        files.download(wc_filename)
    
    # Return summary
    return {
        'video': video_name,
        'total': len(df),
        'positive': pos_count,
        'negative': neg_count,
        'accuracy': acc,
        'cm': (tn, fp, fn, tp)
    }

# ==================================================
# Process Video 1 and Video 2
# ==================================================

# Modify the text column name if needed
TEXT_COL = 'text_clean'   # <-- Change this to match your CSV column name

print("\n" + "="*60)
print("STARTING ANALYSIS FOR TWO VIDEOS")
print("="*60)

# Process first video
result1 = process_video(video_name="Video1", text_column=TEXT_COL)

# Process second video
result2 = process_video(video_name="Video2", text_column=TEXT_COL)

# ==================================================
# Summary Table
# ==================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Video':<10} {'Total':<8} {'Positive':<10} {'Negative':<10} {'Accuracy':<10}")
print("-"*50)
for res in [result1, result2]:
    print(f"{res['video']:<10} {res['total']:<8} {res['positive']:<10} {res['negative']:<10} {res['accuracy']:.2%}")
print("="*60)
