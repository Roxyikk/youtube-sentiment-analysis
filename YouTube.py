from googleapiclient.discovery import build
import json
import csv
from datetime import datetime

API_KEY = "AIzaSyDpviEJ0j5y5slCUMSe-LVqQXOGHkEfGD8"
VIDEO_ID = "73_1biulkYk"
MAX_COMMENTS = 500

youtube = build("youtube", "v3", developerKey=API_KEY)

def get_youtube_comments(video_id, max_comments):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            cmt = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "author": cmt["authorDisplayName"],
                "text": cmt["textDisplay"],
                "like_count": cmt["likeCount"],
                "published_at": cmt["publishedAt"]
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_comments]

if __name__ == "__main__":
    print("Collecting comment data...")
    data = get_youtube_comments(VIDEO_ID, MAX_COMMENTS)

    json_filename = f"youtube_comments_{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    csv_filename = f"youtube_comments_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(csv_filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"\n✅ Data collection completed!")
    print(f"📄 Total comments crawled: {len(data)}")
    print(f"📁 JSON file: {json_filename}")
    print(f"📁 CSV file: {csv_filename}")