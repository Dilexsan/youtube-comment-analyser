from fastapi import FastAPI, HTTPException, Depends
from pydantic import ValidationError
from models import VideoURLRequest, RawCommentList
from services.sentiment_analyzer import predict_sentiment_batch, calculate_positive_score_percentage, get_summary_and_suggestions
from services.youtube_downloader import YoutubeDownloader
from services.youtube_downloader import transform_comments_for_api

app = FastAPI()

# Initialize services
downloader = YoutubeDownloader()

@app.get("/")
def home():
    """A simple health check endpoint."""
    return {"message": "Model API is running."}

@app.post("/video-analyze")
def analyze_youtube_video(video_url_request: VideoURLRequest):
    print(f"Received request to analyze video: {video_url_request.video_url}")

    # Download comments
    try:
        raw_comments = downloader.get_comments_from_url(video_url_request.video_url)
    except Exception as e:
        print(f"Error downloading comments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while downloading comments: {e}"
        )

    total_comments = len(raw_comments)
    if not raw_comments:
        return {
            "total_comments": 0,
            "positive_score": 0.0,
            "summary": "No comments found for this video.",
            "suggestions": "Try analyzing a video with comments."
        }

    print(f"Successfully downloaded {total_comments} comments.")

    # Transform comments and predict sentiment
    try:
        transformed_comments = transform_comments_for_api(raw_comments)
        classified_comments = predict_sentiment_batch(transformed_comments)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction service error: {e}")

    # Calculate final score and generate summary
    positive_score = calculate_positive_score_percentage(classified_comments)
    summary, suggestions = get_summary_and_suggestions(classified_comments, positive_score)

    return {
        "total_comments": total_comments,
        "positive_score": positive_score,
        "summary": summary,
        "suggestions": suggestions
    }