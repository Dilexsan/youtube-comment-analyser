import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")


# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

# Load the model and tokenizer from the local folder
def load_model():
    """Loads the sentiment analysis model and tokenizer."""
    global tokenizer, model, device
    try:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Model could not be loaded. Check your MODEL_PATH and files.")

# Load model at application startup
load_model()

def predict_sentiment_batch(comments: list):
    """Predicts sentiment for a batch of comments."""
    if not model or not tokenizer:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check the model path and files."
        )

    texts = [c['text'] for c in comments]
    
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    predicted_class_ids = probabilities.argmax(dim=1)
    
    classified_comments = []
    for i, comment in enumerate(comments):
        label = model.config.id2label[predicted_class_ids[i].item()]
        status = "p" if label == "LABEL_1" else "n"
        
        classified_comments.append({
            "author": comment['author'],
            "text": comment['text'],
            "likes": comment['likes'],
            "time": comment['published_at'],
            "status": status,
        })
    
    return classified_comments

def calculate_positive_score_percentage(classified_comments: list):
    """Calculates the weighted positive score percentage."""
    positive_total = 0
    negative_total = 0

    for comment in classified_comments:
        if 'status' in comment and 'likes' in comment:
            weight = comment['likes'] + 1
            if comment['status'] == 'p':
                positive_total += weight
            elif comment['status'] == 'n':
                negative_total += weight

    total = positive_total + negative_total
    if total == 0:
        return 0.0
    
    positive_score_percent = (positive_total / total) * 100
    return round(positive_score_percent, 2)

def get_summary_and_suggestions(classified_comments: list, positive_score: float):
    """Generates a summary and suggestions based on sentiment analysis."""
    num_positive = sum(1 for c in classified_comments if c['status'] == 'p')
    num_negative = len(classified_comments) - num_positive
    
    summary = (
        f"The video received {num_positive} positive and {num_negative} negative comments. "
        f"The overall sentiment is {'very positive' if positive_score > 75 else 'positive' if positive_score > 50 else 'mixed' if positive_score > 25 else 'negative'}."
    )
    
    suggestions = (
        "Based on the analysis, keep creating content with a similar style, as it resonates well with your audience. "
        "Engage with top positive comments to build community. Consider addressing common negative feedback in future videos."
    )
    
    return summary, suggestions