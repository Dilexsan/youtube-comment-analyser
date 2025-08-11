from youtube_comment_downloader import YoutubeCommentDownloader
from typing import List
from models import RawComment
import sys
from models import RawComment
from typing import List

def transform_comments_for_api(raw_comments: List[RawComment]) -> list:
    """Transforms raw comment objects into a format suitable for the sentiment analysis API."""
    transformed_list = []
    for comment in raw_comments:
        transformed_list.append({
            "author": comment.author,
            "text": comment.text,
            "likes": int(comment.votes),
            "published_at": comment.time_parsed
        })
    return transformed_list


class YoutubeDownloader:
    """A service for downloading YouTube video comments."""
    def __init__(self):
        self._downloader = YoutubeCommentDownloader()
    
    def get_comments_from_url(self, video_url: str) -> List[RawComment]:
        """
        Downloads comments from a YouTube video URL.
        Sorts comments by top comments (sort_by=1).
        """
        try:
            # The downloader returns a generator, so we convert it to a list
            comment_generator = self._downloader.get_comments_from_url(video_url, sort_by=1)
            raw_comments_dict_list = list(comment_generator)
            
            # Pydantic validation on the raw comments
            raw_comments_list = [RawComment(**c) for c in raw_comments_dict_list]
            return raw_comments_list
        except Exception as e:
            # You might want to handle specific exceptions here
            raise Exception(f"Failed to download comments from URL {video_url}: {e}")