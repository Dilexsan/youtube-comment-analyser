from pydantic import BaseModel, RootModel
from typing import List

# Pydantic models for data validation
class Comment(BaseModel):
    author: str
    text: str
    likes: int
    published_at: float

class CommentListRequest(BaseModel):
    comments: List[Comment]

class RawComment(BaseModel):
    cid: str
    text: str
    time: str
    author: str
    channel: str
    votes: str
    replies: str
    photo: str
    heart: bool
    reply: bool
    time_parsed: float

class RawCommentList(RootModel):
    root: List[RawComment]

class VideoURLRequest(BaseModel):
    video_url: str