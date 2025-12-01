# schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Request for the simple predict endpoint (keeps your existing FastAPI shape)
class InputSchema(BaseModel):
    comments: List[str] = Field(
        ...,
        min_items=1,
        description="List of raw text comments to analyze.",
        example=["I love this product!", "The service was terrible."]
    )

    class Config:
        extra = "forbid"


# Response item for FastAPI-style predict endpoint
class OutputItem(BaseModel):
    comment: str
    prediction: float


class OutputSchema(BaseModel):
    results: List[OutputItem]


# ---- New models used by the migrated Flask endpoints ----

# Single comment with timestamp (for predict_with_timestamps)
class CommentWithTimestamp(BaseModel):
    text: str = Field(..., description="Raw comment text")
    timestamp: Optional[datetime] = Field(
        None, description="ISO timestamp for the comment (optional)"
    )


class PredictWithTimestampsRequest(BaseModel):
    comments: List[CommentWithTimestamp] = Field(..., min_items=1)


# Request for chart/wordcloud/trend endpoints
class SentimentCountsRequest(BaseModel):
    sentiment_counts: dict = Field(
        ...,
        description="Mapping of sentiment value -> count. Keys expected: '1','0','-1' (optional)"
    )


class WordCloudRequest(BaseModel):
    comments: List[str] = Field(..., min_items=1)


class TrendDataItem(BaseModel):
    comment: Optional[str]
    sentiment: int  # -1, 0, 1
    timestamp: datetime


class TrendRequest(BaseModel):
    sentiment_data: List[TrendDataItem] = Field(..., min_items=1)
