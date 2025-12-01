# main.py
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .schemas import (
    InputSchema, OutputSchema, OutputItem,
    PredictWithTimestampsRequest, WordCloudRequest,
    SentimentCountsRequest, TrendRequest
)
from .predict import (
    predict_comments, predict_with_timestamps,
    generate_pie_chart_bytes, generate_wordcloud_bytes,
    generate_trend_graph_bytes
)

import io

# ---------------------------
# FastAPI App Initialization
# ---------------------------
app = FastAPI(
    title="Sentiment Prediction API",
    description="API for predicting sentiment & generating charts/wordclouds.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Utility Endpoints
# ---------------------------
@app.get("/", tags=["Health"])
async def root():
    return {"message": "Sentiment Prediction API is up and running."}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


# ---------------------------
# Prediction Endpoints
# ---------------------------
@app.post("/predict", response_model=OutputSchema, tags=["Prediction"])
async def predict(request: InputSchema) -> OutputSchema:
    """
    Keep existing FastAPI schema output:
    { "results": [ { "comment": "...", "prediction": 1.0 }, ... ] }
    """
    try:
        preds = predict_comments(request.comments)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re)) from re
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    results = [OutputItem(comment=c, prediction=p) for c, p in zip(request.comments, preds)]
    return OutputSchema(results=results)


@app.post("/predict_with_timestamps", response_model=None, tags=["Prediction"])
async def predict_with_timestamps_endpoint(request: PredictWithTimestampsRequest):
    """
    Mirrors your original Flask endpoint (Option A):
    Returns a JSON array of {"comment":..., "sentiment": "...", "timestamp": ...}
    """
    try:
        # prepare list of dicts as expected by predict_with_timestamps
        items = []
        for itm in request.comments:
            # Pydantic converted timestamp to datetime (or None). We will output the original ISO if present.
            ts = itm.timestamp.isoformat() if itm.timestamp is not None else None
            items.append({"text": itm.text, "timestamp": ts})

        response_list = predict_with_timestamps(items)
        # Return a JSON array (not wrapped in "results") to match Flask original
        return JSONResponse(content=response_list, status_code=200)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re)) from re
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


# ---------------------------
# Image / Visualization Endpoints
# ---------------------------
@app.post("/generate_chart", tags=["Visualization"])
async def generate_chart_endpoint(request: SentimentCountsRequest):
    """
    Accepts JSON: {"sentiment_counts": {"1": 10, "0": 5, "-1": 2}}
    Returns PNG pie chart.
    """
    try:
        img_bytes = generate_pie_chart_bytes(request.sentiment_counts)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {exc}") from exc


@app.post("/generate_wordcloud", tags=["Visualization"])
async def generate_wordcloud_endpoint(request: WordCloudRequest):
    """
    Accepts JSON: {"comments": ["a", "b", ...]}
    Returns PNG wordcloud image.
    """
    try:
        img_bytes = generate_wordcloud_bytes(request.comments)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {exc}") from exc


@app.post("/generate_trend_graph", tags=["Visualization"])
async def generate_trend_graph_endpoint(request: TrendRequest):
    """
    Accepts JSON: {"sentiment_data": [{"comment": "...", "sentiment": 1, "timestamp": "2024-01-01T..."}]}
    Returns PNG trend graph.
    """
    try:
        # convert Pydantic TrendDataItem objects to simple dicts (timestamp -> ISO)
        data = []
        for item in request.sentiment_data:
            data.append({
                "comment": item.comment,
                "sentiment": int(item.sentiment),
                "timestamp": item.timestamp.isoformat()
            })

        img_bytes = generate_trend_graph_bytes(data)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {exc}") from exc
