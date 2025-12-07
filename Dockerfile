FROM python:3.12-slim

WORKDIR /app

# Install system packages needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY Backend/ /app/Backend/

# Copy ONLY the Backend requirements.txt
COPY Backend/requirements.txt /app/requirements.txt

# Copy vectorizer (if still needed)
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data
RUN python -m nltk.downloader stopwords wordnet punkt

# Expose port 5000
EXPOSE 5000

# Run fastapi on port 5000
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "5000"]
