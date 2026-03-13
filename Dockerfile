FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "-m", "app.gradio_app"]

# docker build -t book-recommender .
# docker run --env-file .env book-recommender python scripts/vector_db_build.py
# docker run -p 7860:7860 --env-file .env book-recommender