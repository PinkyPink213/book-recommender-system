FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir gradio

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "-m", "app.gradio_app"]

# docker build -t book-recommender .
# docker run --env-file .env book-recommender python scripts/vector_db_build.py
# docker run -p 7860:7860 --env-file .env book-recommender