import os
import gradio as gr
import subprocess
from pathlib import Path
import json 

from rag.pipeline import retrieve_semantic_recommendations
from utils.data_loader import load_books
from utils.query_builder import build_query

import shutil
import subprocess

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = BASE_DIR / "vector_db"
BUILD_SCRIPT = BASE_DIR / "scripts" / "vector_db_build.py"


def ensure_vector_db():

    folder_exists = VECTOR_DB_PATH.exists()
    folder_not_empty = folder_exists and any(VECTOR_DB_PATH.iterdir())

    doc_count = 0

    if folder_not_empty:

        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        db = Chroma(
            collection_name="books",
            persist_directory=str(VECTOR_DB_PATH),
            embedding_function=embeddings
        )

        doc_count = db._collection.count()

    print("Vector DB path:", VECTOR_DB_PATH)
    print("Folder exists:", folder_exists)
    print("Folder not empty:", folder_not_empty)
    print("Docs in DB:", doc_count)

    if not folder_exists or not folder_not_empty or doc_count == 0:

        print("Vector DB missing or empty. Rebuilding...")

        if folder_exists:
            shutil.rmtree(VECTOR_DB_PATH)

        subprocess.run(["python", str(BUILD_SCRIPT)], check=True)

        print("Vector DB build complete")

    else:
        print("Vector DB already populated.")
        
def load_db():
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return Chroma(
        collection_name="books",
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings
    )
    
def chat_recommend(describe, category, chat_history):
    books = load_books()
    answer = retrieve_semantic_recommendations(
        describe,
        category,
        books,
        db_books
    )

    answer = answer.strip()

    if answer.startswith("```"):
        answer = answer.replace("```json", "").replace("```", "").strip()

    data = json.loads(answer)

    gallery = []
    message = "Here are some books you might enjoy:\n\n"

    for item in data["books"]:

        isbn = int(item["isbn"])
        reason = item["reason"]

        book = books[books["isbn13"] == isbn].iloc[0]

        description = book["description"]
        truncated = " ".join(description.split()[:25]) + "..."

        caption = f"{book['title']} by {book['authors']}\n{truncated}"

        gallery.append((book["large_thumbnail"], caption))

        message += f"**{book['title']}** — {reason}\n\n"

    chat_history.append({
        "role": "assistant",
        "content": message
    })

    return chat_history, gallery


def add_user_message(describe, category, chat_history):

    query = build_query(describe, category)

    chat_history.append({
        "role": "user",
        "content": query
    })

    return describe, chat_history


def create_ui():

    books = load_books()

    with gr.Blocks() as dashboard:

        gr.Markdown("# 📚 Semantic Book Recommender")

        describe = gr.Textbox(label="Describe a book")

        category = gr.Dropdown(
            choices=["All"] + sorted(books["simple_categories"].unique()),
            value="All",
            label="Category"
        )


        submit = gr.Button("Find recommendations")

        chatbot = gr.Chatbot(height=200)

        gallery = gr.Gallery(columns=6)

        submit.click(
            fn=add_user_message,
            inputs=[describe, category, chatbot],
            outputs=[describe, chatbot]
        ).then(
            fn=chat_recommend,
            inputs=[describe, category, chatbot],
            outputs=[chatbot, gallery]
        )

    return dashboard


def main():

    ensure_vector_db()
    global db_books
    db_books = load_db()
    dashboard = create_ui()

    port = int(os.environ.get("PORT", 7860))

    dashboard.launch(
        server_name="0.0.0.0",
        server_port=port
    )


if __name__ == "__main__":
    main()