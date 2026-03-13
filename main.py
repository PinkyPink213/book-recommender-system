import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
import gradio as gr
import json
from langchain_core.documents import Document
load_dotenv()


raw_documents = TextLoader("./data/tagged_description.txt").load()
# Split by newline because each line = 1 book
lines = raw_documents[0].page_content.split("\n")

documents = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    parts = line.split(" ", 1)
    isbn = parts[0]
    description = parts[1] if len(parts) > 1 else ""

    documents.append(
        Document(
            page_content=description,
            metadata={"isbn": isbn}
        )
    )
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

db_books = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="books",
    persist_directory="./vector_db"
)

def get_query(describe, category, emotion):
    if describe.strip():
        query = f"I want {describe} in the {category} category with {emotion} emotional tone."
    else:
        query = f"I want a book in the {category} category with {emotion} emotional tone."
    return query

def query_expansion(query):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini")

    prompt = f"""
Generate 3 alternative search queries similar to:
{query}

Return each query on a new line.
"""

    result = llm.invoke(prompt).content
    queries = result.split("\n")

    queries.append(query)

    return queries

def book_recommend(query, context):

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    prompt = f"""
    You are the book recommender assistant.

    Here are candidate books retrieved from the database:

    {context}

    Choose the best 3 books ONLY from the candidate list.
    Do not invent new books.

    Return the result in JSON format:

    {{
    "books":[
        {{
        "title": "...",
        "isbn": "...",
        "reason": "..."
        }}
    ]
    }}
    """

    answer = llm.invoke(prompt) 
    return answer.content
    
def book_reranker(query, docs, top_k=10):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:top_k]]
    return top_docs

def retrieve_semantic_recommendations(describe, category, emotion):

    query = get_query(describe, category, emotion)

    expand_queries = query_expansion(query)

    results = []

    for q in expand_queries:
        docs = db_books.similarity_search(q, k=3)
        results.extend(docs)

    unique_docs = {}

    for doc in results:
        isbn = doc.metadata["isbn"]
        if isbn not in unique_docs:
            unique_docs[isbn] = doc

    dedup_docs = list(unique_docs.values())

    top_docs = book_reranker(query, dedup_docs)

    context = ""

    for doc in top_docs:

        isbn = doc.metadata["isbn"].strip().replace('"', '')
        book = books[books["isbn13"] == int(isbn)].iloc[0]

        context += f"""
    Title: {book['title']}
    Author: {book['authors']}
    ISBN: {book['isbn13']}
    Category: {book['categories']}
    Description: {doc.page_content}

    """

    answer = book_recommend(query, context)

    return answer




def chat_recommend(describe, category, emotion, chat_history):
    books = pd.read_csv("./data/books_with_emotions.csv")

    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    answer = retrieve_semantic_recommendations(
        describe,
        category,
        emotion
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
        truncated_desc = " ".join(description.split()[:25]) + "..."

        authors_split = book["authors"].split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = book["authors"]

        caption = f"{book['title']} by {authors_str}\n{truncated_desc}"

        gallery.append((book["large_thumbnail"], caption))

        message += f"**{book['title']}** — {reason}\n\n"

    chat_history[-1] = (chat_history[-1][0], message)

    return chat_history, gallery

def add_user_message(describe, category, emotion, chat_history):

    query = get_query(describe, category, emotion)

    chat_history.append((query, None))

    return query, chat_history

books = pd.read_csv("./data/books_with_emotions.csv")
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    
    gr.Markdown("# 📚 Semantic Book Recommender")

    

    with gr.Row():
        describe = gr.Textbox(
            label="Describe a book you want",
            placeholder="e.g. forgiveness"
        )

    with gr.Row():
        category = gr.Dropdown(
            choices=["All"] + sorted(books["simple_categories"].unique()),
            label="Select a category",
            value="All"
        )

        emotion = gr.Dropdown(
            choices=["All","Happy","Surprising","Angry","Suspenseful","Sad"],
            label="Select emotional tone",
            value="All"
        )

    submit_button = gr.Button("Find recommendations")
    chatbot = gr.Chatbot(height=200)
    gr.Markdown("## 📚 Recommended Books")

    gallery = gr.Gallery(
        label="Recommended books",
        columns=6,
        rows=2,
        height="auto"
    )

    submit_button.click(
    fn=add_user_message,
    inputs=[describe, category, emotion, chatbot],
    outputs=[describe, chatbot]
).then(
    fn=chat_recommend,
    inputs=[describe, category, emotion, chatbot],
    outputs=[chatbot, gallery]
)

dashboard.launch(server_name="0.0.0.0", server_port=7860)