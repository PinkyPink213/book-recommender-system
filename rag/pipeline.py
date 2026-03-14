from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rag.query_expansion import query_expansion
from rag.reranker import rerank
from rag.generator import book_recommend
from utils.query_builder import build_query


def retrieve_semantic_recommendations(describe, category, books, db_books):

    query = build_query(describe, category)

    queries = query_expansion(query)

    results = []

    # vector search
    for q in queries:

        docs = db_books.similarity_search(q, k=5)

        results.extend(docs)

    # remove duplicates
    unique_docs = {}

    for doc in results:

        isbn = doc.metadata["isbn"]

        if isbn not in unique_docs:
            unique_docs[isbn] = doc

    docs = list(unique_docs.values())

    # --------------------------
    # dataframe metadata filter
    # --------------------------

    filtered_docs = []

    for doc in docs:

        isbn_str = ''.join(filter(str.isdigit, doc.metadata["isbn"]))
        isbn = int(isbn_str)

        book_row = books[books["isbn13"] == isbn]

        if book_row.empty:
            continue

        book = book_row.iloc[0]

        if category != "All" and book["simple_categories"] != category:
            continue

        filtered_docs.append(doc)

    # rerank
    top_docs = rerank(query, filtered_docs, top_k=7)

    # build context
    context = ""

    for doc in top_docs:

        isbn_str = ''.join(filter(str.isdigit, doc.metadata["isbn"]))
        isbn = int(isbn_str)

        book = books[books["isbn13"] == isbn].iloc[0]

        context += f"""
        BOOK
        Title: {book['title']}
        Author: {book['authors']}
        ISBN: {book['isbn13']}
        Category: {book['categories']}
        Description: {doc.page_content}

        """

    return book_recommend(context,query)