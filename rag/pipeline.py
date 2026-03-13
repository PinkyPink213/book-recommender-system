from rag.query_expansion import query_expansion
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import book_recommend
from utils.query_builder import build_query


def retrieve_semantic_recommendations(describe, category, emotion, books):

    query = build_query(describe, category, emotion)

    queries = query_expansion(query)

    results = []

    for q in queries:
        docs = retrieve(q)
        results.extend(docs)

    unique_docs = {}

    for doc in results:

        isbn = doc.metadata["isbn"]

        if isbn not in unique_docs:
            unique_docs[isbn] = doc

    docs = list(unique_docs.values())

    top_docs = rerank(query, docs)

    context = ""

    for doc in top_docs:

        isbn = int(doc.metadata["isbn"])

        book = books[books["isbn13"] == isbn].iloc[0]

        context += f"""
        Title: {book['title']}
        Author: {book['authors']}
        ISBN: {book['isbn13']}
        Category: {book['categories']}
        Description: {doc.page_content}
        """

    return book_recommend(context)