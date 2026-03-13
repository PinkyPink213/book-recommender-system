import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

VECTOR_DB_PATH = "./vector_db"

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

db_books = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)


def retrieve(query, k=20):

    return db_books.similarity_search(query, k=k)