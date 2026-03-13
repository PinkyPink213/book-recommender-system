import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


DATA_PATH = "./data/tagged_description.txt"
VECTOR_DB_PATH = "./vector_db"


def load_documents():

    raw_documents = TextLoader(DATA_PATH).load()

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

    return documents


def build_vector_db():

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("Creating vector database...")

    db = Chroma.from_documents(
        documents,
        embeddings,
        collection_name="books",
        persist_directory=VECTOR_DB_PATH
    )

    db.persist()

    print("Vector DB created successfully!")
    print(f"Saved at: {VECTOR_DB_PATH}")


if __name__ == "__main__":
    build_vector_db()