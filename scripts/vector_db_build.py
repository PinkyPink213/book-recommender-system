from pathlib import Path
import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "tagged_description.txt"
VECTOR_DB_PATH = BASE_DIR / "vector_db"


# --------------------------------------------------
# Load documents
# --------------------------------------------------

def load_documents():

    print("Reading file:", DATA_PATH)

    raw_documents = TextLoader(str(DATA_PATH)).load()

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


# --------------------------------------------------
# Build vector DB
# --------------------------------------------------

def build_vector_db():

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    print("Checking OpenAI API key...")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    print("Creating embeddings model...")

    embeddings = OpenAIEmbeddings(api_key=api_key)

    # remove old DB to avoid corruption
    if VECTOR_DB_PATH.exists():
        print("Removing old vector DB...")
        shutil.rmtree(VECTOR_DB_PATH)

    print("Creating vector database...")

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="books",
        persist_directory=str(VECTOR_DB_PATH)
    )

    print("Inserted docs:", vector_db._collection.count())

    print("Vector DB created successfully!")
    print("Saved at:", VECTOR_DB_PATH)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    build_vector_db()