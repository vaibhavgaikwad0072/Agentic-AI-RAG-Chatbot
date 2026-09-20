import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "agentic-ai-rag"
DIMENSION = 384


def create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created.")

    else:
        print("Pinecone index already exists.")

    return pc.Index(INDEX_NAME)


if __name__ == "__main__":
    index = create_index()

    print("Connected to Pinecone successfully.")
    print("Index:", INDEX_NAME)