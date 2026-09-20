import os

from dotenv import load_dotenv
from pinecone import Pinecone

from app.embeddings import model

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "agentic-ai-rag"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def retrieve(query, top_k=3):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results


if __name__ == "__main__":

    query = "What are the components of an agentic AI system?"

    results = retrieve(query)

    print("\nQuery:")
    print(query)

    print("\nRetrieved chunks:\n")

    for i, match in enumerate(results["matches"], start=1):

        print(f"--- Result {i} ---")
        print("Score:", match["score"])
        print("Page:", match["metadata"]["page"])
        print("Text:")
        print(match["metadata"]["text"])
        print()