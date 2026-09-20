import os
from dotenv import load_dotenv
from pinecone import Pinecone

from app.ingest import load_pdf
from app.chunking import create_chunk
from app.embeddings import create_embedding

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "agentic-ai-rag"

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def upload_vectors():

    # Load PDF
    pages = load_pdf("data/Ebook-Agentic-AI.pdf")

    # Create chunks
    chunks = create_chunk(pages)

    # Create embeddings
    embeddings = create_embedding(chunks)

    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"]
            }
        })

    # Store vectors
    index.upsert(vectors=vectors)

    print(f"Uploaded {len(vectors)} vectors to Pinecone.")


if __name__ == "__main__":
    upload_vectors()