from app.chunking import create_chunk
from app.ingest import load_pdf
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def create_embedding(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings= model.encode(
        texts,
        normalize_embeddings=True
    )
    return embeddings

if __name__ =="__main__":
    pages=load_pdf(PDF_PATH)
    chunks=create_chunk(pages)
    embeddings=create_embedding(chunks)

    print("Total chunks:", len(chunks))

    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])