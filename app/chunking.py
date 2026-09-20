from app.ingest import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

def create_chunk(pages):

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )

    chunks=[]

    for page in pages:
        page_chunks=splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text":chunk,
                "page":page["page"]
            })

    return chunks

if __name__ == "__main__":

    pages = load_pdf(PDF_PATH)

    chunks = create_chunk(pages)

    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))
    for i, chunk in enumerate(chunks):
        if "Goals:" in chunk["text"] or "Environment:" in chunk["text"]:
            print("\n" + "=" * 70)
            print("CHUNK:", i)
            print("PAGE:", chunk["page"])
            print("=" * 70)
            print(chunk["text"])