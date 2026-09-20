import pymupdf

PDF_PATH="data/Ebook-Agentic-AI.pdf"

def load_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages=[]

    for page_number,page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page":page_number+1,
                "text":text
            })

    doc.close()
    return pages
def search_pdf(pages, keyword):
    for page in pages:
        if keyword.lower() in page["text"].lower():
            print("\n" + "=" * 60)
            print("PAGE:", page["page"])
            print("=" * 60)
            print(page["text"][:3000])

if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)

    search_pdf(
    pages,
    "components"
)