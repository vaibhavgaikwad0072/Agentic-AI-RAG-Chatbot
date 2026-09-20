import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.retrieve import retrieve

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)


def generate_answer(question):

    # Retrieve relevant chunks from Pinecone
    results = retrieve(question, top_k=3)

    context = ""

    for match in results["matches"]:
        context += (
            f"\nPage {match['metadata']['page']}:\n"
            f"{match['metadata']['text']}\n"
        )

    prompt = f"""
    You are a helpful assistant answering questions about the Agentic AI eBook.

    Answer the user's question using ONLY the context provided below.

    If the answer cannot be found in the context, say:
    "I could not find this information in the provided knowledge base."

    Do not use outside knowledge.
    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "context": results["matches"]
    }


if __name__ == "__main__":

    question = "What is Agentic AI?"

    result = generate_answer(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved Context:")

    for i, match in enumerate(result["context"], start=1):
        print(f"\n--- Chunk {i} ---")
        print("Page:", match["metadata"]["page"])
        print("Score:", match["score"])
        print(match["metadata"]["text"])