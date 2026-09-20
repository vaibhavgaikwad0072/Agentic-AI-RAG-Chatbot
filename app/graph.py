from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.retrieve import retrieve
from app.rag import llm


class RAGState(TypedDict):
    question: str
    context: list
    answer: str
    confidence: float


def retrieve_context(state: RAGState):

    results = retrieve(
        state["question"],
        top_k=3
    )

    matches = results["matches"]

    if not matches:
        return {
            "context": [],
            "confidence": 0.0
        }

    confidence = sum(
        match["score"] for match in matches
    ) / len(matches)

    
    MIN_SCORE = 0.30

    relevant_matches = [
        match
        for match in matches
        if match["score"] >= MIN_SCORE
    ]

    if not relevant_matches:
        return {
            "context": [],
            "confidence": round(confidence, 4)
        }

    return {
        "context": relevant_matches,
        "confidence": round(confidence, 4)
    }


def generate_answer(state: RAGState):

    context_text = ""

    for match in state["context"]:
        context_text += (
            f"\nPage {match['metadata']['page']}:\n"
            f"{match['metadata']['text']}\n"
        )

    prompt = f"""
    You are an assistant for the Agentic AI eBook.

    Answer the question using ONLY the provided context.

    If the answer is not available in the context, say:
    "I could not find this information in the provided knowledge base."

    Do not use outside knowledge.
    Do not make up information.

    Context:
    {context_text}

    Question:
    {state["question"]}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }



graph_builder = StateGraph(RAGState)


graph_builder.add_node("retrieve", retrieve_context)
graph_builder.add_node("generate", generate_answer)


graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)


graph = graph_builder.compile()


if __name__ == "__main__":

    question = "What is Agentic AI?"

    result = graph.invoke({
        "question": question
    })

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:")
    print(round(result["confidence"], 4))

    print("\nRetrieved Context:")

    for i, match in enumerate(result["context"], start=1):
        print(f"\n--- Chunk {i} ---")
        print("Page:", match["metadata"]["page"])
        print("Score:", match["score"])