from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import graph

app = FastAPI(
    title="RAG Chatbot",
    version="1.0"
)
class ChatRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "message": "RAG Chatbot API is running"
    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({
        "question": request.question
    })

    context = []

    for match in result["context"]:
        context.append({
            "text": match["metadata"]["text"],
            "page": match["metadata"]["page"],
            "score": match["score"]
        })

    return {
        "answer": result["answer"],
        "context": context,
        "confidence": round(result["confidence"], 4)
    }