# Agentic-AI-RAG-Chatbot
A simple RAG chatbot built with Python for the AI Engineering task.

Tech Stack
Python
LangGraph
Pinecone
Sentence Transformers
Groq
FastAPI
Flow
PDF → Chunks → Embeddings → Pinecone
                         ↓
Question → Retrieval → Groq → Answer
Setup
pip install -r requirements.txt

Create .env:

PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key

Run:

python -m app.vector_store
python -m app.upload_vector
uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs
Knowledge Base

Agentic AI – An Executive's Guide to In-depth Understanding of Agentic AI.

Author

Vaibhav Gaikwad
