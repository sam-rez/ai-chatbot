from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAGEngine

app = FastAPI()

rag = RAGEngine()
print("✅ RAGEngine initialized")

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/chat")
def chat(req: ChatRequest):
    return rag.answer(req.question)
