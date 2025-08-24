from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import answer_from_rag
from app.guards import allowed

app = FastAPI(title="Banking Ops RAG Assistant", version="0.1.0")

class Ask(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: Ask):
    if not allowed(req.question):
        raise HTTPException(status_code=400, detail="Blocked by guardrails")
    return {"answer": answer_from_rag(req.question)}
