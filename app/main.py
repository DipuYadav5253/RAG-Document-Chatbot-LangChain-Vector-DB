from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.embeddings import load_and_split, create_vector_store
from app.rag_pipeline import ask_question, ask_question_with_tracking
import shutil
import os

app = FastAPI(
    title="RAG Document Chatbot",
    description="Upload any PDF and ask questions about it using AI",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class QuestionWithTrackingRequest(BaseModel):
    question: str

@app.get("/health")
async def health():
    return {"status": "running", "model": "llama-3.3-70b-versatile"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and build the vector store"""
    os.makedirs("docs", exist_ok=True)
    path = f"docs/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = load_and_split(path)
    create_vector_store(chunks)
    return {
        "message": f"PDF processed successfully!",
        "filename": file.filename,
        "chunks_created": len(chunks)
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    """Ask a question about your uploaded documents"""
    result = ask_question(request.question)
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }

@app.post("/ask-tracked")
async def ask_tracked(request: QuestionWithTrackingRequest):
    """Ask a question with MLflow experiment tracking"""
    result = ask_question_with_tracking(request.question)
    return {
        "question": request.question,
        "answer": result["answer"],
        "response_time_seconds": result["response_time"]
    }