from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.embeddings import load_and_split, create_vector_store
from app.rag_pipeline import ask_question, ask_question_with_tracking
import shutil
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG Document Chatbot",
    description="Upload any PDF and ask questions about it using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    os.makedirs("docs", exist_ok=True)
    path = f"docs/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = load_and_split(path)
    create_vector_store(chunks)
    return {
        "message": "PDF processed successfully!",
        "filename": file.filename,
        "chunks_created": len(chunks)
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    result = ask_question(request.question)
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }

@app.post("/ask-tracked")
async def ask_tracked(request: QuestionWithTrackingRequest):
    result = ask_question_with_tracking(request.question)
    return {
        "question": request.question,
        "answer": result["answer"],
        "response_time_seconds": result["response_time"]
    }

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)