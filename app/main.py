from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QuestionRequest, AnswerResponse, UploadResponse
from app.rag import process_pdf, answer_question
import os

app = FastAPI(
    title="Context-Aware Document Q&A API",
    description="Upload a PDF and ask questions about it using RAG",
    version="1.0.0"
)

# Allow all origins (useful for testing from browser/frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "running", "message": "API is healthy!"}

# ─── Upload PDF ────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # Only allow PDF files
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed!")
    
    try:
        file_bytes = await file.read()  # Read the uploaded file
        chunks_created = process_pdf(file_bytes, file.filename)  # Send to rag.py
        
        return UploadResponse(
            message="PDF processed successfully!",
            filename=file.filename,
            chunks_created=chunks_created
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Ask Question ──────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, filename: str):
    try:
        result = answer_question(
            question=request.question,
            filename=filename,
            provider=request.provider
        )
        
        return AnswerResponse(
            answer=result["answer"],
            provider_used=result["provider_used"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))