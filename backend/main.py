import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add 'core' directory to sys.path to allow imports from it
core_path = os.path.join(os.path.dirname(__file__), "core")
sys.path.append(core_path)

try:
    from pipeline import process_pdf, ask_question
except ImportError as e:
    print(f"Failed to import core modules: {e}")
    # Still allow app to start so the error is visible, but endpoints will fail

app = FastAPI(title="Walki-Talkie PDF API")

# Configure CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process the PDF
        result = process_pdf(file_path)
        return {
            "document_id": result.document_id,
            "filename": result.filename,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
            "status": result.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

@app.post("/api/ask")
async def ask_pdf(request: AskRequest):
    try:
        result = ask_question(request.question, document_id=request.document_id)
        return {"answer": result.answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error asking question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
