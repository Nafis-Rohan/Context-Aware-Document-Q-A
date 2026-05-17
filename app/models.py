from pydantic import BaseModel
from typing import Optional

# What the user sends when asking a question
class QuestionRequest(BaseModel):
    question: str
    provider: Optional[str] = None  # "ollama" or "gemini" (optional, uses .env default)

# What we send back to the user
class AnswerResponse(BaseModel):
    answer: str
    provider_used: str
    sources: list[str]  # Which parts of the PDF were used to answer

# What we send back after uploading a PDF
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int  # How many pieces the PDF was split into