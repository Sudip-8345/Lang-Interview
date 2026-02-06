"""
FastAPI Backend for AI Interview System
With PostgreSQL database integration
"""
import os
import sys
import uuid
import shutil
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import settings
from utils.logger import get_logger
from utils.tts import synthesize as tts_synthesize, list_english_voices
from utils.stt import transcribe as stt_transcribe, preload_whisper_model
from db.database import get_db, init_db, close_db
from services import interview_service

logger = get_logger(__name__)


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    is_complete: bool
    audio_url: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    company_name: Optional[str] = None
    position: Optional[str] = None
    status: str
    is_complete: bool
    created_at: Optional[str] = None
    message_count: int = 0


class EvaluationResponse(BaseModel):
    evaluation: Optional[str] = None
    hr_report: Optional[str] = None
    transcript: Optional[str] = None
    is_complete: bool = False


class TranscribeResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class SessionListItem(BaseModel):
    session_id: str
    company_name: str
    position: str
    status: str
    is_complete: bool
    created_at: str


# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting AI Interview Backend...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Ensure directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs("voice_outputs", exist_ok=True)
    
    # Preload whisper model
    preload_whisper_model(settings.stt_whisper_model)
    
    logger.info(f"Server ready at http://{settings.host}:{settings.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    interview_service.cleanup_workflow_cache()
    await close_db()


# ==================== FastAPI App ====================

app = FastAPI(
    title="AI Interview System API",
    description="Backend API for AI-powered interview system with voice support and PostgreSQL storage",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health & Info Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI Interview System API",
        "version": "2.0.0",
        "database": "PostgreSQL",
        "status": "running"
    }


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check with database status"""
    try:
        # Simple query to check DB connection
        sessions = await interview_service.list_sessions(db, limit=1)
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status
    }


@app.get("/voices")
async def get_voices():
    """Get available TTS voices"""
    try:
        voices = await list_english_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Document Upload Endpoints ====================

@app.post("/upload/jd")
async def upload_jd(file: UploadFile = File(...)):
    """Upload Job Description PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"jd_{file_id}.pdf")
    
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"JD uploaded: {file_path}")
        return {"file_id": file_id, "file_path": file_path, "filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading JD: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload candidate resume PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"resume_{file_id}.pdf")
    
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Resume uploaded: {file_path}")
        return {"file_id": file_id, "file_path": file_path, "filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Session Management Endpoints ====================

@app.post("/session/create", response_model=SessionResponse)
async def create_session(
    jd_path: str = Form(...),
    resume_path: str = Form(...),
    company_name: str = Form("Tech Innovators Inc."),
    position: str = Form("AI Engineer"),
    mode: str = Form("friendly"),
    num_questions: int = Form(3),
    num_followup: int = Form(2),
    db: AsyncSession = Depends(get_db)
):
    """Create a new interview session"""
    session_id = str(uuid.uuid4())
    
    success, message = await interview_service.create_interview_session(
        db=db,
        session_id=session_id,
        jd_path=jd_path,
        resume_path=resume_path,
        company_name=company_name,
        position=position,
        mode=mode,
        num_questions=num_questions,
        num_followup=num_followup
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return SessionResponse(session_id=session_id, message=message)


@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get interview session status"""
    status = await interview_service.get_session_status(db, session_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatusResponse(**status)


@app.get("/sessions", response_model=List[SessionListItem])
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List all interview sessions"""
    sessions = await interview_service.list_sessions(db, limit, offset)
    return [SessionListItem(**s) for s in sessions]


@app.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete an interview session"""
    deleted = await interview_service.delete_interview_session(db, session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully", "session_id": session_id}


# ==================== Interview Endpoints ====================

@app.post("/interview/start")
async def start_interview(
    session_id: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Start the interview and get first AI response"""
    success, response = await interview_service.start_interview(db, session_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=response)
    
    return {"response": response, "is_complete": False}


@app.post("/interview/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """Send a message and get AI response"""
    success, response, is_complete = await interview_service.send_message(
        db, request.session_id, request.message
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=response)
    
    return ChatResponse(response=response, is_complete=is_complete)


@app.post("/interview/chat-with-audio")
async def chat_with_audio(
    session_id: str = Form(...),
    message: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Send a message and get response with audio"""
    success, response, is_complete = await interview_service.send_message(
        db, session_id, message
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=response)
    
    try:
        # Generate audio
        audio_bytes = await tts_synthesize(response)
        
        # Save audio file
        audio_id = str(uuid.uuid4())
        audio_path = f"voice_outputs/{audio_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        
        return {
            "response": response,
            "is_complete": is_complete,
            "audio_url": f"/audio/{audio_id}"
        }
    except Exception as e:
        logger.error(f"TTS error: {e}")
        # Return text response even if TTS fails
        return {
            "response": response,
            "is_complete": is_complete,
            "audio_url": None
        }


@app.get("/interview/{session_id}/evaluation", response_model=EvaluationResponse)
async def get_evaluation(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get interview evaluation and report"""
    result = await interview_service.get_evaluation_results(db, session_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return EvaluationResponse(**result)


# ==================== Voice Endpoints ====================

@app.post("/stt/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text"""
    try:
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        text = await stt_transcribe(audio_data)
        
        return TranscribeResponse(text=text)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Convert text to speech"""
    try:
        audio_bytes = await tts_synthesize(request.text, voice=request.voice)
        
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Get saved audio file"""
    audio_path = f"voice_outputs/{audio_id}.mp3"
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(audio_path, media_type="audio/mpeg")


# ==================== Voice Interview Endpoint ====================

@app.post("/interview/voice")
async def voice_interview(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete voice interview flow:
    1. Transcribe user audio (STT)
    2. Get AI response
    3. Convert response to audio (TTS)
    """
    try:
        # 1. Transcribe audio to text
        audio_data = await audio.read()
        user_text = await stt_transcribe(audio_data)
        logger.info(f"Transcribed: {user_text}")
        
        # Handle low confidence
        if user_text == "[LOW_CONFIDENCE]":
            return {
                "user_text": "",
                "response": "Sorry, I couldn't understand that clearly. Could you please repeat?",
                "is_complete": False,
                "audio_url": None
            }
        
        # 2. Get AI response
        success, response, is_complete = await interview_service.send_message(
            db, session_id, user_text
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=response)
        
        logger.info(f"AI Response: {response[:100]}...")
        
        # 3. Generate speech from response
        response_audio = await tts_synthesize(response)
        
        # Save audio
        audio_id = str(uuid.uuid4())
        audio_path = f"voice_outputs/{audio_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(response_audio)
        
        return {
            "user_text": user_text,
            "response": response,
            "is_complete": is_complete,
            "audio_url": f"/audio/{audio_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice interview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Quick Start Endpoint ====================

@app.post("/quick-start")
async def quick_start(
    jd: UploadFile = File(...),
    resume: UploadFile = File(...),
    company_name: str = Form("Tech Innovators Inc."),
    position: str = Form("AI Engineer"),
    mode: str = Form("friendly"),
    num_questions: int = Form(3),
    num_followup: int = Form(2),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload documents, create session, and start interview in one call.
    """
    # Validate files
    if not jd.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="JD must be a PDF file")
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")
    
    session_id = str(uuid.uuid4())
    
    try:
        # Save files
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        jd_path = os.path.join(settings.upload_dir, f"jd_{session_id}.pdf")
        resume_path = os.path.join(settings.upload_dir, f"resume_{session_id}.pdf")
        
        with open(jd_path, "wb") as f:
            shutil.copyfileobj(jd.file, f)
        
        with open(resume_path, "wb") as f:
            shutil.copyfileobj(resume.file, f)
        
        # Create session
        success, message = await interview_service.create_interview_session(
            db=db,
            session_id=session_id,
            jd_path=jd_path,
            resume_path=resume_path,
            company_name=company_name,
            position=position,
            mode=mode,
            num_questions=num_questions,
            num_followup=num_followup
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # Start interview
        success, response = await interview_service.start_interview(db, session_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=response)
        
        logger.info(f"Quick start session created: {session_id}")
        
        return {
            "session_id": session_id,
            "response": response,
            "is_complete": False
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Run Server ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
