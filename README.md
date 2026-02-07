# LangInterview AI

An AI-powered interview platform that simulates real-world technical interviews using voice and text. Built with LangGraph and LangChain, it analyzes resumes, asks personalized questions, evaluates responses, and generates professional HR reports.

## What It Does

This system acts as a virtual recruiter. Upload a job description and resume, and the AI conducts a structured interview - greeting the candidate, asking about their background and projects, posing technical questions, and wrapping up with feedback. Everything is transcribed, evaluated, and exported as a PDF report.

## Features

- Voice-based interviews with real-time speech-to-text and text-to-speech
- Personalized questions based on uploaded resume
- Multi-stage interview workflow (intro, project discussion, technical questions, evaluation)
- Automatic scoring and feedback generation
- PDF report export for HR review
- Web interface built with Gradio
- REST API backend with FastAPI

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM Orchestration | LangGraph, LangChain |
| Speech-to-Text | Deepgram (primary), Google Speech Recognition (fallback) |
| Text-to-Speech | Edge TTS (primary), gTTS (fallback) |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | Gradio |
| PDF Generation | ReportLab |
| Database | PostgreSQL with SQLAlchemy |

## Installation

```bash
# Clone the repository
git clone https://github.com/Sudip-8345/Lang-Interview.git
cd Lang-Interview

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Running the Application

### Gradio Web Interface
```bash
python app.py
```
Opens at http://localhost:7860

### FastAPI Backend (optional)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API docs at http://localhost:8000/docs

## Configuration

Key environment variables:

| Variable | Description |
|----------|-------------|
| OPENROUTER_API_KEY | Primary LLM provider |
| GROQ_API_KEY | Fallback LLM |
| GOOGLE_API_KEY | Fallback LLM |
| DEEPGRAM_API_KEY | Speech-to-text |
| TTS_VOICE | Edge TTS voice (default: en-IN-NeerjaNeural) |
| DEFAULT_MODE | Interview style: friendly, professional, challenging |
| DEFAULT_NUM_QUESTIONS | Number of technical questions |

## Project Structure

```
LangInterview/
├── app.py                 # Gradio web interface
├── main.py                # FastAPI backend
├── src/
│   ├── agents.py          # LangGraph agents (recruiter, evaluator, report writer)
│   ├── orchastrate.py     # Interview session management
│   ├── prompts.py         # System prompts for AI behavior
│   ├── tools.py           # RAG tools for resume/JD retrieval
│   └── llm.py             # LLM configuration with fallbacks
├── utils/
│   ├── stt.py             # Speech-to-text (Deepgram + Google)
│   ├── tts.py             # Text-to-speech (Edge TTS + gTTS)
│   ├── audio.py           # Audio processing utilities
│   └── config.py          # Settings management
├── services/
│   └── interview_service.py  # Business logic layer
├── db/                    # Database models and CRUD
└── RAG_engine/            # Document indexing and retrieval
```

## How It Works

1. Upload job description and resume (PDF or text)
2. Documents are indexed into ChromaDB vector stores
3. Click Start Interview - AI greets and asks for introduction
4. Speak or type responses; AI asks follow-up questions
5. After all questions, AI ends with feedback and encouragement
6. Evaluation runs automatically, scoring responses
7. HR report generated as downloadable PDF

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /session/create | POST | Create interview session |
| /interview/start | POST | Begin interview |
| /interview/chat | POST | Send message, get response |
| /interview/voice | POST | Full voice interaction (STT to TTS) |
| /stt/transcribe | POST | Convert audio to text |
| /tts/synthesize | POST | Convert text to audio |

## Challenges Faced

- **STT Latency**: Initial implementation using local Whisper was slow. Switched to Deepgram async API for faster transcription, but required careful handling of SDK v5 breaking changes.

- **LLM Tool Leakage**: The AI would sometimes include raw function call syntax in spoken responses. Solved with regex-based cleanup and explicit prompt instructions.

- **Interview Flow Control**: Getting the AI to follow a structured interview format (intro first, then questions, then ending) required iterative prompt engineering. The model often jumped ahead or combined multiple questions.

- **Empty Response Handling**: When the workflow transitions to evaluation/report generation, the final AI message is just tool output. Added fallback messages to provide a proper conversational ending.

- **Gradio API Changes**: Gradio v6 changed the Chatbot component format from tuples to message dictionaries, requiring refactoring of all chat history handling.

## License

MIT
