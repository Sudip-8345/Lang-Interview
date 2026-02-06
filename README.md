# LangInterview AI 🎤🤖

LangInterview AI is an intelligent, end-to-end AI-powered interview platform designed to simulate real-world interviews in both text and voice modes. It dynamically analyzes candidate resumes, asks personalized and technical questions, evaluates responses, and generates professional HR-ready reports in PDF format. The system is built using LangGraph and LangChain to orchestrate a structured, multi-stage interview workflow.

---

## Features

- **Multiple Interview Modes**  
  Conduct interviews in text-based or voice-based modes.

- **Dynamic Resume Analysis**  
  Upload any resume to generate personalized interview questions.

- **Customizable Questions**  
  Upload your own custom interview questions.

- **Voice Interaction**  
  Speak with the AI interviewer using:
  - **STT**: faster-whisper (local) with Google Speech Recognition fallback
  - **TTS**: Edge TTS (high quality) with gTTS fallback

- **Automatic Evaluation**  
  Get detailed scoring and structured feedback on interview performance.

- **Report Generation**  
  Generate comprehensive HR reports with candidate assessment.

- **PDF Export**  
  Download professional, formatted PDF interview reports.

- **REST API**  
  FastAPI backend with comprehensive endpoints for frontend integration.

---

## 🧩 Components

- **LangGraph Workflow**  
  Multi-stage interview process with evaluation and reporting.

- **Speech-to-Text (STT)**
  - Primary: faster-whisper with model caching
  - Fallback: Google Speech Recognition

- **Text-to-Speech (TTS)**  
  - Primary: Edge TTS (Microsoft, high quality voices)
  - Fallback: gTTS (Google Text-to-Speech)

- **Vector Search**  
  Dynamic resume and question analysis using embeddings (ChromaDB).

- **PDF Generation**  
  Professional report formatting and export using ReportLab.

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/Sudip-8345/Lang-Interview.git
cd Lang-Interview
```

### Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## 🚀 Running the Backend

### Start the FastAPI server
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

---

## 📡 API Endpoints

### Health & Info
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/voices` | List available TTS voices |

### Document Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload/jd` | Upload Job Description PDF |
| POST | `/upload/resume` | Upload Resume PDF |

### Interview Session
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/session/create` | Create new interview session |
| GET | `/session/{id}/status` | Get session status |
| DELETE | `/session/{id}` | Delete session |
| POST | `/interview/start` | Start interview |
| POST | `/interview/chat` | Send text message |
| POST | `/interview/chat-with-audio` | Send text, get text+audio response |
| GET | `/interview/{id}/evaluation` | Get evaluation and report |

### Voice
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/stt/transcribe` | Transcribe audio to text |
| POST | `/tts/synthesize` | Convert text to speech |
| POST | `/interview/voice` | Full voice interview (STT→LLM→TTS) |
| GET | `/audio/{id}` | Get saved audio file |

### Quick Start
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/quick-start` | Upload docs, create session, start interview in one call |

---

## 🧠 Interview Process

### Setup
Configure interview parameters and upload resume/questions.

### Introduction
AI introduces itself and asks the candidate to introduce themselves.

### Resume-Based Questions
AI asks questions related to specific projects and experiences from the resume.

### Technical Questions
Role-specific technical questions are asked.

### Follow-up Questions
AI asks follow-up questions for incomplete or unclear answers.

### Evaluation
Candidate responses are evaluated after the interview.

### Report Generation
AI generates a comprehensive HR assessment report.

### PDF Export
Download the report as a professional PDF.

---

## ⚙️ Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key (primary LLM) | - |
| `GROQ_API_KEY` | Groq API key (fallback LLM) | - |
| `GOOGLE_API_KEY` | Google API key (fallback LLM) | - |
| `TTS_VOICE` | Edge TTS voice ID | `en-IN-NeerjaNeural` |
| `STT_WHISPER_MODEL` | Whisper model size (tiny/base/small/medium/large) | `base` |
| `DEFAULT_MODE` | Interview mode (friendly/formal/technical) | `friendly` |
| `DEFAULT_NUM_QUESTIONS` | Number of questions | `3` |

See `.env.example` for all options.

---

## 🔐 API Keys

This project integrates the following external APIs:

- **OpenRouter / Groq / Google** – LLM providers (at least one required)
- **Edge TTS** – High-quality Microsoft voices (free, no API key)
- **gTTS** – Google Text-to-Speech fallback (free, no API key)
- **faster-whisper** – Local speech recognition (free, no API key)

---

## 🛠 Requirements

- Python 3.10+
- FastAPI
- LangGraph & LangChain
- faster-whisper (STT)
- edge-tts & gTTS (TTS)
- ChromaDB (Vector Store)
- ReportLab (PDF generation)
- AssemblyAI (for cloud voice processing)
- ElevenLabs (for text-to-speech)
- FPDF (for PDF generation)
