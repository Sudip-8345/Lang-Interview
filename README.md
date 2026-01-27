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
  - Whisper (local)
  - AssemblyAI (cloud)

- **Automatic Evaluation**  
  Get detailed scoring and structured feedback on interview performance.

- **Report Generation**  
  Generate comprehensive HR reports with candidate assessment.

- **PDF Export**  
  Download professional, formatted PDF interview reports.

---

## 🧩 Components

- **LangGraph Workflow**  
  Multi-stage interview process with evaluation and reporting.

- **Speech-to-Text**
  - Local: Whisper  
  - Cloud: AssemblyAI

- **Text-to-Speech**  
  Realistic voice responses using ElevenLabs.

- **Vector Search**  
  Dynamic resume and question analysis using embeddings.

- **PDF Generation**  
  Professional report formatting and export using FPDF.

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/Sudip-8345/Lang-Interview.git
cd Lang-Interview

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

### Interviewer Mode
- friendly  
- formal  
- technical  

### Position
Job title for the interview.

### Company Name
Organization conducting the interview.

### Number of Questions
Number of technical questions to ask.

### Follow-up Questions
Number of follow-up questions per topic.

### Voice Settings
Model size and language options (for Whisper).

---

## 🔐 API Keys

This project integrates the following external APIs:

- **AssemblyAI** – Cloud-based speech recognition  
- **ElevenLabs** – High-quality AI voice responses  
- **Google Generative AI** – Large Language Model powering the interview logic  

Make sure to set your API keys as environment variables before running the application.

---

## 🛠 Requirements

- Python 3.8+
- Streamlit
- LangGraph
- LangChain
- Google Generative AI
- Whisper (for local voice processing)
- AssemblyAI (for cloud voice processing)
- ElevenLabs (for text-to-speech)
- FPDF (for PDF generation)
