---
title: AskAI.hr
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# AskAI.hr - AI Interview Assistant

A real-time voice-based AI interview system that conducts technical interviews, evaluates candidates, and generates HR reports.

## Features

- Voice-based interview with real-time transcription
- AI interviewer adapts questions based on resume and job description
- Automatic evaluation and HR report generation
- Text fallback option

## Usage

1. Upload Job Description (PDF)
2. Upload Resume (PDF)
3. Click Start Interview
4. Speak your answers
5. View results in Results tab

## Tech Stack

- LangGraph for agent orchestration
- Deepgram for speech-to-text
- Edge-TTS for text-to-speech
- Gradio for UI