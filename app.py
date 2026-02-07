import os
import sys
import uuid
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import settings
from utils.logger import get_logger
from utils.tts import synthesize as tts_synthesize
from utils.stt import transcribe as stt_transcribe
from src.orchastrate import InterviewSession

logger = get_logger(__name__)

# Global session storage
active_sessions = {}


# ==================== Helper Functions ====================

def format_time(seconds: int) -> str:
    """Format seconds into MM:SS display"""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def clean_ai_response(text: str) -> str:
    """Remove any tool/function call syntax from AI response"""
    import re
    
    # Remove <function=...>...</function> tags
    text = re.sub(r'<function=[^>]*>[^<]*</function>', '', text)
    
    # Remove {"function": ...} JSON blocks
    text = re.sub(r'\{"function"[^}]+\}', '', text)
    
    # Remove <tool_call>...</tool_call> tags
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    
    # Remove ```...``` code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    
    # Remove tool result messages (Report saved, file paths, etc.)
    text = re.sub(r'content=[\'"].*?[\'"]\s*name=[\'"].*?[\'"].*', '', text, flags=re.DOTALL)
    text = re.sub(r'✅.*?(saved|created|generated).*?(\.pdf|\.txt|\.docx).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Report.*?saved.*?:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[A-Za-z]:\\[^\s]+', '', text)  # Remove Windows file paths
    text = re.sub(r'/[\w/]+\.[a-z]+', '', text)  # Remove Unix file paths
    
    # Remove tool IDs and technical output
    text = re.sub(r'id=[\'"][^\'"]+[\'"]', '', text)
    text = re.sub(r'tool_call_id=[\'"][^\'"]+[\'"]', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def numpy_to_bytes(audio_tuple: Tuple[int, np.ndarray]) -> bytes:
    """Convert Gradio audio tuple to bytes for STT"""
    import io
    import soundfile as sf
    
    if audio_tuple is None:
        return None
    
    sample_rate, audio_data = audio_tuple
    
    # Normalize audio data
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Write to bytes buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


def save_audio_to_temp_file(audio_bytes: bytes, suffix: str = ".mp3") -> str:
    """Save audio bytes to a temporary file for playback"""
    temp_dir = Path("voice_outputs")
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / f"response_{uuid.uuid4().hex[:8]}{suffix}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    
    return str(temp_path)


# ==================== Interview Session Manager ====================

def format_chat_history(history: List) -> List[dict]:
    """Convert chat history to Gradio messages format"""
    messages = []
    for item in history:
        if isinstance(item, tuple):
            user_msg, ai_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        elif isinstance(item, dict):
            messages.append(item)
    return messages


class GradioInterviewSession:
    """Manages interview session state for Gradio UI"""
    
    def __init__(self):
        self.session_id: str = None
        self.workflow: InterviewSession = None
        self.start_time: float = None
        self.is_active: bool = False
        self.is_complete: bool = False
        self.chat_history: List[dict] = []
        self.jd_path: str = None
        self.resume_path: str = None
        
    def reset(self):
        """Reset session state"""
        self.session_id = None
        self.workflow = None
        self.start_time = None
        self.is_active = False
        self.is_complete = False
        self.chat_history = []


# ==================== Core Interview Functions ====================

async def setup_interview(
    jd_file,
    resume_file,
    company_name: str,
    position: str,
    mode: str,
    num_questions: int,
    num_followup: int,
    session_state: dict
) -> Tuple[str, List, dict, gr.update, gr.update]:
    """Initialize interview session with uploaded documents"""
    
    if jd_file is None or resume_file is None:
        return (
            "⚠️ Please upload both JD and Resume files",
            [],
            session_state,
            gr.update(interactive=False),
            gr.update(visible=False)
        )
    
    try:
        # Save uploaded files
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        session_id = uuid.uuid4().hex[:8]
        jd_path = str(upload_dir / f"{session_id}_jd{Path(jd_file.name).suffix}")
        resume_path = str(upload_dir / f"{session_id}_resume{Path(resume_file.name).suffix}")
        
        # Copy files
        import shutil
        shutil.copy(jd_file.name, jd_path)
        shutil.copy(resume_file.name, resume_path)
        
        # Create interview workflow
        workflow = InterviewSession(
            jd_path=jd_path,
            resume_path=resume_path,
            company_name=company_name or "Tech Company",
            position=position or "Software Engineer",
            mode=mode.lower(),
            num_of_q=num_questions,
            num_of_follow_up=num_followup
        )
        workflow.setup()
        
        # Store in session state
        session_state["session_id"] = session_id
        session_state["workflow"] = workflow
        session_state["jd_path"] = jd_path
        session_state["resume_path"] = resume_path
        session_state["is_active"] = False
        session_state["is_complete"] = False
        session_state["chat_history"] = []
        session_state["start_time"] = None
        
        logger.info(f"Interview session {session_id} initialized")
        
        return (
            f"✅ Session ready! Click 'Start Interview' to begin.\n\n"
            f"📋 **Position:** {position}\n"
            f"🏢 **Company:** {company_name}\n"
            f"🎯 **Mode:** {mode}\n"
            f"❓ **Questions:** {num_questions} (+ {num_followup} follow-ups each)",
            [],
            session_state,
            gr.update(interactive=True, variant="primary"),
            gr.update(visible=True)
        )
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return (
            f"❌ Setup failed: {str(e)}",
            [],
            session_state,
            gr.update(interactive=False),
            gr.update(visible=False)
        )


async def start_interview(session_state: dict) -> Tuple[str, List, str, dict, gr.update, gr.update]:
    """Start the interview and get the first AI message"""
    
    workflow = session_state.get("workflow")
    if not workflow:
        return (
            "⚠️ Session not initialized. Please set up first.",
            [],
            None,
            session_state,
            gr.update(),
            gr.update()
        )
    
    try:
        # Start interview
        response = clean_ai_response(workflow.start_interview("Hello"))
        
        # If response is empty, provide a default greeting
        if not response.strip():
            response = "Hi there! I'm Sarah, and I'll be conducting your interview today. It's great to meet you! To start, could you tell me a bit about yourself?"
        
        # Generate TTS
        audio_bytes = await tts_synthesize(response)
        audio_path = save_audio_to_temp_file(audio_bytes)
        
        # Update session state
        session_state["is_active"] = True
        session_state["start_time"] = time.time()
        session_state["chat_history"] = [
            {"role": "assistant", "content": response}
        ]
        
        logger.info(f"Interview started for session {session_state.get('session_id')}")
        
        return (
            "🎙️ **Interview in Progress**\n\nSpeak your response using the microphone, then click 'Send Response'.",
            session_state["chat_history"],
            audio_path,
            session_state,
            gr.update(interactive=False, variant="secondary"),  # Start button
            gr.update(interactive=True, variant="stop")  # End button
        )
        
    except Exception as e:
        logger.error(f"Start failed: {e}")
        return (
            f"❌ Failed to start: {str(e)}",
            [],
            None,
            session_state,
            gr.update(),
            gr.update()
        )


async def process_audio_response(
    audio_input,
    session_state: dict
) -> Tuple[str, List, str, str, dict]:
    """Process user's audio response and get AI reply"""
    
    workflow = session_state.get("workflow")
    if not workflow or not session_state.get("is_active"):
        return (
            "⚠️ Interview not active. Please start the interview first.",
            session_state.get("chat_history", []),
            None,
            "",
            session_state
        )
    
    if audio_input is None:
        return (
            "⚠️ No audio detected. Please record your response.",
            session_state.get("chat_history", []),
            None,
            "",
            session_state
        )
    
    try:
        # Convert audio to bytes
        audio_bytes = numpy_to_bytes(audio_input)
        
        # Transcribe
        transcript = await stt_transcribe(audio_bytes)
        
        if transcript == "[LOW_CONFIDENCE]":
            return (
                "🔄 Sorry, I couldn't hear that clearly. Please try again.",
                session_state.get("chat_history", []),
                None,
                "",
                session_state
            )
        
        if not transcript.strip():
            return (
                "⚠️ No speech detected. Please speak clearly and try again.",
                session_state.get("chat_history", []),
                None,
                "",
                session_state
            )
        
        logger.info(f"Transcribed: {transcript[:50]}...")
        
        # Get AI response
        response = clean_ai_response(workflow.send_message(transcript))
        is_complete = workflow.is_interview_complete()
        
        # If response is empty (e.g., only tool output), provide a default ending
        if not response.strip() and is_complete:
            response = "That wraps up our interview for today! It was great talking with you. Keep building on your skills, you're doing great. Best of luck, and we'll be in touch soon!"
        elif not response.strip():
            response = "I understand. Let me continue with the next question."
        
        # Generate TTS
        audio_bytes = await tts_synthesize(response)
        audio_path = save_audio_to_temp_file(audio_bytes)
        
        # Update chat history
        chat_history = session_state.get("chat_history", [])
        chat_history.append({"role": "user", "content": transcript})
        chat_history.append({"role": "assistant", "content": response})
        session_state["chat_history"] = chat_history
        
        # Check if interview is complete
        if is_complete:
            session_state["is_complete"] = True
            session_state["is_active"] = False
            
            # Add a clear completion indicator to chat
            chat_history.append({"role": "assistant", "content": "🎉 --- Interview Completed ---"})
            session_state["chat_history"] = chat_history
            
            status = (
                "🎉 **Interview Complete!**\n\n"
                "Great job! The interview has ended.\n"
                "Go to the **Results** tab to see your evaluation and feedback."
            )
        else:
            status = "🎙️ **Your turn** - Record your response"
        
        return (
            status,
            chat_history,
            audio_path,
            transcript,
            session_state
        )
        
    except Exception as e:
        logger.error(f"Process audio failed: {e}")
        return (
            f"❌ Error: {str(e)}",
            session_state.get("chat_history", []),
            None,
            "",
            session_state
        )


async def process_text_response(
    text_input: str,
    session_state: dict
) -> Tuple[str, List, str, dict]:
    """Process user's text response and get AI reply"""
    
    workflow = session_state.get("workflow")
    if not workflow or not session_state.get("is_active"):
        return (
            "⚠️ Interview not active. Please start the interview first.",
            session_state.get("chat_history", []),
            None,
            session_state
        )
    
    if not text_input.strip():
        return (
            "⚠️ Please enter a response.",
            session_state.get("chat_history", []),
            None,
            session_state
        )
    
    try:
        # Get AI response
        response = clean_ai_response(workflow.send_message(text_input))
        is_complete = workflow.is_interview_complete()
        
        # If response is empty (e.g., only tool output), provide a default ending
        if not response.strip() and is_complete:
            response = "That wraps up our interview for today! It was great talking with you. Keep building on your skills, you're doing great. Best of luck, and we'll be in touch soon!"
        elif not response.strip():
            response = "I understand. Let me continue with the next question."
        
        # Generate TTS
        audio_bytes = await tts_synthesize(response)
        audio_path = save_audio_to_temp_file(audio_bytes)
        
        # Update chat history
        chat_history = session_state.get("chat_history", [])
        chat_history.append({"role": "user", "content": text_input})
        chat_history.append({"role": "assistant", "content": response})
        session_state["chat_history"] = chat_history
        
        if is_complete:
            session_state["is_complete"] = True
            session_state["is_active"] = False
            
            # Add a clear completion indicator to chat
            chat_history.append({"role": "assistant", "content": "🎉 --- Interview Completed ---"})
            session_state["chat_history"] = chat_history
            
            status = (
                "🎉 **Interview Complete!**\n\n"
                "Great job! The interview has ended.\n"
                "Go to the **Results** tab to see your evaluation and feedback."
            )
        else:
            status = "🎙️ **Your turn** - Record or type your response"
        
        return (status, chat_history, audio_path, session_state)
        
    except Exception as e:
        logger.error(f"Process text failed: {e}")
        return (
            f"Error: {str(e)}",
            session_state.get("chat_history", []),
            None,
            session_state
        )


async def end_interview(session_state: dict) -> Tuple[str, dict, gr.update, gr.update]:
    """End the interview early"""
    
    session_state["is_active"] = False
    session_state["is_complete"] = True
    
    return (
        "⏹️ **Interview Ended**\n\nYou ended the interview early. Click 'View Results' for partial evaluation.",
        session_state,
        gr.update(interactive=False),  # Start button
        gr.update(interactive=False)   # End button
    )


def get_evaluation_results(session_state: dict) -> Tuple[str, str, str]:
    """Get evaluation and transcript from the completed interview"""
    
    workflow = session_state.get("workflow")
    if not workflow:
        return "No session data", "", ""
    
    evaluation = workflow.get_evaluation() or "Evaluation not available yet."
    hr_report = workflow.get_hr_report() or "HR report not available yet."
    transcript = workflow.get_transcript() or "No transcript available."
    
    return evaluation, hr_report, transcript


def get_timer_display(session_state: dict) -> str:
    """Get current timer display"""
    
    start_time = session_state.get("start_time")
    if not start_time or not session_state.get("is_active"):
        return "00:00"
    
    elapsed = int(time.time() - start_time)
    return format_time(elapsed)


# ==================== Gradio UI ====================

def create_app():
    """Create and configure the Gradio application"""
    
    # Custom CSS for styling
    custom_css = """
    .container { max-width: 1200px; margin: auto; }
    .header { text-align: center; margin-bottom: 20px; }
    .timer-display { 
        font-size: 2em; 
        font-weight: bold; 
        text-align: center; 
        padding: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .status-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .chat-container { min-height: 400px; }
    .control-btn { min-width: 150px; }
    .audio-controls { margin: 10px 0; }
    """
    
    with gr.Blocks(
        title="AI Interview Assistant",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ),
        css=custom_css
    ) as app:
        
        # Session state
        session_state = gr.State({})
        
        # Header
        gr.Markdown(
            """
            # 🎯 AI Interview Assistant
            ### Real-time AI-powered technical interview practice
            ---
            """,
            elem_classes=["header"]
        )
        
        with gr.Tabs() as tabs:
            
            # ==================== Setup Tab ====================
            with gr.Tab("📋 Setup", id="setup"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📄 Upload Documents")
                        jd_file = gr.File(
                            label="Job Description (PDF/TXT)",
                            file_types=[".pdf", ".txt", ".doc", ".docx"],
                            type="filepath"
                        )
                        resume_file = gr.File(
                            label="Your Resume (PDF/TXT)",
                            file_types=[".pdf", ".txt", ".doc", ".docx"],
                            type="filepath"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Interview Settings")
                        company_name = gr.Textbox(
                            label="Company Name",
                            value="Tech Innovators Inc.",
                            placeholder="Enter company name"
                        )
                        position = gr.Textbox(
                            label="Position",
                            value="AI Engineer",
                            placeholder="Enter position title"
                        )
                        mode = gr.Radio(
                            label="Interview Mode",
                            choices=["Friendly", "Professional", "Challenging"],
                            value="Friendly"
                        )
                        
                        with gr.Row():
                            num_questions = gr.Slider(
                                label="Questions",
                                minimum=2,
                                maximum=10,
                                value=3,
                                step=1
                            )
                            num_followup = gr.Slider(
                                label="Follow-ups",
                                minimum=0,
                                maximum=5,
                                value=2,
                                step=1
                            )
                
                setup_btn = gr.Button(
                    "🚀 Initialize Session",
                    variant="primary",
                    size="lg"
                )
                setup_status = gr.Markdown("Upload documents and configure settings to begin.")
            
            # ==================== Interview Tab ====================
            with gr.Tab("🎙️ Interview", id="interview"):
                
                with gr.Row():
                    # Left panel - Controls
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Controls")
                        
                        # Timer
                        timer_display = gr.Markdown(
                            "**⏱️ Duration:** 00:00",
                            elem_classes=["timer-display"]
                        )
                        
                        # Control buttons
                        with gr.Row():
                            start_btn = gr.Button(
                                "▶️ Start Interview",
                                variant="primary",
                                interactive=False,
                                elem_classes=["control-btn"]
                            )
                            end_btn = gr.Button(
                                "⏹️ End Interview",
                                variant="stop",
                                interactive=False,
                                elem_classes=["control-btn"]
                            )
                        
                        gr.Markdown("---")
                        
                        # Status display
                        status_display = gr.Markdown(
                            "📌 **Status:** Waiting to start...",
                            elem_classes=["status-box"]
                        )
                        
                        gr.Markdown("---")
                        
                        # Audio input
                        gr.Markdown("### 🎤 Voice Input")
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record Response",
                            elem_classes=["audio-controls"]
                        )
                        send_audio_btn = gr.Button(
                            "📤 Send Voice Response",
                            variant="secondary"
                        )
                        
                        # Transcription display
                        transcription_box = gr.Textbox(
                            label="📝 Last Transcription",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("---")
                        
                        # Text input (alternative)
                        gr.Markdown("### ⌨️ Text Input (Alternative)")
                        text_input = gr.Textbox(
                            label="Type Response",
                            placeholder="Or type your response here...",
                            lines=3
                        )
                        send_text_btn = gr.Button(
                            "📤 Send Text Response",
                            variant="secondary"
                        )
                    
                    # Right panel - Chat
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 Conversation")
                        chatbot = gr.Chatbot(
                            label="Interview Conversation",
                            height=500,
                            elem_classes=["chat-container"]
                        )
                        
                        # AI audio response
                        gr.Markdown("### 🔊 AI Response Audio")
                        audio_output = gr.Audio(
                            label="AI Speaking",
                            type="filepath",
                            autoplay=True,
                            elem_classes=["audio-controls"]
                        )
            
            # ==================== Results Tab ====================
            with gr.Tab("📊 Results", id="results"):
                gr.Markdown("### 📈 Interview Evaluation")
                
                view_results_btn = gr.Button(
                    "🔄 Load Results",
                    variant="primary"
                )
                
                with gr.Tabs():
                    with gr.Tab("📋 Evaluation"):
                        evaluation_box = gr.Markdown("Complete the interview to see evaluation.")
                    
                    with gr.Tab("📑 HR Report"):
                        hr_report_box = gr.Markdown("Complete the interview to see HR report.")
                    
                    with gr.Tab("📜 Transcript"):
                        transcript_box = gr.Textbox(
                            label="Full Transcript",
                            lines=20,
                            interactive=False
                        )
            
            # ==================== Help Tab ====================
            with gr.Tab("❓ Help", id="help"):
                gr.Markdown(
                    """
                    ## How to Use
                    
                    ### 1️⃣ Setup
                    - Upload your **Job Description** (PDF or text file)
                    - Upload your **Resume** (PDF or text file)
                    - Configure interview settings (company, position, mode)
                    - Click **Initialize Session**
                    
                    ### 2️⃣ Interview
                    - Click **Start Interview** to begin
                    - The AI interviewer will ask questions
                    - Use the **microphone** to record your responses
                    - Or type responses in the text box
                    - Click **Send** after each response
                    
                    ### 3️⃣ Results
                    - After completing the interview, view your **evaluation**
                    - See the **HR report** summary
                    - Review the full **transcript**
                    
                    ---
                    
                    ### 💡 Tips
                    - Speak clearly and at a moderate pace
                    - Wait for the AI to finish speaking before responding
                    - The timer shows your interview duration
                    - You can end the interview early if needed
                    
                    ### 🎯 Interview Modes
                    - **Friendly**: Casual, supportive tone
                    - **Professional**: Standard interview style
                    - **Challenging**: More rigorous questioning
                    """
                )
        
        # ==================== Event Handlers ====================
        
        # Setup
        setup_btn.click(
            fn=setup_interview,
            inputs=[
                jd_file, resume_file, company_name, position,
                mode, num_questions, num_followup, session_state
            ],
            outputs=[setup_status, chatbot, session_state, start_btn, gr.State()]
        )
        
        # Start interview
        start_btn.click(
            fn=start_interview,
            inputs=[session_state],
            outputs=[status_display, chatbot, audio_output, session_state, start_btn, end_btn]
        )
        
        # Send audio response
        send_audio_btn.click(
            fn=process_audio_response,
            inputs=[audio_input, session_state],
            outputs=[status_display, chatbot, audio_output, transcription_box, session_state]
        ).then(
            fn=lambda: None,
            inputs=None,
            outputs=audio_input
        )
        
        # Send text response
        send_text_btn.click(
            fn=process_text_response,
            inputs=[text_input, session_state],
            outputs=[status_display, chatbot, audio_output, session_state]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=text_input
        )
        
        # End interview
        end_btn.click(
            fn=end_interview,
            inputs=[session_state],
            outputs=[status_display, session_state, start_btn, end_btn]
        )
        
        # View results
        view_results_btn.click(
            fn=get_evaluation_results,
            inputs=[session_state],
            outputs=[evaluation_box, hr_report_box, transcript_box]
        )
        
        # Timer update (every 1 second when active)
        timer_update = gr.Timer(1)
        timer_update.tick(
            fn=lambda s: f"**⏱️ Duration:** {get_timer_display(s)}",
            inputs=[session_state],
            outputs=[timer_display]
        )
    
    return app


# ==================== Main Entry ====================

if __name__ == "__main__":
    logger.info("Starting Gradio AI Interview App...")
    
    # Ensure directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("voice_outputs", exist_ok=True)
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
