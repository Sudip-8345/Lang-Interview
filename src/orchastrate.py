import os
import sys

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from RAG_engine.indexer import load_jd_and_resume, split_documents
from RAG_engine.retriever import create_vectorstores, get_retrievers, load_existing_vectorstore
from src.tools import create_jd_tool, create_resume_tool
from src.agents import AgentState, build_interview_workflow
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class InterviewSession:

    def __init__(
        self,
        jd_path: str,
        resume_path: str,
        company_name: str = "Tech Innovators Inc.",
        position: str = "AI Engineer",
        mode: str = None,
        num_of_q: int = None,
        num_of_follow_up: int = None,
        persist_dir: str = "./chroma_db"
    ):
        self.jd_path = jd_path
        self.resume_path = resume_path
        self.company_name = company_name
        self.position = position
        self.mode = mode or settings.default_mode
        self.num_of_q = num_of_q or settings.default_num_questions
        self.num_of_follow_up = num_of_follow_up or settings.default_num_followup
        self.persist_dir = persist_dir
        
        # Will be initialized later
        self.jd_tool = None
        self.resume_tool = None
        self.app = None
        self.current_state = None
        
    def setup(self, force_reindex: bool = False):
  
        logger.info("Setting up interview session...")
        
        # Check if files exist
        if not os.path.exists(self.jd_path):
            raise FileNotFoundError(f"JD/Interview document not found: {self.jd_path}")
        if not os.path.exists(self.resume_path):
            raise FileNotFoundError(f"Resume not found: {self.resume_path}")
        
        # Define collection names based on file names
        jd_collection = os.path.splitext(os.path.basename(self.jd_path))[0].replace(" ", "-")
        resume_collection = os.path.splitext(os.path.basename(self.resume_path))[0].replace(" ", "-")
        
        # Check if vectorstores already exist
        jd_store_path = f"{self.persist_dir}/{jd_collection}"
        resume_store_path = f"{self.persist_dir}/{resume_collection}"
        
        if not force_reindex and os.path.exists(jd_store_path) and os.path.exists(resume_store_path):
            logger.info("Loading existing vectorstores...")
            jd_store = load_existing_vectorstore(jd_collection, self.persist_dir)
            resume_store = load_existing_vectorstore(resume_collection, self.persist_dir)
        else:
            # Load and split documents
            logger.info("Loading and processing documents...")
            pages, resume = load_jd_and_resume(self.jd_path, self.resume_path)
            jd_chunks, resume_chunks = split_documents(pages, resume)
            
            # Create vectorstores
            logger.info("Creating vectorstores...")
            jd_store, resume_store = create_vectorstores(
                jd_chunks, resume_chunks,
                jd_collection=jd_collection,
                resume_collection=resume_collection,
                persist_dir=self.persist_dir
            )
        
        # Create retrievers
        jd_retriever, resume_retriever = get_retrievers(jd_store, resume_store)
        
        # Create tools
        self.jd_tool = create_jd_tool(jd_retriever)
        self.resume_tool = create_resume_tool(resume_retriever)
        
        # Build workflow
        self.app = build_interview_workflow(self.jd_tool, self.resume_tool)
        
        # Initialize state
        self.current_state = {
            "mode": self.mode,
            "num_of_q": self.num_of_q,
            "num_of_follow_up": self.num_of_follow_up,
            "position": self.position,
            "company_name": self.company_name,
            "messages": [],
            "evaluation_result": "",
            "hr_report": ""
        }
        
        logger.info("Interview session setup complete.")
        return self
    
    def start_interview(self, initial_message: str = "Hi") -> str:
        if not self.app:
            raise RuntimeError("Session not set up. Call setup() first.")
        
        self.current_state["messages"] = [HumanMessage(content=initial_message)]
        result = self.app.invoke(self.current_state)
        self.current_state = result
        
        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            return last_msg.content
        return str(last_msg)
    
    def send_message(self, message: str) -> str:
        if not self.app:
            raise RuntimeError("Session not set up. Call setup() first.")
        
        # Add human message to current state
        input_state = {
            **self.current_state,
            "messages": list(self.current_state["messages"]) + [HumanMessage(content=message)]
        }
        
        result = self.app.invoke(input_state)
        self.current_state = result
        
        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            return last_msg.content
        return str(last_msg)
    
    def is_interview_complete(self) -> bool:
        if not self.current_state or not self.current_state.get("messages"):
            return False
        
        # Check if evaluation or HR report was generated (definitive completion)
        if self.current_state.get("evaluation_result"):
            return True
        if self.current_state.get("hr_report"):
            return True
        
        # Check recent AI messages for end phrases (not just the very last)
        messages = self.current_state["messages"]
        end_phrases = [
            "that's it for today",
            "thank you for your time",
            "thank you for joining",
            "this concludes",
            "end of the interview",
            "interview is complete",
            "we're done",
            "that concludes",
            "we'll be in touch",
            "best of luck",
            "good luck",
            "great talking with you",
            "enjoyed learning about",
            "enjoyed our conversation",
            "wraps up our interview",
            "that's all for today",
            "end of our session",
            "take care"
        ]
        
        # Check last 5 AI messages (the farewell might not be the very last)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        for msg in ai_messages[-5:]:
            content_lower = msg.content.lower()
            if any(phrase in content_lower for phrase in end_phrases):
                return True
        
        return False
    
    def get_recruiter_farewell(self) -> str:
        """Get the recruiter's farewell message from the conversation."""
        if not self.current_state or not self.current_state.get("messages"):
            return ""
        
        end_phrases = [
            "that's it for today", "thank you for your time",
            "this concludes", "we'll be in touch", "best of luck",
            "good luck", "great talking with you", "wraps up",
            "take care", "enjoyed"
        ]
        
        messages = self.current_state["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        
        # Search from the end backwards for the farewell message
        for msg in reversed(ai_messages[-10:]):
            content_lower = msg.content.lower()
            if any(phrase in content_lower for phrase in end_phrases):
                return msg.content
        
        return ""
    
    def get_evaluation(self) -> Optional[str]:
        return self.current_state.get("evaluation_result") if self.current_state else None
    
    def get_hr_report(self) -> Optional[str]:
        return self.current_state.get("hr_report") if self.current_state else None
    
    def get_transcript(self) -> str:
        if not self.current_state or not self.current_state.get("messages"):
            return ""
        
        transcript = []
        for msg in self.current_state["messages"]:
            if isinstance(msg, HumanMessage):
                transcript.append(f"Candidate: {msg.content}")
            elif isinstance(msg, AIMessage):
                transcript.append(f"Recruiter: {msg.content}")
        
        return "\n\n".join(transcript)


def run_interactive_interview(
    jd_path: str,
    resume_path: str,
    company_name: str = "Tech Innovators Inc.",
    position: str = "AI Engineer",
    **kwargs
) -> AgentState:
    
    session = InterviewSession(
        jd_path=jd_path,
        resume_path=resume_path,
        company_name=company_name,
        position=position,
        **kwargs
    )
    session.setup()
    
    print("=" * 60)
    print(f"AI Interview Session - {company_name}")
    print(f"Position: {position}")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    
    # Start interview
    print("\n⏳ Waiting for AI response (this may take a moment)...")
    response = session.start_interview("Hi")
    print(f"\nRecruiter: {response}\n")
    
    while not session.is_interview_complete():
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Interview ended by user.")
            break
        
        if not user_input:
            continue
        
        try:
            response = session.send_message(user_input)
            print(f"\nRecruiter: {response}\n")
        except Exception as e:
            logger.error(f"Error during interview: {e}")
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Interview Complete!")
    
    evaluation = session.get_evaluation()
    if evaluation:
        print("\n--- Evaluation ---")
        print(evaluation)
    
    hr_report = session.get_hr_report()
    if hr_report:
        print("\n--- HR Report ---")
        print(hr_report)
    
    return session.current_state


# For backward compatibility
def generate_interview_response(query: str, jd_path: str, resume_path: str) -> str:
    
    session = InterviewSession(jd_path=jd_path, resume_path=resume_path)
    session.setup()
    return session.send_message(query)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) >= 3:
        jd_file = sys.argv[1]
        resume_file = sys.argv[2]
    else:
        # Default paths using project root
        jd_file = os.path.join(project_root, "utils", "interview-llm-hoang.pdf")
        resume_file = os.path.join(project_root, "utils", "Sudip_AI_Eng.pdf")
    
    run_interactive_interview(
        jd_path=jd_file,
        resume_path=resume_file,
        company_name="Data Solutions Inc.",
        position="AI/ML Engineer"
    )
    