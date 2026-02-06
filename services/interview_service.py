import os
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from db import crud
from db.models import SessionStatus, MessageRole
from src.orchastrate import InterviewSession as WorkflowSession
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory cache for active workflow sessions
# This stores the LangGraph workflow state that can't be serialized to DB
_workflow_cache: Dict[str, WorkflowSession] = {}


async def create_interview_session(
    db: AsyncSession,
    session_id: str,
    jd_path: str,
    resume_path: str,
    company_name: str = "Tech Innovators Inc.",
    position: str = "AI Engineer",
    mode: str = "friendly",
    num_questions: int = 3,
    num_followup: int = 2
) -> Tuple[bool, str]:
    try:
        if not os.path.exists(jd_path):
            return False, f"JD file not found: {jd_path}"
        if not os.path.exists(resume_path):
            return False, f"Resume file not found: {resume_path}"
        
        # Create DB session record
        db_session = await crud.create_session(
            db=db,
            session_id=session_id,
            company_name=company_name,
            position=position,
            mode=mode,
            num_questions=num_questions,
            num_followup=num_followup,
            jd_path=jd_path,
            resume_path=resume_path
        )
        
        # Initialize workflow
        workflow = WorkflowSession(
            jd_path=jd_path,
            resume_path=resume_path,
            company_name=company_name,
            position=position,
            mode=mode,
            num_of_q=num_questions,
            num_of_follow_up=num_followup
        )
        workflow.setup()
        
        # Cache the workflow
        _workflow_cache[session_id] = workflow
        
        logger.info(f"Interview session created: {session_id}")
        return True, "Session created successfully"
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return False, str(e)


async def start_interview(
    db: AsyncSession,
    session_id: str
) -> Tuple[bool, str]:
    try:
        workflow = _workflow_cache.get(session_id)
        if not workflow:
            # Try to restore from DB
            restored = await _restore_workflow(db, session_id)
            if not restored:
                return False, "Session not found"
            workflow = _workflow_cache[session_id]
        
        # Start interview
        response = workflow.start_interview("Hi")
        
        # Update DB status
        await crud.update_session_status(db, session_id, SessionStatus.IN_PROGRESS)
        
        # Save messages to DB
        await crud.add_message(db, session_id, MessageRole.HUMAN, "Hi")
        await crud.add_message(db, session_id, MessageRole.AI, response)
        
        return True, response
        
    except Exception as e:
        logger.error(f"Failed to start interview: {e}")
        return False, str(e)


async def send_message(
    db: AsyncSession,
    session_id: str,
    message: str
) -> Tuple[bool, str, bool]:
    try:
        workflow = _workflow_cache.get(session_id)
        if not workflow:
            restored = await _restore_workflow(db, session_id)
            if not restored:
                return False, "Session not found", False
            workflow = _workflow_cache[session_id]
        
        # Get AI response
        response = workflow.send_message(message)
        is_complete = workflow.is_interview_complete()
        
        # Save messages to DB
        await crud.add_message(db, session_id, MessageRole.HUMAN, message)
        await crud.add_message(db, session_id, MessageRole.AI, response)
        
        # Update status if complete
        if is_complete:
            await crud.update_session_status(db, session_id, SessionStatus.COMPLETED, is_complete=True)
            
            # Save evaluation results
            evaluation = workflow.get_evaluation()
            hr_report = workflow.get_hr_report()
            if evaluation or hr_report:
                await crud.update_session_evaluation(db, session_id, evaluation, hr_report)
        
        return True, response, is_complete
        
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return False, str(e), False


async def get_session_status(
    db: AsyncSession,
    session_id: str
) -> Optional[Dict[str, Any]]:
    session = await crud.get_session(db, session_id)
    if not session:
        return None
    
    return {
        "session_id": session.id,
        "company_name": session.company_name,
        "position": session.position,
        "status": session.status.value,
        "is_complete": session.is_complete,
        "created_at": session.created_at.isoformat(),
        "message_count": len(session.messages) if session.messages else 0
    }


async def get_evaluation_results(
    db: AsyncSession,
    session_id: str
) -> Optional[Dict[str, Any]]:
    session = await crud.get_session(db, session_id)
    if not session:
        return None
    
    # Try to get from workflow cache first (may have more recent data)
    workflow = _workflow_cache.get(session_id)
    
    transcript = await crud.get_transcript(db, session_id)
    
    return {
        "evaluation": workflow.get_evaluation() if workflow else session.evaluation_result,
        "hr_report": workflow.get_hr_report() if workflow else session.hr_report,
        "transcript": transcript,
        "is_complete": session.is_complete
    }


async def delete_interview_session(
    db: AsyncSession,
    session_id: str
) -> bool:
    # Remove from cache
    if session_id in _workflow_cache:
        del _workflow_cache[session_id]
    
    # Delete from DB
    return await crud.delete_session(db, session_id)


async def list_sessions(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0
) -> list:
    sessions = await crud.list_sessions(db, limit, offset)
    return [
        {
            "session_id": s.id,
            "company_name": s.company_name,
            "position": s.position,
            "status": s.status.value,
            "is_complete": s.is_complete,
            "created_at": s.created_at.isoformat()
        }
        for s in sessions
    ]


async def _restore_workflow(
    db: AsyncSession,
    session_id: str
) -> bool:
    """
    Restore a workflow from database
    Used when workflow is not in cache (e.g., after server restart)
    """
    session = await crud.get_session_with_messages(db, session_id)
    if not session:
        return False
    
    try:
        # Recreate workflow
        workflow = WorkflowSession(
            jd_path=session.jd_path,
            resume_path=session.resume_path,
            company_name=session.company_name,
            position=session.position,
            mode=session.mode,
            num_of_q=session.num_questions,
            num_of_follow_up=session.num_followup
        )
        workflow.setup()
        
        # Replay messages to restore state
        if session.messages:
            # Start with first exchange
            first_human = next((m for m in session.messages if m.role == MessageRole.HUMAN), None)
            if first_human:
                workflow.start_interview(first_human.content)
            
            # Replay remaining human messages
            for msg in session.messages:
                if msg.role == MessageRole.HUMAN and msg != first_human:
                    workflow.send_message(msg.content)
        
        _workflow_cache[session_id] = workflow
        logger.info(f"Restored workflow for session: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to restore workflow: {e}")
        return False


def cleanup_workflow_cache(session_id: str = None):
    if session_id:
        if session_id in _workflow_cache:
            del _workflow_cache[session_id]
    else:
        _workflow_cache.clear()
