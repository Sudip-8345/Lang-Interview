from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from operator import add

from src.prompts import interviewer_prompt, evaluator_prompt, report_writer_prompt
from src.llm import get_default_llm, get_eval_llm
from src.tools import create_jd_tool, create_resume_tool, save_report_as_pdf, report_writer_tools
from utils.logger import get_logger

logger = get_logger(__name__)


# ===================== STATE DEFINITION =====================
class AgentState(TypedDict):
    mode: str
    num_of_q: int
    num_of_follow_up: int
    position: str
    evaluation_result: Annotated[str, add]
    hr_report: Annotated[str, add]
    company_name: str
    messages: Annotated[list, add_messages]


# ===================== RECRUITER AGENT =====================
def create_recruiter_agent(tools: List):
    
    def recruiter(state: AgentState) -> AgentState:
        """The recruiter agent conducts interviews using the interviewer prompt and tools."""
        sys_prompt = SystemMessage(content=interviewer_prompt.format(
            mode=state['mode'],
            company_name=state['company_name'],
            position=state['position'],
            number_of_questions=state['num_of_q'],
            number_of_follow_up=state['num_of_follow_up']
        ))
        all_messages = [sys_prompt] + list(state['messages'])
        llm = get_default_llm()
        result = llm.bind_tools(tools).invoke(all_messages)
        
        # If the model made tool calls, execute tools and get a clean response
        if hasattr(result, 'tool_calls') and result.tool_calls:
            tool_node = ToolNode(tools)
            tool_results = tool_node.invoke({"messages": [result]})
            
            updated_messages = all_messages + [result] + tool_results['messages']
            final_result = llm.bind_tools(tools).invoke(updated_messages)
            return {"messages": [result] + tool_results['messages'] + [final_result]}
        
        return {"messages": [result]}
    
    return recruiter


# ===================== EVALUATOR AGENT =====================
def create_evaluator_agent(jd_tool):
    
    def evaluator(state: AgentState) -> AgentState:
        """The evaluator agent assesses candidate responses using the evaluator prompt."""
        sys_prompt = evaluator_prompt.format(
            num_of_q=state['num_of_q'],
            num_of_follow_up=state['num_of_follow_up'],
            position=state['position']
        )
        sys_msg = SystemMessage(content=sys_prompt)
        
        # Build interview transcript
        interview_base = []
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                interview_base.append('Candidate: ' + str(msg.content))
            elif isinstance(msg, AIMessage):
                interview_base.append('Interviewer: ' + str(msg.content))
        
        all_messages = [sys_msg, HumanMessage(content='\n'.join(interview_base))]
        evallm = get_eval_llm()
        evallm_with_tools = evallm.bind_tools([jd_tool])
        results = evallm_with_tools.invoke(all_messages)
        
        return {
            'messages': [AIMessage(content=results.content)], 
            'evaluation_result': results.content
        }
    
    return evaluator


# ===================== REPORT WRITER AGENT =====================
def report_writer(state: AgentState) -> AgentState:
    """Generates a report based on the interview transcript and evaluation."""
    interviewer_transcript = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            interviewer_transcript.append('Candidate: ' + str(m.content))
        elif isinstance(m, AIMessage):
            # Exclude evaluation results from transcript
            if 'Evaluation:\n1. Introduction question' not in m.content:
                interviewer_transcript.append('AI Recruiter: ' + str(m.content))
    
    # Get evaluation report
    evaluation_report = [
        m.content for m in state["messages"] 
        if isinstance(m, AIMessage) and 'Evaluation:\n1. Introduction question' in m.content
    ]
    
    sys_prompt = report_writer_prompt.format(
        position=state['position'],
        company_name=state['company_name'],
        interview_transcript='\n'.join(interviewer_transcript),
        evaluation_report='\n'.join(evaluation_report) if evaluation_report else state.get('evaluation_result', '')
    )
    sys_message = SystemMessage(content=sys_prompt)
    all_messages = [sys_message, HumanMessage(content='Generate the report now.')]
    llm = get_default_llm()
    result = llm.bind_tools(report_writer_tools).invoke(all_messages)
    
    return {"messages": [result], "hr_report": result.content}


# ===================== CONDITIONAL EDGES =====================
# Phrases that indicate the interview has ended
END_PHRASES = [
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
    "enjoyed our conversation"
]

def is_interview_ended(content: str) -> bool:
    """Check if the message indicates the interview has ended."""
    content_lower = content.lower()
    return any(phrase in content_lower for phrase in END_PHRASES)


def custom_tools_condition(state: AgentState) -> str:
    """Decides the next step based on the last message."""
    last_message = state['messages'][-1] if state['messages'] else None
    
    if last_message is None:
        return 'WAIT_FOR_HUMAN'
    
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        elif is_interview_ended(last_message.content):
            return "END_CONVERSATION"
    
    return 'WAIT_FOR_HUMAN'


# ===================== WORKFLOW BUILDER =====================
def build_interview_workflow(jd_tool, resume_tool):
    
    tools = [jd_tool, resume_tool]
    
    # Create agents
    recruiter = create_recruiter_agent(tools)
    evaluator = create_evaluator_agent(jd_tool)
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("recruiter", recruiter)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("evaluator_tools", ToolNode([jd_tool]))
    workflow.add_node("report_writer", report_writer)
    workflow.add_node("report_writer_tools", ToolNode([save_report_as_pdf]))
    
    # Set entry point
    workflow.set_entry_point("recruiter")
    
    # Add conditional edges for recruiter
    workflow.add_conditional_edges(
        "recruiter",
        custom_tools_condition,
        {
            "tools": "tools",
            "END_CONVERSATION": "evaluator",
            "WAIT_FOR_HUMAN": END
        }
    )
    
    # Add edge from tools back to recruiter
    workflow.add_edge("tools", "recruiter")
    
    # Add edges for evaluator
    workflow.add_conditional_edges(
        "evaluator",
        tools_condition,
        {
            "tools": "evaluator_tools",
            END: "report_writer"
        }
    )
    workflow.add_edge("evaluator_tools", "evaluator")
    
    # Add edges for report writer
    workflow.add_conditional_edges(
        "report_writer",
        tools_condition,
        {
            "tools": "report_writer_tools",
            END: END
        }
    )
    workflow.add_edge("report_writer_tools", END)
    
    # Compile
    app = workflow.compile()
    logger.info("Interview workflow compiled successfully.")
    
    return app
