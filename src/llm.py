from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid errors when API keys are missing
_ChatOpenAI = None
_ChatGroq = None
_ChatGoogleGenerativeAI = None

def _import_llm_classes():
    """Lazy import LLM classes only when needed"""
    global _ChatOpenAI, _ChatGroq, _ChatGoogleGenerativeAI
    if _ChatOpenAI is None:
        from langchain_openai import ChatOpenAI as CO
        _ChatOpenAI = CO
    if _ChatGroq is None:
        from langchain_groq import ChatGroq as CG
        _ChatGroq = CG
    if _ChatGoogleGenerativeAI is None:
        from langchain_google_genai import ChatGoogleGenerativeAI as CGG
        _ChatGoogleGenerativeAI = CGG


def get_llm(temperature: float = 0.6):
    """Get LLM instance with fallback chain"""
    _import_llm_classes()
    
    # Primary: Groq
    if settings.groq_api_key:
        try:
            llm_instance = _ChatGroq(
                model_name=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=temperature
            )
            logger.info(f"Initialized LLM via Groq: {settings.groq_model}")
            return llm_instance
        except Exception as e:
            logger.warning(f"Groq LLM initialization failed: {e}")
    
    # Fallback: Google
    if settings.google_api_key:
        try:
            llm_instance = _ChatGoogleGenerativeAI(
                model=settings.google_model,
                google_api_key=settings.google_api_key,
                temperature=temperature
            )
            logger.info(f"Initialized LLM via Google: {settings.google_model}")
            return llm_instance
        except Exception as e:
            logger.error(f"Google LLM initialization failed: {e}")

    # Fallback: OpenRouter
    if settings.openrouter_api_key:
        try:
            llm_instance = _ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                temperature=temperature
            )
            logger.info(f"Initialized LLM via OpenRouter: {settings.openrouter_model}")
            return llm_instance
        except Exception as e:
            logger.warning(f"OpenRouter LLM initialization failed: {e}")
            
    raise RuntimeError("No LLM could be initialized. Check your API keys.")


# Lazy LLM instances - initialized on first access
_llm = None
_evallm = None

def get_default_llm():
    """Get default LLM with lazy initialization"""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def get_eval_llm():
    """Get evaluator LLM with lazy initialization"""
    global _evallm
    if _evallm is None:
        _evallm = get_llm(temperature=0.3)
    return _evallm
