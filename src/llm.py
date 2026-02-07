from utils.config import settings
from utils.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

logger = get_logger(__name__)


def get_llm(temperature: float = 0.6):
    
    # Primary: Groq
    if settings.groq_api_key:
        try:
            llm = ChatGroq(
                model_name=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=temperature
            )
            logger.info(f"Initialized LLM via Groq: {settings.groq_model}")
            return llm
        except Exception as e:
            logger.warning(f"Groq LLM initialization failed: {e}")
    
    # Fallback: Google
    if settings.google_api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.google_model,
                google_api_key=settings.google_api_key,
                temperature=temperature
            )
            logger.info(f"Initialized LLM via Google: {settings.google_model}")
            return llm
        except Exception as e:
            logger.error(f"Google LLM initialization failed: {e}")

    # Fallback: OpenRouter
    if settings.openrouter_api_key:
        try:
            llm = ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                temperature=temperature
            )
            logger.info(f"Initialized LLM via OpenRouter: {settings.openrouter_model}")
            return llm
        except Exception as e:
            logger.warning(f"OpenRouter LLM initialization failed: {e}")
            
    raise RuntimeError("No LLM could be initialized. Check your API keys.")

# Default LLM instance
llm = get_llm()

# Evaluator LLM 
evallm = get_llm(temperature=0.3)
