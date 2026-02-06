from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from os import getenv
from utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model():
    try:
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        logger.info("Initialized OpenAI Embeddings via OpenRouter.")
        return embedding
    except Exception as e:
        logger.warning(f"OpenAI Embeddings failed: {e}. Falling back to Google.")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            api_key=getenv("GOOGLE_API_KEY")
        )


def create_vectorstores(jd_chunks, resume_chunks, 
                        jd_collection="interview-jd",
                        resume_collection="candidate-resume",
                        persist_dir="./chroma_db"):

    embedding = get_embedding_model()
    
    try:
        jd_store = Chroma.from_documents(
            documents=jd_chunks,
            embedding=embedding,
            collection_name=jd_collection,
            persist_directory=f"{persist_dir}/{jd_collection}"
        )
        logger.info(f"Created JD vector store with {len(jd_chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error creating JD vector store: {e}")
        raise RuntimeError(f"Failed to create JD vector store: {e}")
    
    try:
        resume_store = Chroma.from_documents(
            documents=resume_chunks,
            embedding=embedding,
            collection_name=resume_collection,
            persist_directory=f"{persist_dir}/{resume_collection}"
        )
        logger.info(f"Created resume vector store with {len(resume_chunks)} chunks.")
    except Exception as e:
        logger.error(f"Error creating resume vector store: {e}")
        raise RuntimeError(f"Failed to create resume vector store: {e}")
    
    return jd_store, resume_store


def get_retrievers(jd_store, resume_store, search_type="similarity"):
    
    jd_retriever = jd_store.as_retriever(search_type=search_type)
    resume_retriever = resume_store.as_retriever(search_type=search_type)
    
    logger.info(f"Created retrievers with search_type='{search_type}'.")
    return jd_retriever, resume_retriever


def load_existing_vectorstore(collection_name, persist_dir="./chroma_db"):

    embedding = get_embedding_model()
    
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=f"{persist_dir}/{collection_name}"
        )
        logger.info(f"Loaded existing vector store: {collection_name}")
        return store
    except Exception as e:
        logger.error(f"Error loading vector store '{collection_name}': {e}")
        raise RuntimeError(f"Failed to load vector store: {e}")