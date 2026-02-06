from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)


def load_jd_and_resume(jd_path: str, resume_path: str) -> Tuple[List[Document], List[Document]]:
    
    jd_loader = PyPDFLoader(jd_path)
    resume_loader = PyPDFLoader(resume_path)
    
    try:
        pages = jd_loader.load()
        resume = resume_loader.load()
        logger.info(f"Loaded {len(pages)} pages from JD and {len(resume)} pages from resume.")
        return pages, resume
    except Exception as e:
        logger.error(f"Error loading PDFs: {e}")
        raise RuntimeError(f"Failed to load documents: {e}")


def split_documents(
    pages: List[Document], 
    resume: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[Document], List[Document]]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    page_chunks = text_splitter.split_documents(pages)
    resume_chunks = text_splitter.split_documents(resume)

    logger.info(f"JD split into {len(page_chunks)} chunks")
    logger.info(f"Resume split into {len(resume_chunks)} chunks")
    
    return page_chunks, resume_chunks