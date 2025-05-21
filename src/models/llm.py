"""LLM model configuration for the video QA system."""

import os
from langchain_groq import ChatGroq
from langchain.schema.language_model import BaseLanguageModel
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import GROQ_API_KEY, LLM_MODEL


def get_llm() -> BaseLanguageModel:
    """Get the configured LLM model.
    
    Returns:
        A configured language model instance.
    """
    # Ensure API key is set
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize the LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
    )
    
    return llm