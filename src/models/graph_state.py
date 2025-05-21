"""Definition of the state used in the LangGraph workflow."""

from typing import TypedDict


class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    
    # Input query
    initial_query: str
    
    # Language detection
    query_language: str
    
    # Processed query
    final_query: str
    
    # Enhanced query after processing
    new_query: str
    
    # Retrieved information
    Rag_search: str
    web_research: str
    context: str
    
    # Final answer
    final_answer: str