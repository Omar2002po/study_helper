"""LangGraph workflow definition for the multi-agent video QA system."""

from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END

from src.models.graph_state import GraphState
from src.models.agents import (
    language_detection_chain,
    translation_agent,
    query_enhancement_agent,
    rag_retrieval_agent,
    web_retrieval_agent,
    answer_generation_agent,
)


def route_by_language(state: GraphState) -> str:
    """Route the workflow based on detected language.
    
    Args:
        state: The current state of the graph.
        
    Returns:
        Next node identifier based on language.
    """
    # Get the initial query
    initial_query = state['initial_query']
    
    # Create and invoke the language detector
    language_detector = language_detection_chain(state)
    query_language = language_detector.invoke({"initial_query": initial_query})
    print(f"Detected language: {query_language}")
    
    # Update the state with the detected language
    state["query_language"] = query_language
    
    # Route based on language
    if query_language.lower() == 'english':
        state["final_query"] = state["initial_query"]
        print("Language: English")
        return "english"
    else:
        state["query_language"] = 'another'
        print("Language: Non-English")
        return "another"


def create_workflow(vector_store):
    """Create the LangGraph workflow for the QA system.
    
    Args:
        vector_store: The vector store to use for retrievals.
        
    Returns:
        A compiled LangGraph workflow.
    """
    # Create the workflow with the defined state
    workflow = StateGraph(GraphState)
    
    # Add nodes for each agent
    workflow.add_node("translation", lambda state: translation_agent(state))
    workflow.add_node("query_enhancement", lambda state: query_enhancement_agent(state))
    workflow.add_node("rag_retrieval", lambda state: rag_retrieval_agent(state, vector_store))
    workflow.add_node("web_retrieval", lambda state: web_retrieval_agent(state))
    workflow.add_node("answer_generation", lambda state: answer_generation_agent(state))
    
    # Define the conditional routing from the start based on language
    workflow.add_conditional_edges(
        START,
        route_by_language,
        {
            "english": "query_enhancement",
            "another": "translation",
        },
    )
    
    # Define the rest of the workflow
    workflow.add_edge("translation", "query_enhancement")
    workflow.add_edge("query_enhancement", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "web_retrieval")
    workflow.add_edge("web_retrieval", "answer_generation")
    workflow.add_edge("answer_generation", END)
    
    # Compile the workflow
    return workflow.compile()