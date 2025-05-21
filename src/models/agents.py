"""Agent definitions for the video QA system."""

import json
import requests
import urllib.parse
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.schema.document import Document

from src.config import TAVILY_API_KEY
from src.models.llm import get_llm


# Get the LLM
llm = get_llm()

# Initialize Tavily retriever
web_retriever = TavilySearchAPIRetriever(k=2, api_key=TAVILY_API_KEY)


def language_detection_chain(state):
    """Create a chain to detect if text is in English or another language.
    
    Args:
        state: The current state of the graph.
        
    Returns:
        A language detection chain.
    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a Language Detection Agent. Your task is to determine if the given text is in English or not.

<|eot_id|><|start_header_id|>user<|end_header_id|> Analyze the following text and 
determine if it's in English or not. If it's in English, respond with the string "english". 
If it's not in English (i.e., it's in any other language), respond with the string "another".

Only respond with either "english" or "another", nothing else.

Text to analyze: {initial_query}

<|eot_id|> <|start_header_id|>assistant<|end_header_id|> """,
        input_variables=["initial_query"],
    )

    return prompt | llm | StrOutputParser()


def translation_agent(state):
    """Translate non-English text to English.
    
    Args:
        state: The current state of the graph.
        
    Returns:
        Updated state with translated query.
    """
    text = state["initial_query"]
    base_url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": "ar|en"  # Arabic to English by default, could be made dynamic
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data["responseStatus"] == 200:
            state["final_query"] = data["responseData"]["translatedText"]
        else:
            state["final_query"] = f"Translation error: {data['responseStatus']}"
    else:
        state["final_query"] = f"HTTP error: {response.status_code}"
    
    print(f"Translated query: {state['final_query']}")
    return state


def query_enhancement_agent(state):
    """Enhance the query for better search results.
    
    Args:
        state: The current state of the graph.
        
    Returns:
        Updated state with enhanced query.
    """
    template = """You are a helpful assistant that improve search queries based on a single input query.

Generate a single improved search query related to: {question}

Only respond with the generated query nothing else.
Output (single query):"""
    
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = prompt_rag_fusion | llm | StrOutputParser()
    
    question = state["final_query"]
    enhanced_query = generate_queries.invoke({"question": question})
    
    state['new_query'] = enhanced_query
    print(f"Enhanced query: {enhanced_query}")
    
    return state


def rag_retrieval_agent(state, vector_store):
    """Retrieve relevant information from the vector store.
    
    Args:
        state: The current state of the graph.
        vector_store: The vector store to retrieve from.
        
    Returns:
        Updated state with retrieved information.
    """
    print("---RAG SEARCH INFO RETRIEVAL---")
    new_query = state["new_query"]
 
    print(f"Searching for: {new_query}")

    # Retrieve documents from vector store
    docs = vector_store.similarity_search(new_query, k=3)
    
    # Join the documents' content
    combined_content = "\n".join([doc.page_content for doc in docs])
    
    # Add to state
    state["Rag_search"] = combined_content
    
    print(f"Retrieved {len(docs)} documents from vector store")
    return state


def web_retrieval_agent(state):
    """Retrieve information from the web using Tavily.
    
    Args:
        state: The current state of the graph.
        
    Returns:
        Updated state with web research results.
    """
    web_results = web_retriever.invoke(state["new_query"])
    
    # Convert to string if it's a list of documents
    if isinstance(web_results, list):
        web_context = "\n".join([doc.page_content for doc in web_results])
    else:
        web_context = str(web_results)
    
    state["web_research"] = web_context
    
    # Combine RAG and web results
    state["context"] = f"Information from document: {state['Rag_search']}\n\nInformation from web: {state['web_research']}"
    
    print("Retrieved information from web")
    return state


def answer_generation_agent(state):
    """Generate a final answer based on retrieved information.
    
    Args:
        state: The current state of the graph.
        
    Returns:
        Updated state with final answer.
    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )

    chain = (
        {"context": lambda x: state['context'], "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    state["final_answer"] = chain.invoke(state["new_query"])
    print(f"Generated answer: {state['final_answer']}")
    
    return state