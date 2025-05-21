"""Vector store operations for storing and retrieving document embeddings."""

import os
import time
from typing import List, Optional
from langchain_community.vectorstores.pinecone import (Pinecone)
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from pinecone import Pinecone, ServerlessSpec
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class TranscriptionVectorStore:
    """Class for managing the vector store for transcription data."""
    
    def __init__(self):
        """Initialize the vector store."""
        # Set up embeddings
        gpt4all_kwargs = {'allow_download': 'True'}
        self.embeddings = GPT4AllEmbeddings(
            model_name=EMBEDDING_MODEL,
            gpt4all_kwargs=gpt4all_kwargs
        )
        
        # Set up text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        self._initialize_index()
        
        # Initialize vector store
        self.vector_store = None
    
    def _initialize_index(self) -> None:
        """Initialize the Pinecone index."""
        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in self.pinecone_client.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            # Wait for index to be ready
            while not self.pinecone_client.describe_index(PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)
            print(f"Pinecone index {PINECONE_INDEX_NAME} created successfully")
        else:
            print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
    
    def load_transcription(self, transcription_path: str) -> List[Document]:
        """Load and split a transcription into documents.
        
        Args:
            transcription_path: Path to the transcription file.
            
        Returns:
            List of Document objects.
        """
        # Read the transcription
        with open(transcription_path, "r", encoding="utf-8") as file:
            transcription_text = file.read()
        
        # Create a document
        doc = Document(page_content=transcription_text, metadata={"source": transcription_path})
        
        # Split the document
        documents = self.text_splitter.split_documents([doc])
        print(f"Split transcription into {len(documents)} chunks")
        
        return documents
    
    def index_documents(self, documents: List[Document]) -> VectorStore:
        """Index documents in the vector store.
        
        Args:
            documents: List of Document objects to index.
            
        Returns:
            The vector store instance.
        """
        # Create Pinecone Vector Store
        self.vector_store = Pinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        print(f"Indexed {len(documents)} documents in Pinecone")
        
        return self.vector_store
    from langchain_community.vectorstores import Pinecone

    def get_retriever(self, k: int = 4):
        """Get a retriever for the vector store.
        
        Args:
            k: Number of documents to retrieve.
            
        Returns:
            A retriever instance.
        """
        if self.vector_store is None:
            # Connect to existing index
            self.vector_store = Pinecone(
                index_name=PINECONE_INDEX_NAME,
                embedding=self.embeddings,
            )
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})