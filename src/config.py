"""Configuration settings for the video QA system."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Configuration
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2.gguf2.f16.gguf"
EMBEDDING_DIMENSION = 384

# Pinecone Configuration
PINECONE_INDEX_NAME = "meta"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20

# Device Configuration
USE_CUDA = True