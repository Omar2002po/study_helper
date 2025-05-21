from setuptools import setup, find_packages

setup(
    name="video_qa",
    version="0.1.0",
    description="Video transcription and question answering system with LangGraph multi-agent RAG",
    author="Omar Abdelnasser",
    author_email="omarabdelnasser313@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-groq",
        "groq",
        "pinecone-client",
        "gpt4all",
        "torch",
        "torchaudio",
        "pytube",
        "whisper",
        "langchain-community",
        "pydantic",
        "python-dotenv",
    ],
    python_requires=">=3.8",
)