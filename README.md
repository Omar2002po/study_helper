# Video Transcription and Q&A System

This project provides a system for transcribing YouTube videos and enabling question-answering based on the video content. It uses LangGraph for multi-agent orchestration and RAG (Retrieval-Augmented Generation) for answering questions about the video content.

## Features

- YouTube video downloading
- Audio extraction and transcription using Whisper
- Vector storage of transcription chunks in Pinecone
- RAG-based question answering system
- Multi-agent system orchestrated by LangGraph

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/video-qa.git
cd video-qa
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Then edit the `.env` file to include your API keys for:
- GROQ_API_KEY
- PINECONE_API_KEY
- TAVILY_API_KEY

## Usage

### Command Line Interface

```bash
python -m src.main --video_url "https://www.youtube.com/watch?v=your_video_id"
```

### Python API

```python
from src.pipeline import VideoQAPipeline

# Initialize pipeline
pipeline = VideoQAPipeline()

# Process a video
pipeline.process_video("https://www.youtube.com/watch?v=your_video_id")

# Ask questions about the video
answer = pipeline.ask_question("What is the main topic of this video?")
print(answer)
```

## Project Structure

```
project/
├── data/               # Raw and processed data
├── notebooks/          # Original notebooks for reference
├── src/                # Source code
│   ├── __init__.py
│   ├── data/           # Data processing scripts
│   ├── models/         # Model definition and training
│   ├── utils/          # Helper functions
│   └── visualization/  # Plotting functions
├── tests/              # Unit tests
├── README.md           # Project documentation
├── requirements.txt    # Dependencies
└── setup.py            # For packaging
```

## License

[Your chosen license]