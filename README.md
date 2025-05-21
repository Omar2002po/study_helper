# 🎥 Video Transcription and Q\&A System

This project provides an end-to-end pipeline that allows users to input a YouTube video URL and interactively ask questions about its content. It combines powerful tools like OpenAI Whisper, Pinecone, and Tavily search, orchestrated via a multi-agent system built with LangGraph to deliver accurate, context-rich answers.

---

## 🚀 Features

✅ **YouTube Video Input**
➡ Accepts a YouTube URL and automatically downloads the video.

✅ **Audio Transcription**
➡ Extracts and transcribes audio using [OpenAI Whisper](https://github.com/openai/whisper).

✅ **Vector Storage with Pinecone**
➡ Transcription is chunked, embedded, and stored in a Pinecone vector database for efficient retrieval.

✅ **Retrieval-Augmented Generation (RAG)**
➡ Questions are answered using information retrieved from the transcription and web search results.

✅ **Web Search Augmentation**
➡ Uses [Tavily](https://www.tavily.com/) to retrieve the latest and most relevant information to supplement video-based answers.

✅ **LangGraph Multi-Agent Orchestration**
➡ Coordinates specialized agents (e.g., transcriber, retriever, web-searcher, generator) for modular, scalable reasoning.

✅ **Prompt Engineering Optimization**
➡ Custom prompts are designed to enhance the answer quality by guiding agent behavior intelligently.

---

## 🧠 System Overview

```
         ┌────────────┐
         │ YouTube URL│
         └─────┬──────┘
               ▼
      ┌─────────────────┐
      │ Download & Audio│
      │ Extraction       │
      └─────┬────────────┘
            ▼
    ┌────────────────────┐
    │ Whisper Transcriber│
    └─────┬──────────────┘
          ▼
    ┌──────────────────────┐
    │ Chunk & Embed Text   │──┐
    └────┬─────────────────┘  │
         ▼                    │
   ┌──────────────────────┐   │
   │ Store in Pinecone    │   │
   └──────────────────────┘   │
         ▼                    │
 ┌──────────────────────┐     ▼
 │ LangGraph Agents     │◄────┘
 └────────┬─────────────┘
          ▼
  ┌────────────────────────┐
  │ Tavily Web Search Agent│
  └────────┬───────────────┘
           ▼
   ┌─────────────────────────┐
   │ Answer Generation Agent │
   └─────────────────────────┘
           ▼
    🧑‍💻 Final Answer Output
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/video-qa.git
cd video-qa
```

2. **Install dependencies**

```bash
pip install -e .
```

3. **Configure environment variables**

```bash
cp .env.example .env
```

Update `.env` with your credentials:

* `GROQ_API_KEY`
* `PINECONE_API_KEY`
* `TAVILY_API_KEY`

---

## 🛠 Usage

### 🖥 CLI Interface

```bash
python -m src.main --video_url "https://www.youtube.com/watch?v=your_video_id"
```

### 🐍 Python API

```python
from src.pipeline import VideoQAPipeline

pipeline = VideoQAPipeline()

# Step 1: Process video
pipeline.process_video("https://www.youtube.com/watch?v=your_video_id")

# Step 2: Ask a question
answer = pipeline.ask_question("What is the main topic of this video?")
print(answer)
```

---

## 🗂 Project Structure

```
project/
├── data/               # Raw and processed data
├── notebooks/          # Reference notebooks
├── src/                # Core source code
│   ├── __init__.py
│   ├── data/           # Data handling
│   ├── models/         # LangGraph agents and prompts
│   ├── utils/          # Helper utilities
│   └── visualization/  # Optional visualization tools
├── tests/              # Unit and integration tests
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── setup.py            # Packaging info
```

---

## 🔧 Tech Stack

* **LangGraph** – Multi-agent workflow coordination
* **Whisper** – Automatic Speech Recognition (ASR)
* **Pinecone** – Vector database for fast semantic search
* **Tavily API** – Live web search augmentation
* **GROQ** – Language models for answer generation

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
Feel free to use, modify, and distribute it with attribution.

---

