"""Complete pipeline for the video QA system."""

import os
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.transcription import VideoTranscriber
from models.vectorstore import TranscriptionVectorStore
from models.workflow import create_workflow
from models.graph_state import GraphState


class VideoQAPipeline:
    """Pipeline for video transcription and question answering."""
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.transcriber = VideoTranscriber()
        self.vector_store = TranscriptionVectorStore()
        self.workflow = None
        self.transcription_path = None
    
    def process_video(self, video_url: str, output_dir: str = "data") -> str:
        """Process a video: download, transcribe, and index.
        
        Args:
            video_url: URL of the YouTube video.
            output_dir: Directory to save outputs.
            
        Returns:
            Path to the transcription file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Transcribe the video
        transcription = self.transcriber.process_youtube_video(video_url, output_dir)
        self.transcription_path = os.path.join(output_dir, "transcription.txt")
        
        # Index the transcription
        self.load_transcription(self.transcription_path)
        
        return self.transcription_path
    
    def load_transcription(self, transcription_path: str) -> None:
        """Load and index an existing transcription.
        
        Args:
            transcription_path: Path to the transcription file.
        """
        self.transcription_path = transcription_path
        
        # Split and index the transcription
        documents = self.vector_store.load_transcription(transcription_path)
        vector_store = self.vector_store.index_documents(documents)
        
        # Create the workflow
        self.workflow = create_workflow(vector_store)
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the transcribed video.
        
        Args:
            question: The question to ask.
            
        Returns:
            The answer to the question.
        """
        if not self.workflow:
            if not self.transcription_path or not os.path.exists(self.transcription_path):
                raise ValueError("No transcription has been loaded. Please process a video first.")
            
            # Try to load the transcription if workflow isn't initialized
            self.load_transcription(self.transcription_path)
        
        # Prepare the input for the workflow
        inputs = {"initial_query": question}
        
        # Execute the workflow
        final_state = None
        for output in self.workflow.stream(inputs):
            final_state = output
        
        # Return the final answer
        if final_state and "final_answer" in final_state:
            return final_state["final_answer"]
        else:
            return "I couldn't generate an answer for your question." # Model definition and training