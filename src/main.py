"""Main entry point for the video QA system."""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.transcription import VideoTranscriber
from src.models.vectorstore import TranscriptionVectorStore
from src.pipeline import VideoQAPipeline


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Video Transcription and QA System")
    parser.add_argument(
        "--video_url", 
        type=str, 
        help="YouTube video URL to process"
    )
    parser.add_argument(
        "--transcription_path", 
        type=str, 
        default=None,
        help="Path to existing transcription file (if already transcribed)"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default=None,
        help="Question to ask about the video"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Initialize the pipeline
    pipeline = VideoQAPipeline()
    
    # Process video or use existing transcription
    if args.video_url:
        print(f"Processing video: {args.video_url}")
        pipeline.process_video(args.video_url)
    elif args.transcription_path:
        if not os.path.exists(args.transcription_path):
            print(f"Transcription file not found: {args.transcription_path}")
            return
        print(f"Using existing transcription: {args.transcription_path}")
        pipeline.load_transcription(args.transcription_path)
    else:
        print("Please provide either a video URL or path to an existing transcription file.")
        return
    
    # Ask question if provided
    if args.question:
        print(f"\nQuestion: {args.question}")
        answer = pipeline.ask_question(args.question)
        print(f"\nAnswer: {answer}")
    else:
        # Interactive mode
        print("\nEnter your questions about the video (type 'exit' to quit):")
        while True:
            question = input("\nQuestion: ")
            if question.lower() in ["exit", "quit", "q"]:
                break
            
            answer = pipeline.ask_question(question)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()