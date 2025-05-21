"""Example script to process a video and ask questions about it."""

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import VideoQAPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process a video and ask questions about it")
    parser.add_argument(
        "--video_url", 
        type=str, 
        default="https://www.youtube.com/watch?v=cdiD-9MMpb0",
        help="YouTube video URL to process"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Enter interactive question-answering mode after processing"
    )
    return parser.parse_args()


def main():
    """Run the example."""
    args = parse_args()
    
    print(f"Processing video: {args.video_url}")
    
    # Initialize the pipeline
    pipeline = VideoQAPipeline()
    
    # Process the video
    pipeline.process_video(args.video_url)
    
    if args.interactive:
        print("\n=== Interactive Question-Answering Mode ===")
        print("Type 'exit' to quit.")
        
        while True:
            question = input("\nAsk a question about the video: ")
            if question.lower() in ["exit", "quit", "q"]:
                break
            
            answer = pipeline.ask_question(question)
            print(f"\nAnswer: {answer}")
    else:
        # Ask a default question
        question = "What is the main topic of this video?"
        print(f"\nAsking: {question}")
        answer = pipeline.ask_question(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()