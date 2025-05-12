#!/usr/bin/env python3
"""
Dynamic Agent Orchestrator - Main Entry Point
"""
import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    exit(1)

def main():
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Interactive shell
    print("Welcome to Dynamic Agent Orchestrator!")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        try:
            response = orchestrator.process_request(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            print(f"\nError: {e}")
    
    print("Thank you for using Dynamic Agent Orchestrator!")

if __name__ == "__main__":
    main() 