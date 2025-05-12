#!/usr/bin/env python3
"""
Example usage of the Dynamic Agent Orchestrator
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
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

def run_example():
    """Run example queries through the orchestrator"""
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Example 1: Weather and recipe request
    print("\n\n==== Example 1: Weather and Recipe Request ====")
    query1 = "What's the weather like in London and can you find me a simple pasta recipe?"
    print(f"User: {query1}")
    
    response1 = orchestrator.process_request(query1)
    print(f"Agent: {response1}")
    
    # Example 2: Calculator request
    print("\n\n==== Example 2: Calculator Request ====")
    query2 = "Calculate the area of a circle with radius 5"
    print(f"User: {query2}")
    
    response2 = orchestrator.process_request(query2)
    print(f"Agent: {response2}")
    
    # Example 3: Calendar check
    print("\n\n==== Example 3: Calendar Request ====")
    query3 = "What events do I have scheduled today?"
    print(f"User: {query3}")
    
    response3 = orchestrator.process_request(query3)
    print(f"Agent: {response3}")
    
    # Example 4: Search and research
    print("\n\n==== Example 4: Search and Research Request ====")
    query4 = "Research the history of artificial intelligence and summarize key developments"
    print(f"User: {query4}")
    
    response4 = orchestrator.process_request(query4)
    print(f"Agent: {response4}")
    
    # Example 5: Tool generation (requesting something not in our predefined tools)
    print("\n\n==== Example 5: Tool Generation Request ====")
    query5 = "Can you translate 'Hello, how are you?' to Spanish, French, and German?"
    print(f"User: {query5}")
    
    response5 = orchestrator.process_request(query5)
    print(f"Agent: {response5}")

if __name__ == "__main__":
    run_example() 