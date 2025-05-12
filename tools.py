"""
Tools Module - Contains tools that can be used by agents
"""
import logging
import json
import requests
from datetime import datetime
from custom_tool import Tool


logger = logging.getLogger(__name__)

# Weather Tool
def get_weather(location):
    """
    Get the current weather for a location
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        dict: Weather information
    """
    logger.info(f"Getting weather for {location}")
    
    # In a real implementation, this would call a weather API
    # For demo purposes, we'll return mock data
    weather_data = {
        "location": location,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45,
        "wind_speed": 5,
        "forecast": "Clear skies throughout the day"
    }
    
    return json.dumps(weather_data)

# Recipe Tool
def find_recipe(query, cuisine=None, dietary=None):
    """
    Find recipes matching a query
    
    Args:
        query: The recipe search query
        cuisine: Optional cuisine type
        dietary: Optional dietary restrictions
        
    Returns:
        dict: Recipe information
    """
    logger.info(f"Finding recipe for {query}, cuisine={cuisine}, dietary={dietary}")
    
    # In a real implementation, this would call a recipe API
    # For demo purposes, we'll return mock data
    recipe_data = {
        "query": query,
        "cuisine": cuisine,
        "dietary": dietary,
        "results": [
            {
                "name": f"Best {query.title()} Recipe",
                "ingredients": ["Ingredient 1", "Ingredient 2", "Ingredient 3"],
                "instructions": ["Step 1", "Step 2", "Step 3"],
                "prep_time": "15 minutes",
                "cook_time": "30 minutes"
            },
            {
                "name": f"Quick {query.title()} Recipe",
                "ingredients": ["Ingredient A", "Ingredient B", "Ingredient C"],
                "instructions": ["Step 1", "Step 2", "Step 3"],
                "prep_time": "10 minutes",
                "cook_time": "20 minutes"
            }
        ]
    }
    
    return json.dumps(recipe_data)

# Calendar Tool
def check_calendar(date=None):
    """
    Check calendar for events
    
    Args:
        date: Date to check (defaults to today)
        
    Returns:
        dict: Calendar events for the specified date
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Checking calendar for {date}")
    
    # In a real implementation, this would call a calendar API
    # For demo purposes, we'll return mock data
    calendar_data = {
        "date": date,
        "events": [
            {
                "title": "Team Meeting",
                "time": "10:00 AM - 11:00 AM",
                "location": "Conference Room A"
            },
            {
                "title": "Lunch with Client",
                "time": "12:30 PM - 1:30 PM",
                "location": "Downtown Cafe"
            },
            {
                "title": "Project Review",
                "time": "3:00 PM - 4:00 PM",
                "location": "Virtual Meeting Room"
            }
        ]
    }
    
    return json.dumps(calendar_data)

# Search Tool
def search_web(query):
    """
    Search the web for information
    
    Args:
        query: Search query
        
    Returns:
        dict: Search results
    """
    logger.info(f"Searching web for: {query}")
    
    # In a real implementation, this would call a search API
    # For demo purposes, we'll return mock data
    search_data = {
        "query": query,
        "results": [
            {
                "title": f"Information about {query}",
                "snippet": f"This is a summary of information about {query} that would be returned by a search engine.",
                "url": f"https://example.com/result1"
            },
            {
                "title": f"More on {query}",
                "snippet": f"Additional details related to {query} that provide more context and information.",
                "url": f"https://example.com/result2"
            },
            {
                "title": f"{query} - Wikipedia",
                "snippet": f"Wikipedia article about {query} with comprehensive background and details.",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
            }
        ]
    }
    
    return json.dumps(search_data)

# Calculator Tool
def calculate(expression):
    """
    Evaluate a mathematical expression
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        dict: Result of the calculation
    """
    logger.info(f"Calculating: {expression}")
    
    try:
        # Using eval is generally not recommended in production code without proper validation
        # This is just for demonstration purposes
        result = eval(expression)
        
        return json.dumps({
            "expression": expression,
            "result": result
        })
    except Exception as e:
        return json.dumps({
            "expression": expression,
            "error": str(e)
        })

def get_all_tools():
    """
    Get all available tools
    
    Returns:
        list: List of all Tool objects
    """
    # Define tools using ADK Tool class
    weather_tool = Tool(
        name="get_weather",
        description="Get current weather information for a specified location",
        function=get_weather
    )
    
    recipe_tool = Tool(
        name="find_recipe",
        description="Find recipes based on query, optional cuisine type, and dietary restrictions",
        function=find_recipe
    )
    
    calendar_tool = Tool(
        name="check_calendar",
        description="Check calendar events for a specified date (defaults to today)",
        function=check_calendar
    )
    
    search_tool = Tool(
        name="search_web",
        description="Search the web for information on a specified query",
        function=search_web
    )
    
    calculator_tool = Tool(
        name="calculate",
        description="Evaluate a mathematical expression and return the result",
        function=calculate
    )
    
    return [
        weather_tool,
        recipe_tool,
        calendar_tool,
        search_tool,
        calculator_tool
    ] 