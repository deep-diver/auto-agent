import datetime

# --- Tool Definitions ---

# === Time Tool ===
def get_current_time() -> str:
    """
    Returns the current date and time in ISO format.
    """
    print("---> Calling get_current_time()")
    return datetime.datetime.now().isoformat()

get_current_time_schema = {
    "name": "get_current_time",
    "description": "Gets the current date and time.",
    "parameters": {
        "type": "object",
        "properties": {}, # No parameters needed for this tool
        "required": []
    }
}

# === Calculator Tools ===

# Add function
def add(a: float, b: float) -> float:
    """Calculates the sum of two numbers."""
    print(f"---> Calling add(a={a}, b={b})")
    return a + b

add_schema = {
    "name": "add",
    "description": "Adds two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number."},
            "b": {"type": "number", "description": "The second number."}
        },
        "required": ["a", "b"]
    }
}

# Subtract function
def subtract(a: float, b: float) -> float:
    """Calculates the difference between two numbers (a - b)."""
    print(f"---> Calling subtract(a={a}, b={b})")
    return a - b

subtract_schema = {
    "name": "subtract",
    "description": "Subtracts the second number from the first number (a - b).",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number (minuend)."},
            "b": {"type": "number", "description": "The second number (subtrahend)."}
        },
        "required": ["a", "b"]
    }
}

# === Available Tools Dictionary ===
# This dictionary maps tool names to their function and schema
available_tools = {
    "get_current_time": {
        "function": get_current_time,
        "schema": get_current_time_schema
    },
    "add": {
        "function": add,
        "schema": add_schema
    },
    "subtract": {
        "function": subtract,
        "schema": subtract_schema
    }
} 