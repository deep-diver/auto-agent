import google.generativeai as genai
import os
import json

# Import definitions from separate files
from tools import available_tools
# from agent_profiles import agent_profiles # We might not need this directly anymore if always dynamic

# --- Registry for Dynamically Created Tools ---
dynamically_created_tools_registry = {}

# --- API Key Loading and Environment Setup ---
loaded_api_key_names = []
api_keys_file_path = ".api_keys.json"
if os.path.exists(api_keys_file_path):
    try:
        with open(api_keys_file_path, 'r') as f:
            api_keys_from_file = json.load(f)
        if isinstance(api_keys_from_file, dict):
            for key_name, key_value in api_keys_from_file.items():
                os.environ[key_name] = str(key_value) # Ensure it's a string for environ
                loaded_api_key_names.append(key_name)
            if loaded_api_key_names:
                print(f"Successfully loaded and set environment variables for API keys: {loaded_api_key_names}")
        else:
            print(f"Warning: Content of {api_keys_file_path} is not a valid JSON dictionary.")
    except Exception as e:
        print(f"Warning: Could not load or parse {api_keys_file_path}: {e}")
else:
    print(f"Info: API key file '{api_keys_file_path}' not found. No external API keys loaded into environment.")

# --- Configure your Gemini API Key ---
# IMPORTANT: Set the GOOGLE_API_KEY environment variable in your system
# or replace "YOUR_GOOGLE_API_KEY" with your actual key for testing (not recommended for production).
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback for local testing if env var not set, but prompt user
        print("WARNING: GOOGLE_API_KEY environment variable not found.")
        api_key = input("Please enter your GOOGLE_API_KEY: ")
        if not api_key: # If user still doesn't provide it
            raise ValueError("API key not provided. Please set the GOOGLE_API_KEY environment variable or enter it when prompted.")
    genai.configure(api_key=api_key)
    print("Gemini API Key configured.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Potentially exit or raise further if API key is critical for all operations
    # For now, we'll allow the script to continue so other parts can be developed,
    # but API calls will fail.

# --- Tool Definitions ---

# 1. Tool Function Implementation
def get_current_time() -> str:
    """
    Returns the current date and time in ISO format.
    """
    return datetime.datetime.now().isoformat()

# 2. Tool Schema for Gemini
get_current_time_schema = {
    "name": "get_current_time",
    "description": "Gets the current date and time.",
    "parameters": {
        "type": "object",
        "properties": {}, # No parameters needed for this tool
        "required": []
    }
}

# --- Calculator Tools ---

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

# Store tools for later lookup
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

# --- Agent Profiles Definitions ---

# Define profiles for different agent capabilities
agent_profiles = {
    "TimeAgent": {
        "description": "Handles requests about the current date and time.",
        "system_prompt": "You are a helpful assistant specializing in time-related queries. When asked for the current time or date, use the available `get_current_time` tool.",
        "associated_tools": ["get_current_time"]
    },
    "CalculatorAgent": {
        "description": "Handles arithmetic calculations like addition and subtraction.",
        "system_prompt": "You are a helpful assistant specializing in calculations. Use the `add` tool for addition and the `subtract` tool for subtraction. Clearly state the result of the calculation.",
        "associated_tools": ["add", "subtract"]
    },
    # Future agent profiles will be added here
    # e.g., "WeatherAgent", "GeneralQAAgent"
}

# --- Dynamic Generation (Prompt + Tool Code + Registry Awareness + API Key Awareness) ---
def generate_agent_details(user_request: str, predefined_tools_dict: dict, dynamic_tools_registry: dict, available_env_api_keys: list):
    """
    Generates prompt, identifies needed tools, and potentially generates new tool code.
    Informs LLM about available API keys in the environment.
    WARNING: Generates executable code - EXTREMELY RISKY.
    """
    print(f"\n--- Generating Agent Details (API Key Aware, Registry Aware, Dynamic Code Risk!) for: '{user_request}' ---")

    predefined_tool_descriptions = "\n".join([
        f"- Name: {name}\n  Description: {data['schema']['description']}"
        for name, data in predefined_tools_dict.items()
    ])
    dynamic_tool_names = list(dynamic_tools_registry.keys())
    dynamic_tool_listing = "\n".join([f"- {name}" for name in dynamic_tool_names]) if dynamic_tool_names else "(None yet)"
    
    api_key_listing_for_prompt = ", ".join(available_env_api_keys) if available_env_api_keys else "(None loaded)"

    prompt = f"""
Analyze the user request considering available tools and API keys. Decide action based on:
(a) Pre-defined tools.
(b) Dynamically created tools (already available).
(c) A mix of pre-defined and dynamic tools.
(d) Generating code for a NEW tool (if nothing existing is suitable). If the new tool needs an API key, it MUST try to use one listed as available in the environment.

Your goal is to:
1. Identify names of PRE-DEFINED tools needed.
2. Identify names of DYNAMICALLY CREATED tools needed.
3. If NEITHER existing type is suitable, generate complete, Python code (with imports) for a NEW function. If this function requires an API key, it MUST attempt to retrieve it from the environment using os.environ.get('KEY_NAME'). Provide the function name(s) defined.
4. Generate a concise system prompt for an AI assistant to fulfill the request, instructing it to use the identified tools.

User Request: "{user_request}"

Pre-defined Tools (Prefer these if suitable):
{predefined_tool_descriptions}

Dynamically Created Tools (Available by name):
{dynamic_tool_listing}

API Keys Available in Environment (for generated code via os.environ.get()):
{api_key_listing_for_prompt}

Please respond ONLY with a valid JSON object containing:
- "generated_prompt": (string) The system prompt for the assistant.
- "required_predefined_tools": (list of strings) Names from Pre-defined Tools.
- "required_dynamic_tools": (list of strings) Names from Dynamically Created Tools.
- "generated_tool_code": (string or null) Code for a NEW tool, or null if not needed.
- "required_new_tool_names": (list of strings) Name(s) from generated_tool_code.

Example (New Tool Needing API Key):
Request: "What is the weather in London?" (Assume WEATHER_API_KEY is listed as available)
Output:
{{
  "generated_prompt": "You are a weather assistant. Use the 'get_weather' tool for London.",
  "required_predefined_tools": [],
  "required_dynamic_tools": [],
  "generated_tool_code": "import requests\nimport os\n\ndef get_weather(city):\n    print(f'---> Dynamically generated tool get_weather({{city}}) called')\n    api_key = os.environ.get('WEATHER_API_KEY')\n    if not api_key: return 'Error: WEATHER_API_KEY not found in environment for generated tool.'\n    try:\n        url = f'https://api.openweathermap.org/data/2.5/weather?q={{city}}&appid={{api_key}}&units=metric'\n        response = requests.get(url)\n        response.raise_for_status()\n        data = response.json()\n        return f'Weather in {{city}}: {{data['weather'][0]['description']}}, Temp: {{data['main']['temp']}}C (Live via generated tool)'\n    except Exception as e:\n        return f'Error in generated weather tool: {{e}}'",
  "required_new_tool_names": ["get_weather"]
}}

Example (Pre-defined Tool):
Request: "What is 10 minus 3?"
Output:
{{
  "generated_prompt": "You are a calculator. Use the 'subtract' tool to calculate 10 minus 3.",
  "required_predefined_tools": ["subtract"],
  "required_dynamic_tools": [],
  "generated_tool_code": null,
  "required_new_tool_names": []
}}

Example (Using Previously Generated Dynamic Tool):
Request: "Now what is the weather in Paris?" (Assume 'get_weather' was generated in the previous turn)
Output:
{{
  "generated_prompt": "You are a weather assistant. Use the previously generated 'get_weather' tool for Paris.",
  "required_predefined_tools": [],
  "required_dynamic_tools": ["get_weather"],
  "generated_tool_code": null,
  "required_new_tool_names": []
}}

JSON Response:
"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        response = model.generate_content(prompt)
        cleaned_response_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        generation_result = json.loads(cleaned_response_text)

        # Basic validation - check for all expected keys
        expected_keys = ['generated_prompt', 'required_predefined_tools', 'required_dynamic_tools', 'generated_tool_code', 'required_new_tool_names']
        if isinstance(generation_result, dict) and all(k in generation_result for k in expected_keys):
            print(f"Agent details generated (API Key Aware, Registry Aware).")
            print(f"  - Prompt: '{generation_result['generated_prompt']}'")
            print(f"  - Required Predefined Tools: {generation_result['required_predefined_tools']}")
            print(f"  - Required Dynamic Tools: {generation_result['required_dynamic_tools']}")
            print(f"  - New Tool Code Generated: {'Yes' if generation_result['generated_tool_code'] else 'No'}")
            print(f"  - New Tool Names: {generation_result['required_new_tool_names']}")
            return generation_result
        else:
            print(f"Error: Invalid JSON structure from generator: {generation_result}")
            return None

    except json.JSONDecodeError as json_err:
        print(f"Error parsing JSON: {json_err}")
        print(f"Raw response: {response.text}")
        return None
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

# --- Router Logic --- 
# (Old route_request function should be deleted from here if it wasn't automatically)

# --- Worker Logic (Modified for Dynamic Tools) --- 
def execute_request_with_worker(user_request: str, system_prompt: str, tool_functions: list):
    """
    Executes the user request using the provided system prompt and a list of actual tool functions.
    WARNING: Assumes tool_functions may contain dynamically generated code, which is risky.

    Args:
        user_request: The original user request.
        system_prompt: The dynamically generated system prompt for the worker.
        tool_functions: A list of callable function objects to be made available to Gemini.

    Returns:
        The final text response from the Gemini model, or an error message.
    """
    print(f"\n--- Executing Request with Dynamic Prompt & Tools (Dynamic Code Risk!) ---")
    print(f"System Prompt:\n'''{system_prompt}'''")
    print(f"Providing tools to worker: {[f.__name__ for f in tool_functions] if tool_functions else 'None'}")

    try:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-pro-latest',
            system_instruction=system_prompt,
            tools=tool_functions # Pass the actual list of callable functions
        )

        chat = model.start_chat(enable_automatic_function_calling=True)
        print(f"Sending message to worker: '{user_request}'")
        response = chat.send_message(user_request)

        final_text = "".join([part.text for part in response.parts if hasattr(part, 'text')])
        print(f"Worker response received.")
        return final_text

    except Exception as e:
        print(f"Error during worker execution: {e}")
        return f"Error during worker execution: {e}"

# --- Basic Gemini Interaction (can be removed or kept for other purposes) ---
def generate_text_with_gemini(prompt_text: str):
    """
    Generates text using the Gemini Pro model.
    (This function will be adapted later to use function calling)
    """
    try:
        # For basic text generation, use a standard model
        # We'll use the same model as routing for consistency
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Error generating text: {e}"

# --- Main Execution (Updated for API Key File and passing key names) ---
if __name__ == "__main__":
    print("\n" + "*"*80)
    print("WARNING: THIS SCRIPT WILL ATTEMPT TO EXECUTE PYTHON CODE GENERATED BY AN LLM.")
    print("THIS IS EXTREMELY DANGEROUS AND CAN LEAD TO SECURITY VULNERABILITIES OR SYSTEM DAMAGE.")
    print("PROCEED WITH EXTREME CAUTION AND ONLY IN A CONTROLLED, ISOLATED ENVIRONMENT.")
    print("DO NOT RUN THIS ON ANY SYSTEM WITH SENSITIVE DATA.")
    print("YOU HAVE BEEN WARNED.")
    print("" + "*"*80 + "\n")
    
    # input("Press Enter to continue if you understand the risks and wish to proceed...") # Optional safety break

    print("--- Starting Auto-Agent Test (Dynamic Tool Code + Registry + API Keys - EXTREMELY RISKY!) ---")
    
    # Loop for multiple turns to test registry
    turn_count = 0
    while True:
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")
        
        # Get user input (or use predefined for testing)
        if turn_count == 1:
             test_request = "What is the weather in London?" # Should trigger NEW tool generation
        elif turn_count == 2:
             test_request = "Now what is the weather in Paris?" # Should REUSE generated tool
        elif turn_count == 3:
             test_request = "What is 100 - 11?" # Should use PREDEFINED tool
        else:
            try:
                test_request = input("Enter your request (or type 'quit'): ")
                if test_request.lower() == 'quit':
                    break
            except EOFError: # Handle case where input is piped
                break 

        if not test_request:
            continue
            
        print(f"User Request: '{test_request}'")

        # 1. Generate agent details, passing the CURRENT registry state AND available API key names
        generation_details = generate_agent_details(test_request, available_tools, dynamically_created_tools_registry, loaded_api_key_names)

        if generation_details:
            generated_prompt = generation_details['generated_prompt']
            required_predefined_tools = generation_details['required_predefined_tools']
            required_dynamic_tools = generation_details['required_dynamic_tools']
            generated_tool_code_str = generation_details['generated_tool_code']
            required_new_tool_names = generation_details['required_new_tool_names']

            all_tool_functions_for_worker = []

            # Add required PREDEFINED tools
            for tool_name in required_predefined_tools:
                if tool_name in available_tools:
                    all_tool_functions_for_worker.append(available_tools[tool_name]['function'])
                else:
                    print(f"Warning: Generator specified pre-defined tool '{tool_name}' not found.")
            
            # Add required DYNAMIC tools from registry
            for tool_name in required_dynamic_tools:
                if tool_name in dynamically_created_tools_registry:
                    all_tool_functions_for_worker.append(dynamically_created_tools_registry[tool_name])
                    print(f"Reusing dynamically generated tool: '{tool_name}'")
                else:
                    print(f"Warning: Generator specified dynamic tool '{tool_name}' not found in registry.")

            # DANGEROUS PART: Execute NEW dynamically generated code if present
            if generated_tool_code_str:
                print("\n" + "="*30 + " EXECUTING DYNAMICALLY GENERATED CODE " + "="*30)
                print("Code to be executed (via exec()):\n---\n" + generated_tool_code_str + "\n---")
                try:
                    exec(generated_tool_code_str, globals())
                    print("Dynamic code execution attempted.")
                    
                    # Add newly defined functions to the list AND the registry
                    for new_tool_name in required_new_tool_names:
                        if new_tool_name in globals() and callable(globals()[new_tool_name]):
                            new_func_obj = globals()[new_tool_name]
                            all_tool_functions_for_worker.append(new_func_obj)
                            # --- ADD TO REGISTRY --- 
                            dynamically_created_tools_registry[new_tool_name] = new_func_obj
                            print(f"Successfully registered dynamically generated function: '{new_tool_name}'")
                        else:
                            print(f"Error: Dynamic function '{new_tool_name}' not callable after exec().")
                except Exception as e:
                    print(f"CRITICAL ERROR EXECUTING DYNAMIC CODE: {e}")
                    generated_prompt = f"Error executing dynamic tool code: {e}"
                    all_tool_functions_for_worker = [] # Clear tools on error
                print("" + "="*80 + "\n")

            # 2. Execute with Worker 
            final_response = execute_request_with_worker(
                user_request=test_request,
                system_prompt=generated_prompt,
                tool_functions=all_tool_functions_for_worker # Pass the combined list
            )
            print("\n--- Final Response --- ")
            print(final_response)
        else:
            print(f"\nCould not generate agent details for request: '{test_request}'")
        
        print(f"\nCurrent Dynamic Registry: {list(dynamically_created_tools_registry.keys())}")

    print("\n--- Test Complete --- ")
    # print(f"\nAvailable PRE-DEFINED Tools (from tools.py): {list(available_tools.keys())}")

    print(f"\nAvailable PRE-DEFINED Tools (from tools.py): {list(available_tools.keys())}")

    # Optional: Print available items after import
    # print(f"\nAvailable Tools: {list(available_tools.keys())}")
    # print(f"Available Profiles: {list(agent_profiles.keys())}") 