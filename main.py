import google.generativeai as genai
import os
import json as json_module
import datetime
import yaml
import toml
import time # Added for retry delays
import tool_manager # Import the new module
import traceback
from unittest.mock import patch # For sandboxed testing
from typing import Generator # For type hinting the generator
import random

# Import definitions from separate files
from tools import available_tools
# from agent_profiles import agent_profiles # We might not need this directly anymore if always dynamic

PROMPT_CONFIG_PATH = "prompts.toml"

def get_prompt_template(template_name: str) -> str | None:
    """
    Loads a specific prompt template string from the TOML configuration file.
    Returns None if the file or template is not found.
    """
    try:
        with open(PROMPT_CONFIG_PATH, 'r') as f:
            prompts_config = toml.load(f)
        print(f"DEBUG: Keys found in '{PROMPT_CONFIG_PATH}': {list(prompts_config.keys())}") # DEBUG line
        if template_name not in prompts_config:
            print(f"Error: Template key '{template_name}' not found in '{PROMPT_CONFIG_PATH}'. Available keys: {list(prompts_config.keys())}")
            return None
        return prompts_config.get(template_name)
    except FileNotFoundError:
        print(f"Error: Prompt configuration file '{PROMPT_CONFIG_PATH}' not found.")
        return None
    except toml.TomlDecodeError as e: # Be more specific for TOML parsing errors
        print(f"Error decoding TOML from '{PROMPT_CONFIG_PATH}': {e}")
        return None
    except Exception as e:
        print(f"Error loading prompt template '{template_name}' from '{PROMPT_CONFIG_PATH}'. Exception type: {type(e).__name__}, Details: {e}")
        return None

def refresh_api_keys_and_configure_gemini(api_keys_file_path=".api_keys.yaml"):
    """
    Reads API keys from the specified YAML file, updates environment variables,
    and attempts to configure the Gemini API using GOOGLE_API_KEY from the file.
    Returns a dictionary of loaded key details and a list of loaded key names.
    """
    current_loaded_api_keys_details = {}
    current_loaded_api_key_names = []
    google_api_key_from_file = None

    print(f"\nRefreshing API keys from '{api_keys_file_path}' for the current request...")

    if os.path.exists(api_keys_file_path):
        try:
            with open(api_keys_file_path, 'r', encoding='utf-8') as f:
                api_keys_from_file = yaml.safe_load(f)
            
            if isinstance(api_keys_from_file, dict):
                for key_name, key_data in api_keys_from_file.items():
                    if isinstance(key_data, dict) and 'key_value' in key_data:
                        key_value_str = str(key_data['key_value'])
                        os.environ[key_name] = key_value_str # Update environment
                        current_loaded_api_key_names.append(key_name)
                        current_loaded_api_keys_details[key_name] = {
                            "service_name": key_data.get("service_name", "N/A"),
                            "description": key_data.get("description", "N/A"),
                            "usage": key_data.get("usage", "# No usage example provided"),
                            "output_format_example": key_data.get("output_format_example", "# No output format example provided") # Add output format field
                        }
                        if key_name == "GOOGLE_API_KEY":
                            google_api_key_from_file = key_value_str
                    else:
                        print(f"Warning: Invalid format for key '{key_name}' in {api_keys_file_path}. Skipping.")
                
                if current_loaded_api_key_names:
                    print(f"Environment variables updated for API keys: {current_loaded_api_key_names}")
                else:
                    print(f"No valid API keys found in {api_keys_file_path} to load.")

            else:
                print(f"Warning: Content of {api_keys_file_path} is not a valid YAML dictionary.")
        except yaml.YAMLError as e:
            print(f"Error loading or parsing {api_keys_file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {api_keys_file_path}: {e}")
    else:
        print(f"Info: API key file '{api_keys_file_path}' not found. No API keys loaded for this request.")

    # Configure Gemini API based on what was loaded (or not) from the file for THIS request
    if google_api_key_from_file:
        try:
            genai.configure(api_key=google_api_key_from_file)
            print("Gemini API Key configured successfully using GOOGLE_API_KEY from file for this request.")
        except Exception as e:
            print(f"Error configuring Gemini API with key from file: {e}. Calls to Gemini may fail.")
    else:
        print("WARNING: GOOGLE_API_KEY not found in .api_keys.yaml for this request. "
              "Gemini API may not be configured or may use a previous/external configuration. "
              "This could lead to unexpected behavior if the key is required.")

    return current_loaded_api_keys_details, current_loaded_api_key_names

# --- API Key Loading and Environment Setup (Initial - will be refreshed per request) ---
loaded_api_key_names = []
api_keys_file_path_initial = ".api_keys.yaml" # Changed extension
loaded_api_keys_details = {} # Store details for later use

if os.path.exists(api_keys_file_path_initial):
    try:
        with open(api_keys_file_path_initial, 'r', encoding='utf-8') as f:
            api_keys_from_file = yaml.safe_load(f) # Use yaml.safe_load
        if isinstance(api_keys_from_file, dict):
            for key_name, key_data in api_keys_from_file.items():
                # Validate the nested structure
                if isinstance(key_data, dict) and 'key_value' in key_data:
                    key_value = key_data['key_value']
                    os.environ[key_name] = str(key_value) # Ensure it's a string for environ
                    loaded_api_key_names.append(key_name)
                    # Store the full details
                    loaded_api_keys_details[key_name] = {
                        "service_name": key_data.get("service_name", "N/A"), # Default if missing
                        "description": key_data.get("description", "N/A"), # Default if missing
                        "usage": key_data.get("usage", "# No usage example provided"), # Add usage field
                        "output_format_example": key_data.get("output_format_example", "# No output format example provided") # Add output format field
                    }
                else:
                     print(f"Warning: Initial load - Invalid format for key '{key_name}' in {api_keys_file_path_initial}. Expected a dictionary with at least a 'key_value'. Skipping.")
            if loaded_api_key_names:
                print(f"Successfully performed initial load and set environment variables for API keys: {loaded_api_key_names}")
        else:
            print(f"Warning: Initial load - Content of {api_keys_file_path_initial} is not a valid YAML dictionary.") # Updated warning
    except yaml.YAMLError as e: # Catch YAML specific errors
        print(f"Warning: Initial load - Could not load or parse {api_keys_file_path_initial}: {e}")
    except Exception as e: # Catch other potential errors
        print(f"Warning: Initial load - Unexpected error processing {api_keys_file_path_initial}: {e}")
else:
    print(f"Info: Initial load - API key file '{api_keys_file_path_initial}' not found. No external API keys loaded into environment initially.")

# --- Configure your Gemini API Key (Initial attempt - will be refreshed per request) ---
try:
    api_key_initial_gemini = os.environ.get("GOOGLE_API_KEY")
    if api_key_initial_gemini:
        genai.configure(api_key=api_key_initial_gemini)
        print("Gemini API Key configured initially (will be refreshed per request).")
    else:
        # This warning will likely be superseded by the refresh function's warning if key is still missing
        print("WARNING: GOOGLE_API_KEY not found in environment during initial setup.") 
except Exception as e:
    print(f"Error configuring Gemini API initially: {e}")

# --- Load dynamic tools at startup using the manager --- 
dynamic_tool_registry = tool_manager.load_dynamic_tools()

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
def generate_agent_details(user_request: str, predefined_tools_dict: dict, dynamic_tools_registry: dict, api_key_details: dict):
    """
    Generates prompt, identifies needed tools, and potentially generates new tool code.
    Informs LLM about available API keys (name, service, description) in the environment.
    WARNING: Generates executable code - EXTREMELY RISKY.
    """
    print(f"\n--- Generating Agent Details (API Key Aware, Registry Aware, Dynamic Code Risk!) for: '{user_request}' ---")

    predefined_tool_descriptions = "\n".join([
        f"- Name: {name}\n  Description: {data['schema']['description']}"
        for name, data in predefined_tools_dict.items()
    ])
    dynamic_tool_names = list(dynamic_tools_registry.keys())
    dynamic_tool_listing = "\n".join([f"- {name}" for name in dynamic_tool_names]) if dynamic_tool_names else "(None yet)"

    # Format API key details for the prompt
    api_key_prompt_parts = []
    if api_key_details:
        for key_name, details in api_key_details.items():
            service = details.get("service_name", "N/A")
            desc = details.get("description", "N/A")
            usage = details.get("usage", "# N/A") # Get usage snippet
            output_example = details.get("output_format_example", "# N/A") # Get output example
            # Format the output example within a code block for clarity
            output_block = f"```json\n{output_example}\n```" if output_example.strip().startswith('{') or output_example.strip().startswith('[') else f"```\n{output_example}\n```"
            api_key_prompt_parts.append(
                f"- Env Var Name: {key_name}\n"
                f"  Service: {service}\n"
                f"  Description: {desc}\n"
                f"  Usage Snippet:\n```python\n{usage}\n```\n"
                f"  Output Format Example:\n{output_block}"
            )
    api_key_listing_for_prompt = "\n".join(api_key_prompt_parts) if api_key_prompt_parts else "(None loaded or details missing)"

    # Load the base prompt template from TOML file
    prompt_template = get_prompt_template("generate_agent_details_prompt")
    if not prompt_template:
        print("Error: Could not load prompt template for generate_agent_details. Using a fallback or failing.")
        # Fallback to a very basic prompt or raise an error
        # For now, let's return None to indicate failure to generate details
        return None 

    # Populate the template with dynamic values
    prompt = prompt_template.format(
        user_request=user_request,
        predefined_tool_descriptions=predefined_tool_descriptions,
        dynamic_tool_listing=dynamic_tool_listing,
        api_key_listing_for_prompt=api_key_listing_for_prompt
    )

    try:
        # Consistent model usage, maybe parameterize later?
        model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06') # Using flash for potentially faster/cheaper generation
        response = model.generate_content(prompt)
        # Improved cleaning for potential markdown code blocks
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith('```json'):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith('```'):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()

        generation_result = json_module.loads(cleaned_response_text)

        # Basic validation - check for all expected keys
        expected_keys = [
            'generated_prompt', 
            'required_predefined_tools', 
            'required_dynamic_tools', 
            'generated_tool_code', 
            'required_new_tool_names',
            'example_dummy_arguments' # Added for testing new tools
            # 'generalized_from_tools' # Removed this key for now
            ]
        if isinstance(generation_result, dict) and all(k in generation_result for k in expected_keys):
            print(f"Agent details generated (API Key Aware, Registry Aware). Details:")
            print(f"  - Prompt: '{generation_result['generated_prompt']}'")
            print(f"  - Required Predefined Tools: {generation_result['required_predefined_tools']}")
            print(f"  - Required Dynamic Tools: {generation_result['required_dynamic_tools']}")
            print(f"  - New Tool Code Generated: {'Yes' if generation_result['generated_tool_code'] else 'No'}")
            print(f"  - New Tool Names: {generation_result['required_new_tool_names']}")
            return generation_result
        else:
            print(f"Error: Invalid JSON structure from generator:")
            print(generation_result)
            return None

    except json_module.JSONDecodeError as json_err:
        print(f"Error parsing JSON: {json_err}")
        print(f"Raw response: {response.text}")
        return None
    except Exception as e:
        print(f"Error during generation: {e}")
        # Consider logging response.prompt_feedback if available and relevant
        # print(f"Prompt Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
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
        model = genai.GenerativeModel('gemini-1.5-pro-latest') # Ensure this model is appropriate
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Error generating text: {e}"

# --- Core Agent Logic for Web/External Calls ---
def handle_web_request(user_query: str) -> Generator[str, None, None]:
    """
    Handles a single user request, performing all necessary steps.
    Includes retry logic for tool generation/testing and refactoring.
    This function is a GENERATOR. It yields log strings prefixed with "log: "
    and finally yields the agent reply string prefixed with "reply: ".
    """
    MAX_RETRIES = 5 # Define max retries for critical steps
    print(f"\n--- Handling Web Request (Generator): '{user_query}' ---") # Server console log
    yield f"log: Processing request: '{user_query}'..."

    # === Refresh API Keys and Configure Gemini for this request ===
    yield "log: Refreshing API keys and configuring Gemini..."
    current_api_key_details, loaded_key_names = refresh_api_keys_and_configure_gemini()
    if loaded_key_names:
        yield f"log: API keys loaded/refreshed: {', '.join(loaded_key_names)}"
    else:
        yield "log: No new API keys loaded from file. Using existing environment configuration if available."
    if "GOOGLE_API_KEY" in os.environ:
        yield "log: Gemini API should be configured."
    else:
        yield "log: WARNING: GOOGLE_API_KEY not found. Gemini calls may fail."

    # --- Retry Loop for Tool Generation & Testing ---
    generation_details = None
    generated_tool_code_str = None # Initialize here
    tool_code_is_valid = False # Assume invalid until tested successfully
    exec_globals = {} # Define scope for exec outside loop if needed later for adding tools
    newly_added_tool_names = [] # Track names of successfully tested NEW tools

    for attempt in range(1, MAX_RETRIES + 1):
        yield f"log: Attempt {attempt}/{MAX_RETRIES} to generate/validate tool..."
        # --- Attempt to generate agent details ---
        current_generation_details = generate_agent_details(user_query, available_tools, dynamic_tool_registry, current_api_key_details)

        if not current_generation_details:
            yield f"log: Generation failed on attempt {attempt}. Retrying..."
            time.sleep(1)
            continue

        # Generation succeeded for this attempt, store details
        generation_details = current_generation_details # Store the latest successful generation
        generated_prompt = generation_details['generated_prompt']
        required_predefined_tools = generation_details['required_predefined_tools']
        required_dynamic_tools = generation_details['required_dynamic_tools']
        current_generated_code_str = generation_details['generated_tool_code']
        current_required_new_names = generation_details['required_new_tool_names']
        dummy_args = generation_details.get("example_dummy_arguments")

        yield "log: Agent details generated successfully this attempt."
        if current_required_new_names: yield f"log: New tools specified: {current_required_new_names}"

        # If no new code generated this attempt, validation is trivial
        if not current_generated_code_str:
            yield "log: No new tool code generated. Validation successful."
            generated_tool_code_str = None # Ensure this is None if no code
            required_new_tool_names = []
            tool_code_is_valid = True
            break # Exit retry loop

        # --- Attempt to execute and test the generated code ---
        yield f"log: Testing generated code (Attempt {attempt})..."
        print("\n" + "="*30 + f" EXECUTING/TESTING DYNAMIC CODE (Attempt {attempt}) " + "="*30)
        is_current_code_valid = False
        current_exec_globals = {**globals()}
        if 'requests' not in current_exec_globals:
            try:
                import requests
                current_exec_globals['requests'] = requests
            except ImportError:
                pass
        if 'json' not in current_exec_globals:
            try:
                import json
                current_exec_globals['json'] = json
            except ImportError:
                pass

        try:
            exec(current_generated_code_str, current_exec_globals)
            yield f"log: Dynamic code execution successful (Attempt {attempt})."

            if current_required_new_names:
                primary_new_tool_name = current_required_new_names[0]
                if primary_new_tool_name in current_exec_globals and callable(current_exec_globals[primary_new_tool_name]):
                    new_func_obj = current_exec_globals[primary_new_tool_name]
                    yield f"log: Testing primary tool: {primary_new_tool_name}..."

                    if isinstance(dummy_args, dict):
                        with patch.dict(os.environ, {'TEST_API_KEY_GENERIC': 'dummy_test_value', 'OPEN_WEATHER_API_KEY': 'dummy_owm_key', 'SERPER_API_KEY': 'dummy_serper_key'}, clear=False):
                            try:
                                print(f"Testing {primary_new_tool_name} with dummy args: {dummy_args}")
                                test_result = new_func_obj(**dummy_args)
                                print(f"Test successful for {primary_new_tool_name}. Result: {test_result}")
                                yield f"log: Test successful for {primary_new_tool_name}."
                                is_current_code_valid = True # Test passed!
                            except Exception as test_e:
                                print(f"Test failed for {primary_new_tool_name}: {test_e}")
                                print(traceback.format_exc())
                                yield f"log: Test failed for {primary_new_tool_name}: {test_e}"
                                is_current_code_valid = False
                    else:
                        print(f"Warning: No dummy args for {primary_new_tool_name}. Skipping test.")
                        yield f"log: Warning: No dummy args for {primary_new_tool_name}. Test skipped."
                        is_current_code_valid = False # Consider test failed if skipped
                else:
                    print(f"Error: Primary function '{primary_new_tool_name}' not found/callable.")
                    yield f"log: Error: Primary function '{primary_new_tool_name}' not found/callable."
                    is_current_code_valid = False
            else: # Code generated but no names specified
                print("Warning: Code generated, but no tool names specified. Discarding code.")
                yield "log: Warning: Code generated, but no new tool names. Treating as valid (no tool added)."
                is_current_code_valid = True # Allow proceeding without adding tool

        except Exception as e_exec:
            print(f"CRITICAL ERROR EXECUTING CODE (Attempt {attempt}): {e_exec}")
            print(traceback.format_exc())
            yield f"log: Error executing dynamic code: {e_exec}"
            is_current_code_valid = False

        print("" + "="*80 + "\n")

        # If validation passed this attempt, store results and break loop
        if is_current_code_valid:
            yield f"log: Tool generation/validation successful on attempt {attempt}."
            tool_code_is_valid = True
            generated_tool_code_str = current_generated_code_str # Store the valid code
            required_new_tool_names = current_required_new_names # Store the corresponding names
            exec_globals = current_exec_globals # Store the globals where valid code was exec'd
            newly_added_tool_names = current_required_new_names # Track names for saving
            break # Exit the generation/testing retry loop
        else:
            yield f"log: Validation failed on attempt {attempt}. Retrying if possible..."
            time.sleep(1)

    # --- End of Retry Loop for Generation & Testing ---

    # Check if generation ultimately failed after all retries
    if not tool_code_is_valid:
        error_message = f"Failed to generate and validate tool after {MAX_RETRIES} attempts for: '{user_query}'"
        print(error_message)
        yield f"log: Error: {error_message}"
        yield f"reply: {error_message}"
        return

    # --- Proceed with validated/existing tools ---
    yield f"log: Using generated prompt: '{generated_prompt[:100]}...'"
    if required_predefined_tools: yield f"log: Adding predefined tools: {required_predefined_tools}"
    if required_dynamic_tools: yield f"log: Adding existing dynamic tools: {required_dynamic_tools}"

    all_tool_functions_for_worker = []
    tools_confirmed_for_worker = [] # New list to track tools passed to worker

    # Add required PREDEFINED tools
    for tool_name in required_predefined_tools:
        if tool_name in available_tools:
            all_tool_functions_for_worker.append(available_tools[tool_name]['function'])
            tools_confirmed_for_worker.append(tool_name) # Track tool
        else:
            print(f"Warning: Predefined tool '{tool_name}' not found by generator.")
            yield f"log: Warning: Predefined tool '{tool_name}' not found by generator."
    
    # Add required DYNAMIC tools from registry
    for tool_name in required_dynamic_tools: # Tools identified by generator as needed from registry
        if tool_name in dynamic_tool_registry:
            all_tool_functions_for_worker.append(dynamic_tool_registry[tool_name])
            tools_confirmed_for_worker.append(tool_name) # Track tool
        else:
            print(f"Warning: Required dynamic tool '{tool_name}' not found in registry.")
            yield f"log: Warning: Required dynamic tool '{tool_name}' not found in registry."

    # Add the successfully validated NEW tools (if any) to the worker list and registry
    successfully_added_tools = []
    if tool_code_is_valid and generated_tool_code_str and required_new_tool_names:
        yield f"log: Adding validated new tool(s) to worker: {required_new_tool_names}"
        for tool_name in required_new_tool_names:
            if tool_name in exec_globals and callable(exec_globals[tool_name]):
                tool_func = exec_globals[tool_name]
                all_tool_functions_for_worker.append(tool_func)
                dynamic_tool_registry[tool_name] = tool_func # Update live registry
                successfully_added_tools.append(tool_name) # Track for saving
                tools_confirmed_for_worker.append(tool_name) # Track tool
            else:
                print(f"Error: Validated tool function '{tool_name}' missing from exec_globals.")
                yield f"log: Error: Could not find validated function '{tool_name}' to add."

    # --- Save the newly generated tool to persistent storage (using manager) ---
    if successfully_added_tools and generated_tool_code_str:
        yield f"log: Saving new tool(s) to persistent registry: {', '.join(successfully_added_tools)}"
        tool_manager.save_dynamic_tool(successfully_added_tools, generated_tool_code_str)

        # --- Only attempt to refactor the tool registry with a 30% chance ---
        should_attempt_refactor = random.random() <= 0.3  # 30% chance to attempt refactoring
        
        if should_attempt_refactor:
            yield f"log: Randomly decided to attempt tool refactoring (30% chance)..."
            refactor_success = False
            for ref_attempt in range(1, MAX_RETRIES + 1):
                yield f"log: Attempt {ref_attempt}/{MAX_RETRIES} to refactor tool registry..."
                try:
                    refactor_result = tool_manager.refactor_tool_registry()
                    refactor_success = True # Assume success if no exception
                    yield f"log: Refactoring attempt {ref_attempt} successful."
                    # Log details from refactor result if any
                    if isinstance(refactor_result, list):
                        for ref_log in refactor_result: yield f"log: [Refactor] {ref_log}"
                    elif isinstance(refactor_result, str): yield f"log: [Refactor] {refactor_result}"
                    break # Exit refactor retry loop on success
                except Exception as ref_e:
                    yield f"log: Refactoring attempt {ref_attempt} failed: {ref_e}"
                    if ref_attempt < MAX_RETRIES:
                        yield f"log: Retrying refactoring..."
                        time.sleep(1)
                    else:
                        yield f"log: Refactoring failed after {MAX_RETRIES} attempts."
                        print(f"ERROR: Refactoring failed after {MAX_RETRIES} attempts.")
        else:
            yield f"log: Skipping tool refactoring (70% chance to skip)."

    # 2. Execute with Worker
    yield "log: Executing request with Gemini worker and available tools..."
    final_response = execute_request_with_worker(
        user_request=user_query, # Use the original user_query
        system_prompt=generated_prompt,
        tool_functions=all_tool_functions_for_worker
    )
    print("\n--- Final Response (from handle_web_request) --- ")
    print(final_response)
    print(f"Current Dynamic Registry (after web request): {list(dynamic_tool_registry.keys())}")
    yield "log: Request processing complete."

    # Yield summary of tools actually used by the worker
    if tools_confirmed_for_worker:
        unique_tools_used = sorted(list(set(tools_confirmed_for_worker)))
        yield f"log: tools_used_summary: {json_module.dumps(unique_tools_used)}"

    yield f"reply: {final_response}"

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

        # === Refresh API Keys and Configure Gemini for this turn ===
        current_api_key_details, _ = refresh_api_keys_and_configure_gemini()
        # We capture the list of names too, but don't explicitly use it here (it's used in the refresh func for print)
        # If Gemini wasn't configured (no key found/error), subsequent API calls might fail.
        
        # Get user input (or use predefined for testing)
        if turn_count == 1:
             test_request = "What is the weather in London?" # Should trigger NEW tool generation
        elif turn_count == 2:
             test_request = "Now what is the weather in Paris?" # Should REUSE generated tool
        elif turn_count == 3:
             test_request = "What is 100 - 11?" # Should use PREDEFINED tool
        elif turn_count == 4: # Add a new test case for search
             test_request = "What is the capital of France?" # Should trigger NEW tool generation using Serper
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

        # In CLI mode, we call the new handler function.
        # This makes the CLI behave similarly to the web request flow.
        # For CLI, the callback can simply be a print statement.
        # def cli_log_callback(message):
        # print(f"CLI LOG: {message}")

        # final_cli_response = handle_web_request(test_request, cli_log_callback)
        print(f"User Request (CLI): '{test_request}'")
        final_reply_from_cli = None
        for message_part in handle_web_request(test_request):
            if message_part.startswith("log: "):
                print(f"CLI LOG: {message_part[5:]}")
            elif message_part.startswith("reply: "):
                final_reply_from_cli = message_part[7:]
                print(f"\n--- CLI Final Response ---")
                print(final_reply_from_cli)
            else:
                print(f"CLI UNKNOWN: {message_part}")
        
        if final_reply_from_cli is None:
            print("\n--- CLI Final Response ---")
            print("(No final reply yielded by agent)")

        # The handle_web_request function already prints details via server console logs.
        # The callback prints CLI logs. The final response is printed below.
        # print("\n--- CLI Final Response ---")
        # print(final_cli_response)
        # The dynamic_tool_registry is updated by handle_web_request directly.
        # print(f"
# CLI - Current Dynamic Registry: {list(dynamic_tool_registry.keys())}")


    print("\n--- Test Complete --- ")
    # print(f"\nAvailable PRE-DEFINED Tools (from tools.py): {list(available_tools.keys())}")

    print(f"\nAvailable PRE-DEFINED Tools (from tools.py): {list(available_tools.keys())}")

    # Optional: Print available items after import
    # print(f"\nAvailable Tools: {list(available_tools.keys())}")
    # print(f"Available Profiles: {list(agent_profiles.keys())}") 