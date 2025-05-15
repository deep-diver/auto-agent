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
def generate_agent_details(user_request: str,
                           predefined_tools_dict: dict,
                           dynamic_tools_registry: dict,
                           api_key_details: dict,
                           previous_tool_code_to_refine: str | None = None,
                           feedback_for_refinement: str | None = None):
    """
    Generates prompt, identifies needed tools, and potentially generates new tool code.
    If previous_tool_code_to_refine and feedback_for_refinement are provided,
    it instructs the LLM to refine the existing tool code.
    Informs LLM about available API keys (name, service, description) in the environment.
    WARNING: Generates executable code - EXTREMELY RISKY.
    """
    print(f"\n--- Generating Agent Details (API Key Aware, Registry Aware, Dynamic Code Risk!) for: '{user_request}' ---")
    if previous_tool_code_to_refine and feedback_for_refinement:
        print(f"Attempting to refine previous tool code based on feedback.")
        print(f"Previous code snippet: {previous_tool_code_to_refine[:200]}..." if previous_tool_code_to_refine else "No previous code.")
        print(f"Feedback: {feedback_for_refinement[:200]}..." if feedback_for_refinement else "No feedback.")

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
    # This template should instruct Gemini on the JSON output format.
    base_prompt_template = get_prompt_template("generate_agent_details_prompt")
    if not base_prompt_template:
        print("Error: Could not load prompt template 'generate_agent_details_prompt'. Cannot generate agent details.")
        return None 

    # Add refinement instructions if feedback is provided
    refinement_instructions = ""
    if previous_tool_code_to_refine and feedback_for_refinement:
        refinement_instructions = f"""

--------------------------------------------------------------------------------
**TASK: REFINE EXISTING TOOL or CREATE NEW TOOL BASED ON FEEDBACK**

You are in a refinement loop. A previous attempt to generate or use a tool has been made.
Your task is to EITHER refine the Python code of the PREVIOUS tool OR generate Python code for a COMPLETELY NEW tool if the previous approach was misguided.
Base your decision on the provided feedback.

**Previous Tool Code (Python):**
```python
{previous_tool_code_to_refine}
```

**Feedback on Previous Tool/Attempt:**
{feedback_for_refinement}

**Instructions for this refinement iteration:**
1.  Analyze the original user request, the previous tool code, and the feedback.
2.  If the feedback suggests improving the `Previous Tool Code`, provide the **complete, refined Python code** for the tool in the `generated_tool_code` field of your JSON response. Ensure the refined tool still aims to solve the original user request.
3.  If the feedback suggests the `Previous Tool Code` was a wrong approach or a new tool is needed, generate Python code for a **new tool** in `generated_tool_code`.
4.  If the feedback suggests NO tool is needed, or the existing non-dynamic tools are sufficient, leave `generated_tool_code` as an empty string or null, and set `required_new_tool_names` to an empty list.
5.  Update `generated_prompt` (system prompt for the agent that will use the tool), `required_predefined_tools`, `required_dynamic_tools`, and `required_new_tool_names` according to your new plan.
6.  Provide `example_dummy_arguments` if you generate or refine a tool.
7.  Ensure your entire response is a single JSON object.
--------------------------------------------------------------------------------
"""
    elif previous_tool_code_to_refine and not feedback_for_refinement:
        # This case might indicate a tool was generated, but evaluation failed or didn't give specific feedback.
        # We might instruct to try re-generating or just note it.
        refinement_instructions = f"""

--------------------------------------------------------------------------------
**NOTE: PREVIOUS TOOL EXISTS BUT NO SPECIFIC FEEDBACK**

A tool was generated in a previous iteration:
```python
{previous_tool_code_to_refine}
```
However, no specific feedback for improvement was provided.
Re-evaluate if this tool is appropriate for the user request. You may choose to:
1. Keep using this tool (do not regenerate code, but ensure it's listed in required_dynamic_tools if it was saved).
2. Generate a new, different tool if you think this one is not suitable.
3. Decide no tool is needed.
Proceed with generating the JSON response as usual.
--------------------------------------------------------------------------------
"""


    # Populate the template with dynamic values
    # The base_prompt_template should contain placeholders for {user_request}, {predefined_tool_descriptions}, etc.
    # And now we add the refinement_instructions at an appropriate place (e.g., before the JSON output instructions).
    # For this to work best, 'generate_agent_details_prompt' should probably have a placeholder like {refinement_section}
    # If not, we can append. For now, let's assume we append it before the part asking for JSON.
    
    # A more robust way would be to design the .toml prompt to include an optional refinement section.
    # For now, we will prepend the refinement instructions to the user request for context.
    # This is a simplification; ideally, the main prompt template would have a dedicated section for these instructions.
    
    contextual_user_request = user_request
    if refinement_instructions:
        contextual_user_request = f"{refinement_instructions}\n\nOriginal User Request: {user_request}"


    prompt = base_prompt_template.format(
        user_request=contextual_user_request, # Pass the possibly augmented user request
        predefined_tool_descriptions=predefined_tool_descriptions,
        dynamic_tool_listing=dynamic_tool_listing,
        api_key_listing_for_prompt=api_key_listing_for_prompt
        # If your .toml prompt has a {refinement_section} placeholder, you'd add:
        # refinement_section=refinement_instructions
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
MAX_REFINEMENT_ITERATIONS = 3 # Configurable: Max times to refine and re-evaluate
MAX_TOOL_GENERATION_RETRIES = 3 # Renamed your MAX_RETRIES for clarity

def handle_web_request(user_query: str) -> Generator[str, None, None]:
    """
    Handles a single user request, performing all necessary steps.
    Includes an outer loop for iterative refinement of generated tools based on evaluation,
    and an inner loop for retrying tool generation/testing.
    This function is a GENERATOR. It yields log strings prefixed with "log: "
    and finally yields the agent reply string prefixed with "reply: ".
    """
    yield f"log: Processing request: '{user_query}'..."
    original_user_request = user_query # Keep the original request

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

    current_tool_code_str = None # Holds the code of the tool being refined/evaluated
    feedback_history = []
    agent_response_text = "No response generated yet." # Default if all iterations fail early
    tools_confirmed_for_last_successful_worker = []


    for refinement_iteration in range(1, MAX_REFINEMENT_ITERATIONS + 1):
        yield f"log: --- Refinement Iteration {refinement_iteration}/{MAX_REFINEMENT_ITERATIONS} ---"

        # 1. TOOL GENERATION / REFINEMENT STAGE
        generation_details_this_iteration = None
        generated_tool_code_for_this_iteration = None
        tool_code_is_valid_for_this_iteration = False
        exec_globals_for_this_iteration = {} # Scope for exec for this iteration's new tool
        newly_added_tool_names_this_iteration = [] # Names of tools from this iteration's generation

        # Prepare arguments for generate_agent_details (or its refined version)
        # Note: generate_agent_details will need to be modified to accept and use
        # previous_tool_code_to_refine and feedback_for_refinement.
        tool_generation_args = {
            "user_request": original_user_request,
            "predefined_tools_dict": available_tools,
            "dynamic_tools_registry": dynamic_tool_registry,
            "api_key_details": current_api_key_details,
            "previous_tool_code_to_refine": current_tool_code_str, # Pass current tool for refinement
            "feedback_for_refinement": feedback_history[-1] if feedback_history else None
        }

        for attempt in range(1, MAX_TOOL_GENERATION_RETRIES + 1):
            yield f"log: [Refinement Iteration {refinement_iteration}] Tool generation/validation attempt {attempt}/{MAX_TOOL_GENERATION_RETRIES}..."
            
            # CALLING generate_agent_details (needs modification to use new args)
            current_generation_details_dict = generate_agent_details(**tool_generation_args)

            if not current_generation_details_dict:
                yield f"log: [Refinement Iteration {refinement_iteration}] Tool generation failed (generate_agent_details returned None) on attempt {attempt}."
                if attempt < MAX_TOOL_GENERATION_RETRIES:
                    yield f"log: Retrying tool generation..."
            time.sleep(1)
            continue
                else: # All retries for tool generation failed
                    yield f"log: [Refinement Iteration {refinement_iteration}] All {MAX_TOOL_GENERATION_RETRIES} attempts to generate tool details failed."
                    # This is a critical failure for this refinement iteration if a tool was expected.
                    # We might decide to break the refinement loop or try to proceed without a tool.
                    # For now, we'll mark tool code as invalid and let the outer logic handle it.
                    tool_code_is_valid_for_this_iteration = False 
                    break # Break from tool generation retries

            generation_details_this_iteration = current_generation_details_dict
            system_prompt_for_worker = generation_details_this_iteration['generated_prompt']
            required_predefined_tools = generation_details_this_iteration['required_predefined_tools']
            required_dynamic_tools_from_registry = generation_details_this_iteration['required_dynamic_tools']
            potential_new_tool_code = generation_details_this_iteration['generated_tool_code']
            required_new_tool_names_from_gen = generation_details_this_iteration['required_new_tool_names']
            dummy_args_for_test = generation_details_this_iteration.get("example_dummy_arguments")

            yield f"log: [Refinement Iteration {refinement_iteration}] Agent details generated attempt {attempt}."
            if required_new_tool_names_from_gen:
                yield f"log: Potential new tools: {required_new_tool_names_from_gen}"

            if not potential_new_tool_code: # No new tool code was suggested by generate_agent_details
                yield f"log: [Refinement Iteration {refinement_iteration}] No new tool code provided by generator. Validation of 'no new code' is successful."
                generated_tool_code_for_this_iteration = None
                newly_added_tool_names_this_iteration = []
                tool_code_is_valid_for_this_iteration = True # Valid in the sense that no new code was expected/generated
                break # Exit tool generation/validation retry loop

            # --- Test the generated code if new code was provided ---
            yield f"log: [Refinement Iteration {refinement_iteration}] Testing generated code (Attempt {attempt})..."
            print(f"\\n{'='*30} EXECUTING/TESTING DYNAMIC CODE (Ref.Iter {refinement_iteration}, Attempt {attempt}) {'='*30}")
            
            current_code_actually_valid = False # Flag for this specific code validation block
            current_exec_globals = {**globals()} # Fresh globals for each exec attempt
        if 'requests' not in current_exec_globals:
                try: import requests; current_exec_globals['requests'] = requests
                except ImportError: pass
            if 'json' not in current_exec_globals: # json is json_module in outer scope
                try: import json as json_std; current_exec_globals['json'] = json_std
                except ImportError: pass


            try:
                exec(potential_new_tool_code, current_exec_globals)
                yield f"log: [Refinement Iteration {refinement_iteration}] Dynamic code execution syntax OK (Attempt {attempt})."

                if required_new_tool_names_from_gen:
                    primary_new_tool_name = required_new_tool_names_from_gen[0]
                if primary_new_tool_name in current_exec_globals and callable(current_exec_globals[primary_new_tool_name]):
                    new_func_obj = current_exec_globals[primary_new_tool_name]
                        yield f"log: [Refinement Iteration {refinement_iteration}] Testing primary tool: {primary_new_tool_name}..."

                        if isinstance(dummy_args_for_test, dict):
                            # Temporarily patch os.environ for testing if keys are needed by tool
                            # Ensure this doesn't leak or permanently alter os.environ outside tests
                        with patch.dict(os.environ, {'TEST_API_KEY_GENERIC': 'dummy_test_value', 'OPEN_WEATHER_API_KEY': 'dummy_owm_key', 'SERPER_API_KEY': 'dummy_serper_key'}, clear=False):
                            try:
                                    print(f"Testing {primary_new_tool_name} with dummy args: {dummy_args_for_test}")
                                    test_result = new_func_obj(**dummy_args_for_test)
                                print(f"Test successful for {primary_new_tool_name}. Result: {test_result}")
                                    yield f"log: [Refinement Iteration {refinement_iteration}] Test successful for {primary_new_tool_name}."
                                    current_code_actually_valid = True
                            except Exception as test_e:
                                print(f"Test failed for {primary_new_tool_name}: {test_e}")
                                print(traceback.format_exc())
                                    yield f"log: [Refinement Iteration {refinement_iteration}] Test FAILED for {primary_new_tool_name}: {test_e}"
                                    current_code_actually_valid = False
                        else: # No dummy args, or not a dict
                            if required_new_tool_names_from_gen: # Only warn if a tool was expected
                                print(f"Warning: No suitable dummy args for {primary_new_tool_name}. Cannot perform functional test.")
                                yield f"log: [Refinement Iteration {refinement_iteration}] Warning: No dummy args for {primary_new_tool_name}. Functional test skipped."
                            current_code_actually_valid = True # Syntax was OK, but treat as valid if no test possible / no tool expected
                    else:
                        print(f"Error: Primary function '{primary_new_tool_name}' not found or not callable after exec.")
                        yield f"log: [Refinement Iteration {refinement_iteration}] Error: Primary function '{primary_new_tool_name}' not found/callable."
                        current_code_actually_valid = False
                else: # Code generated but no tool names specified by generator. This is unusual.
                    print("Warning: Code generated, but no tool names specified by generator. Cannot test or use.")
                    yield f"log: [Refinement Iteration {refinement_iteration}] Warning: Code generated, but no new tool names. Discarding code."
                    current_code_actually_valid = False # Or True if we want to allow proceeding without a new tool

        except Exception as e_exec:
                print(f"CRITICAL ERROR EXECUTING CODE (Ref.Iter {refinement_iteration}, Attempt {attempt}): {e_exec}")
            print(traceback.format_exc())
                yield f"log: [Refinement Iteration {refinement_iteration}] Error executing dynamic code: {e_exec}"
                current_code_actually_valid = False
            
            print(f"{'='*80}\\n")

            if current_code_actually_valid:
                yield f"log: [Refinement Iteration {refinement_iteration}] Tool code validated successfully on attempt {attempt}."
                tool_code_is_valid_for_this_iteration = True
                generated_tool_code_for_this_iteration = potential_new_tool_code
                newly_added_tool_names_this_iteration = required_new_tool_names_from_gen
                exec_globals_for_this_iteration = current_exec_globals # Save the globals where the valid code ran
                break # Exit tool generation/validation retry loop
            else: # Validation failed this attempt
                yield f"log: [Refinement Iteration {refinement_iteration}] Tool code validation failed on attempt {attempt}."
                if attempt < MAX_TOOL_GENERATION_RETRIES:
                    yield f"log: Retrying tool generation/validation..."
                    time.sleep(1) # Wait before retrying
                # else: implicitly, loop will end, and tool_code_is_valid_for_this_iteration will be false
        # --- End of Tool Generation/Validation Retry Loop ---

        if not tool_code_is_valid_for_this_iteration and required_new_tool_names_from_gen: # If new tool code was expected but all attempts to validate it failed.
            error_message = f"Failed to generate and validate a NEW tool after {MAX_TOOL_GENERATION_RETRIES} attempts in refinement iteration {refinement_iteration}."
        yield f"log: Error: {error_message}"
            feedback_history.append(f"Tool generation/validation failed: {error_message}. Last attempted code (if any): {potential_new_tool_code}")
            current_tool_code_str = None # Reset current_tool_code_str as the last attempt was bad
            if refinement_iteration == MAX_REFINEMENT_ITERATIONS:
                yield f"reply: {error_message} Cannot provide a solution."
        return
            continue # To the next refinement iteration, hoping for better luck or different strategy

        # Update current_tool_code_str with the successfully validated code for this iteration (if any)
        # This will be used by the evaluator and for the next refinement cycle.
        if tool_code_is_valid_for_this_iteration and generated_tool_code_for_this_iteration:
            current_tool_code_str = generated_tool_code_for_this_iteration
        # If no new code was generated (e.g. tool_code_is_valid_for_this_iteration is True but generated_tool_code_for_this_iteration is None)
        # then current_tool_code_str retains its value from the previous iteration (or None if first iteration).

        # Add successfully validated NEW tools to the dynamic_tool_registry and save them
        if tool_code_is_valid_for_this_iteration and generated_tool_code_for_this_iteration and newly_added_tool_names_this_iteration:
            yield f"log: [Refinement Iteration {refinement_iteration}] Adding validated new tool(s) to registry: {newly_added_tool_names_this_iteration}"
            successfully_added_to_registry_this_iter = []
            for tool_name in newly_added_tool_names_this_iteration:
                if tool_name in exec_globals_for_this_iteration and callable(exec_globals_for_this_iteration[tool_name]):
                    tool_func = exec_globals_for_this_iteration[tool_name]
                    dynamic_tool_registry[tool_name] = tool_func # Update live registry
                    successfully_added_to_registry_this_iter.append(tool_name)
                else:
                    yield f"log: [Refinement Iteration {refinement_iteration}] Error: Validated function '{tool_name}' missing from exec_globals. Cannot add to registry."
            
            if successfully_added_to_registry_this_iter: # Only save if tools were actually added
                yield f"log: [Refinement Iteration {refinement_iteration}] Saving new tool(s) to persistent registry: {successfully_added_to_registry_this_iter}"
                # Assuming save_dynamic_tool takes list of names and the single code block for them
                tool_manager.save_dynamic_tool(successfully_added_to_registry_this_iter, generated_tool_code_for_this_iteration)
                
                # Optional: Attempt to refactor tool registry (your existing logic)
                if random.random() <= 0.3: # 30% chance
                    yield f"log: [Refinement Iteration {refinement_iteration}] Attempting tool registry refactoring..."
                    try:
                        refactor_logs = tool_manager.refactor_tool_registry()
                        yield f"log: [Refinement Iteration {refinement_iteration}] Tool registry refactoring successful."
                        if isinstance(refactor_logs, list):
                            for r_log in refactor_logs: yield f"log: [Refactor] {r_log}"
                        elif isinstance(refactor_logs, str): yield f"log: [Refactor] {refactor_logs}"
                    except Exception as ref_e:
                        yield f"log: [Refinement Iteration {refinement_iteration}] Tool registry refactoring failed: {ref_e}"
        elif not newly_added_tool_names_this_iteration:
             yield f"log: [Refinement Iteration {refinement_iteration}] No new tools were generated in this iteration to add/save."


        # 2. AGENT EXECUTION STAGE
        yield f"log: [Refinement Iteration {refinement_iteration}] Preparing toolset for worker..."
        all_tool_functions_for_worker_this_iter = []
        tools_confirmed_for_worker_this_iter = []

        # Add PREDEFINED tools required by the current agent plan
    for tool_name in required_predefined_tools:
        if tool_name in available_tools:
                all_tool_functions_for_worker_this_iter.append(available_tools[tool_name]['function'])
                tools_confirmed_for_worker_this_iter.append(tool_name)
        else:
                yield f"log: [Refinement Iteration {refinement_iteration}] Warning: Predefined tool '{tool_name}' required by plan not found."
    
        # Add existing DYNAMIC tools required by the current agent plan
        for tool_name in required_dynamic_tools_from_registry:
        if tool_name in dynamic_tool_registry:
                all_tool_functions_for_worker_this_iter.append(dynamic_tool_registry[tool_name])
                tools_confirmed_for_worker_this_iter.append(tool_name)
        else:
                yield f"log: [Refinement Iteration {refinement_iteration}] Warning: Required dynamic tool '{tool_name}' from registry not found."

        # Add the NEWLY validated and registered tools for THIS iteration
        for tool_name in newly_added_tool_names_this_iteration: # These are names from the successful generation
            if tool_name in dynamic_tool_registry: # Check if it was successfully added to registry
                all_tool_functions_for_worker_this_iter.append(dynamic_tool_registry[tool_name])
                tools_confirmed_for_worker_this_iter.append(tool_name)
            else:
                 yield f"log: [Refinement Iteration {refinement_iteration}] Warning: Newly generated tool '{tool_name}' not found in dynamic registry for worker."
        
        yield f"log: [Refinement Iteration {refinement_iteration}] Tools for worker: {tools_confirmed_for_worker_this_iter}"
        yield f"log: [Refinement Iteration {refinement_iteration}] Executing request with Gemini worker using system prompt: '{system_prompt_for_worker[:100]}...'"
        
        agent_response_text = execute_request_with_worker( # This is your existing function
            user_request=original_user_request,
            system_prompt=system_prompt_for_worker,
            tool_functions=all_tool_functions_for_worker_this_iter
        )
        tools_confirmed_for_last_successful_worker = list(set(tools_confirmed_for_worker_this_iter)) # Update for this run

        yield f"log: [Refinement Iteration {refinement_iteration}] Worker response: '{agent_response_text[:200]}...'"

        # 3. RESPONSE EVALUATION STAGE
        yield f"log: [Refinement Iteration {refinement_iteration}] Gemini evaluating the agent's response and tool performance..."
        
        eval_tool_name_display = (newly_added_tool_names_this_iteration[0] 
                                  if newly_added_tool_names_this_iteration else 
                                  (required_dynamic_tools_from_registry[0] if required_dynamic_tools_from_registry else "N/A"))
        
        evaluation_prompt = f"""
Original user request: {original_user_request}
Tool focused on in this iteration (if applicable): {eval_tool_name_display}
Tool code (if new/refined in this iteration):\n```python\n{current_tool_code_str if current_tool_code_str and newly_added_tool_names_this_iteration else 'N/A (Used existing or no specific tool code generated this round)'}\n```
Agent's response to user: {agent_response_text}
Tools available to agent during execution: {tools_confirmed_for_last_successful_worker}

Considering the original user request, please evaluate:
1. Is the agent's response sufficient and accurate to fully address the user's request?
2. If a dynamic tool was generated/refined or a specific existing one was targeted, did it perform correctly and as intended to help answer the request?
3. If the response is NOT sufficient OR the tool is flawed/ineffective:
   - Provide specific, actionable feedback for improving the Python tool's code (if a tool was involved).
   - If no tool was generated but one is needed, suggest what kind of tool should be created.
   - If a tool was used but it's the wrong approach, suggest a different tool or how to fix the current one.

Respond ONLY in JSON format with the following keys:
- "is_sufficient": boolean (true if the response fully satisfies the user request, false otherwise)
- "reasoning": string (your detailed explanation for why the response is sufficient or not)
- "feedback_for_tool_improvement": string (specific feedback to improve/change the Python tool code for the next iteration, or what tool to create if none was made. Null if no tool improvement is needed or possible.)
"""
        yield f"log: [Refinement Iteration {refinement_iteration}] Sending evaluation prompt to Gemini..."
        raw_evaluation_output = generate_text_with_gemini(evaluation_prompt) # Your existing function
        yield f"log: [Refinement Iteration {refinement_iteration}] Raw evaluation from Gemini: {raw_evaluation_output}"

        try:
            # Attempt to remove markdown ```json ... ``` if present
            cleaned_eval_output = raw_evaluation_output.strip()
            if cleaned_eval_output.startswith("```json"):
                cleaned_eval_output = cleaned_eval_output[7:]
            if cleaned_eval_output.endswith("```"):
                cleaned_eval_output = cleaned_eval_output[:-3]
            
            evaluation_result = json_module.loads(cleaned_eval_output.strip())
            is_sufficient = evaluation_result.get("is_sufficient", False)
            reasoning = evaluation_result.get("reasoning", "No reasoning provided by evaluator.")
            feedback_for_improvement = evaluation_result.get("feedback_for_tool_improvement")
        except Exception as e:
            yield f"log: [Refinement Iteration {refinement_iteration}] Error parsing evaluation JSON: {e}. Raw: '{raw_evaluation_output}'. Assuming response is not sufficient."
            is_sufficient = False
            reasoning = f"Evaluation parsing error: {e}. The evaluator's response was not valid JSON."
            feedback_for_improvement = "The evaluator's response was not valid JSON. Please attempt to regenerate/refine the tool or rephrase the request, focusing on clear instructions for the tool's behavior."

        yield f"log: [Refinement Iteration {refinement_iteration}] Evaluation: Sufficient - {is_sufficient}. Reasoning: {reasoning}"

        if is_sufficient:
            yield f"log: Response deemed sufficient in iteration {refinement_iteration}."
            if tools_confirmed_for_last_successful_worker:
                 yield f"log: tools_used_summary: {json_module.dumps(tools_confirmed_for_last_successful_worker)}"
            yield f"reply: {agent_response_text}"
            return

        # 4. PREPARE FOR NEXT ITERATION (if response was not sufficient)
        current_feedback = feedback_for_improvement
        if not current_feedback: # If Gemini didn't provide specific tool feedback, use the reasoning.
            current_feedback = f"Response was insufficient. Evaluator reasoning: {reasoning}. No specific tool improvement suggestion was provided."
        
        feedback_history.append(current_feedback)
        yield f"log: [Refinement Iteration {refinement_iteration}] Feedback for next iteration: {current_feedback}"

        if not current_tool_code_str and not feedback_for_improvement: 
            # This means no tool was generated/used, and the evaluator didn't suggest creating one.
            yield f"log: [Refinement Iteration {refinement_iteration}] Evaluator did not provide feedback for tool creation, and no tool currently exists. Cannot effectively refine."
            if tools_confirmed_for_last_successful_worker:
                yield f"log: tools_used_summary: {json_module.dumps(tools_confirmed_for_last_successful_worker)}"
            yield f"reply: {agent_response_text} (Note: Unable to achieve a sufficient result after {refinement_iteration} attempts. Last reasoning: {reasoning})"
            return
        
        # If there was tool code, but no specific feedback for improvement, the existing code will be tried again
        # with the hope that the general feedback (now in feedback_history) helps generate_agent_details make better plan.

    # MAX_REFINEMENT_ITERATIONS Reached
    yield f"log: Maximum refinement iterations ({MAX_REFINEMENT_ITERATIONS}) reached."
    final_reasoning_or_feedback = feedback_history[-1] if feedback_history else "Max iterations reached without achieving a sufficient response."
    if tools_confirmed_for_last_successful_worker:
        yield f"log: tools_used_summary: {json_module.dumps(tools_confirmed_for_last_successful_worker)}"
    yield f"reply: {agent_response_text} (Note: Response after {MAX_REFINEMENT_ITERATIONS} refinement attempts. Last feedback/reasoning: {final_reasoning_or_feedback})"


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