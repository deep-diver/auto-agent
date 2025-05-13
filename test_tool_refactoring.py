import os
import yaml
import shutil
import tool_manager
import google.generativeai as genai # For configuring API key

# --- Test Configuration ---
TEST_REGISTRY_FILE = "test_dynamic_tool_registry.yaml"
ORIGINAL_REGISTRY_FILE_IN_TOOL_MANAGER = tool_manager.persistent_tool_registry_path

# Dummy API key for testing (replace with a valid one if live calls are made)
# Ensure GOOGLE_API_KEY is set in your environment for the test to run genai models
# You might want to load this from your .api_keys.yaml for convenience in a real test setup

EXAMPLE_TOOLS_FOR_REFACTORING = [
    {
        "tool_names": ["get_london_weather"],
        "code": """import os
import requests
def get_london_weather():
    api_key = os.environ.get('OPEN_WEATHER_API_KEY')
    if not api_key: return 'Error: Key not found'
    try:
        response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}&units=metric')
        response.raise_for_status()
        data = response.json()
        return f"London: {data['weather'][0]['description']}, {data['main']['temp']}C"
    except Exception as e:
        return f'Error: {e}'"""
    },
    {
        "tool_names": ["get_paris_weather"],
        "code": """import os
import requests
def get_paris_weather():
    api_key = os.environ.get('OPEN_WEATHER_API_KEY')
    if not api_key: return 'Error: Key not found'
    try:
        response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?q=Paris&appid={api_key}&units=metric')
        response.raise_for_status()
        data = response.json()
        return f"Paris: {data['weather'][0]['description']}, {data['main']['temp']}C"
    except Exception as e:
        return f'Error: {e}'"""
    },
    {
        "tool_names": ["get_time_now"], # A non-related tool
        "code": """import datetime
def get_time_now():
    return datetime.datetime.now().isoformat()"""
    }
]

def setup_test_environment():
    """Sets up the test environment for refactoring.
       - Configures Gemini API key.
       - Creates a dummy tool registry file for the test.
       - Points tool_manager to use this test registry file.
    """
    print("--- Setting up test environment ---")
    
    # 1. Configure Gemini API (essential for the refactor_tool_registry call)
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("WARNING: GOOGLE_API_KEY not found in environment. Test will likely fail if LLM call is made.")
        # Attempt to load from .api_keys.yaml as a fallback for local testing convenience
        try:
            if os.path.exists(".api_keys.yaml"):
                with open(".api_keys.yaml", 'r') as f_key:
                    keys = yaml.safe_load(f_key)
                    if "GOOGLE_API_KEY" in keys and "key_value" in keys["GOOGLE_API_KEY"]:
                        google_api_key = str(keys["GOOGLE_API_KEY"]["key_value"])
                        os.environ["GOOGLE_API_KEY"] = google_api_key # Set it for genai
                        print("Loaded GOOGLE_API_KEY from .api_keys.yaml for test setup.")
        except Exception as e:
            print(f"Could not load GOOGLE_API_KEY from .api_keys.yaml: {e}")

    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            print("Gemini API configured for test.")
        except Exception as e:
            print(f"Error configuring Gemini API for test: {e}. Test may fail.")
    else:
        print("Critical: No GOOGLE_API_KEY. LLM calls in refactoring will fail.")

    # 2. Create a dummy registry file for the test
    with open(TEST_REGISTRY_FILE, 'w') as f:
        yaml.dump(EXAMPLE_TOOLS_FOR_REFACTORING, f, sort_keys=False, indent=2, default_flow_style=False)
    print(f"Created test registry: {TEST_REGISTRY_FILE} with example tools.")

    # 3. Point tool_manager to use this test registry file
    tool_manager.persistent_tool_registry_path = TEST_REGISTRY_FILE
    print(f"Tool manager now using: {tool_manager.persistent_tool_registry_path}")

def cleanup_test_environment():
    """Cleans up test files and restores original tool_manager path."""
    print("\n--- Cleaning up test environment ---")
    if os.path.exists(TEST_REGISTRY_FILE):
        os.remove(TEST_REGISTRY_FILE)
        print(f"Removed test registry: {TEST_REGISTRY_FILE}")
    tool_manager.persistent_tool_registry_path = ORIGINAL_REGISTRY_FILE_IN_TOOL_MANAGER
    print(f"Tool manager restored to: {tool_manager.persistent_tool_registry_path}")

def main():
    setup_test_environment()
    
    print("\n--- Running tool_manager.refactor_tool_registry() ---")
    tool_manager.refactor_tool_registry()
    
    print("\n--- Checking results --- ")
    if os.path.exists(TEST_REGISTRY_FILE):
        with open(TEST_REGISTRY_FILE, 'r') as f:
            final_registry_content = yaml.safe_load(f)
        print("Final content of test registry file:")
        print(yaml.dump(final_registry_content, sort_keys=False, indent=2, default_flow_style=False))
    else:
        print(f"Test registry file '{TEST_REGISTRY_FILE}' not found after refactoring attempt.")
        
    cleanup_test_environment()
    print("\n--- Test complete --- ")

if __name__ == "__main__":
    # Ensure prompts.toml exists in the current path or accessible by main.py for get_prompt_template
    # This test relies on tool_manager.py importing get_prompt_template from main.py
    if not os.path.exists("prompts.toml"):
        print("ERROR: prompts.toml not found in the current directory. ")
        print("       The refactoring test needs this file because tool_manager.py imports get_prompt_template from main.py,")
        print("       which expects prompts.toml to be co-located.")
        print("       Please ensure prompts.toml from the main project is present here or create a dummy one.")
    else:
        main() 