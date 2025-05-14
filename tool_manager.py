import os
import yaml
import traceback # For more detailed error printing during exec
import json
import random

# Path to the persistent tool registry file
persistent_tool_registry_path = "dynamic_tool_registry.yaml"

def load_dynamic_tools() -> dict:
    """
    Loads dynamically generated tools from the persistent YAML registry file,
    executes their code in a controlled scope, and returns a dictionary
    mapping tool names to their function objects.
    """
    loaded_tools_registry = {}
    if not os.path.exists(persistent_tool_registry_path):
        print(f"Info (Tool Manager): Persistent tool registry '{persistent_tool_registry_path}' not found. No dynamic tools pre-loaded.")
        return loaded_tools_registry

    try:
        with open(persistent_tool_registry_path, 'r') as f:
            persisted_tools = yaml.safe_load(f)

        if not persisted_tools: # Handles empty file
            print(f"Info (Tool Manager): Persistent tool registry '{persistent_tool_registry_path}' is empty.")
            return loaded_tools_registry

        if not isinstance(persisted_tools, list):
            print(f"Warning (Tool Manager): Persistent tool registry '{persistent_tool_registry_path}' is not a valid list. Skipping load.")
            return loaded_tools_registry

        loaded_count = 0
        # Prepare a shared globals dict for exec, including necessary imports
        # NOTE: This is still risky. Ideally, generated code shouldn't need unexpected imports.
        # Consider making required imports explicit in the generated code string itself.
        shared_globals = {'os': os, 'yaml': yaml, '__builtins__': __builtins__} 
        
        # Add commonly needed modules to shared_globals
        modules_to_import = [
            ('requests', 'requests'),
            ('json', 'json'),
            ('datetime', 'datetime'),
            ('time', 'time'),
            ('random', 'random'),
            ('re', 're'),
            ('sys', 'sys'),
            ('math', 'math'),
            ('pathlib', 'Path'),
            ('urllib.parse', 'parse'),
            ('urllib.request', 'request'),
            ('urllib.error', 'error'),
            ('collections', 'defaultdict'),
            ('collections', 'Counter'),
            ('collections', 'deque'),
            ('PyPDF2', 'PdfReader'),
        ]
        
        for module_name, import_name in modules_to_import:
            try:
                if '.' in module_name:
                    # For submodules like urllib.parse
                    main_module, sub_module = module_name.split('.', 1)
                    module = __import__(main_module, fromlist=[sub_module])
                    sub_mod = getattr(module, sub_module)
                    shared_globals[import_name] = sub_mod
                else:
                    # For regular modules
                    module = __import__(module_name)
                    shared_globals[module_name] = module
                    
                    # Special case for collections to import specific classes
                    if module_name == 'collections' and import_name != 'collections':
                        shared_globals[import_name] = getattr(module, import_name)
            except (ImportError, AttributeError) as e:
                print(f"Note: Could not import {module_name} for dynamic tools: {e}")


        for tool_entry in persisted_tools:
            if isinstance(tool_entry, dict) and 'tool_names' in tool_entry and 'code' in tool_entry:
                tool_names = tool_entry['tool_names']
                code_str = tool_entry['code']
                if not isinstance(tool_names, list):
                    print(f"Warning (Tool Manager): tool_names for code block is not a list in '{persistent_tool_registry_path}'. Skipping entry.")
                    continue
                
                local_namespace = {}
                try:
                    # Execute the code within the prepared globals and a local namespace
                    exec(code_str, shared_globals, local_namespace)

                    for tool_name in tool_names:
                        if tool_name in local_namespace and callable(local_namespace[tool_name]):
                            loaded_tools_registry[tool_name] = local_namespace[tool_name]
                            loaded_count += 1
                            # print(f"Successfully loaded and registered persisted dynamic tool: '{tool_name}'") # Keep less verbose
                        else:
                            print(f"Warning (Tool Manager): Persisted tool function '{tool_name}' not found or not callable after exec() from '{persistent_tool_registry_path}'.")
                            print(f"  Available in local namespace after exec: {list(local_namespace.keys())}")
                except Exception as e:
                    print(f"Error executing persisted tool code for '{tool_names}' from '{persistent_tool_registry_path}': {e}")
                    print(traceback.format_exc()) # Print full traceback
            else:
                print(f"Warning (Tool Manager): Invalid tool entry format in '{persistent_tool_registry_path}'. Skipping: {tool_entry}")

        if loaded_count > 0:
            print(f"--- (Tool Manager) Successfully loaded {loaded_count} persisted dynamic tool functions. ---")
        else:
            print(f"--- (Tool Manager) No dynamic tools successfully loaded from '{persistent_tool_registry_path}'. ---")

    except yaml.YAMLError as e:
        print(f"Error parsing persistent tool registry '{persistent_tool_registry_path}': {e}")
    except Exception as e:
        print(f"Unexpected error loading persistent tool registry: {e}")
        print(traceback.format_exc())

    return loaded_tools_registry

def save_dynamic_tool(new_tool_names: list, generated_tool_code_str: str):
    """
    Saves a newly generated tool's names and code to the persistent YAML registry.
    Checks for existing tool names to avoid duplicates in the file (simple check by name).
    """
    if not generated_tool_code_str or not new_tool_names:
        print("Info (Tool Manager): Tool code or names are empty, not saving to persistent registry.")
        return

    # Add a warning message about imports
    print("IMPORTANT: Make sure your tool code includes ALL necessary imports explicitly.")
    print("           Tools should be self-contained to work properly after page reloads.")
    
    # Check if the code contains import statements
    if not any(line.strip().startswith('import ') or line.strip().startswith('from ') 
               for line in generated_tool_code_str.split('\n')):
        print("WARNING: No import statements detected in tool code. This may cause issues after page reload.")

    new_entry = {
        "tool_names": new_tool_names,
        "code": generated_tool_code_str
    }

    try:
        all_persisted_tools = []
        if os.path.exists(persistent_tool_registry_path):
            with open(persistent_tool_registry_path, 'r') as f:
                # Use load_all for potentially multiple documents, though we expect a list
                loaded_content = yaml.safe_load(f) 
                if isinstance(loaded_content, list):
                    all_persisted_tools = loaded_content
                elif loaded_content is not None: # File exists but is not a list or is invalid
                    print(f"Warning (Tool Manager): Persistent tool registry '{persistent_tool_registry_path}' content is not a list. Overwriting potentially invalid data.")
                    all_persisted_tools = [] # Start fresh if format was wrong

        # Check if any of the new tool names already exist among persisted tools
        existing_names = set()
        for existing_tool in all_persisted_tools:
            if isinstance(existing_tool, dict) and isinstance(existing_tool.get("tool_names"), list):
                for name in existing_tool["tool_names"]:
                    existing_names.add(name)

        # Check for overlap
        has_overlap = False
        for name in new_tool_names:
            if name in existing_names:
                print(f"Warning (Tool Manager): Tool with name '{name}' already exists in '{persistent_tool_registry_path}'. Not saving to avoid duplicates.")
                has_overlap = True
                break # No need to check further if one overlap is found
        
        if has_overlap:
            return # Do not save if any name overlaps

        # If no overlap, append and write
        all_persisted_tools.append(new_entry)

        with open(persistent_tool_registry_path, 'w') as f:
            yaml.dump(all_persisted_tools, f, sort_keys=False, indent=2, default_flow_style=False)
        print(f"(Tool Manager) Successfully saved new dynamic tool(s) '{new_tool_names}' to '{persistent_tool_registry_path}'.")

    except yaml.YAMLError as e:
        print(f"Error writing to persistent tool registry '{persistent_tool_registry_path}': {e}")
    except Exception as e:
        print(f"Unexpected error saving to persistent tool registry: {e}")
        print(traceback.format_exc())


def remove_dynamic_tools(tool_names_to_remove: list):
    """
    Removes tools with the specified names from the persistent YAML registry.
    """
    if not tool_names_to_remove:
        print("Info (Tool Manager): No tool names provided for removal.")
        return

    removed_count = 0
    if not os.path.exists(persistent_tool_registry_path):
        print(f"Warning (Tool Manager): Persistent tool registry '{persistent_tool_registry_path}' not found. Cannot remove tools.")
        return

    try:
        all_persisted_tools = []
        with open(persistent_tool_registry_path, 'r') as f:
            loaded_content = yaml.safe_load(f)
            if isinstance(loaded_content, list):
                all_persisted_tools = loaded_content
            else:
                 print(f"Warning (Tool Manager): Content of '{persistent_tool_registry_path}' is not a list. Cannot process removals.")
                 return # Cannot proceed if the format is wrong

        # Filter out the tools to remove
        updated_tools_list = []
        names_to_remove_set = set(tool_names_to_remove)
        
        for tool_entry in all_persisted_tools:
            if isinstance(tool_entry, dict) and isinstance(tool_entry.get("tool_names"), list):
                # Check if any of the tool's names are in the removal set
                entry_names_set = set(tool_entry["tool_names"])
                if not entry_names_set.intersection(names_to_remove_set):
                    # Keep the entry if none of its names match the removal list
                    updated_tools_list.append(tool_entry)
                else:
                    # Log which tool is being removed
                    print(f"(Tool Manager) Removing tool entry with names '{tool_entry['tool_names']}' from registry.")
                    removed_count += 1
            else:
                 # Keep potentially malformed entries? Or discard? Let's keep them for now.
                 updated_tools_list.append(tool_entry)

        # Write the updated list back to the file
        if removed_count > 0:
            with open(persistent_tool_registry_path, 'w') as f:
                yaml.dump(updated_tools_list, f, sort_keys=False, indent=2, default_flow_style=False)
            print(f"(Tool Manager) Successfully removed {removed_count} tool entr(y/ies) associated with names '{tool_names_to_remove}' from '{persistent_tool_registry_path}'.")
        else:
            print(f"(Tool Manager) No tools matching names '{tool_names_to_remove}' found in '{persistent_tool_registry_path}'.")


    except yaml.YAMLError as e:
        print(f"Error during processing or writing for tool removal in '{persistent_tool_registry_path}': {e}")
    except Exception as e:
        print(f"Unexpected error removing tools from persistent registry: {e}")
        print(traceback.format_exc())

def refactor_tool_registry():
    """
    Analyzes the existing dynamic tools for generalization opportunities using an LLM.
    If a valid generalization is proposed, it saves the new tool and removes the old ones.
    This function interacts with the persistent storage (YAML file).
    Only runs 20% of the time to avoid over-aggressive refactoring.
    """
    # Add a random check to only refactor 20% of the time
    if random.random() > 0.2:  # 80% chance to skip refactoring
        print("\n--- (Tool Manager) Skipping refactoring attempt (80% probability) ---")
        return
        
    print("\n--- (Tool Manager) Attempting to refactor dynamic tool registry... ---")
    if not os.path.exists(persistent_tool_registry_path):
        print("Info (Tool Manager): Registry file not found, nothing to refactor.")
        return

    all_persisted_tools = []
    try:
        with open(persistent_tool_registry_path, 'r') as f:
            loaded_content = yaml.safe_load(f)
            if isinstance(loaded_content, list):
                all_persisted_tools = loaded_content
            else:
                print(f"Warning (Tool Manager): Registry content is not a list. Cannot refactor.")
                return
        
        if not all_persisted_tools or len(all_persisted_tools) < 2: # Need at least 2 tools for meaningful generalization
            print("Info (Tool Manager): Fewer than 2 tools in registry, skipping refactoring attempt.")
            return

        # Prepare tool codes for the prompt
        tool_code_blocks = []
        for i, tool_entry in enumerate(all_persisted_tools):
            if isinstance(tool_entry, dict) and 'tool_names' in tool_entry and 'code' in tool_entry:
                tool_code_blocks.append(f"Tool {i+1} (Names: {tool_entry['tool_names']}):\n```python\n{tool_entry['code']}\n```")
            else:
                tool_code_blocks.append(f"Tool {i+1} (Malformed entry):\n{str(tool_entry)}")
        
        tool_codes_for_prompt = "\n\n".join(tool_code_blocks)

        # Load the refactoring prompt template
        # Assuming get_prompt_template can be accessed or re-implemented here if this were a truly isolated module
        # For now, let's assume we might need to pass genai and get_prompt_template or make this part of a class
        # Quickest path: re-import and use get_prompt_template from main (or a shared utility)
        # This indicates a potential need for better structuring of prompt loading if tool_manager is fully independent.
        try:
            from main import get_prompt_template, PROMPT_CONFIG_PATH # Temporary, for direct use
            import google.generativeai as genai # Also temporary for direct use
            # Potentially need to re-configure genai if API key isn't globally set from main's initial load
            # This dependency on main suggests refactoring might be better initiated from main, or prompts need a shared loader.
        except ImportError:
            print("Error (Tool Manager): Could not import get_prompt_template from main.py for refactoring. Cannot proceed.")
            return
        
        refactor_prompt_template = get_prompt_template("refactor_dynamic_tools_prompt")
        if not refactor_prompt_template:
            print("Error (Tool Manager): Could not load 'refactor_dynamic_tools_prompt'. Cannot proceed.")
            return

        prompt_for_llm = refactor_prompt_template.format(tool_codes_for_prompt=tool_codes_for_prompt)

        # Call LLM for refactoring suggestions
        model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model
        print("Info (Tool Manager): Sending tool codes to LLM for refactoring analysis...")
        response = model.generate_content(prompt_for_llm)

        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith('```json'):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith('```'):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()

        refactor_proposal = json.loads(cleaned_response_text)

        # Validate and process the proposal
        new_code = refactor_proposal.get("new_generalized_tool_code")
        new_names = refactor_proposal.get("new_generalized_tool_names")
        old_names_to_remove = refactor_proposal.get("tools_to_remove_after_generalization")

        if new_code and new_names and old_names_to_remove:
            print(f"Info (Tool Manager): LLM proposed generalization. New tool: '{new_names}', Replaces: '{old_names_to_remove}'")
            # Save the new generalized tool
            # Note: save_dynamic_tool checks for duplicates by name. If the new generalized name conflicts, it won't save.
            # This is generally good, but ensure LLM doesn't propose existing names unless intended for an update (which this isn't).
            save_dynamic_tool(new_names, new_code) 
            
            # Remove the old specific tools
            remove_dynamic_tools(old_names_to_remove)
            print("--- (Tool Manager) Refactoring applied. --- ")
        else:
            print("Info (Tool Manager): LLM proposed no generalization, or proposal was incomplete.")

    except FileNotFoundError:
        print(f"Error (Tool Manager): Registry file '{persistent_tool_registry_path}' disappeared during refactoring.")
    except yaml.YAMLError as e:
        print(f"Error (Tool Manager): YAML error during refactoring: {e}")
    except json.JSONDecodeError as json_err:
        print(f"Error (Tool Manager): Parsing JSON response from LLM for refactoring: {json_err}")
        print(f"Raw LLM Refactor Response: {response.text}")
    except Exception as e:
        print(f"Error (Tool Manager): Unexpected error during tool refactoring: {e}")
        print(traceback.format_exc()) 

def fix_tool_imports(tool_name=None, add_imports=None):
    """
    Adds missing imports to existing tools in the registry.
    
    Args:
        tool_name (str, optional): Specific tool name to fix. If None, fixes all tools.
        add_imports (list, optional): List of import statements to add. If None, adds common imports.
    
    Returns:
        int: Number of tools updated
    """
    if not os.path.exists(persistent_tool_registry_path):
        print(f"Warning (Tool Manager): Persistent tool registry '{persistent_tool_registry_path}' not found.")
        return 0
    
    # Default common imports to add if not specified
    if add_imports is None:
        add_imports = [
            "import os",
            "import json",
            "import requests",
            "import datetime",
            "import time",
            "import re",
            "import random"
        ]
    
    try:
        # Load the current registry
        with open(persistent_tool_registry_path, 'r') as f:
            all_persisted_tools = yaml.safe_load(f)
        
        if not isinstance(all_persisted_tools, list):
            print(f"Warning (Tool Manager): Registry content is not a list. Cannot fix imports.")
            return 0
        
        updated_count = 0
        
        # Process each tool entry
        for tool_entry in all_persisted_tools:
            if not isinstance(tool_entry, dict) or 'tool_names' not in tool_entry or 'code' not in tool_entry:
                continue
            
            # Skip if we're targeting a specific tool and this isn't it
            if tool_name and tool_name not in tool_entry['tool_names']:
                continue
            
            code = tool_entry['code']
            code_lines = code.split('\n')
            
            # Find existing imports
            existing_imports = []
            for line in code_lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    existing_imports.append(line)
            
            # Determine which imports to add
            imports_to_add = []
            for imp in add_imports:
                # Check if this import or a similar one already exists
                if not any(existing_imp.startswith(imp.split()[0]) for existing_imp in existing_imports):
                    imports_to_add.append(imp)
            
            if imports_to_add:
                # Add imports at the beginning of the code
                new_code = '\n'.join(imports_to_add) + '\n\n' + code
                tool_entry['code'] = new_code
                updated_count += 1
                print(f"Added imports to tool(s): {tool_entry['tool_names']}")
        
        # Save the updated registry if changes were made
        if updated_count > 0:
            with open(persistent_tool_registry_path, 'w') as f:
                yaml.dump(all_persisted_tools, f, sort_keys=False, indent=2, default_flow_style=False)
            print(f"(Tool Manager) Successfully updated imports for {updated_count} tool entries.")
        else:
            print("(Tool Manager) No tools needed import updates.")
        
        return updated_count
    
    except Exception as e:
        print(f"Error fixing imports: {e}")
        print(traceback.format_exc())
        return 0 