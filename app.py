from flask import Flask, render_template, request, jsonify, Response, session
import os
import main as agent_core # Assuming main.py can be imported and its core logic refactored
import json # For formatting SSE data if it's complex, though we'll use strings for now
import time # For potential keep-alive pings if needed, though not critical for basic logs
import inspect # <-- Add inspect module
import yaml # Add yaml import
import tool_manager # Import the tool_manager module
import ast # For parsing Python code to get docstrings
import db_manager # Import the database manager for chat history

# Explicitly set static folder
app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))  # For session management

def get_first_docstring_line_from_code(code_str: str) -> str | None:
    """Helper to extract the first line of a docstring from a code string."""
    if not code_str:
        return None
    try:
        tree = ast.parse(code_str)
        # Try to find a function or class definition first
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    first_line = docstring.strip().split('\n')[0].strip()
                    return first_line if first_line else None # Ensure it's not empty
        # If no function/class docstring, try module level
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            first_line = module_docstring.strip().split('\n')[0].strip()
            return first_line if first_line else None
    except SyntaxError:
        return None # Code couldn't be parsed
    return None

# Initial check for Google API Key at app startup for early warning.
# The core logic in agent_core.handle_web_request will handle per-request refresh and configuration.
if not os.environ.get("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY not set in environment at Flask app startup. "
          "The agent may fail if it cannot load the key from .api_keys.yaml during requests.")
else:
    # We can optionally try to configure genai here, but agent_core will do it per request anyway.
    # For now, just confirm it's set.
    print("Flask App: GOOGLE_API_KEY is set in the environment. Configuration will be handled by agent_core.")

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Create a new conversation if none exists in session
    if 'conversation_id' not in session:
        session['conversation_id'] = db_manager.create_conversation()
    
    return render_template('index.html')

@app.route('/new_conversation')
def new_conversation():
    """Create a new conversation and store its ID in the session."""
    # Create a new conversation
    conversation_id = db_manager.create_conversation()
    session['conversation_id'] = conversation_id
    
    return jsonify({
        'success': True,
        'conversation_id': conversation_id
    })

@app.route('/send_message') # Changed to GET, message via query param for SSE
def send_message_sse():
    """Handles user messages and streams logs and final reply via SSE."""
    user_message = request.args.get('message')
    if not user_message:
        # For SSE, errors ideally should also be streamed if the connection is open,
        # but for a missing message param, a simple HTTP error is fine before streaming starts.
        return jsonify({'error': 'No message provided'}), 400
    
    # Ensure we have a conversation ID in the session
    if 'conversation_id' not in session:
        session['conversation_id'] = db_manager.create_conversation()
    
    conversation_id = session['conversation_id']
    
    # Store the user message in the database
    db_manager.add_message(
        conversation_id=conversation_id,
        content=user_message,
        role='user'
    )
    
    # Set default title based on first message if not already set
    conversation = db_manager.get_conversation(conversation_id)
    if conversation and conversation.get('title') == 'New Conversation' and len(conversation.get('messages', [])) <= 1:
        title = db_manager.extract_title_from_first_message(conversation_id)
        if title:
            db_manager.update_conversation_title(conversation_id, title)

    print(f"WebUI User Message (for SSE): {user_message}")

    def generate_sse_stream():
        # The old stream_log_callback is no longer needed here,
        # as handle_web_request will yield log/reply strings directly.
        logs = []
        tools_used = []
        agent_reply = None

        try:
            # agent_core.handle_web_request is now a generator
            for message_part in agent_core.handle_web_request(user_message):
                if message_part.startswith("log: "):
                    log_content = message_part[5:] # Remove "log: " prefix
                    formatted_log = str(log_content).replace('\n', '\\n')
                    logs.append(log_content)
                    yield f"data: {formatted_log}\n\n"
                    
                    # Check for tools used in logs
                    if log_content.startswith("tools_used_summary: "):
                        try:
                            tools_json = log_content[len("tools_used_summary: "):]
                            tools_list = json.loads(tools_json)
                            if isinstance(tools_list, list):
                                tools_used.extend(tools_list)
                        except json.JSONDecodeError:
                            print(f"Error parsing tools used JSON: {tools_json}")
                            
                elif message_part.startswith("reply: "):
                    reply_content = message_part[7:].strip() # Remove prefix AND trim whitespace
                    formatted_final_reply = str(reply_content).replace('\n', '\\n')
                    agent_reply = reply_content
                    yield f"event: final_reply\ndata: {formatted_final_reply}\n\n"
                else:
                    # Should not happen if main.py adheres to the prefix convention
                    print(f"SSE: Unknown message part received: {message_part}")
                    formatted_unknown = str(message_part).replace('\n', '\\n')
                    yield f"data: [UNKNOWN] {formatted_unknown}\n\n" 

        except Exception as e:
            print(f"Error during SSE stream generation or agent_core.handle_web_request iteration: {e}")
            import traceback
            traceback.print_exc()
            # Send an error event to the client
            error_message = f"An error occurred: {str(e)}".replace('\n', '\\n')
            yield f"event: stream_error\ndata: {error_message}\n\n"
            agent_reply = f"Error: {str(e)}"
        finally:
            # Store the agent's response in the database
            if agent_reply:
                db_manager.add_message(
                    conversation_id=conversation_id,
                    content=agent_reply,
                    role='agent',
                    logs=logs,
                    tools_used=tools_used if tools_used else None
                )
            
            # Optionally, send a stream close event, though client usually detects closure
            yield "event: stream_end\ndata: Stream ended\n\n"
            # print("SSE STREAM ENDED")
            
    return Response(generate_sse_stream(), mimetype='text/event-stream')

@app.route('/get_conversations')
def get_conversations():
    """Get a list of all conversations."""
    conversations = db_manager.get_all_conversations()
    return jsonify(conversations)

@app.route('/get_conversation/<conversation_id>')
def get_specific_conversation(conversation_id):
    """Get a specific conversation by ID."""
    conversation = db_manager.get_conversation(conversation_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    return jsonify(conversation)

@app.route('/load_conversation/<conversation_id>')
def load_conversation(conversation_id):
    """Set the current conversation in the session."""
    # Check if the conversation exists first
    conversation = db_manager.get_conversation(conversation_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Set in session
    session['conversation_id'] = conversation_id
    
    return jsonify({
        'success': True,
        'conversation': conversation
    })

@app.route('/update_conversation_title/<conversation_id>', methods=['POST'])
def update_title(conversation_id):
    """Update a conversation's title."""
    data = request.json
    if not data or 'title' not in data:
        return jsonify({'error': 'Title is required'}), 400
    
    success = db_manager.update_conversation_title(conversation_id, data['title'])
    if not success:
        return jsonify({'error': 'Conversation not found or title not updated'}), 404
    
    return jsonify({'success': True})

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
def delete_specific_conversation(conversation_id):
    """Delete a conversation."""
    success = db_manager.delete_conversation(conversation_id)
    if not success:
        return jsonify({'error': 'Conversation not found or not deleted'}), 404
    
    # If we just deleted the active conversation, create a new one
    if session.get('conversation_id') == conversation_id:
        session['conversation_id'] = db_manager.create_conversation()
    
    return jsonify({'success': True})

@app.route('/delete_dynamic_tool/<tool_name>', methods=['DELETE'])
def delete_dynamic_tool(tool_name):
    """Delete a dynamic tool by name."""
    if not tool_name:
        return jsonify({'error': 'Tool name is required'}), 400
    
    try:
        # Check if the tool exists in YAML or in memory
        tool_exists = False
        in_yaml = False
        
        # Check YAML file
        if os.path.exists(tool_manager.persistent_tool_registry_path):
            with open(tool_manager.persistent_tool_registry_path, 'r') as f:
                persisted_tools = yaml.safe_load(f)
                if isinstance(persisted_tools, list):
                    for tool_entry in persisted_tools:
                        if isinstance(tool_entry, dict) and 'tool_names' in tool_entry:
                            if tool_name in tool_entry['tool_names']:
                                tool_exists = True
                                in_yaml = True
                                break
                                
        # Check in-memory registry 
        in_memory = hasattr(agent_core, 'dynamic_tool_registry') and tool_name in agent_core.dynamic_tool_registry
        if in_memory:
            tool_exists = True
        
        if not tool_exists:
            return jsonify({'error': f'Tool "{tool_name}" not found in YAML or memory'}), 404
        
        # Remove from YAML if present there
        if in_yaml:
            tool_manager.remove_dynamic_tools([tool_name])
        
        # Remove from in-memory registry if present there
        if in_memory:
            del agent_core.dynamic_tool_registry[tool_name]
            print(f"Removed tool '{tool_name}' from in-memory registry")
        
        # Return success with details of where it was removed from
        removal_locations = []
        if in_yaml:
            removal_locations.append("YAML registry")
        if in_memory:
            removal_locations.append("in-memory registry")
            
        return jsonify({
            'success': True, 
            'message': f'Tool "{tool_name}" successfully deleted from: {", ".join(removal_locations)}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Error deleting tool: {str(e)}'}), 500

@app.route('/get_system_info')
def get_system_info():
    """Returns information about available tools and API keys."""
    predefined_tools_info = []
    if hasattr(agent_core, 'available_tools') and isinstance(agent_core.available_tools, dict):
        for name, data in agent_core.available_tools.items():
            description = data.get('schema', {}).get('description', 'No description')
            predefined_tools_info.append({"name": name, "description": description})

    # Dynamic tools - read directly from the YAML file to show its persisted state "as is"
    dynamic_tools_from_yaml_info = []
    if os.path.exists(tool_manager.persistent_tool_registry_path):
        try:
            with open(tool_manager.persistent_tool_registry_path, 'r', encoding='utf-8') as f:
                persisted_tools_yaml = yaml.safe_load(f)
            
            if isinstance(persisted_tools_yaml, list):
                for tool_entry in persisted_tools_yaml: # Each entry is a code block with one or more names
                    if isinstance(tool_entry, dict) and \
                       isinstance(tool_entry.get('tool_names'), list) and tool_entry['tool_names']:
                        
                        code_content = tool_entry.get('code')
                        base_description_for_block = "User-generated tool" # Default for this code block
                        if code_content:
                            first_line_doc = get_first_docstring_line_from_code(code_content)
                            if first_line_doc:
                                base_description_for_block = first_line_doc
                        
                        all_names_for_this_block = tool_entry['tool_names']
                        
                        for current_tool_name in all_names_for_this_block:
                            final_description = base_description_for_block
                            
                            if len(all_names_for_this_block) > 1:
                                other_aliases = [name for name in all_names_for_this_block if name != current_tool_name]
                                if other_aliases:
                                    final_description += f" (Also known as: {', '.join(other_aliases)})"
                            
                            dynamic_tools_from_yaml_info.append({
                                "name": current_tool_name, 
                                "description": final_description 
                            })
            else:
                # Handle case where YAML is not a list (e.g., empty or malformed)
                if persisted_tools_yaml is not None:
                    print(f"Warning (/get_system_info): Content of {tool_manager.persistent_tool_registry_path} is not a list.")
                dynamic_tools_from_yaml_info.append({"name": "Registry Status", "description": f"Note: {os.path.basename(tool_manager.persistent_tool_registry_path)} is empty or not a list."})
        except Exception as e:
            print(f"Error reading or parsing {tool_manager.persistent_tool_registry_path} in /get_system_info: {e}")
            dynamic_tools_from_yaml_info.append({"name": "Error", "description": f"Could not load from {os.path.basename(tool_manager.persistent_tool_registry_path)}."})
    else:
        dynamic_tools_from_yaml_info.append({"name": "Registry Status", "description": f"{os.path.basename(tool_manager.persistent_tool_registry_path)} not found."})

    # Get current API key details by calling the refresh function from agent_core
    api_key_details_info = []
    if hasattr(agent_core, 'refresh_api_keys_and_configure_gemini'):
        try:
            api_details, _ = agent_core.refresh_api_keys_and_configure_gemini()
            if isinstance(api_details, dict):
                for key_name, details in api_details.items():
                    api_key_details_info.append({
                        "name": key_name,
                        "service": details.get("service_name", "N/A"),
                        "description": details.get("description", "N/A")
                    })
        except Exception as e:
            print(f"Error fetching API key details for /get_system_info: {e}")

    return jsonify({
        'predefined_tools': predefined_tools_info,
        'dynamic_tools': dynamic_tools_from_yaml_info,
        'api_keys': api_key_details_info,
        'active_conversation_id': session.get('conversation_id')
    })

@app.route('/get_tool_details/<tool_name>')
def get_tool_details(tool_name):
    """Returns details (schema or code) for a specific tool."""
    tool_details = None
    tool_type = None
    description = "N/A"
    details_content = "# Tool details not found or error loading."
    details_format = "plaintext" # Default format

    # Check predefined tools
    if hasattr(agent_core, 'available_tools') and tool_name in agent_core.available_tools:
        tool_data = agent_core.available_tools[tool_name]
        schema = tool_data.get('schema', {})
        description = schema.get('description', 'No description available.')
        function_obj = tool_data.get('function')
        tool_type = "predefined"

        if function_obj:
            try:
                details_content = inspect.getsource(function_obj)
                details_format = "python"
            except (TypeError, OSError) as e:
                print(f"Could not get source for {tool_name}: {e}. Falling back to schema.")
                # Fallback to showing schema if source isn't available
                try:
                    details_content = json.dumps(schema, indent=4)
                    details_format = "json"
                except Exception as json_e:
                    print(f"Error serializing schema for {tool_name} after failing to get source: {json_e}")
                    details_content = f"# Error getting source code or serializing schema: {e}"
                    details_format = "plaintext"
        else:
            # If function object is missing, try showing schema
            print(f"Warning: Function object missing for predefined tool {tool_name}. Trying schema.")
            try:
                details_content = json.dumps(schema, indent=4)
                details_format = "json"
            except Exception as json_e:
                print(f"Error serializing schema for {tool_name} (no function obj): {json_e}")
                details_content = f"# Tool function missing and error serializing schema: {json_e}"
                details_format = "plaintext"

    # Check dynamic tools if not found in predefined
    elif hasattr(agent_core, 'dynamic_tool_registry') and tool_name in agent_core.dynamic_tool_registry:
        tool_type = "dynamic"
        description = "User-generated dynamic tool. Source from persisted YAML registry:"
        details_format = "python" # Assume python code is stored

        try:
            if not os.path.exists(tool_manager.persistent_tool_registry_path):
                details_content = f"# Dynamic tool registry file not found: {tool_manager.persistent_tool_registry_path}"
                details_format = "plaintext"
            else:
                with open(tool_manager.persistent_tool_registry_path, 'r') as f:
                    persisted_tools = yaml.safe_load(f)
                
                found_code_block = None
                if isinstance(persisted_tools, list):
                    for tool_entry in persisted_tools:
                        if isinstance(tool_entry, dict) and \
                           isinstance(tool_entry.get('tool_names'), list) and \
                           tool_name in tool_entry['tool_names'] and \
                           'code' in tool_entry:
                            found_code_block = tool_entry['code']
                            # Optional: Update description if more specific one is stored in YAML for this entry
                            # description = tool_entry.get('entry_description', description) 
                            break 
                
                if found_code_block:
                    details_content = found_code_block
                else:
                    # This might happen if the tool is in dynamic_tool_registry (loaded in memory)
                    # but its entry was somehow removed from YAML or YAML is out of sync.
                    # Or if the tool was generated but not persisted correctly.
                    details_content = f"# Code for dynamic tool '{tool_name}' not found in {tool_manager.persistent_tool_registry_path}. Tool might be in memory only or registry is out of sync."
                    details_format = "plaintext"
                    # Fallback: Try inspect.getsource for in-memory version as a last resort
                    function_obj = agent_core.dynamic_tool_registry.get(tool_name)
                    if function_obj:
                        try:
                            details_content += "\n\n# Fallback: Source code of in-memory function:\n" + inspect.getsource(function_obj)
                            details_format = "python" # It's python if inspect worked
                        except Exception as inspect_e:
                            details_content += f"\n# Failed to get source of in-memory function: {inspect_e}"
                            details_format = "plaintext"
        except Exception as e:
            print(f"Error reading or parsing dynamic tool registry for {tool_name}: {e}")
            details_content = f"# Error accessing or parsing tool registry for '{tool_name}': {e}"
            details_format = "plaintext"
    
    if tool_type:
        return jsonify({
            "name": tool_name,
            "type": tool_type, # "predefined" or "dynamic"
            "description": description,
            "details_format": details_format, # "python", "json", or "plaintext"
            "details": details_content
        })
    else:
        return jsonify({"error": "Tool not found"}), 404

# Debug route to check if static files are accessible
@app.route('/check_static')
def check_static():
    static_path = app.static_folder
    assets_path = os.path.join(static_path, 'assets')
    
    files_info = []
    if os.path.exists(assets_path):
        for filename in os.listdir(assets_path):
            file_path = os.path.join(assets_path, filename)
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 'N/A'
            files_info.append({
                'name': filename,
                'path': f'/static/assets/{filename}',
                'size': file_size,
                'exists': os.path.isfile(file_path)
            })
    
    return jsonify({
        'static_folder': static_path,
        'assets_folder': assets_path,
        'assets_folder_exists': os.path.exists(assets_path),
        'files': files_info
    })

if __name__ == '__main__':
    # Create a templates folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Check if index.html exists, if not, create a dummy one
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('<h1>Please use the main index.html for SSE functionality</h1>')
        print("Created dummy templates/index.html - Note: SSE requires the full JS client.")

    # Print debug information about static files
    static_path = app.static_folder
    assets_path = os.path.join(static_path, 'assets')
    print(f"Flask static folder: {static_path}")
    print(f"Assets folder: {assets_path}")
    print(f"Assets folder exists: {os.path.exists(assets_path)}")
    
    if os.path.exists(assets_path):
        print("Files in assets folder:")
        for filename in os.listdir(assets_path):
            file_path = os.path.join(assets_path, filename)
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 'N/A'
            print(f"  - {filename} ({file_size} bytes)")

    app.run(debug=True, threaded=True) # threaded=True can be important for SSE with dev server 