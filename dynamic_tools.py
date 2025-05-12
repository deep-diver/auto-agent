"""
Dynamic Tools Module - For generating and loading tools at runtime
"""
import logging
import importlib.util
import sys
import os
import re
import google.generativeai as genai
from google.adk.tools import Tool

logger = logging.getLogger(__name__)

class DynamicToolGenerator:
    """Class for generating and managing dynamic tools"""
    
    def __init__(self, model_name="gemini-1.5-pro"):
        """Initialize the tool generator"""
        self.model_name = model_name
        self.tools_dir = "generated_tools"
        
        # Create directory for generated tools if it doesn't exist
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir)
        
        # Make sure there's an __init__.py file in the tools directory
        init_file = os.path.join(self.tools_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Generated tools package\n")
    
    def generate_tool(self, tool_name, tool_description, tool_function_description):
        """
        Generate a tool using Gemini API
        
        Args:
            tool_name: Name of the tool
            tool_description: Description of what the tool does
            tool_function_description: Detailed description of the function's behavior
            
        Returns:
            Tool: A loaded ADK Tool object or None if generation failed
        """
        logger.info(f"Generating tool: {tool_name}")
        
        # Clean tool name for use as a Python identifier
        clean_name = self._clean_name(tool_name)
        
        # Generate code for the tool
        try:
            code = self._generate_code(clean_name, tool_description, tool_function_description)
            if not code:
                logger.error(f"Failed to generate code for tool: {clean_name}")
                return None
            
            # Save the code to a Python file
            file_path = self._save_code(clean_name, code)
            
            # Load the tool
            return self._load_tool(clean_name, file_path, tool_description)
            
        except Exception as e:
            logger.error(f"Error generating tool {clean_name}: {e}")
            return None
    
    def _clean_name(self, name):
        """
        Clean a name for use as a Python identifier
        
        Args:
            name: Original name
            
        Returns:
            str: Cleaned name suitable for use as a Python identifier
        """
        # Replace spaces and special characters with underscores
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it starts with a letter or underscore
        if not clean[0].isalpha() and clean[0] != '_':
            clean = 'tool_' + clean
        return clean.lower()
    
    def _generate_code(self, clean_name, tool_description, function_description):
        """
        Generate Python code for a tool using Gemini API
        
        Args:
            clean_name: Clean name for the function
            tool_description: Description of what the tool does
            function_description: Detailed description of the function's behavior
            
        Returns:
            str: Generated Python code for the tool
        """
        prompt = f"""
        Generate Python code for a function that can be used as a tool in an AI agent.
        
        Function name: {clean_name}
        
        Tool description: {tool_description}
        
        Function behavior: {function_description}
        
        Requirements:
        1. The function should be well-documented with docstrings
        2. Include appropriate error handling
        3. Return results as a JSON string
        4. Use mock data or simple implementations where external APIs would normally be needed
        5. Keep the implementation simple and focused
        
        Only respond with the Python code, no explanations or markdown.
        """
        
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        
        # Extract just the Python code
        code_text = response.text
        
        # Try to remove any markdown code block syntax if present
        code_text = re.sub(r'^```python\n', '', code_text)
        code_text = re.sub(r'^```\n', '', code_text)
        code_text = re.sub(r'\n```$', '', code_text)
        
        return code_text
    
    def _save_code(self, clean_name, code):
        """
        Save generated code to a Python file
        
        Args:
            clean_name: Clean name for the function and file
            code: Generated Python code
            
        Returns:
            str: Path to the saved Python file
        """
        file_name = f"{clean_name}.py"
        file_path = os.path.join(self.tools_dir, file_name)
        
        with open(file_path, "w") as f:
            f.write(code)
        
        logger.info(f"Saved generated code to {file_path}")
        return file_path
    
    def _load_tool(self, clean_name, file_path, description):
        """
        Dynamically load a tool from a Python file
        
        Args:
            clean_name: Clean name for the function
            file_path: Path to the Python file
            description: Description of the tool
            
        Returns:
            Tool: ADK Tool object
        """
        # Import the module dynamically
        module_name = f"generated_tools.{clean_name}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get the function from the module
        function = getattr(module, clean_name)
        
        # Create and return the Tool
        tool = Tool(
            name=clean_name,
            description=description,
            function=function
        )
        
        logger.info(f"Successfully loaded tool: {clean_name}")
        return tool 