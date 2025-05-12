"""
Custom Tool implementation to replace the missing ADK Tool class
"""
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class Tool:
    """
    Custom Tool class that wraps a function and provides metadata for agents.
    
    This class simulates the behavior of the ADK Tool class but is implemented
    from scratch since ADK doesn't provide it directly.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict] = None
    ):
        """
        Initialize a new tool with metadata and the wrapped function.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is invoked
            parameters: Optional parameter specifications if not inferred from function
        """
        self.name = name
        self.description = description
        self.function = function
        
        # Get function signature if parameters not provided
        if parameters is None:
            self.parameters = self._infer_parameters_from_function(function)
        else:
            self.parameters = parameters
    
    def _infer_parameters_from_function(self, function: Callable) -> Dict:
        """
        Infer parameters from function signature
        
        Args:
            function: The function to analyze
            
        Returns:
            dict: Parameter information dictionary
        """
        params = {}
        sig = inspect.signature(function)
        
        for name, param in sig.parameters.items():
            param_info = {
                "type": "string",  # Default type
                "description": f"Parameter {name}",
                "required": param.default == inspect.Parameter.empty
            }
            
            # Get type annotation if available
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_info["type"] = "string"
                elif param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list or param.annotation == List:
                    param_info["type"] = "array"
                elif param.annotation == dict or param.annotation == Dict:
                    param_info["type"] = "object"
            
            params[name] = param_info
        
        return params
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the underlying function with the provided arguments
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: The result of the function call
        """
        try:
            logger.info(f"Invoking tool: {self.name}")
            return self.function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error invoking tool {self.name}: {e}")
            raise 