"""
Orchestrator for Dynamic Agent Generation
"""
import logging
import google.generativeai as genai
from google.adk.agents import Agent, Runner
from google.adk.tools import Tool
from tools import get_all_tools
from dynamic_tools import DynamicToolGenerator
from agent_registry import AgentRegistry
import time

logger = logging.getLogger(__name__)

class Orchestrator:
    """Orchestrator for dynamically configuring and running AI agents"""
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.available_tools = get_all_tools()
        self.tool_generator = DynamicToolGenerator()
        self.agent_registry = AgentRegistry()
        self.gemini_model = "gemini-1.5-pro"  # Use appropriate Gemini model
        logger.info("Orchestrator initialized")
        
    def process_request(self, user_input):
        """
        Process a user request by dynamically configuring and running an agent
        
        Args:
            user_input (str): The user's input query
            
        Returns:
            str: The response from the agent
        """
        logger.info(f"Processing request: {user_input}")
        
        # 1. Check if we have a similar agent already in registry
        required_tools_prediction = self._predict_required_tools(user_input)
        similar_agent = self.agent_registry.find_similar_agent(
            required_tools=set(required_tools_prediction),
            query=user_input
        )
        
        if similar_agent:
            logger.info(f"Using existing agent: {similar_agent.name}")
            
            # Get the tool objects for the selected tools
            tool_name_to_object = {tool.name: tool for tool in self.available_tools}
            selected_tools = [
                tool_name_to_object[tool_name] 
                for tool_name in similar_agent.tools 
                if tool_name in tool_name_to_object
            ]
            
            # Run the agent with the existing configuration
            agent_config = {
                "name": similar_agent.name,
                "description": similar_agent.description,
                "instruction": similar_agent.instruction,
                "tools": similar_agent.tools
            }
            
            return self._run_agent(user_input, agent_config, selected_tools)
        
        # 2. Generate agent configuration using Gemini
        agent_config = self._generate_agent_config(user_input)
        logger.info(f"Generated agent config: {agent_config}")
        
        # 3. Select appropriate tools
        selected_tools = self._select_tools(user_input, agent_config)
        logger.info(f"Selected tools: {[tool.name for tool in selected_tools]}")
        
        # 4. Check if we need to generate any new tools
        if "new_tools" in agent_config and agent_config["new_tools"]:
            for new_tool in agent_config["new_tools"]:
                generated_tool = self._generate_new_tool(new_tool)
                if generated_tool:
                    selected_tools.append(generated_tool)
                    self.available_tools.append(generated_tool)
                    if "tools" in agent_config:
                        agent_config["tools"].append(generated_tool.name)
        
        # 5. Register the agent configuration for future use
        self.agent_registry.register_agent(agent_config)
        
        # 6. Configure and run the agent
        response = self._run_agent(user_input, agent_config, selected_tools)
        
        return response
    
    def _predict_required_tools(self, user_input):
        """
        Predict which tools might be required for a user query
        
        Args:
            user_input (str): The user's input query
            
        Returns:
            list: List of predicted tool names
        """
        # List available tools for context
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.available_tools
        ])
        
        prompt = f"""
        Based on the following user request, determine which tools from the available list would be needed to address it completely.

        User request: "{user_input}"

        Available tools:
        {tool_descriptions}

        Return a comma-separated list of just the tool names that are needed, with no additional explanation.
        For example: "get_weather,search_web"
        """
        
        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            
            # Parse the response to get the tool names
            tools = [t.strip() for t in response.text.split(",")]
            return tools
            
        except Exception as e:
            logger.error(f"Error predicting required tools: {e}")
            return []
    
    def _generate_agent_config(self, user_input):
        """
        Use Gemini to generate agent configuration
        
        Args:
            user_input (str): The user's input query
            
        Returns:
            dict: A dictionary containing the agent configuration
        """
        # List available tools for context
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.available_tools
        ])
        
        prompt = f"""
        Based on the following user request, generate a configuration for an AI agent that can best address it.

        User request: "{user_input}"

        Available tools:
        {tool_descriptions}

        Generate a JSON object with the following fields:
        - name: A short name for the agent
        - description: A concise description of the agent's purpose
        - instruction: Detailed instructions for the agent on how to handle the request
        - tools: A list of tool names from the available tools that would be useful for this request
        - new_tools: (Optional) A list of objects describing new tools that would be helpful but aren't in the available list. Each object should have:
          * name: A descriptive name for the tool
          * description: What the tool does
          * function_description: Detailed explanation of how the tool should work

        Keep the instructions detailed but concise.
        """
        
        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            
            # Extract the JSON content - this is simplified and would need proper parsing
            import json
            import re
            
            # Find JSON content within the response
            content_text = response.text
            json_match = re.search(r'```json\n(.*?)\n```', content_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = content_text
            
            # Clean up and parse JSON
            lines = []
            for line in json_text.split('\n'):
                if line.strip() and not line.strip().startswith('//'):
                    lines.append(line)
            
            cleaned_json = '\n'.join(lines)
            return json.loads(cleaned_json)
        
        except Exception as e:
            logger.error(f"Error generating agent configuration: {e}")
            # Fallback to a basic configuration
            return {
                "name": "fallback_assistant",
                "description": "A general assistant that helps with user requests",
                "instruction": "You are a helpful assistant. Answer the user's questions to the best of your ability.",
                "tools": []
            }
    
    def _select_tools(self, user_input, agent_config):
        """
        Select appropriate tools based on the agent configuration
        
        Args:
            user_input (str): The user's input query
            agent_config (dict): The generated agent configuration
            
        Returns:
            list: List of selected Tool objects
        """
        selected_tools = []
        
        # Extract tool names from agent configuration
        requested_tool_names = agent_config.get("tools", [])
        
        # Map tool names to actual tool objects
        tool_name_to_object = {tool.name: tool for tool in self.available_tools}
        
        for tool_name in requested_tool_names:
            if tool_name in tool_name_to_object:
                selected_tools.append(tool_name_to_object[tool_name])
                logger.info(f"Selected tool: {tool_name}")
            else:
                logger.warning(f"Requested tool not available: {tool_name}")
        
        return selected_tools
    
    def _generate_new_tool(self, new_tool_config):
        """
        Generate a new tool based on the configuration
        
        Args:
            new_tool_config (dict): Configuration for the new tool
            
        Returns:
            Tool: The generated tool or None if generation failed
        """
        try:
            name = new_tool_config["name"]
            description = new_tool_config["description"]
            function_description = new_tool_config["function_description"]
            
            logger.info(f"Generating new tool: {name}")
            
            generated_tool = self.tool_generator.generate_tool(
                name, description, function_description
            )
            
            if generated_tool:
                logger.info(f"Successfully generated new tool: {name}")
                return generated_tool
            else:
                logger.warning(f"Failed to generate tool: {name}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating new tool: {e}")
            return None
    
    def _run_agent(self, user_input, agent_config, selected_tools):
        """
        Configure and run an ADK agent
        
        Args:
            user_input (str): The user's input query
            agent_config (dict): The generated agent configuration
            selected_tools (list): List of selected Tool objects
            
        Returns:
            str: The response from the agent
        """
        # Create a unique agent ID
        agent_id = f"{agent_config['name']}_{int(time.time())}"
        
        # Create the agent
        agent = Agent(
            name=agent_id,
            model=self.gemini_model,
            description=agent_config["description"],
            instruction=agent_config["instruction"],
            tools=selected_tools
        )
        
        logger.info(f"Created agent: {agent.name}")
        
        # Create a runner
        runner = Runner(agent)
        
        # Run the agent with the user input
        try:
            response = runner.run(user_input)
            logger.info("Agent execution completed successfully")
            return response.message.content
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"I encountered an error while processing your request: {e}" 