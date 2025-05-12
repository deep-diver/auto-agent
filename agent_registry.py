"""
Agent Registry - Store and retrieve agent configurations
"""
import logging
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Data class for agent configuration"""
    name: str
    description: str
    instruction: str
    tools: List[str]
    timestamp: float
    usage_count: int = 0
    last_used: float = 0

class AgentRegistry:
    """Registry for storing and retrieving agent configurations"""
    
    def __init__(self, registry_file="agent_registry.json"):
        """Initialize the agent registry"""
        self.registry_file = registry_file
        self.registry: Dict[str, AgentConfig] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load registry from file"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for key, value in data.items():
                    self.registry[key] = AgentConfig(**value)
                
                logger.info(f"Loaded {len(self.registry)} agent configurations from registry")
            except Exception as e:
                logger.error(f"Error loading registry file: {e}")
        else:
            logger.info("No registry file found. Starting with empty registry")
    
    def save_registry(self):
        """Save registry to file"""
        try:
            # Convert AgentConfig objects to dictionaries
            data = {key: value.__dict__ for key, value in self.registry.items()}
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.registry)} agent configurations to registry")
        except Exception as e:
            logger.error(f"Error saving registry file: {e}")
    
    def register_agent(self, config: dict) -> str:
        """
        Register an agent configuration in the registry
        
        Args:
            config: Dictionary containing agent configuration
            
        Returns:
            str: Registry key for the agent
        """
        # Create a fingerprint/key for this configuration
        tools_str = ",".join(sorted(config.get("tools", [])))
        key = f"{config['name']}_{tools_str}"
        
        # Create AgentConfig object
        agent_config = AgentConfig(
            name=config["name"],
            description=config["description"],
            instruction=config["instruction"],
            tools=config.get("tools", []),
            timestamp=time.time(),
            usage_count=1,
            last_used=time.time()
        )
        
        # Add to registry
        self.registry[key] = agent_config
        
        # Save registry
        self.save_registry()
        
        logger.info(f"Registered agent configuration: {key}")
        return key
    
    def find_similar_agent(self, required_tools: Set[str], query: str, similarity_threshold=0.7) -> Optional[AgentConfig]:
        """
        Find a similar agent configuration in the registry
        
        Args:
            required_tools: Set of required tool names
            query: User query to match
            similarity_threshold: Threshold for considering an agent similar
            
        Returns:
            Optional[AgentConfig]: Agent configuration if a similar one is found, None otherwise
        """
        if not self.registry:
            return None
        
        best_match = None
        best_score = 0
        
        for key, config in self.registry.items():
            # Check if this configuration has all the required tools
            config_tools = set(config.tools)
            if not required_tools.issubset(config_tools):
                continue
            
            # Simple similarity score based on tool match ratio
            # In a real implementation, this could use more sophisticated NLP
            tool_score = len(required_tools) / max(len(config_tools), 1) if config_tools else 0
            
            # If the tool score is above threshold, consider it a match
            if tool_score > similarity_threshold:
                if tool_score > best_score:
                    best_score = tool_score
                    best_match = config
        
        if best_match:
            # Update usage statistics
            best_match.usage_count += 1
            best_match.last_used = time.time()
            self.save_registry()
            
            logger.info(f"Found similar agent: {best_match.name} (score: {best_score:.2f})")
        
        return best_match
    
    def get_all_registered_agents(self) -> List[AgentConfig]:
        """
        Get all registered agent configurations
        
        Returns:
            List[AgentConfig]: List of all agent configurations
        """
        return list(self.registry.values()) 