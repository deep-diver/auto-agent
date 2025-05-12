# Dynamic Agent Orchestrator

A system that dynamically configures and generates AI agents based on user requests using Google's Agent Development Kit (ADK).

## Project Overview

The Dynamic Agent Orchestrator (DAO) is a flexible system that understands user requests and dynamically configures appropriate AI agents to handle them. It selects from pre-defined tools, can generate new tools on-demand, and remembers previously successful agent configurations for future reuse.

### Key Features

- **Dynamic Agent Configuration**: Uses the Gemini API to generate tailored agent descriptions and instructions based on user queries.
- **Pre-defined Tool Integration**: Automatically selects the most appropriate tools for each agent based on the user's request.
- **Tool Generation**: Can generate new tool code using the Gemini API when needed functionality isn't available.
- **Agent Registry**: Remembers previously successful agent configurations and reuses them for similar requests.

## Getting Started

### Prerequisites

- Python 3.8+
- Google API key for Gemini

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd dynamic-agent-orchestrator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your-api-key
   ```

### Usage

Run the main application:

```
python main.py
```

This will start an interactive session where you can enter requests for the system to process.

Example interactions:

```
User: What's the weather like in London and can you find me a simple pasta recipe?
Agent: [Configures an agent with weather and recipe tools and provides the response]

User: Help me calculate the area of a circle with radius 5
Agent: [Configures a calculator agent and provides the response]
```

## How It Works

1. The system receives a user request.
2. It checks if a similar agent has been configured before.
3. If not, it uses Gemini to generate an appropriate agent configuration.
4. The orchestrator selects or generates the required tools.
5. An ADK agent is configured with these tools and the generated instructions.
6. The agent processes the request and returns a response.
7. The configuration is saved in the registry for future use.

## Project Structure

- `main.py`: Main entry point for the application
- `orchestrator.py`: Core orchestration logic
- `tools.py`: Pre-defined tools for agents
- `dynamic_tools.py`: Tool generation functionality
- `agent_registry.py`: Registry for storing and retrieving agent configurations

## Future Enhancements

- Web interface using Flask or Streamlit
- More sophisticated agent similarity detection
- Hierarchical agent configurations for complex tasks
- Integration with external APIs for real functionality (weather, recipes, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google's Agent Development Kit (ADK) for agent framework
- Google Gemini API for natural language understanding and code generation 