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
    # --- Add New Agent Profiles Below --- #

    # Example: General QA Agent (if needed later)
    # "GeneralQAAgent": {
    #     "description": "Handles general questions and conversation that do not require specific tools.",
    #     "system_prompt": "You are a general helpful assistant. Answer the user's query directly.",
    #     "associated_tools": []
    # },

    # --- End of Agent Profiles --- #
} 