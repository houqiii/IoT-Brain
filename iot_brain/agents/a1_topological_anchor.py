# iot_brain/agents/a1_topological_anchor.py

from typing import Dict, Any, Optional
from pathlib import Path

# Import the base class from the same directory.
from .base_agent import BaseAgent

class TopologicalAnchor(BaseAgent):
    """
    The first agent in the IoT-Brain pipeline: Topological Anchor.
    
    This agent is responsible for the initial parsing of a user's natural language
    query. Its primary function is to identify all mentioned spatio-temporal entities
    and transform them into a structured, preliminary Spatial Trajectory Graph (STG).
    This STG serves as the foundational data structure for all subsequent agents.
    """
    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initializes the TopologicalAnchor agent.

        Args:
            system_prompt (str): The system prompt string that defines the agent's role and instructions.
            model_name (str): The name of the Large Language Model to be used (e.g., "gpt-4-turbo").
            api_key (Optional[str]): The API key for the LLM service.
            base_url (Optional[str]): A custom base URL for the LLM API endpoint.
        """
        # Call the constructor of the parent BaseAgent class.
        super().__init__(
            agent_name="TopologicalAnchor",
            system_prompt=system_prompt,  # Pass the prompt string directly.
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )

    def generate_stg(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Generates the initial Spatial Trajectory Graph (STG) from a user query.

        This method takes the raw user query, sends it to the LLM via the base
        `execute` method, and expects a structured JSON object in return.

        Args:
            user_query (str): The original natural language query from the user.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the parsed STG JSON object if successful.
                                      Returns None if the generation or validation fails.
        """
        print(f"[{self.agent_name}] Generating STG for query: '{user_query[:50]}...'")
        
        # Call the parent's execute method, specifying that a JSON output is expected.
        # The user_query will be automatically wrapped into the user role message.
        stg_json = self.execute(user_input=user_query, json_output=True)

        # Validate the received JSON structure before returning it.
        if stg_json and self._validate_stg(stg_json):
            print(f"[{self.agent_name}] STG generation successful.")
            return stg_json
        else:
            print(f"[{self.agent_name}] Failed to generate a valid STG.")
            return None

    def _validate_stg(self, stg_json: Dict[str, Any]) -> bool:
        """
        A simple validation function to ensure the returned JSON has the necessary top-level keys.
        
        This can be extended for deeper structural validation as needed.

        Args:
            stg_json (Dict[str, Any]): The JSON object (as a dictionary) returned by the LLM.

        Returns:
            bool: True if the STG is valid, False otherwise.
        """
        required_keys = ['objective', 'nodes', 'edges']
        for key in required_keys:
            if key not in stg_json:
                print(f"[{self.agent_name}] Validation Error: Missing required key '{key}' in STG.")
                return False
        return True
