# iot_brain/agents/base_agent.py

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the official OpenAI library and its specific error classes.
from openai import OpenAI, APIError, APITimeoutError

class BaseAgent:
    """
    A generic base class for all agents in the IoT-Brain framework.
    
    This class handles the common functionalities partageed by all agents, such as:
    - Initialization with a system prompt string.
    - Configuration of the OpenAI client.
    - A core `execute` method for making API calls to the LLM.
    - Robust error handling with retries for API calls.
    - A utility for parsing JSON from the LLM's response.
    
    Subclasses should inherit from this class and implement their specific logic.
    """
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        request_timeout: int = 120
    ):
        """
        Initializes the BaseAgent.

        Args:
            agent_name (str): The name of the agent (e.g., "TopologicalAnchor"), used for logging.
            system_prompt (str): The complete system prompt string that defines the agent's role.
            model_name (str): The name of the LLM model to be used.
            api_key (Optional[str]): The API key for the LLM service. If not provided, it will
                                     be sourced from the OPENAI_API_KEY environment variable.
            base_url (Optional[str]): A custom base URL for the LLM API endpoint.
            max_retries (int): The maximum number of times to retry a failed API call.
            request_timeout (int): The timeout in seconds for each API request.
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # Ensure the provided system prompt is a valid, non-empty string.
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError(f"System prompt for agent '{agent_name}' must be a non-empty string.")
        self.system_prompt = system_prompt

        # Initialize the OpenAI client. It prioritizes the passed `api_key`.
        # If `api_key` is None, it falls back to the environment variable.
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url
        )
        # Log successful initialization.
        print(f"[{self.agent_name}] Agent initialized with model '{self.model_name}'.")

    def _create_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        Constructs the list of messages to be sent to the Chat Completions API.
        
        Subclasses can override this method to support more complex conversation histories.

        Args:
            user_input (str): The content for the 'user' role message.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries in the required format.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

    def execute(self, user_input: str, json_output: bool = True) -> Optional[Any]:
        """
        Executes the core logic of the agent: calls the LLM and processes its response.

        Args:
            user_input (str): The input content from the user or a previous agent.
            json_output (bool): If True, expects the LLM to return a JSON object and will
                                attempt to parse it. If False, returns the raw text content.

        Returns:
            Optional[Any]: The LLM's response content (as a dictionary if `json_output` is True,
                           or as a string otherwise). Returns None if the API call fails
                           after all retries.
        """
        messages = self._create_messages(user_input)
        
        for attempt in range(self.max_retries):
            try:
                print(f"[{self.agent_name}] Executing... (Attempt {attempt + 1}/{self.max_retries})")
                
                # Set the `response_format` argument only if JSON output is requested.
                # This improves compatibility with models that may not support this argument.
                response_format_arg = {"type": "json_object"} if json_output else None
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=self.request_timeout,
                    response_format=response_format_arg,
                )

                content = response.choices[0].message.content
                
                if not content:
                    print(f"[{self.agent_name}] Warning: Received empty content from API. Retrying...")
                    time.sleep(2 ** attempt)
                    continue

                if json_output:
                    parsed = self._parse_json(content)
                    if parsed:
                        return parsed
                    else:
                        print(f"[{self.agent_name}] Failed to parse expected JSON, retrying...")
                        continue
                else:
                    # If raw text is requested, return it directly.
                    return content

            except (APITimeoutError, APIError) as e:
                print(f"[{self.agent_name}] API Error: {e}. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            except Exception as e:
                import traceback
                print(f"[{self.agent_name}] An unexpected error occurred in execute: {e}")
                traceback.print_exc()
                return None
        
        print(f"[{self.agent_name}] Failed to get a valid response after {self.max_retries} retries.")
        return None

    def _parse_json(self, content: str) -> Optional[Dict]:
        """
        A robust JSON parser that handles potential formatting issues.
        
        It first tries to parse the content directly. If that fails, it looks for a
        JSON object embedded within a Markdown code block (```json ... ```).

        Args:
            content (str): The string content returned by the LLM.

        Returns:
            Optional[Dict]: A dictionary if parsing is successful, otherwise None.
        """
        try:
            # First, attempt a direct parse.
            return json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, search for a Markdown JSON block.
            match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    print(f"[{self.agent_name}] Error: Failed to parse JSON from Markdown block. Content: {match.group(1)}. Error: {e}")
                    return None
            
            print(f"[{self.agent_name}] Error: Content is not a valid JSON object and no Markdown block found. Content: {content}")
            return None