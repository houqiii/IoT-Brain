import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base_agent import BaseAgent
from ..memory.programming_memory import ProgrammingMemory

class SchedulingSynthesizer(BaseAgent):
    """
    The Scheduling Synthesizer Agent.

    This agent's sole purpose is to act as a graph-to-code compiler. It takes the
    fully grounded and verified Spatial Trajectory Graph (STG) and translates it
    into an executable Python script using a predefined Execution API Pool.
    It leverages a ProgrammingMemory to find similar successful examples, using them
    as in-context learning prompts to improve the quality and robustness of the
    generated code.
    """

    def __init__(
        self,
        system_prompt: str,
        execution_api_pool_prompt: str,
        icl_examples_prompt: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        memory_path: Optional[str | Path] = None
    ):
        """
        Initializes the SchedulingSynthesizer Agent.

        Args:
            system_prompt (str): The main system prompt template, which contains placeholders.
            execution_api_pool_prompt (str): The documentation for the execution API pool.
            icl_examples_prompt (str): Static in-context learning examples.
            model_name (str): The name of the LLM model to use.
            api_key (Optional[str]): OpenAI API key.
            base_url (Optional[str]): Custom base URL for the API.
            memory_path (Optional[str | Path]): Path to the programming memory JSON file.
        """
        super().__init__(
            agent_name="SchedulingSynthesizer",
            system_prompt=system_prompt,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
        

        self.execution_api_pool_prompt = execution_api_pool_prompt
        self.icl_examples_prompt = icl_examples_prompt

        if memory_path is None:

            memory_path = Path.cwd() / "programming_memory.json"
        self.memory = ProgrammingMemory(memory_file_path=memory_path)

    def _prepare_final_prompt(self, grounded_stg_json: str, similar_examples: List[Dict[str, str]]) -> str:
        """
        Constructs the final, complete prompt to be sent to the LLM by formatting the main system prompt.
        """
        icl_section = ""
        if similar_examples:
            print(f"[{self.agent_name}] Injecting {len(similar_examples)} similar example(s) into the prompt for ICL.")
            example_texts = []
            for i, example in enumerate(similar_examples):
                example_text = (
                    f"--- SIMILAR EXAMPLE {i+1} ---\n"
                    f"QUERY: {example['original_query']}\n"
                    f"SUCCESSFUL SCRIPT:\n```python\n{example['executed_script']}\n```"
                )
                example_texts.append(example_text)
            icl_section = "\n\n".join(example_texts)

        final_prompt = self.system_prompt.format(
            EXECUTION_API_POOL=self.execution_api_pool_prompt,
            ICL_EXAMPLES=self.icl_examples_prompt,
            SIMILAR_DYNAMIC_EXAMPLES=icl_section,
            GROUNDED_STG_JSON=grounded_stg_json
        )
        return final_prompt

    def _extract_python_code(self, llm_response: str) -> Optional[str]:
        """
        Extracts the Python code block from the LLM's response.
        """
        # Regex to find a python code block, being robust to variations.
        match = re.search(r"```python\s*([\s\S]+?)\s*```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback for plain code without markdown
        # Be cautious with this, as it might grab non-code text.
        # It's better to enforce the markdown format in the prompt.
        print(f"[{self.agent_name}] Warning: Could not find Python markdown block. Returning raw content.")
        return llm_response.strip()

    def generate_script(self, grounded_stg: Dict[str, Any]) -> Optional[str]:
        """
        The main method to generate an executable Python script from a grounded STG.

        Args:
            grounded_stg (Dict[str, Any]): The full, verified output from the GroundingVerifier.

        Returns:
            Optional[str]: A string containing the executable Python script, or None on failure.
        """
        original_query = grounded_stg.get("original_query", "No query provided.")
        print(f"[{self.agent_name}] Starting script generation for query: '{original_query[:50]}...'")

        # 1. Retrieve similar examples from programming memory
        similar_examples = self.memory.find_similar_examples(original_query, top_k=1, score_threshold=0.9)

        # 2. Prepare the full user input for the LLM
        grounded_stg_json_str = json.dumps(grounded_stg, indent=2)
        # Note: We are not using the standard _create_messages here because the prompt is
        # highly structured and built dynamically. We'll pass the full content as the user message.
        final_user_content = self._prepare_final_prompt(grounded_stg_json_str, similar_examples)
        
        messages = [
            # The system prompt is now just a high-level instruction,
            # while the detailed structure is in the user message.
            {"role": "system", "content": "You are an expert Python programmer acting as a graph-to-code compiler. Your sole function is to translate a fully specified and verified plan into an executable Python script."},
            {"role": "user", "content": final_user_content}
        ]

        # 3. Execute the LLM call
        # We override the default execute method's message creation for more control
        # Let's call the client directly
        try:
            print(f"[{self.agent_name}] Executing LLM call for code generation...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.request_timeout,
            )
            llm_response_content = response.choices[0].message.content
        except Exception as e:
            print(f"[{self.agent_name}] LLM API call failed: {e}")
            return None

        if not llm_response_content:
            print(f"[{self.agent_name}] Error: Received empty response from LLM.")
            return None
        
        # 4. Extract and return the Python code
        python_script = self._extract_python_code(llm_response_content)
        
        if python_script:
            print(f"[{self.agent_name}] Script generation successful.")
            return python_script
        else:
            print(f"[{self.agent_name}] Failed to extract a valid Python script from LLM response.")
            return None

    def add_to_memory(self, original_query: str, executed_script: str):
        """
        Public method to allow the controller to add a successful script to memory.
        """
        self.memory.add_successful_script(original_query, executed_script)

