import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .base_agent import BaseAgent
from ..memory.geospatial_memory import GeospatialMemory
# from ..utils.api_toolkit import VerificationToolkit

class GroundingVerifier(BaseAgent):
    """
    The Grounding Verifier Agent.
    ...
    """

    def __init__(
        self,
        system_prompt: str,
        verification_toolkit: Any,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        memory_path: Optional[str | Path] = None
    ):
        """
        Initializes the GroundingVerifier Agent.

        Args:
            system_prompt (str): The complete system prompt, including rules and tool docs.
            verification_toolkit (Any): An instantiated VerificationToolkit object.
            model_name (str): The name of the LLM model to use.
            api_key (Optional[str]): OpenAI API key.
            base_url (Optional[str]): Custom base URL for the API.
            memory_path (Optional[str | Path]): Path to the geospatial memory JSON file.
        """
        super().__init__(
            agent_name="GroundingVerifier",
            system_prompt=system_prompt,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
        if memory_path is None:
            memory_path = Path.cwd() / "geospatial_memory.json"
        
        self.memory = GeospatialMemory(memory_file_path=memory_path)
        self.toolkit = verification_toolkit 
        
        self.conversation_history = []

    def _create_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _extract_thought_and_action(self, llm_response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts separate 'Thought' and 'Action' parts from the LLM's response.
        This version fixes the greedy regex for Action.
        """
        # Remove markdown bolding for easier parsing
        clean_response = llm_response.replace('**', '')

        # Regex for Thought: Find "Thought:" and capture everything until "Action:" or "Final Thought:"
        thought_match = re.search(r"Thought:(.*?)(?=Action:|Final Thought:|$)", clean_response, re.DOTALL | re.IGNORECASE)
        
        # Regex for Action: Find "Action:" and greedily capture everything after it.
        # We use ([\s\S]+) which is a greedy match for any character including newlines.
        # We also remove the optional backticks from the regex itself and strip them later for robustness.
        action_match = re.search(r"Action:\s*([\s\S]+)", clean_response, re.IGNORECASE)
        
        thought = thought_match.group(1).strip() if thought_match else None
        
        if action_match:
            # Get the full action block and strip surrounding backticks and whitespace
            action_str = action_match.group(1).strip().strip('`').strip()
        else:
            action_str = None

        if not thought:
             print("[Verifier WARNING] Could not parse 'Thought' from LLM response.")
        if not action_str:
             print("[Verifier WARNING] Could not parse 'Action' from LLM response.")

        # Handle the case where the action is explicitly 'None'
        if action_str and action_str.lower() == 'none':
            return thought, None
            
        return thought, action_str

    def _execute_action(self, action_string: str) -> str:
        """
        Dynamically executes a tool call string and ensures the result is logged and stored in memory.
        """
        if not action_string or not action_string.startswith("verifier_toolkit."):
            return f"Error: Invalid action format. Expected 'verifier_toolkit.function(...)', but got '{action_string}'."
        
        try:
            result = eval(action_string, {"verifier_toolkit": self.toolkit})
            self._update_memory_from_action(action_string, result)
            return str(result)
        except Exception as e:
            error_message = f"Error: Failed to execute action '{action_string}'. Details: {e}"
            return error_message

    def _update_memory_from_action(self, action_string: str, result: Any):
        """
        Parses the action and its result to update the geospatial memory.
        It will NOT store results that indicate an error.
        """

        if isinstance(result, str) and "error" in result.lower():
            print(f"[{self.agent_name}] Tool call resulted in an error. Skipping memory update.")
            return
        if isinstance(result, dict) and "error" in result:
            print(f"[{self.agent_name}] Tool call resulted in an error. Skipping memory update.")
            return

        try:
            params = dict(re.findall(r"(\w+)\s*=\s*'([^']+)'", action_string))
            location_name = params.get('location_name') or params.get('start_location')
            scenario_name = params.get('scenario_name')

            if not location_name:
                return

            floor = re.search(r'(\d+F)', location_name).group(1) if location_name and 'F' in location_name else None
            location_type = 'indoor' if floor else 'outdoor'

            fact_to_store = {}
            if "cameras_verify" in action_string:
                count_match = re.search(r'(\d+)', str(result))
                if count_match:
                    fact_to_store = {"camera_count": int(count_match.group(1))}
            elif "facilities_verify" in action_string:

                if isinstance(result, dict) and "facilities" in result:
                    fact_to_store = {"facilities": result["facilities"]}
                else: 
                    facilities = dict(re.findall(r"([\w_]+): (\d+)", str(result)))
                    if facilities:
                        fact_to_store = {"facilities": facilities}
            elif "doors_verify" in action_string:
                if isinstance(result, dict) and "doors" in result:
                     fact_to_store = {"doors": result["doors"]}
                else: 
                    doors = [d.strip() for d in str(result).replace("Doors:", "").split(',') if d.strip()]
                    if doors:
                        fact_to_store = {"doors": doors}
            
            if fact_to_store:
                self.memory.add_verified_info(location_name, floor, scenario_name, location_type, fact_to_store)
        except Exception as e:
            print(f"[{self.agent_name}] Warning: Could not update memory from action '{action_string}'. Reason: {e}")

    def _check_memory_and_prepare_input(self, hypothesized_stg: Dict) -> str:
        memory_content = self.memory.memory_data
        if not memory_content:
            return "Geospatial memory is currently empty. All hypotheses must be verified via tool calls."
        
        memory_summary = (
            f"PRE-VERIFICATION MEMORY CHECK:\n"
            f"The following facts have been previously verified and are stored in memory. "
            f"You can use this information to skip verification steps if a hypothesis is already answered.\n"
            f"```json\n{json.dumps(memory_content, indent=2)}\n```"
        )
        return memory_summary

    def verify_stg(self, hypothesized_stg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the TAO loop with a clear, step-by-step process.
        """
        self.conversation_history = []
        grounded_stg = hypothesized_stg.copy()
        verification_log = []

        memory_check_result = self._check_memory_and_prepare_input(hypothesized_stg)
        current_input = (
            f"INITIAL INPUT:\n```json\n{json.dumps(hypothesized_stg, indent=2)}\n```\n\n"
            f"{memory_check_result}"
        )
        
        max_turns = 15
        for turn in range(max_turns):
            print(f"\n{'-'*20} [VERIFIER TAO Turn {turn + 1}] {'-'*20}")
            

            llm_response = self.execute(user_input=current_input, json_output=False)
            
            print(f"\n--- RAW LLM RESPONSE ---\n{llm_response}\n------------------------\n")

            if not llm_response:
                print(f"[{self.agent_name}] CRITICAL ERROR: LLM returned an empty response. Aborting.")
                break

            self.conversation_history.append({"role": "user", "content": current_input})
            self.conversation_history.append({"role": "assistant", "content": llm_response})

            thought, action = self._extract_thought_and_action(llm_response)
            
            print(f"PARSED THOUGHT: {thought}")
            verification_log.append(f"Thought: {thought}")

            if "Ending thought:" in llm_response:
                print(f"\n[{self.agent_name}] 'Ending thought' detected. Verification loop complete.")
                break
            
            if action:
                print(f"PARSED ACTION: {action}")
                verification_log.append(f"Action: {action}")
                # Execute the action and get the observation
                print(f"\n>>> EXECUTING ACTION...")
                observation = self._execute_action(action)
                print(f"<<< ACTION RESULT (OBSERVATION): {observation}\n")
                
                verification_log.append(f"Observation: '{observation}'")
                current_input = f"Observation: `{observation}`"
            else:
                print(f"[{self.agent_name}] WARNING: No further action specified and no 'Ending thought'. Terminating loop.")
                break
        else:
            print(f"[{self.agent_name}] WARNING: Reached max turns ({max_turns}). Terminating loop.")

        grounded_stg['verification_log'] = verification_log
        return grounded_stg