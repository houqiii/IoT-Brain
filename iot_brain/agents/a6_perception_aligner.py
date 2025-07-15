# iot_brain/agents/a6_perception_aligner.py

import os
import json
import base64
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from .base_agent import BaseAgent 

# Pillow (PIL) is needed for image handling
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is not installed. Please install it with 'pip install Pillow'")


class PerceptionAligner(BaseAgent):
    """
    The Perception Aligner Agent.

    This agent is the final execution stage of the IoT-Brain framework. Its primary
    function is to analyze a chronological stream of video frames, guided by the
    original user query, to find a definitive answer. It operates on a frame-by-frame
    basis, making a decision at each step whether to continue processing or terminate
    with a successful finding.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str, # This should ideally be a Vision-Language Model (VLM)
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initializes the PerceptionAligner Agent.

        Args:
            system_prompt (str): The system prompt defining the agent's behavior.
            model_name (str): The name of the Vision-Language Model to use (e.g., "gpt-4-vision-preview").
            api_key (Optional[str]): The API key.
            base_url (Optional[str]): The custom base URL for the API.
        """
        # Note: The BaseAgent's execute method is designed for text. We'll need a custom
        # way to handle image inputs here.
        super().__init__(
            agent_name="PerceptionAligner",
            system_prompt=system_prompt,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )

    def _encode_image_to_base64(self, image_path: str | Path) -> str:
        """Encodes an image file to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"[{self.agent_name}] Error: Image file not found at {image_path}")
            return ""
        except Exception as e:
            print(f"[{self.agent_name}] Error encoding image: {e}")
            return ""

    def analyze_frame(
        self,
        image_path: str | Path,
        user_query: str,
        plan_context: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Analyzes a single video frame to decide on the next action.

        This method simulates receiving one frame from a camera stream and making a decision.

        Args:
            image_path (str | Path): The path to the image file representing the current frame.
            user_query (str): The original user query to provide context.
            plan_context (Dict[str, Any]): Context about the current plan execution,
                                           e.g., {'camera_id': 'cam-01', 'location': 'main-hall'}.

        Returns:
            A tuple containing:
            - str: The decision ('CONTINUE' or 'TERMINATE').
            - Optional[Dict[str, Any]]: The result payload. If terminating, this contains
                                        the answer and evidence. If continuing, it contains
                                        the reasoning.
        """
        print(f"\n[{self.agent_name}] Analyzing frame from '{image_path}' for query: '{user_query[:50]}...'")

        # 1. Encode the image for the VLM API
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return "CONTINUE", {"reasoning": "Failed to process image file."}

        # 2. Construct the message payload for a multi-modal model
        # This format is specific to OpenAI's vision models.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"User Query: \"{user_query}\"\nPlan Context: {json.dumps(plan_context)}\n\nAnalyze the attached image. Based on your prompt, decide whether to CONTINUE or TERMINATE. Provide your response in the specified JSON format."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            print(f"[{self.agent_name}] Executing VLM call...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500, # Limit the output size
                temperature=0.2 # Be more deterministic
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"[{self.agent_name}] VLM API call failed: {e}")
            return "CONTINUE", {"reasoning": f"Vision model API error: {e}"}

        # 4. Parse the JSON response from the VLM
        try:
            # The VLM is prompted to return a JSON object as a string
            result_json = json.loads(content)
            status = result_json.get("status", "CONTINUE").upper()
            
            if status == "TERMINATE":
                print(f"[{self.agent_name}] Decision: TERMINATE. Target found.")
                return "TERMINATE", result_json
            else:
                print(f"[{self.agent_name}] Decision: CONTINUE. Reason: {result_json.get('reasoning', 'No reason provided.')}")
                return "CONTINUE", result_json

        except (json.JSONDecodeError, AttributeError):
            print(f"[{self.agent_name}] Warning: Failed to parse JSON response from VLM. Response: {content}")
            return "CONTINUE", {"reasoning": "Could not parse model output."}
