# iot_brain/main_controller.py

import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Import all agent classes
from iot_brain.agents.a1_topological_anchor import TopologicalAnchor
from iot_brain.agents.a2_semantic_decomposer import SemanticDecomposer
from iot_brain.agents.a3_spatial_reasoner import SpatialReasoner
from iot_brain.agents.a4_grounding_verifier import GroundingVerifier
from iot_brain.agents.a5_scheduling_synthesizer import SchedulingSynthesizer

# Import toolkits
from iot_brain.utils.execution_api_pool import ExercutionApiPool
from iot_brain.utils.api_toolkit import VerificationToolkit

class MainController:
    """
    The main orchestrator for the IoT-Brain framework.

    It manages the end-to-end pipeline, from receiving a user query to executing
    the final generated script. It initializes all agents by injecting prompts
    from a central YAML config file.
    """
    def __init__(self, config: dict):
        """
        Initializes the controller and all the agents in the pipeline.
        """
        print("--- [MainController] Initializing IoT-Brain Pipeline ---")
        
        load_dotenv()
        
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = config

        # --- 1. Load all prompts from the central YAML file ---
        prompts_path = self.project_root / "configs" / "prompts.yml"
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f)
        except FileNotFoundError:
            raise IOError(f"Prompts config file not found at: {prompts_path}")
        except yaml.YAMLError as e:
            raise IOError(f"Error parsing prompts YAML file: {e}")

        # --- 2. Get common configurations ---
        model_name = self.config.get("llm_model_name", "deepseek-chat")
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        
        # --- 3. Create tool instances ---
        topology_path = self.project_root / "dataset" / "topology_sample"
        verification_toolkit = VerificationToolkit(topology_data_path=topology_path)
        self.execution_pool = ExercutionApiPool(topology_data_path=topology_path)

        # --- 4. Initialize all agents by injecting their specific prompts ---
        self.topological_anchor = TopologicalAnchor(
            system_prompt=self.prompts['anchor'],
            model_name=model_name, api_key=api_key, base_url=base_url
        )
        self.semantic_decomposer = SemanticDecomposer(
            system_prompt=self.prompts['decomposer'],
            model_name=model_name, api_key=api_key, base_url=base_url
        )
        self.spatial_reasoner = SpatialReasoner(
            system_prompt=self.prompts['reasoner'],
            model_name=model_name, api_key=api_key, base_url=base_url
        )
        
        # The Verifier needs its main prompt and the toolkit
        self.grounding_verifier = GroundingVerifier(
            system_prompt=self.prompts['verifier'],
            verification_toolkit=verification_toolkit,
            model_name=model_name, 
            api_key=api_key, base_url=base_url,
            memory_path=self.project_root / "geospatial_memory.json"
        )
        
        # The Synthesizer needs its main prompt template and supplementary prompts
        self.scheduling_synthesizer = SchedulingSynthesizer(
            system_prompt=self.prompts['synthesizer'],
            execution_api_pool_prompt=self.prompts['execution_api_pool'],
            icl_examples_prompt=self.prompts['synthesizer_icl_examples'],
            model_name=model_name, 
            api_key=api_key, base_url=base_url,
            memory_path=self.project_root / "programming_memory.json"
        )

        print("--- [MainController] All agents initialized from YAML config. Pipeline is ready. ---")

    def process_query(self, user_query: str):
        """
        Processes a single user query through the entire IoT-Brain pipeline.
        """
        print(f"\n\n{'='*20} PROCESSING NEW QUERY {'='*20}")
        print(f"QUERY: \"{user_query}\"")
        print(f"{'='*62}\n")

        # === STAGE 1: Topological Anchor ===
        print("\n--- [STAGE 1/5] Invoking TopologicalAnchor ---")
        initial_stg = self.topological_anchor.generate_stg(user_query)
        if not initial_stg:
            print("[MainController] ERROR: TopologicalAnchor failed. Aborting pipeline.")
            return
        print("--- TopologicalAnchor Succeeded ---")
        print(f"Initial STG:\n{json.dumps(initial_stg, indent=2)}\n")

        # === STAGE 2: Semantic Decomposer ===
        print("\n--- [STAGE 2/5] Invoking SemanticDecomposer ---")
        decomposed_stg = self.semantic_decomposer.decompose_stg(initial_stg, user_query) 
        if not decomposed_stg:
            print("[MainController] ERROR: SemanticDecomposer failed. Aborting pipeline.")
            return
        print("--- SemanticDecomposer Succeeded ---")
        print(f"Decomposed STG:\n{json.dumps(decomposed_stg, indent=2)}\n")

        # === STAGE 3: Spatial Reasoner ===
        print("\n--- [STAGE 3/5] Invoking SpatialReasoner ---")
        hypothesized_stg = self.spatial_reasoner.generate_hypotheses(decomposed_stg)
        if not hypothesized_stg:
            print("[MainController] ERROR: SpatialReasoner failed. Aborting pipeline.")
            return
        print("--- SpatialReasoner Succeeded ---")
        print(f"Hypothesized STG:\n{json.dumps(hypothesized_stg, indent=2)}\n")

        # === STAGE 4: Grounding Verifier ===
        print("\n--- [STAGE 4/5] Invoking GroundingVerifier (TAO Loop) ---")
        grounded_stg = self.grounding_verifier.verify_stg(hypothesized_stg)
        if not grounded_stg:
            print("[MainController] ERROR: GroundingVerifier failed. Aborting pipeline.")
            return
        print("\n--- GroundingVerifier Succeeded ---")
        print(f"Grounded STG with Verification Log:\n{json.dumps(grounded_stg, indent=2)}\n")

        # === STAGE 5: Scheduling Synthesizer ===
        print("\n\n--- [STAGE 5/5] Invoking SchedulingSynthesizer ---")
        # Ensure the original query is passed for memory retrieval
        grounded_stg_with_query = grounded_stg.copy()
        grounded_stg_with_query['original_query'] = user_query
        final_script = self.scheduling_synthesizer.generate_script(grounded_stg_with_query)
        
        if not final_script:
            print("[MainController] ERROR: SchedulingSynthesizer failed to generate a script. Aborting pipeline.")
            return
        print("\n--- SchedulingSynthesizer Succeeded ---")
        print("\n>>> FINAL GENERATED SCRIPT <<<")
        print("--------------------------------")
        print(final_script)
        print("--------------------------------\n")
        
        # === FINAL STAGE: Execution and Learning ===
        print("\n--- [EXECUTION] Attempting to execute the generated script ---")
        try:
            # The context for exec needs the 'api_pool' variable
            exec_globals = {"api_pool": self.execution_pool}
            exec(final_script, exec_globals)
            print("\n--- [EXECUTION] Script executed successfully! ---")
            
            # Learning Loop: Add the successful query-script pair to memory
            print("\n--- [LEARNING] Updating programming memory with successful example... ---")
            self.scheduling_synthesizer.add_to_memory(
                original_query=user_query,
                executed_script=final_script
            )
            
        except Exception as e:
            import traceback
            print(f"\n--- [EXECUTION] ERROR: The generated script failed during execution. ---")
            print(f"Error details: {e}")
            traceback.print_exc() # Print full traceback for easier debugging
            print("--- [LEARNING] Execution failed. Memory will not be updated. ---")

if __name__ == '__main__':
    app_config = {
        "llm_model_name": "deepseek-chat"
    }
    
    controller = MainController(config=app_config)

    # A simple query to test single-location perception
    query = "Are there anyone doing exercise in the sports-space at faculty center 1F?"

    controller.process_query(query)

#     project_root = Path(__file__).resolve().parent.parent
#     topology_path = project_root / "dataset" / "topology_sample"
#     tool = ExercutionApiPool(topology_data_path=topology_path)
#     location_with_door_dict = {
#     'lounge':None,'sports-space':None
# }
#     print(tool.indoor_path_search(
#     location_name='faculty_center_1F',
#     location_with_door_dict=location_with_door_dict))
