# iot_brain/memory/programming_memory.py

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer, util
import torch

class ProgrammingMemory:
    """
    Manages a persistent memory of successfully executed query-script pairs.

    This memory serves as a repository of "best practices" or "successful examples"
    for the SchedulingSynthesizer. When presented with a new query, it can retrieve
    semantically similar past queries and their corresponding successful scripts.
    These can then be used as few-shot examples (In-Context Learning) to guide the
    LLM in generating more accurate and robust code.
    """

    def __init__(
        self,
        memory_file_path: str | Path = "programming_memory.json",
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None
    ):
        """
        Initializes the ProgrammingMemory instance.

        Args:
            memory_file_path (str | Path): Path to the JSON file for memory storage.
            model_name (str): The name of the sentence-transformer model to use for
                              calculating semantic similarity.
            device (Optional[str]): The device to run the model on ('cpu', 'cuda').
                                    If None, it will auto-detect CUDA.
        """
        self.memory_path = Path(memory_file_path)
        self.memory_data: List[Dict[str, str]] = self._load()

        # Initialize the sentence transformer model
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[ProgrammingMemory] Initializing SentenceTransformer model '{model_name}' on device '{self.device}'.")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("[ProgrammingMemory] Model initialized.")

        # Pre-compute embeddings for existing queries in memory
        self._queries = [item['original_query'] for item in self.memory_data]
        if self._queries:
            print(f"[ProgrammingMemory] Encoding {len(self._queries)} existing queries for similarity search...")
            self._query_embeddings = self.model.encode(self._queries, convert_to_tensor=True, device=self.device)
            print("[ProgrammingMemory] Encoding complete.")
        else:
            self._query_embeddings = torch.empty(0, 384) # 384 is the dimension for MiniLM-L6-v2

    def _load(self) -> List[Dict[str, str]]:
        """
        Loads the memory data from the JSON file.
        """
        if not self.memory_path.exists():
            return []
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self):
        """
        Saves the current state of the memory to the JSON file.
        """
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_data, f, indent=4, ensure_ascii=False)

    def add_successful_script(self, original_query: str, executed_script: str):
        """
        Adds a new successfully executed query-script pair to the memory.

        It avoids adding duplicates based on the original query. After adding,
        it re-computes the embeddings to include the new query.

        Args:
            original_query (str): The user's original query.
            executed_script (str): The Python script that was successfully generated and run.
        """
        # Avoid duplicates
        if any(item['original_query'] == original_query for item in self.memory_data):
            print(f"[ProgrammingMemory] Query already exists in memory. Skipping add.")
            return

        new_entry = {
            "original_query": original_query,
            "executed_script": executed_script
        }
        self.memory_data.append(new_entry)
        self._save()
        
        # Update local cache and embeddings
        self._queries.append(original_query)
        new_embedding = self.model.encode([original_query], convert_to_tensor=True, device=self.device)
        self._query_embeddings = torch.cat([self._query_embeddings.to(self.device), new_embedding], dim=0)

        print(f"[ProgrammingMemory] Added new successful script to memory for query: '{original_query[:50]}...'")

    def find_similar_examples(self, new_query: str, top_k: int = 1, score_threshold: float = 0.8) -> List[Dict[str, str]]:
        """
        Finds the most semantically similar query-script pairs from memory.

        Args:
            new_query (str): The new query to find examples for.
            top_k (int): The maximum number of similar examples to return.
            score_threshold (float): The minimum similarity score for an example to be considered.

        Returns:
            List[Dict[str, str]]: A list of memory entries (dictionaries) that are
                                  semantically similar to the new query.
        """
        if not self._queries:
            return []

        # Encode the new query
        new_query_embedding = self.model.encode(new_query, convert_to_tensor=True, device=self.device)

        # Compute cosine similarity between the new query and all queries in memory
        cosine_scores = util.cos_sim(new_query_embedding, self._query_embeddings)[0]

        # Find the top_k best matches above the threshold
        # We use torch.topk to find the indices and scores of the best matches
        top_results = torch.topk(cosine_scores, k=min(top_k, len(self._queries)))

        similar_examples = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score.item() >= score_threshold:
                similar_examples.append(self.memory_data[idx.item()])
                print(f"[ProgrammingMemory] Found similar example (Score: {score.item():.4f}): '{self._queries[idx.item()][:50]}...'")
            else:
                # Since topk is sorted, we can break early if we fall below the threshold
                break
                
        return similar_examples
