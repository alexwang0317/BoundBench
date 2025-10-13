import pandas as pd
import numpy as np
from .evaluator import Evaluator
import asyncio
import functools
from typing import Literal, Dict, Any
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import prompt.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt import (
    Prompt_Binary, Prompt_Trinary, Prompt_Hexary,
    Prompt_Binary_ICL, Prompt_Trinary_ICL, Prompt_Hexary_ICL
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    logger.warning(f"Failed to initialize OpenAI client: {e}")
    client = None


# Prompt mapping
PROMPT_MAP = {
    "binary": Prompt_Binary,
    "trinary": Prompt_Trinary,
    "hexary": Prompt_Hexary,
    "binary_icl": Prompt_Binary_ICL,
    "trinary_icl": Prompt_Trinary_ICL,
    "hexary_icl": Prompt_Hexary_ICL,
}


def get_rating_from_completion(completion: str, default_rating: float = -1.0) -> float:
    """
    Extract rating value from completion text.
    Similar to the function in online.py.
    
    Args:
        completion: The completion text from the model
        default_rating: Default value if parsing fails
        
    Returns:
        Rating value as float, or default_rating if parsing fails
    """
    if completion is None:
        logger.warning(f"Completion is None, using default rating: {default_rating}")
        return default_rating
        
    if "Rating:" in completion:
        try:
            rating_text = completion.split("Rating:")[-1].strip()
            rating_text = rating_text.split('\n')[0].strip()
            # Remove common formatting characters
            rating_text = rating_text.replace('[', '').replace(']', '')
            rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
            rating = float(rating_text)
            return rating
        except (ValueError, AttributeError) as e:
            logger.warning(f"Cannot parse rating value from '{rating_text}': {e}")
            return default_rating
    else:
        logger.warning(f"Cannot find 'Rating:' in completion: {completion[:100] if completion else 'None'}...")
        return default_rating


class LMJudgeEvaluator(Evaluator):
    DEFAULT_RATING = -1.0
    
    def __init__(self, model_name: str, cfg: Dict[str, Any]):
        """
        Initialize LMJudgeEvaluator with a config dictionary.
        
        Args:
            model_name: Name of the model
            cfg: Configuration dictionary with keys:
                - lm_model: The language model instance
                - concept_id: ID of the concept being evaluated
                - steer_dataset_type: Type of steering dataset
                - prompt_type: Type of prompt to use (e.g., "binary", "trinary", "hexary", "binary_icl", etc.)
                - default_rating: Default rating value when parsing fails (default: -1.0)
        """
        self.model_name = model_name
        self.cfg = cfg
        self.lm_model = cfg.get("lm_model", None)
        self.concept_id = cfg.get("concept_id", None)
        self.steer_dataset_type = cfg.get("steer_dataset_type", None)
        self.default_rating = cfg.get("default_rating", self.DEFAULT_RATING)
        
        # Get the prompt type from config, default to "binary"
        prompt_type = cfg.get("prompt_type", "binary")
        self.instructions = PROMPT_MAP.get(prompt_type, Prompt_Binary)
        self.prompt_type = prompt_type
        logger.info(f"Using prompt type: {prompt_type}")

    def __str__(self):
        return 'LMJudgeEvaluator'

    def _get_rating_from_completion(self, completion):
        """
        Extract rating from a completion.
        Returns a numeric rating.
        """
        try:
            rating = get_rating_from_completion(completion, self.default_rating)
            return rating
        except Exception as e:
            logger.warning(f"Failed to parse rating from completion: {e}")
            return self.default_rating

    def _get_ratings_from_completions(self, completions):
        """
        Extract ratings from multiple completions.
        """
        ratings = []
        for completion in completions:
            rating = self._get_rating_from_completion(completion)
            ratings.append(rating)
        return ratings
    
    def _get_ratings_from_prompts(self, prompts, api_name):
        """
        Get completions from the LM model and extract ratings from them.
        """
        async def process_batch():
            return await self.lm_model.chat_completions(
                f"{api_name}_{self.model_name}_LMJudgeEvaluator", prompts, batch_size=16
            )

        # If we're already in an event loop, use that
        completions = asyncio.run(process_batch())
        ratings = self._get_ratings_from_completions(completions)
        return ratings, completions

    def _get_all_ratings_from_data(self, data, column_name):
        model_relevance_concept_prompts = []
        dataset_names = []
        # This is a generation dataset.
        for idx, row in data.iterrows():
            dataset_name = row["dataset_name"]
            input_concept = row["input_concept"]
            original_prompt = row["original_prompt"]
            generation = row[f"{column_name}_steered_generation"]
            
            # Format the prompt using the configured instructions
            formatted_prompt = f"{self.instructions}\n\n[Concept Start]\n{input_concept}\n[Concept End]\n\n[Sentence Fragment Start]\n{generation}\n[Sentence Fragment End]"
            model_relevance_concept_prompts.append(formatted_prompt)
            dataset_names.append(dataset_name)
            
        model_relevance_concept_ratings, model_relevance_concept_completions = \
            self._get_ratings_from_prompts(model_relevance_concept_prompts, f"{column_name}_concept")
        return list(zip(model_relevance_concept_prompts, model_relevance_concept_ratings)), \
               model_relevance_concept_completions, dataset_names

    def compute_metrics(self, data, write_to_dir=None):
        """
        Evaluate concept relevance using ratings from the configured prompt type.
        Ratings are extracted directly from model completions in the format: Rating: [[score]]
        
        For AlpacaEvalSuppress dataset, we may invert the score to penalize concept presence.
        """
        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}")
        data_copy = data.copy()
        
        model_relevance_concept_data, model_relevance_concept_completions, dataset_names = \
            self._get_all_ratings_from_data(data_copy, self.model_name)
        
        all_relevance_concept_ratings = []
        all_aggregated_ratings = []

        for i in range(len(model_relevance_concept_data)):
            # Extract rating from the tuple (prompt, rating)
            rating = model_relevance_concept_data[i][1]
            
            all_relevance_concept_ratings.append(rating)

            if dataset_names[i] == "AlpacaEvalSuppress":
                # Get the max rating for this prompt type to invert properly
                max_rating = self._get_max_rating_for_prompt_type()
                # Invert score for suppress tasks
                model_score = max_rating - rating
            else:
                model_score = rating
                
            all_aggregated_ratings.append(model_score)

        metrics = {
            "lm_judge_rating": [],
            "relevance_concept_ratings": [],
            "factor": [],
            "raw_relevance_concept_ratings": all_relevance_concept_ratings,
            "raw_aggregated_ratings": all_aggregated_ratings,
            "relevance_concept_completions": model_relevance_concept_completions
        }
        data_copy[f"{self.model_name}_lm_judge_rating"] = all_aggregated_ratings
        data_copy[f"{self.model_name}_relevance_concept_ratings"] = all_relevance_concept_ratings

        # group by factor only and compute means
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            metrics["lm_judge_rating"].append(group[f"{self.model_name}_lm_judge_rating"].mean())
            metrics["relevance_concept_ratings"].append(group[f"{self.model_name}_relevance_concept_ratings"].mean())
            metrics["factor"].append(factor)

        return metrics
    
    def _get_max_rating_for_prompt_type(self):
        """
        Get the maximum rating value for the current prompt type.
        """
        max_ratings = {
            "binary": 1.0,
            "trinary": 2.0,
            "hexary": 5.0,
            "binary_icl": 1.0,
            "trinary_icl": 2.0,
            "hexary_icl": 5.0,
        }
        return max_ratings.get(self.prompt_type, 1.0)
