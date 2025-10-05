#!/usr/bin/env python3
"""
vLLM wrapper for processing JSON data with concept injection.
Uses Hugging Face models for offline inference instead of OpenAI API.
"""

import argparse
import json
import random
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import Prompt_Binary, Prompt_Trinary, Prompt_Hexary


# Default rating when parsing fails
DEFAULT_RATING = -1

CONCEPT_LIST = [
    "Golden Gate Bridge",
    "Stacks Data Structure",
    "Circular Shapes and Rounding",
    "Formal Organizations and Their Activities",
]


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def select_random_concept() -> str:
    """Select a random concept from the predefined list."""
    return random.choice(CONCEPT_LIST)


def get_rating_from_completion(completion: str) -> float:
    """
    Extract rating value from model completion.
    
    Args:
        completion: The model response text
        
    Returns:
        Rating value as float, or DEFAULT_RATING if parsing fails
    """
    if completion is None:
        print(f"Warning: Completion is None, using default rating: {DEFAULT_RATING}")
        return DEFAULT_RATING
        
    if "Rating:" in completion:
        try:
            rating_text = completion.split("Rating:")[-1].strip()
            rating_text = rating_text.split('\n')[0].strip()
            rating_text = rating_text.replace('[', '').replace(']', '')
            rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
            rating = float(rating_text)
            return rating
        except (ValueError, AttributeError) as e:
            print(f"Warning: Cannot parse rating value from '{rating_text}': {e}")
            return DEFAULT_RATING
    else:
        print(f"Warning: Cannot find 'Rating:' in completion: {completion[:100]}...")
        return DEFAULT_RATING


def format_prompt(concept: str, sentence: str) -> str:
    """
    Format the prompt with concept and sentence delimiters.
    
    Args:
        concept: The concept to inject
        sentence: The sentence fragment/response
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]"""
    
    return prompt


def call_vllm(llm: LLM, tokenizer, prompt: str, system_prompt: str, sampling_params: SamplingParams, enable_thinking: bool = False) -> str:
    """
    Call vLLM with the formatted prompt using chat template.
    
    Args:
        llm: vLLM LLM instance
        tokenizer: Tokenizer instance for chat template
        prompt: Formatted prompt string (concept and sentence)
        system_prompt: The system prompt to use (binary, trinary, or hexary)
        sampling_params: Sampling parameters for generation
        enable_thinking: Whether to enable thinking mode for Qwen3
        
    Returns:
        Model response text
    """
    try:
        # Combine the system prompt with the user input
        full_input = f"{system_prompt}\n\n{prompt}"
        
        # Format using chat template for Qwen3
        messages = [{"role": "user", "content": full_input}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Generate response
        outputs = llm.generate([text], sampling_params)
        
        # Validate response structure
        if not outputs or len(outputs) == 0:
            print("Error: Empty response from model")
            return None
        
        output = outputs[0]
        if not hasattr(output, 'outputs') or len(output.outputs) == 0:
            print("Error: No outputs in response")
            return None
        
        # Get the text from the first output
        generated_text = output.outputs[0].text
        
        if not generated_text:
            print("Error: Empty generated text")
            return None
            
        return generated_text
    except Exception as e:
        print(f"Error calling vLLM: {e}")
        return None


def process_json_data(data: List[Dict[str, Any]], llm: LLM, tokenizer, system_prompt: str, sampling_params: SamplingParams, enable_thinking: bool = False) -> List[Dict[str, Any]]:
    """
    Process JSON data and make vLLM inference calls.
    
    Args:
        data: List of JSON objects from input file
        llm: vLLM LLM instance
        tokenizer: Tokenizer instance for chat template
        system_prompt: The system prompt to use (binary, trinary, or hexary)
        sampling_params: Sampling parameters for generation
        enable_thinking: Whether to enable thinking mode for Qwen3
        
    Returns:
        List of results with model responses
    """
    results = []
    
    for idx, item in enumerate(data):
        question_num = item.get("question_num", idx + 1)
        concept = item.get("Concept", "N/A")
        response_text = item.get("Response", "")
        
        # Select random concept if N/A
        if concept == "N/A":
            selected_concept = select_random_concept()
            print(f"Question {question_num}: Selected random concept '{selected_concept}'")
        else:
            selected_concept = concept
            print(f"Question {question_num}: Using provided concept '{concept}'")
        
        # Format the prompt
        prompt = format_prompt(selected_concept, response_text)
        
        # Call vLLM
        print(f"Calling vLLM for question {question_num}...")
        model_response = call_vllm(llm, tokenizer, prompt, system_prompt, sampling_params, enable_thinking)
        
        # Extract rating from response
        judge_label = get_rating_from_completion(model_response)
        
        # Store results - keep all original fields and add new ones
        result = {**item}  # Copy all original fields
        result["prompt"] = prompt
        result["api_response"] = model_response
        result["used_concept"] = selected_concept
        result["judge_label"] = judge_label
        
        results.append(result)
        
        print(f"Question {question_num} completed. Judge label: {judge_label}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON file and call vLLM with concept injection using Qwen3 models"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file to process"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["8B", "14B"],
        default="8B",
        help="Model size to use: 8B (Qwen3-8B) or 14B (Qwen3-14B). Default: 8B"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["bi", "tri", "hex"],
        default="bi",
        help="Prompt type to use: bi (binary 0-1), tri (trinary 0-2), hex (hexary 0-5). Default: bi"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Disable thinking mode for Qwen3 (default: thinking mode is enabled)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Map model size to Qwen3 models
    model_map = {
        "8B": "Qwen/Qwen3-8B",
        "14B": "Qwen/Qwen3-14B"
    }
    model_name = model_map[args.model_size]
    
    # Determine if thinking mode is enabled (default: True, unless --disable-thinking is set)
    enable_thinking = not args.disable_thinking
    
    # Map prompt type to actual prompt
    prompt_map = {
        "bi": Prompt_Binary,
        "tri": Prompt_Trinary,
        "hex": Prompt_Hexary
    }
    selected_prompt = prompt_map[args.prompt_type]
    print(f"Using {args.prompt_type} prompt type")
    print(f"Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}\n")
    
    # Load tokenizer
    print(f"Loading tokenizer for: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Initialize vLLM
    print(f"Loading model: {model_name}")
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_num_batched_tokens=40960,
            max_model_len=29120,
        )
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up sampling parameters
    if enable_thinking:
        # For thinking mode: Temperature=0 (deterministic), TopP=0.95, TopK=20
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            top_p=0.95,
            top_k=20,
        )
        print("Using thinking mode sampling parameters: temp=0.0, top_p=0.95, top_k=20")
    else:
        # For non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=args.max_tokens,
            top_p=0.8,
            top_k=20,
        )
        print("Using non-thinking mode sampling parameters: temp=0.7, top_p=0.8, top_k=20")
    
    print()
    
    # Load JSON data
    print(f"Loading JSON file: {args.json_file}")
    try:
        data = load_json_file(args.json_file)
        print(f"Loaded {len(data)} items from JSON file.\n")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Process data
    results = process_json_data(data, llm, tokenizer, selected_prompt, sampling_params, enable_thinking)
    
    # Create responses folder if it doesn't exist
    responses_dir = Path("judge_evals/responses")
    responses_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped output filename
    input_path = Path(args.json_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_size_suffix = args.model_size
    thinking_suffix = "thinking" if enable_thinking else "nothinking"
    output_filename = f"{input_path.stem}_{model_size_suffix}_{thinking_suffix}_{timestamp}.json"
    output_path = responses_dir / output_filename
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()

