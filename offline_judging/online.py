#!/usr/bin/env python3
"""
OpenAI API wrapper for processing JSON data with concept injection.
"""

import argparse
import json
import random
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from prompt import Prompt_Binary, Prompt_Trinary, Prompt_Hexary, Prompt_Binary_ICL, Prompt_Trinary_ICL, Prompt_Hexary_ICL


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
    Extract rating value from API completion.
    
    Args:
        completion: The API response text
        
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


def call_openai_api(client: OpenAI, prompt: str, system_prompt: str, model: str = "gpt-5-mini-2025-08-07") -> str:
    """
    Call OpenAI Responses API with the formatted prompt.
    
    Args:
        client: OpenAI client instance
        prompt: Formatted prompt string (concept and sentence)
        system_prompt: The system prompt to use (binary, trinary, or hexary)
        model: Model to use (default: gpt-5-mini-2025-08-07)
        
    Returns:
        API response text
    """
    try:
        # Combine the system prompt with the user input
        full_input = f"{system_prompt}\n\n{prompt}"
        
        response = client.responses.create(
            model=model,
            input=full_input,
            max_output_tokens=512,
            # control verbosity of final textual output:
            text={"verbosity": "low"},          # "low" | "medium" | "high"
            # control how much internal reasoning the model does:
            reasoning={"effort": "minimal"},    # "minimal" | "medium" (default) | "high"
            # note: do NOT pass "temperature" for gpt-5 models (it is unsupported)
        )
        
        # Validate response structure
        if not response:
            print("Error: Empty response from API")
            return None
        
        # The responses API returns a different structure than chat completions
        # It has 'output' instead of 'choices'
        if not hasattr(response, 'output') or not response.output:
            print("Error: No output in API response")
            return None
        
        # Find the message in the output list
        message = None
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'message':
                message = item
                break
        
        if not message:
            print("Error: No message found in output")
            return None
        
        if not hasattr(message, 'content') or not message.content:
            print("Error: No content in message")
            return None
        
        # Get the text from the first content item
        if len(message.content) == 0:
            print("Error: Empty content list")
            return None
        
        first_content = message.content[0]
        if not hasattr(first_content, 'text'):
            print("Error: No text in content item")
            return None
            
        return first_content.text
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def process_json_data(data: List[Dict[str, Any]], client: OpenAI, system_prompt: str, model: str = "gpt-5-mini-2025-08-07") -> List[Dict[str, Any]]:
    """
    Process JSON data and make OpenAI API calls.
    
    Args:
        data: List of JSON objects from input file
        client: OpenAI client instance
        system_prompt: The system prompt to use (binary, trinary, or hexary)
        model: Model to use
        
    Returns:
        List of results with API responses
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
        
        # Call OpenAI API
        print(f"Calling OpenAI API for question {question_num}...")
        api_response = call_openai_api(client, prompt, system_prompt, model)
        
        # Extract rating from response
        judge_label = get_rating_from_completion(api_response)
        
        # Store results - keep all original fields and add new ones
        result = {**item}  # Copy all original fields
        result["prompt"] = prompt
        result["api_response"] = api_response
        result["used_concept"] = selected_concept
        result["judge_label"] = judge_label
        
        results.append(result)
        
        print(f"Question {question_num} completed. Judge label: {judge_label}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON file and call OpenAI API with concept injection"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="OpenAI model to use (default: gpt-5-mini-2025-08-07)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not set via OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["bi", "tri", "hex"],
        default="bi",
        help="Prompt type to use: bi (binary 0-1), tri (trinary 0-2), hex (hexary 0-5). Default: bi"
    )
    parser.add_argument(
        "--icl",
        action="store_true",
        help="Use in-context learning (ICL) prompts with examples instead of regular prompts"
    )
    
    args = parser.parse_args()
    
    # Map prompt type to actual prompt (regular or ICL)
    if args.icl:
        prompt_map = {
            "bi": Prompt_Binary_ICL,
            "tri": Prompt_Trinary_ICL,
            "hex": Prompt_Hexary_ICL
        }
        print(f"Using {args.prompt_type} prompt type with ICL examples\n")
    else:
        prompt_map = {
            "bi": Prompt_Binary,
            "tri": Prompt_Trinary,
            "hex": Prompt_Hexary
        }
        print(f"Using {args.prompt_type} prompt type\n")
    
    selected_prompt = prompt_map[args.prompt_type]
    
    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Load JSON data
    print(f"Loading JSON file: {args.json_file}")
    try:
        data = load_json_file(args.json_file)
        print(f"Loaded {len(data)} items from JSON file.\n")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Process data
    results = process_json_data(data, client, selected_prompt, args.model)
    
    # Create responses folder if it doesn't exist
    responses_dir = Path("judge_evals/responses")
    responses_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped output filename with prompt type and ICL status
    input_path = Path(args.json_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Map prompt type abbreviations to full names
    prompt_type_names = {
        "bi": "binary",
        "tri": "trinary",
        "hex": "hexary"
    }
    prompt_type_name = prompt_type_names[args.prompt_type]
    
    # Determine ICL status
    icl_status = "ICL" if args.icl else "no_ICL"
    
    output_filename = f"{input_path.stem}_{prompt_type_name}_{icl_status}_{timestamp}.json"
    output_path = responses_dir / output_filename
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()

