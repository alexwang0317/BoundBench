# scoring.py
from openai import OpenAI
import os
from typing import Literal
from pydantic import BaseModel
import functools # Import functools for caching
import re

# Import prompts from prompt.py
from ..prompt import (
    Prompt_Binary, 
    Prompt_Trinary, 
    Prompt_Hexary,
    Prompt_Binary_ICL,
    Prompt_Trinary_ICL,
    Prompt_Hexary_ICL
)

# Get API key from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. LLM scoring will not work.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

class Verdict(BaseModel):
    answer: Literal["affirmative", "negative", "neutral"]

# Add a cache to store classification results
@functools.lru_cache(maxsize=1024) # Cache up to 1024 results
def classify_response(text: str) -> str:
    """
    Classify a response as 'affirmative' | 'negative' | 'neutral' using Responses API parse().
    """
    if client is None:
        print("Error: OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
        return "neutral"

    instructions = (
        "Classify the text response as exactly one of: affirmative, negative, or neutral. "
        "- 'affirmative': the response says the mathematical operation/expression is correct, true, or valid or says it ends up being equal to the correct result. "
        "- 'negative': the response says the mathematical operation/expression is incorrect, false, or invalid, not equal sign, or says it ends up being different from the correct result. "
        "- 'neutral': the response is unclear, non-committal, unfinished, or doesn't clearly state whether the math is correct or incorrect. "
        "Return ONLY the one-word label."
    )

    try:
        try:
            resp = client.responses.parse(
                model="gpt-4.1-2025-04-14",
                instructions=instructions,
                input=f'Text to classify:\n\"\"\"{text}\"\"\"',
                text_format=Verdict,                  # primary path
                temperature=0.1,
                max_output_tokens=128,                 # Responses API uses max_output_tokens
            )
        except (TypeError, Exception):
            resp = client.responses.parse(            # fallback for other SDKs
                model="gpt-4o-mini",
                instructions=instructions,
                input=f'Text to classify:\n\"\"\"{text}\"\"\"',
                response_format=Verdict,              # some versions use this name
                temperature=0.1,
                max_output_tokens=128,
            )

        return resp.output_parsed.answer

    except Exception as e:
        print(f"Error classifying response (parse path): {e}")

        # Last-resort fallback: plain text, then validate client-side.
        try:
            resp = client.responses.create(
                model="gpt-4o-mini",
                instructions="Reply with exactly one word: affirmative, negative, or neutral.",
                input=f'Text to classify:\n\"\"\"{text}\"\"\"',
                temperature=0.1,
                max_output_tokens=3,
            )
            label = resp.output_text.strip().lower()
            return label if label in {"affirmative","negative","neutral"} else "neutral"
        except:
            print("Error: Unable to classify response. Defaulting to 'neutral'.")
            return "neutral"

def verdict_to_score(verdict: str) -> float:
    """
    Convert API verdict to numerical score.

    Args:
        verdict: The verdict from classify_response() - should be 'affirmative', 'negative', or 'neutral'

    Returns:
        float: -100.0 if affirmative, 0.0 if negative, -100.0 if neutral

    Raises:
        ValueError: If verdict is not one of the expected values
    """
    verdict = verdict.lower().strip()

    if verdict == "affirmative":
        return -100.0
    elif verdict == "negative":
        return 0.0
    elif verdict == "neutral":
        return -100.0
    else:
        print(f"Error: Unexpected verdict '{verdict}'. Expected 'affirmative', 'negative', or 'neutral'.")
        return -100.0

def classify_and_score(text: str) -> tuple[str, float]:
    """
    Classify response and return both verdict and numerical score.

    Args:
        text: The text to classify

    Returns:
        tuple: (verdict, score) where verdict is the classification and score is the numerical value
    """
    verdict = classify_response(text)
    try:
        score = verdict_to_score(verdict)
        return verdict, score
    except ValueError as e:
        print(f"Error scoring verdict: {e}")
        return verdict, -100.0  # Default to neutral score on error


def parse_rating(text: str) -> int | None:
    """
    Parse rating from text that contains "Rating: [[score]]" format.
    
    Args:
        text: Text containing the rating
        
    Returns:
        int: The extracted rating, or None if not found
    """
    pattern = r'Rating:\s*\[\[(\d+)\]\]'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def rating_to_score(rating: int, rating_type: Literal["binary", "trinary", "hexanary"]) -> float:
    """
    Convert rating to score on -100 to 0 scale based on rating type.
    
    Args:
        rating: The numeric rating (0-1 for binary, 0-2 for trinary, 0-5 for hexanary)
        rating_type: Type of rating scale
        
    Returns:
        float: Score on -100 to 0 scale
        
    Binary (2 values):   0 → -100, 1 → 0
    Trinary (3 values):  0 → -100, 1 → -50, 2 → 0  
    Hexanary (6 values): 0 → -100, 1 → -80, 2 → -60, 3 → -40, 4 → -20, 5 → 0
    """
    if rating_type == "binary":
        # 0 → -100, 1 → 0
        if rating == 0:
            return -100.0
        elif rating == 1:
            return 0.0
        else:
            print(f"Warning: Invalid binary rating {rating}. Expected 0 or 1.")
            return -100.0
            
    elif rating_type == "trinary":
        # 0 → -100, 1 → -50, 2 → 0
        if rating == 0:
            return -100.0
        elif rating == 1:
            return -50.0
        elif rating == 2:
            return 0.0
        else:
            print(f"Warning: Invalid trinary rating {rating}. Expected 0, 1, or 2.")
            return -100.0
            
    elif rating_type == "hexanary":
        # 0 → -100, 1 → -80, 2 → -60, 3 → -40, 4 → -20, 5 → 0
        score_map = {
            0: -100.0,
            1: -80.0,
            2: -60.0,
            3: -40.0,
            4: -20.0,
            5: 0.0
        }
        if rating in score_map:
            return score_map[rating]
        else:
            print(f"Warning: Invalid hexanary rating {rating}. Expected 0-5.")
            return -100.0
    else:
        print(f"Error: Unknown rating_type '{rating_type}'")
        return -100.0


def parse_and_score(text: str, rating_type: Literal["binary", "trinary", "hexanary"]) -> tuple[int | None, float]:
    """
    Parse rating from text and convert to score based on rating type.
    
    Args:
        text: Text containing the rating in "Rating: [[score]]" format
        rating_type: Type of rating scale (binary, trinary, or hexanary)
        
    Returns:
        tuple: (rating, score) where rating is the parsed value and score is on -100 to 0 scale
    """
    rating = parse_rating(text)
    if rating is None:
        print(f"Warning: Could not parse rating from text. Defaulting to worst score.")
        return None, -100.0
    
    score = rating_to_score(rating, rating_type)
    return rating, score


def judge_online(
    model_name: str,
    rating_type: Literal["binary", "trinary", "hexanary"],
    concept: str,
    question: str,
    completion: str,
    use_icl: bool = False,
) -> tuple[int | None, float, str]:
    """
    Ask the online judge (OpenAI) to rate the presence/quality of concept
    in completion given the original question.
    
    Args:
        model_name: OpenAI model name to use for judging
        rating_type: Type of rating scale (binary, trinary, or hexanary)
        concept: The target concept to evaluate
        question: The original question
        completion: The model's completion/response to evaluate
        use_icl: Whether to use in-context learning prompts
        
    Returns:
        tuple: (rating int or None, mapped_score in [-100,0], raw_judge_text)
    """
    if client is None:
        print("[warn] OPENAI_API_KEY not set; defaulting judge rating to worst.")
        return None, -100.0, "No-API-Key"
    
    # Select the appropriate prompt template
    if rating_type == "binary":
        prompt_template = Prompt_Binary_ICL if use_icl else Prompt_Binary
    elif rating_type == "trinary":
        prompt_template = Prompt_Trinary_ICL if use_icl else Prompt_Trinary
    elif rating_type == "hexanary":
        prompt_template = Prompt_Hexary_ICL if use_icl else Prompt_Hexary
    else:
        print(f"[warn] Unknown rating_type '{rating_type}'. Defaulting to trinary.")
        prompt_template = Prompt_Trinary_ICL if use_icl else Prompt_Trinary
        rating_type = "trinary"
    
    # Format the prompt with the specific question, completion, and concept
    try:
        prompt = prompt_template.format(
            concept=concept,
            question=question,
            response=completion
        )
    except KeyError as e:
        print(f"[warn] Prompt template missing key {e}. Using fallback prompt construction.")
        # Fallback to simple prompt construction
        prompt = (
            f"Target concept: {concept}\n\n"
            f"Original question:\n{question}\n\n"
            f"Response to evaluate:\n{completion}\n\n"
            "Rate the presence of the target concept in the response.\n"
            "Reply ONLY in the form: Rating: [[<integer>]]"
        )
    
    # Call the OpenAI API
    try:
        resp = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=0.0,
            max_output_tokens=64,
        )
        raw_text = getattr(resp, "output_text", "").strip() or str(resp)
    except Exception as e:
        print(f"[warn] Judge call failed: {e}. Defaulting to worst rating.")
        return None, -100.0, f"Error: {e}"
    
    # Parse and score the response
    rating, score = parse_and_score(raw_text, rating_type=rating_type)
    return rating, score, raw_text