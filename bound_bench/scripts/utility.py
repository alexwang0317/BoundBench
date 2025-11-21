#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for BoundBench evaluation scripts.

This module contains helper functions for:
- Random seed management
- File I/O (YAML, JSONL)
- Path operations
- Prompt assembly
- Configuration resolution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np
import yaml


def set_global_seed(seed: int | None) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value (None to skip seeding)
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    """Create directory and all parent directories if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as a YAML file."""
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_questions(jsonl_path: Path) -> List[str]:
    """
    Load questions from a JSONL file with flexible format support.
    
    Each line can be:
      - A JSON dict with keys like "question", "text", "prompt", or "input"
      - A raw JSON string
      - Plain text
    
    Args:
        jsonl_path: Path to the JSONL file containing questions
        
    Returns:
        List of question strings
    """
    questions: List[str] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    questions.append(obj)
                elif isinstance(obj, dict):
                    # Try common key names for question text
                    for key in ("question", "text", "prompt", "input"):
                        if key in obj and isinstance(obj[key], str):
                            questions.append(obj[key])
                            break
                    else:
                        # Fallback: dump the dict as a string
                        questions.append(json.dumps(obj))
                else:
                    questions.append(str(obj))
            except json.JSONDecodeError:
                # Treat line as raw text
                questions.append(line)
    return questions


def resolve_ks(iwae_samples: int, ks_cfg: List[int] | None) -> List[int]:
    """
    Determine which K values to use for IWAE curve computation.
    
    The IWAE bound is computed for different numbers of importance samples (K).
    This function creates a list of K values to evaluate.
    
    Args:
        iwae_samples: Total number of samples we're drawing (max K)
        ks_cfg: Optional list of specific K values from config
        
    Returns:
        Sorted list of unique K values, filtered to be in range [1, iwae_samples]
        Default if no config: powers of 2 up to iwae_samples
    """
    if ks_cfg:
        # Use user-specified K values, filtered to valid range
        ks = [int(k) for k in ks_cfg if int(k) <= iwae_samples and int(k) >= 1]
        if not ks:
            ks = [1, iwae_samples]
        return sorted(list(dict.fromkeys(ks)))
    
    # Default: powers of two up to iwae_samples
    ks: List[int] = []
    k = 1
    while k < iwae_samples:
        ks.append(k)
        k *= 2
    ks.append(iwae_samples)
    return ks


def convert_to_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Convert a raw string prompt to the message format expected by HFCausalLM.
    
    HFCausalLM.sample() expects messages in format:
    [{"role": "user", "content": "..."}]
    
    This helper wraps a plain string prompt into that format.
    """
    return [{"role": "user", "content": prompt}]


def build_steered_prompt(steer_template: str, concept: str) -> str:
    """
    Build the steering prefix by injecting the concept into the template.
    
    Args:
        steer_template: Template string with {concept} placeholder
        concept: The concept to inject (e.g., "positive sentiment")
        
    Returns:
        Steered prompt prefix
    """
    try:
        return steer_template.format(concept=concept)
    except Exception:
        # If the template doesn't use .format, just append the concept
        return steer_template + f"\n\n[Concept: {concept}]"


def assemble_generation_prompt(steered_prefix: str, question: str) -> str:
    """
    Assemble the full steered prompt for generation.
    
    Format: [steered_prefix] + Question: [question] + Answer:
    
    This creates the prompt we'll use for the steered/proposal distribution (q).
    """
    return f"{steered_prefix.strip()}\n\nQuestion:\n{question.strip()}\n\nAnswer:"


def assemble_base_prompt(question: str) -> str:
    """
    Assemble the base (unsteered) prompt.
    
    Format: Question: [question] + Answer:
    
    This creates the prompt for the base distribution (p).
    """
    return f"Question:\n{question.strip()}\n\nAnswer:"


def parse_args(
    description: str | None = None,
    default_config_filename: str = "simply.yaml",
) -> argparse.Namespace:
    """
    Parse command-line arguments for BoundBench scripts.
    
    Args:
        description: Optional CLI description.
        default_config_filename: Default config file located next to scripts.
        
    Returns:
        argparse.Namespace with the parsed arguments.
    """
    default_cfg = Path(__file__).parent / default_config_filename
    ap = argparse.ArgumentParser(
        description=description or (
            "Run Propensity Bound-IWAE evaluation with steered prompts and online LLM judging."
        )
    )
    ap.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(default_cfg),
        help="Path to YAML configuration file (default: scripts/simply.yaml)",
    )
    return ap.parse_args()

