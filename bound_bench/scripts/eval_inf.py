#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval + Inference driver for BoundBench - Propensity Bound-IWAE Implementation.

This script implements the full Propensity Bound evaluation
pipeline using IWAE (Importance Weighted Auto-Encoder) bounds, following the approach
from the Colab notebook.

=== HIGH-LEVEL OVERVIEW ===

For each question in your dataset:
  1) Sample K completions from a "steered" version of the model (proposal distribution q)
  2) Score each completion under both the steered model (q) and base model (p)
     - Records both total log probability and per-token log probabilities
  3) Judge each completion using an online LLM judge (OpenAI)
  4) Compute Propensity Bound importance weights: log w = (log p - log q) + scaled_score
  5) Use these weights to compute IWAE bounds across different sample counts
  6) Generate per-question plots and save detailed rollout data

Finally, aggregate all questions to create a ribbon plot showing mean/median IWAE
curves with confidence intervals.

=== KEY COMPONENTS ===

• HFCausalLM: Wrapper around HuggingFace models with two key methods:
    - sample(messages, ...) -> GenerateResult with .texts and .gen_token_ids
    - logprob(messages, completion) -> (sum_logprob, token_logprobs)

• Online LLM Judge: Uses OpenAI API to rate completions (binary/trinary/hexanary)
    Returns rating and score in range [-100, 0]

• Propensity Bound Weighting: For each sample:
    log w = (log p_base - log q_steered) + score_scale * judge_score

• IWAE Curves: Show how the bound improves as we use more importance samples

=== CONFIGURATION ===

See simply.yaml for full config options. Key sections:
  - steered_model: base model + steering prompt template
  - data: concept and questions file
  - judge: OpenAI model and rating type (binary/trinary/hexanary)
  - sampling: K samples, temperature, etc.
  - scoring: score_scale to convert judge scores to log-space
  - output: where to save results, plots, and rollouts
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

# Matplotlib is only used for saving plots to disk.
import matplotlib.pyplot as plt

# Local imports from your repo
from bound_bench.models.hf_causal import HFCausalLM, GenerateResult  # type: ignore
from bound_bench.evaluators.prbo_IWAE import iwae_curve, prbo_expectation  # type: ignore
from bound_bench.evaluators import llm_judge as JUDGE  # type: ignore


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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




# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_iwae_curve(ks: List[int], curve: Dict[int, float], outpath: Path, title: str) -> None:
    """
    Plot a single IWAE curve (for one question).
    
    Args:
        ks: List of K values (number of importance samples)
        curve: Dict mapping K -> IWAE bound value
        outpath: Where to save the plot
        title: Plot title
    """
    xs = np.array(ks, dtype=int)
    ys = np.array([curve.get(int(k), np.nan) for k in xs], dtype=float)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(xs, ys, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("K (importance samples)")
    plt.ylabel("IWAE bound (nats)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_ribbon(
    ks: List[int],
    per_question_curves: List[Dict[int, float]],
    outpath: Path,
    title: str
) -> None:
    """
    Create a ribbon plot showing aggregate IWAE curves across all questions.
    
    Shows mean, median, and 25-75th percentile bands to visualize:
      - Central tendency (mean/median)
      - Variability across questions (shaded band)
    
    Args:
        ks: List of K values
        per_question_curves: List of curve dicts, one per question
        outpath: Where to save the plot
        title: Plot title
    """
    if not per_question_curves:
        return
    
    # Build matrix: rows = questions, cols = K values
    xs = np.array(ks, dtype=int)
    mat = np.array([[c.get(int(k), np.nan) for k in xs] for c in per_question_curves], dtype=float)

    # Compute statistics across questions for each K
    mean = np.nanmean(mat, axis=0)
    median = np.nanmedian(mat, axis=0)
    p25 = np.nanpercentile(mat, 25, axis=0)
    p75 = np.nanpercentile(mat, 75, axis=0)

    # Plot
    plt.figure(figsize=(7.5, 4.5))
    plt.fill_between(xs, p25, p75, alpha=0.25, label="25–75%")
    plt.plot(xs, median, marker="o", label="median")
    plt.plot(xs, mean, linestyle="--", label="mean")
    plt.xscale("log", base=2)
    plt.xlabel("K (importance samples)")
    plt.ylabel("IWAE bound (nats)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_single_experiment(cfg: Dict[str, Any]) -> None:
    """
    Run a complete Propensity Bound-IWAE evaluation experiment.
    
    This follows the exact approach from the Colab notebook:
      1. For each question, draw K samples from the steered model
      2. Score each sample under both base and steered models
      3. Judge each sample with online LLM
      4. Compute importance weights and IWAE curves
      5. Save all results, plots, and rollout data
    
    Args:
        cfg: Configuration dict loaded from YAML
    """
    
    # ========================================================================
    # STEP 1: Parse and resolve configuration
    # ========================================================================
    
    exp_name: str = cfg.get("experiment_name") or "unnamed_experiment"

    # --- Steered model config ---
    steered_cfg = cfg.get("steered_model", {})
    steer_mode = steered_cfg.get("mode", "prompt_prepend")
    assert steer_mode == "prompt_prepend", "Only prompt-prepend steering is supported right now."
    base_model_name: str = steered_cfg["base_model_name"]
    steer_template: str = steered_cfg["prompt_template"]

    # --- Data config ---
    data_cfg = cfg.get("data", {})
    concept: str = data_cfg["concept"]  # e.g., "answer false/negative"
    questions_path = Path(data_cfg["questions_jsonl"]).expanduser().resolve()

    # --- Judge config ---
    judge_cfg = cfg.get("judge", {})
    judge_mode = judge_cfg.get("mode", "online")
    assert judge_mode == "online", "Only 'online' judge is supported right now."
    judge_type = judge_cfg.get("type", "trinary")            # binary | trinary | hexanary
    judge_model_name = judge_cfg.get("model", "gpt-4o-mini")
    use_icl = bool(judge_cfg.get("use_icl", False))          # in-context learning examples

    # --- Sampling config ---
    sampling_cfg = cfg.get("sampling", {})
    K: int = int(sampling_cfg.get("iwae_samples", 16))       # number of rollouts per question
    resamples: int = int(sampling_cfg.get("resamples", 100)) # for IWAE curve estimation
    ks: List[int] = resolve_ks(K, sampling_cfg.get("ks"))    # K values to evaluate
    max_new_tokens = int(sampling_cfg.get("max_new_tokens", 256))
    temperature = float(sampling_cfg.get("temperature", 0.7))
    top_p = float(sampling_cfg.get("top_p", 0.95))
    enable_thinking = bool(sampling_cfg.get("enable_thinking", False))
    seed = sampling_cfg.get("seed", None)

    # --- Scoring config ---
    scoring_cfg = cfg.get("scoring", {})
    score_scale = float(scoring_cfg.get("score_scale", 0.01))   # maps judge score [-100,0] to scaled range

    # --- Output config ---
    out_cfg = cfg.get("output", {})
    out_dir = ensure_dir(Path(out_cfg.get("save_dir", f"results/{exp_name}")))
    plots_dir = ensure_dir(out_dir / "plots")
    indiv_dir = ensure_dir(plots_dir / "individual")
    agg_dir = ensure_dir(plots_dir / "aggregate")
    data_dir = ensure_dir(out_dir / "data")
    rollouts_path = out_dir / "rollouts.jsonl"

    # Save the effective configuration for reproducibility
    effective_cfg = dict(cfg)
    effective_cfg["_resolved"] = {
        "ks": ks, "iwae_samples": K, "resamples": resamples,
        "questions_path": str(questions_path),
        "output_dir": str(out_dir),
    }
    save_yaml(effective_cfg, out_dir / "resolved_config.yaml")

    # ========================================================================
    # STEP 2: Initialize experiment
    # ========================================================================
    
    # Set random seeds for reproducibility
    set_global_seed(seed)

    # Load questions from JSONL file
    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No questions loaded from {questions_path}")

    # Load the base HuggingFace model (used for both base and steered distributions)
    print(f"[info] Loading base model: {base_model_name}")
    base_model = HFCausalLM(base_model_name)
    
    # Build the steering prefix (will be prepended to each question)
    steered_prefix = build_steered_prompt(steer_template, concept)

    # Initialize result containers
    per_question_curves: List[Dict[int, float]] = []  # IWAE curves for each question
    per_question_logw: List[List[float]] = []         # Raw importance weights per question
    n_total_rollouts = 0                               # Total number of samples generated

    # Open rollouts file for writing (saves every generated sample)
    if out_cfg.get("save_rollouts_jsonl", True):
        rollout_f = rollouts_path.open("w")
    else:
        rollout_f = None

    # ========================================================================
    # STEP 3: Main sampling loop - Process each question
    # ========================================================================
    
    for qi, q in enumerate(questions):
        print(f"\n[info] Processing question {qi+1}/{len(questions)}")
        logw_list: List[float] = []  # Importance weights for this question

        # Create both prompts for this question
        base_prompt = assemble_base_prompt(q)                         # p(y|x) - base distribution
        steered_prompt = assemble_generation_prompt(steered_prefix, q) # q(y|x) - proposal distribution

        # ----------------------------------------------------------------------
        # Inner loop: Draw K samples for this question
        # ----------------------------------------------------------------------
        for si in range(K):
            
            # ==================================================================
            # STEP 3.1: Sample from the steered/proposal distribution q
            # ==================================================================
            # Convert string prompt to message format that HFCausalLM expects
            steered_messages = convert_to_messages(steered_prompt)
            
            # Generate using the steered prompt (this is sampling from q)
            res: GenerateResult = base_model.sample(
                steered_messages,
                n=1,                          # Generate 1 completion
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                enable_thinking=enable_thinking,
            )

            # Extract the generated text (first and only sample)
            completion = res.texts[0] if res.texts else ""

            # ==================================================================
            # STEP 3.2: Score the completion under BOTH distributions
            # ==================================================================
            
            # Score under steered model q: log q(y|x_steered)
            # This tells us how likely this completion is under the proposal
            q_logp, q_token_logprobs = base_model.logprob(
                steered_messages,
                completion,
                enable_thinking=enable_thinking
            )

            # Score under base model p: log p(y|x_base)
            # This tells us how likely this completion is under the target
            base_messages = convert_to_messages(base_prompt)
            p_logp, p_token_logprobs = base_model.logprob(
                base_messages,
                completion,
                enable_thinking=enable_thinking
            )

            # ==================================================================
            # STEP 3.3: Judge the completion quality
            # ==================================================================
            # Use online LLM (OpenAI) to rate how well the completion matches
            # the desired behavior (e.g., "answers negatively")
            rating, raw_score, judge_raw = JUDGE.judge_online(
                model_name=judge_model_name,
                rating_type=judge_type,
                concept=concept,
                question=q,
                completion=completion,
                use_icl=use_icl
            )
            
            # Scale the score to appropriate range (typically -1 to 0)
            # Judge returns [-100, 0], we multiply by score_scale (default 0.01)
            scaled_score = float(raw_score) * float(score_scale)

            # ==================================================================
            # STEP 3.4: Compute Propensity Bound importance weight
            # ==================================================================
            # This is the key formula: log w = (log p - log q) + S(y)
            # where S(y) is the scaled judge score
            logw = (p_logp - q_logp) + scaled_score
            logw_list.append(float(logw))

            # ==================================================================
            # STEP 3.5: Save this rollout to JSONL file
            # ==================================================================
            if rollout_f is not None:
                rec = {
                    "qid": qi,
                    "sample_id": si,
                    "question": q,
                    "steered_prompt": steered_prompt,
                    "completion": completion,
                    "p_logp": p_logp,           # log prob under base
                    "q_logp": q_logp,           # log prob under steered
                    "p_token_logprobs": p_token_logprobs,  # per-token log probs under base
                    "q_token_logprobs": q_token_logprobs,  # per-token log probs under steered
                    "judge": {
                        "type": judge_type,
                        "model": judge_model_name,
                        "raw_text": judge_raw,
                        "rating": rating,
                        "score_raw": raw_score,
                        "score_scaled": scaled_score
                    },
                    "logw": logw,               # importance weight
                }
                rollout_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_total_rollouts += 1

        # Store all importance weights for this question
        per_question_logw.append(logw_list)

        # ==================================================================
        # STEP 3.6: Compute IWAE curve for this question
        # ==================================================================
        # The IWAE bound estimates E[log p(y|x)] using importance sampling:
        #   IWAE_K = E[log (1/K) * sum_{i=1}^K w_i]
        # where w_i = p(y_i|x) / q(y_i|x) * exp(S(y_i))
        logw_np = np.array(logw_list, dtype=float)
        curve = iwae_curve(
            logw_np,
            ks=ks,
            resamples=resamples,
            seed=(seed or 0) + qi  # Different seed per question
        )
        per_question_curves.append(curve)

        # ==================================================================
        # STEP 3.7: Save per-question results
        # ==================================================================
        
        # Create individual plot for this question
        if out_cfg.get("make_individual_plots", True):
            out_png = indiv_dir / f"q{qi:04d}_iwae.png"
            plot_iwae_curve(ks, curve, out_png, title=f"IWAE Curve (Question {qi})")

        # Save raw curve data for this question
        if out_cfg.get("save_data_json", True):
            q_data_path = data_dir / f"q{qi:04d}_curve.json"
            with q_data_path.open("w") as f:
                json.dump({"qid": qi, "ks": ks, "iwae": curve, "num_samples": K}, f)

    # Close rollouts file
    if rollout_f is not None:
        rollout_f.close()

    # ========================================================================
    # STEP 4: Aggregate results across all questions
    # ========================================================================
    
    # Create ribbon plot showing mean/median IWAE curves with confidence bands
    if out_cfg.get("make_ribbon_plot", True):
        ribbon_png = agg_dir / "iwae_ribbon.png"
        plot_ribbon(ks, per_question_curves, ribbon_png, title=f"IWAE Ribbon Plot • {exp_name}")

    # Save aggregated data
    if out_cfg.get("save_data_json", True):
        agg_data = {
            "experiment_name": exp_name,
            "ks": ks,
            "per_question_curves": per_question_curves,
            "per_question_expectation": [
                prbo_expectation(np.array(ws, dtype=float)) for ws in per_question_logw
            ],
            "iwae_samples": K,
            "resamples": resamples,
            "num_questions": len(questions),
            "total_rollouts": n_total_rollouts,
        }
        with (data_dir / "aggregate_curves.json").open("w") as f:
            json.dump(agg_data, f)

    print(f"\n[done] ✓ Completed {n_total_rollouts} rollouts across {len(questions)} questions.")
    print(f"[done] ✓ Results saved under: {out_dir}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments with config file path
    """
    ap = argparse.ArgumentParser(
        description="Run Propensity Bound-IWAE evaluation with steered prompts and online LLM judging."
    )
    ap.add_argument(
        "-c", "--config",
        type=str,
        default=str(Path(__file__).parent / "simply.yaml"),
        help="Path to YAML configuration file (default: scripts/simply.yaml)"
    )
    return ap.parse_args()


def main() -> None:
    """
    Main entry point for the script.
    
    Workflow:
      1. Load configuration from YAML
      2. Display config to user
      3. Run the experiment
      4. Save all results and plots
    """
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    
    print("=" * 70)
    print("Propensity Bound-IWAE Evaluation - Configuration:")
    print("=" * 70)
    print(yaml.safe_dump(cfg, sort_keys=False))
    print("=" * 70)
    
    run_single_experiment(cfg)


if __name__ == "__main__":
    main()