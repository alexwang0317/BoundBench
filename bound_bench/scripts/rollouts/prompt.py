from __future__ import annotations

"""
Generate prompt-steered rollouts that capture raw token log probabilities.

This mirrors the sampling path from eval_inf.py but omits judging, weight
computation, and plotting so downstream analysis can consume the saved JSON
rollouts directly.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from bound_bench.models.hf_causal import GenerateResult, HFCausalLM, HFCausalLMvLLM
from dotenv import load_dotenv

try:
    from ..utility import (  # type: ignore
        assemble_base_prompt,
        assemble_generation_prompt,
        build_steered_prompt,
        convert_to_messages,
        ensure_dir,
        load_questions,
        load_yaml,
        parse_args,
        save_yaml,
        set_global_seed,
    )
except (ImportError, ValueError):
    # Fallback if running as script and relative import fails
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utility import (  # type: ignore
        assemble_base_prompt,
        assemble_generation_prompt,
        build_steered_prompt,
        convert_to_messages,
        ensure_dir,
        load_questions,
        load_yaml,
        parse_args,
        save_yaml,
        set_global_seed,
    )


def single_experiment(cfg: dict[str, Any]) -> None:
    """
    Run a rollouts-only experiment that records base/steered token log probabilities.
    """

    exp_name: str = cfg.get("experiment_name") or "unnamed_experiment"

    steered_cfg = cfg.get("steered_model", {})
    steer_mode = steered_cfg.get("mode", "prompt_prepend")
    assert steer_mode == "prompt_prepend", "Only prompt-prepend steering is supported right now."
    base_model_name: str = steered_cfg["base_model_name"]
    steer_template: str = steered_cfg["prompt_template"]


    data_cfg = cfg.get("data", {})
    concept: str = data_cfg["concept"]  # taken from concept dict 
    questions_path = Path(data_cfg["questions_jsonl"]).expanduser().resolve()

    # note here that we do not get the judge at all

    sampling_cfg = cfg.get("sampling", {})
    K: int = int(sampling_cfg.get("iwae_samples", 16))       # number of rollouts per question
    max_new_tokens = int(sampling_cfg.get("max_new_tokens", 512))
    temperature = float(sampling_cfg.get("temperature", 0.7))
    top_p = float(sampling_cfg.get("top_p", 0.95))
    enable_thinking = bool(sampling_cfg.get("enable_thinking", False))
    seed = sampling_cfg.get("seed", None)

    # --- Outputs --- 

    out_cfg = cfg.get("output", {})
    out_dir = ensure_dir(Path(out_cfg.get("save_dir", f"results/{exp_name}")))
    data_dir = ensure_dir(out_dir / "data")
    rollouts_path = out_dir / "rollouts.jsonl"

    # Save the effective configuration for reproducibility
    effective_cfg = dict(cfg)
    effective_cfg["_resolved"] = {
        "iwae_samples": K,
        "questions_path": str(questions_path),
        "output_dir": str(out_dir),
    }

    save_yaml(effective_cfg, out_dir / "resolved_config.yaml")

    # --- Initialize --- 

    set_global_seed(seed)

    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No questions loaded from {questions_path}")

    print(f"[info] Loading base model: {base_model_name}")
    
    if torch.cuda.is_available():
        try:
            print("[info] Attempting to load model with vLLM...")
            base_model = HFCausalLMvLLM(base_model_name)
            print("[info] Successfully loaded model with vLLM.")
        except Exception as e:
            print(f"[warning] vLLM load failed ({e}). Falling back to standard HFCausalLM.")
            base_model = HFCausalLM(base_model_name)
    else:
        print("[info] CUDA not available. Loading model with standard HFCausalLM.")
        base_model = HFCausalLM(base_model_name)

    steered_prefix = build_steered_prompt(steer_template, concept)

    n_total_rollouts = 0

    if out_cfg.get("save_rollouts_jsonl", True):
        rollout_f = rollouts_path.open("w")
    else:
        rollout_f = None


    # --- Main loop --- 

    for qi, q in enumerate(questions):
        print(f"\n[info] Processing question {qi+1}/{len(questions)}")

        base_prompt = assemble_base_prompt(q)                         # p(y|x) - base distribution
        steered_prompt = assemble_generation_prompt(steered_prefix, q) # q(y|x) - proposal distribution
        question_rollouts: List[Dict[str, Any]] = []

        # --- Inner loop: Draw K samples for this question --- 
        for si in range(K):
            steered_messages = convert_to_messages(steered_prompt)
            base_messages = convert_to_messages(base_prompt)

            res: GenerateResult = base_model.sample(
                steered_messages,
                n=1,                          # Generate 1 completion
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                enable_thinking=enable_thinking,
            )

            completion = res.texts[0] if res.texts else ""

            q_logp, q_token_logprobs = base_model.logprob(
                steered_messages,
                completion,
                enable_thinking=enable_thinking
            )

            p_logp, p_token_logprobs = base_model.logprob(
                base_messages,
                completion,
                enable_thinking=enable_thinking
            )

            record = {
                "qid": qi,
                "sample_id": si,
                "question": q,
                "steered_prompt": steered_prompt,
                "base_prompt": base_prompt,
                "completion": completion,
                "p_logp": p_logp,
                "q_logp": q_logp,
                "p_token_logprobs": p_token_logprobs,
                "q_token_logprobs": q_token_logprobs,
            }
            question_rollouts.append(record)

            if rollout_f is not None:
                rollout_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_total_rollouts += 1

        if out_cfg.get("save_data_json", True):
            q_data_path = data_dir / f"q{qi:04d}_rollouts.json"
            with q_data_path.open("w") as f:
                json.dump(
                    {
                        "qid": qi,
                        "question": q,
                        "steered_prompt": steered_prompt,
                        "base_prompt": base_prompt,
                        "rollouts": question_rollouts,
                    },
                    f,
                    ensure_ascii=False,
                )

    if rollout_f is not None:
        rollout_f.close()
    # --- Post-processing --- 

    print(f"\n[done] ✓ Completed {n_total_rollouts} rollouts across {len(questions)} questions.")
    print(f"[done] ✓ Results saved under: {out_dir}")


def main() -> None:
    """
    Entry point mirroring eval_inf.py but saving only rollout log-prob data.
    """
    load_dotenv()
    args = parse_args(
        description="Generate rollouts and token log probabilities without judging or plotting."
    )
    cfg_path = Path(args.config).expanduser().resolve()
    cfg_obj = load_yaml(cfg_path)

    cfg_list = cfg_obj if isinstance(cfg_obj, list) else [cfg_obj]
    for cfg in cfg_list:
        exp_name = cfg.get("experiment_name", "unnamed_experiment")
        print(f"\n[info] Starting rollouts for config: {exp_name}")
        single_experiment(cfg)
        print(f"[done] ✓ Completed rollouts for: {exp_name}")

if __name__ == "__main__":
    main()