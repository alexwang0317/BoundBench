"""
A script to demonstrate rollouts of a pyvene base model with a steering intervention.

This script loads a pre-trained model, applies a "happy" vector intervention
at a specific layer, and generates text from a series of prompts with
varying intervention strengths (magnitudes).

Modified to run on multiple models and perform 100 rollouts per model with timing.
"""
import torch
import pyvene as pv
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import statistics
from typing import List, Dict, Tuple

class HappyIntervention(pv.ConstantSourceIntervention):
    """
    An intervention that adds a 'happy' vector to the model's representation.
    """
    def __init__(self, happy_vector, **kwargs):
        # keep_last_dim=True is essential for getting token-level representations
        super().__init__(**kwargs, keep_last_dim=True)
        self.happy_vector = happy_vector
        self.called_counter = 0

    def forward(self, base, source=None, subspaces=None, **kwargs):
        if subspaces is not None and subspaces.get("logging", False):
            print(f"(called {self.called_counter} times) incoming reprs shape: {base.shape}")
        
        # The actual intervention: add the happy vector scaled by magnitude
        if subspaces is not None and "mag" in subspaces:
            base = base + subspaces["mag"] * self.happy_vector
        
        self.called_counter += 1
        return base

    def reset_counter(self):
        """Resets the internal counter."""
        self.called_counter = 0


def run_model_rollouts(model_name: str, num_rollouts: int = 100) -> Dict[str, float]:
    """
    Runs rollouts for a specific model and returns timing statistics.

    Args:
        model_name: HuggingFace model name to load
        num_rollouts: Number of rollouts to perform

    Returns:
        Dictionary containing timing statistics
    """
    print(f"\n{'='*80}")
    print(f"STARTING ROLLOUTS FOR MODEL: {model_name}")
    print(f"TARGET ROLLOUTS: {num_rollouts}")
    print(f"{'='*80}")

    # --- 1. Model and Tokenizer Setup ---
    print("Loading model and tokenizer...")
    start_time = time.time()
    # Use bfloat16 for memory efficiency if available
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model_load_time = time.time() - start_time
    print(f"Model loaded on {device} in {model_load_time:.2f}s")

    # --- 2. Intervention Vector Setup ---
    print("Extracting 'happy' vector...")
    with torch.no_grad():
        happy_id = tokenizer("happy")['input_ids'][-1]
        happy_vector = model.model.embed_tokens.weight[happy_id].clone().detach()

    # --- 3. Pyvene Intervenable Model Setup ---
    # We need to pass the happy_vector to our custom intervention.
    happy_intervention = HappyIntervention(
        happy_vector,
        embed_dim=model.config.hidden_size,
        low_rank_dimension=1
    )

    # Configure the intervention point
    pv_config = pv.IntervenableConfig(
        representations=[
            {
                "layer": 20,
                "component": "model.layers[20].output",
                "intervention": happy_intervention,
            }
        ],
        intervention_types=pv.ConstantSourceIntervention,
    )

    pv_model = pv.IntervenableModel(pv_config, model)
    pv_model.set_device(device)

    # --- 4. Expanded Rollout Configuration for 100 rollouts ---
    # Create more prompts and magnitudes to reach 100 rollouts
    base_prompts = [
        "Write a story for me about a dragon.",
        "What is the meaning of life?",
        "Tell me a joke about computers.",
        "Explain quantum physics in simple terms.",
        "Describe your favorite vacation spot.",
        "How do you make chocolate chip cookies?",
        "What's the best way to learn a new language?",
        "Write a haiku about coding.",
        "Explain machine learning to a child.",
        "What would you do with a million dollars?",
        "Describe the perfect day.",
        "How does photosynthesis work?",
        "Write a short poem about friendship.",
        "What's the most interesting historical fact you know?",
        "Explain how computers work.",
        "Describe a futuristic city.",
        "How do you stay motivated?",
        "Write about your favorite hobby.",
        "What's the meaning of happiness?",
        "Explain the stock market simply."
    ]

    # Generate more magnitudes including negative values for variety
    base_magnitudes = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    # To reach 100 rollouts, we'll use combinations of prompts and magnitudes
    # 20 prompts * 5 magnitudes = 100 rollouts
    prompts = base_prompts[:20]  # Take first 20 prompts
    magnitudes = base_magnitudes  # 8 magnitudes

    generation_config = {
        "max_new_tokens": 60,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    }

    # --- 5. Perform Rollouts with Timing ---
    rollout_times = []
    total_rollouts = len(prompts) * len(magnitudes)

    print(f"\nStarting {total_rollouts} rollouts...")
    print(f"Prompts: {len(prompts)}, Magnitudes: {len(magnitudes)}")

    for i, prompt_text in enumerate(prompts):
        for j, mag in enumerate(magnitudes):
            rollout_start = time.time()

            # Format prompt for the model
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

            # Reset the intervention's call counter for a clean slate
            happy_intervention.reset_counter()

            _, generations = pv_model.generate(
                inputs,
                unit_locations=None,  # Apply intervention at every forward pass (token)
                intervene_on_prompt=True,
                subspaces=[{"mag": mag, "logging": False}], # Pass magnitude to intervention
                **generation_config,
            )

            # Decode and print the generated text
            generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)
            # Remove the prompt from the output for clarity
            output_only = generated_text[len(chat_prompt.replace(tokenizer.bos_token, '')):]
            output_clean = output_only.strip()

            rollout_time = time.time() - rollout_start
            rollout_times.append(rollout_time)

            # Progress tracking
            current_rollout = i * len(magnitudes) + j + 1
            if current_rollout % 10 == 0:  # Print progress every 10 rollouts
                recent_times = rollout_times[-10:] if len(rollout_times) >= 10 else rollout_times
                avg_time = statistics.mean(recent_times)
                print(f"Completed rollout {current_rollout}/{total_rollouts} in {rollout_time:.2f}s (recent avg: {avg_time:.2f}s)")

    # --- 6. Calculate and Return Statistics ---
    total_time = sum(rollout_times)
    avg_time = statistics.mean(rollout_times)
    median_time = statistics.median(rollout_times)
    min_time = min(rollout_times)
    max_time = max(rollout_times)
    std_dev = statistics.stdev(rollout_times) if len(rollout_times) > 1 else 0

    stats = {
        "model_name": model_name,
        "total_rollouts": len(rollout_times),
        "model_load_time": model_load_time,
        "total_time": total_time,
        "avg_time_per_rollout": avg_time,
        "median_time_per_rollout": median_time,
        "min_time_per_rollout": min_time,
        "max_time_per_rollout": max_time,
        "std_dev_time": std_dev,
        "throughput_rollouts_per_sec": len(rollout_times) / total_time if total_time > 0 else 0
    }

    return stats


def run_rollouts():
    """
    Main function to run rollouts on multiple models and print timing statistics.
    """
    models_to_test = [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it"
    ]

    num_rollouts_per_model = 100

    all_stats = []

    for model_name in models_to_test:
        model_stats = run_model_rollouts(model_name, num_rollouts_per_model)
        all_stats.append(model_stats)

    # --- Print Summary Statistics ---
    print(f"\n{'='*100}")
    print("FINAL SUMMARY STATISTICS")
    print(f"{'='*100}")

    for stats in all_stats:
        print(f"\nModel: {stats['model_name']}")
        print(f"  Total rollouts: {stats['total_rollouts']}")
        print(f"  Model load time: {stats['model_load_time']:.2f}s")
        print(f"  Total generation time: {stats['total_time']:.2f}s")
        print(f"  Average time per rollout: {stats['avg_time_per_rollout']:.3f}s")
        print(f"  Median time per rollout: {stats['median_time_per_rollout']:.3f}s")
        print(f"  Min/Max time per rollout: {stats['min_time_per_rollout']:.3f}s / {stats['max_time_per_rollout']:.3f}s")
        print(f"  Std dev time: {stats['std_dev_time']:.3f}s")
        print(f"  Throughput: {stats['throughput_rollouts_per_sec']:.2f} rollouts/sec")

    # Compare models
    if len(all_stats) > 1:
        print(f"\n{'='*100}")
        print("MODEL COMPARISON")
        print(f"{'='*100}")

        for i, stats in enumerate(all_stats):
            speedup = "N/A"
            if i > 0:
                prev_throughput = all_stats[i-1]['throughput_rollouts_per_sec']
                if prev_throughput > 0:
                    speedup = f"{stats['throughput_rollouts_per_sec'] / prev_throughput:.2f}x"

            print(f"  {stats['model_name']}:")
            print(f"    Throughput: {stats['throughput_rollouts_per_sec']:.2f} rollouts/sec")
            if speedup != "N/A":
                print(f"    Speedup vs {all_stats[i-1]['model_name']}: {speedup}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    run_rollouts()
