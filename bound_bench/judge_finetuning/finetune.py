
# script for fine-tuning model
# pip install -U transformers peft trl datasets accelerate

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

# Load all the data
with open("./judge_output.jsonl", "r") as f: 
    raw_data = [json.loads(line) for line in f]

# Create the prompt template
JUDGE_PROMPT = """Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.

Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out references to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{response}
[Sentence Fragment End]"""

# Create input-output pairs
training_data = []
for datapoint in raw_data:
    input_text = JUDGE_PROMPT.format(
        concept=datapoint["concept"],
        response=datapoint["response"]
    )
    output_text = datapoint["judge_rollout"]
    
    training_data.append({
        "input": input_text,
        "output": output_text
    })

# Split the training_data into train and test subsets
train_data = training_data[:4500]
test_data = training_data[4500:]

print(f"Created {len(training_data)} total examples")
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Model configuration
MODEL_ID = "Qwen/Qwen3-8B"   

# 1) Create dataset from training pairs
ds_train = Dataset.from_list(train_data)
ds_test = Dataset.from_list(test_data)

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.pad_token = tok.eos_token

def to_text(ex):
    chat = [
        {"role": "user", "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    # Single training string with Qwen's chat formatting
    return {"text": tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)}

ds_train = ds_train.map(to_text)
ds_test = ds_test.map(to_text)

# 2) Load model in BF16 (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# 3) LoRA config (attention + MLP)
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# 4) Train — loss on assistant tokens only
args = SFTConfig(
    output_dir="qwen-judge-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
)

# Format function to extract text field
def formatting_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    peft_config=lora,
    args=args,
    formatting_func=formatting_func,
)

trainer.train()

# 5) Save adapter
trainer.model.save_pretrained("qwen-judge-lora-adapter")
tok.save_pretrained("qwen-judge-lora-adapter")

print("Training complete! Adapter saved to qwen-judge-lora-adapter/")

i