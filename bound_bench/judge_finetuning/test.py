
# script for fine-tuning model
import json
import re
from peft import PeftModel
from vllm import LLM, SamplingParams
from dataclasses import dataclass, asdict

base_model =  "Qwen/Qwen3-8B"   
adapter_path = "./"

@dataclass
class Question:
    question: str 
    dataset: str 
    sub_dataset: str | None = None

@dataclass
class JudgeDataPoint: 
    question: Question
    level: int
    concept: str
    response: str

@dataclass
class JudgedDataPoint(JudgeDataPoint):
    judge_rollout: str  

@dataclass
class JudgeComparison(JudgedDataPoint):
    finetune_rollout: str = ""
    judge_rating: int = -1
    finetune_rating: int = -1

sampling_params = SamplingParams(
    temperature=0,
    top_p=0.95,
    max_tokens=512,
    n=1
)

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

# Create input-output pairs and reconstruct full datapoints
training_data = []
full_datapoints = []
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
    
    # Reconstruct the full datapoint
    question = Question(
        question=datapoint["question"]["question"],
        dataset=datapoint["question"]["dataset"],
        sub_dataset=datapoint["question"].get("sub_dataset")
    )
    full_datapoints.append(JudgedDataPoint(
        question=question,
        level=datapoint.get("level", -1),
        concept=datapoint["concept"],
        response=datapoint["response"],
        judge_rollout=output_text
    ))

# Split the training_data into train and test subsets
train_data = training_data[:4500]
test_data = training_data[4500:]
test_datapoints = full_datapoints[4500:]

# now load an adapter
print("Loading model and adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained("merged_model")

print("Loading finetuned judge with vLLM...")
finetuned_judge = LLM(model="merged_model")

print(f"Generating samples for {len(test_data)} test examples...")

finetuned_outputs = finetuned_judge.generate(
    [item["input"] for item in test_data], 
    sampling_params=sampling_params,
)

def extract_rating(text: str) -> int:
    """Extract rating from text in format 'Rating: [[#]]'. Returns -1 if not found."""
    match = re.search(r'Rating:\s*\[\[(\d+)\]\]', text)
    if match:
        return int(match.group(1))
    return -1

def serialize_comparison(comp: JudgeComparison) -> dict:
    """Serialize JudgeComparison to dict, handling nested Question dataclass."""
    return {
        "question": asdict(comp.question),
        "level": comp.level,
        "concept": comp.concept,
        "response": comp.response,
        "judge_rollout": comp.judge_rollout,
        "finetune_rollout": comp.finetune_rollout,
        "judge_rating": comp.judge_rating,
        "finetune_rating": comp.finetune_rating,
    }

# Process outputs and create JudgeComparison objects
print("\nProcessing outputs and extracting ratings...")
comparisons = []
for i, (output, datapoint) in enumerate(zip(finetuned_outputs, test_datapoints)):
    finetune_text = output.outputs[0].text.strip()
    
    # Extract ratings
    judge_rating = extract_rating(datapoint.judge_rollout)
    finetune_rating = extract_rating(finetune_text)
    
    comparison = JudgeComparison(
        question=datapoint.question,
        level=datapoint.level,
        concept=datapoint.concept,
        response=datapoint.response,
        judge_rollout=datapoint.judge_rollout,
        finetune_rollout=finetune_text,
        judge_rating=judge_rating,
        finetune_rating=finetune_rating
    )
    comparisons.append(comparison)
    
    # Print first few examples
    if i < 5:
        print(f"\n{'='*80}")
        print(f"Example {i+1}/{len(test_datapoints)}")
        print(f"Concept: {datapoint.concept[:60]}...")
        print(f"Judge Rating: {judge_rating}")
        print(f"Finetune Rating: {finetune_rating}")
        print(f"Judge: {datapoint.judge_rollout[:100]}...")
        print(f"Finetune: {finetune_text[:100]}...")
        print(f"{'='*80}\n")

# Save to jsonl
print(f"\nWriting {len(comparisons)} comparisons to judge_comparison.jsonl...")
with open("judge_comparison.jsonl", "w") as f:
    for comp in comparisons:
        f.write(json.dumps(serialize_comparison(comp)) + "\n")

print("Done! Results saved to judge_comparison.jsonl")
print(f"Total comparisons: {len(comparisons)}")

# Print some statistics
judge_ratings = [c.judge_rating for c in comparisons if c.judge_rating != -1]
finetune_ratings = [c.finetune_rating for c in comparisons if c.finetune_rating != -1]
print(f"\nStatistics:")
print(f"Judge ratings parsed: {len(judge_ratings)}/{len(comparisons)}")
print(f"Finetune ratings parsed: {len(finetune_ratings)}/{len(comparisons)}")
if judge_ratings:
    print(f"Judge rating distribution: {dict((r, judge_ratings.count(r)) for r in set(judge_ratings))}")
if finetune_ratings:
    print(f"Finetune rating distribution: {dict((r, finetune_ratings.count(r)) for r in set(finetune_ratings))}")