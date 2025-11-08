# file used to generate the (input, outpute) pairs for fine-tuning. 
import textwrap
from openai import OpenAI
from dataclasses import dataclass, asdict
import time
from pprint import pprint
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load OpenAI API key from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running this script.")

client = OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-5-mini"

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

# list of questions
questions: list[Question] = []

concepts: list[str] = []

final_judge_input: list[JudgeDataPoint] = []

datapoints: list[JudgeDataPoint] = []

def extract_text(resp) -> str:
    """
    Robust extractor: returns assistant text if present; otherwise falls back to 
    any reasoning summary so you don't lose data.
    """
    # 1) Fast path: helper
    if getattr(resp, "output_text", None):
        return resp.output_text.strip()

    # 2) Manual walk of output items
    pieces = []
    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)

        # assistant message text
        if t == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                    pieces.append(c.text)

        # optional: include reasoning text if no message shows up
        elif t == "reasoning":
            # Some SDK builds expose item.content (list of reasoning chunks) OR item.reasoning
            # Try both; keep it minimal so it doesn't become your training target accidentally.
            rtxt = getattr(item, "reasoning", None)
            if isinstance(rtxt, str) and rtxt.strip():
                pieces.append(f"[reasoning] {rtxt.strip()}")
            else:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "reasoning") and getattr(c, "text", None):
                        pieces.append(f"[reasoning] {c.text}")

    return "\n".join(pieces).strip()


def concept_rollout(query: str):
    return client.responses.create(
        model = model,
        instructions= textwrap.dedent("""You are an expert writer, you will be given a question which you will answer, and you will be given a concept which 
        you will incorporate into the answer to a varying degree or level with each of the levels defined as such: 
    
        Level 0 = concept not mentioned at all, just answer the question
        Level 1 = concept barely hinted at or extremely tangential
        Level 2 = concept present but minimal or forced
        Level 3 = concept moderately incorporated but somewhat awkward
        Level 4 = concept clearly present and reasonably well-integrated
        Level 5 = concept fully and naturally incorporated throughout

        From now, I will give you a question, a concept, and a concept level. You will answer the question incorporating the concept based on the
        levels that I provided above.
        """),
        input = query,
        text = {"verbosity": "low"},  # Ensure text output is generated
        reasoning = {"effort": "low"},  # Use minimal reasoning to save tokens for actual response
        max_output_tokens=1024,  # Increased from 256 to allow for both reasoning and message
    )   



# populating the questions dataset
with open("../../bound_bench/data/data_300.json", "r") as f: 
    dataset = json.load(f)



for dataset_name, queries in dataset.items():
    for question_id, question_data in queries.items(): 
        if dataset_name == "AlpacaEval": 
            questions.append(Question(question=question_data["instruction"], 
                                        dataset="AlpacaEval", 
                                        sub_dataset=question_data["dataset"]))
        elif dataset_name == "GSM8K":
            questions.append(Question(question=question_data["question"],
                                        dataset="GSM8K",
                                        sub_dataset=None,
                                    ))
        elif dataset_name == "MBPP":
            questions.append(Question(question=question_data["text"],
                                        dataset="MBPP",
                                        sub_dataset=None, 
                                        ))
        else:
            pprint(f"Could not find dataset: {dataset_name}")



with open("../../bound_bench/data/concepts_500.json", "r") as f:
    concept_dict = json.load(f)
    # getting all concepts present - extract string values from the dict
    concepts.extend(concept_dict.values())



# Generate samples using random sampling strategy
visited = set()
samples_to_generate = []

print(f"Generating {5000} unique random samples...")
print(f"Available: {len(questions)} questions, {len(concepts)} concepts, 6 levels (0-5)")

while len(visited) < 5000:
    # Random sample from each
    concept = random.choice(concepts)
    level = random.choice([0, 1, 2, 3, 4, 5])
    question = random.choice(questions)
    
    # Create unique key
    sample_key = (question.question, concept, level)
    
    if sample_key in visited:
        continue
    
    visited.add(sample_key)
    samples_to_generate.append((question, concept, level))
    
print(f"Generated {len(samples_to_generate)} unique samples")


def process_single_sample(sample_data):
    """Process a single (question, concept, level) sample."""
    question, concept, level = sample_data
    
    formatted_query = textwrap.dedent(f""" 
    Given the following information: 
        Question: {question.question}
        Concept: {concept}
        Concept Level: {level}  
    Write me a answer to the question that incorporates that concept at that level. 
    """)
    
    try:
        response = concept_rollout(query=formatted_query)
        response_text = extract_text(response)
        
        # Debug aid if it's still empty
        if not response_text:
            print(f"WARNING: Empty response for question '{question.question[:30]}...', concept '{concept[:30]}...', level {level}")
        
        return JudgeDataPoint(
            question=question,
            level=level,
            concept=concept,
            response=response_text
        )
    except Exception as e:
        print(f"Error processing sample: {e}")
        return JudgeDataPoint(
            question=question,
            level=level,
            concept=concept,
            response=f"Error: {e}"
        )


# Process samples in parallel batches
print(f"\nProcessing {len(samples_to_generate)} samples with parallel API calls...")
max_workers = 10  # Number of parallel threads

call_counter = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    futures = {executor.submit(process_single_sample, sample): sample for sample in samples_to_generate}
    
    # Process completed tasks with progress bar
    for future in tqdm(as_completed(futures), total=len(samples_to_generate), desc="API calls"):
        call_counter += 1
        
        try:
            result = future.result()
            datapoints.append(result)
            
            # Print details for first 5 calls and every 10th call after
            if call_counter <= 5 or call_counter % 10 == 0:
                sample = futures[future]
                question, concept, level = sample
                print(f"\n{'='*80}")
                print(f"API Call #{call_counter}")
                print(f"Question: {question.question[:60]}...")
                print(f"Concept: {concept[:60]}...")
                print(f"Level: {level}")
                print(f"Response: {result.response if result.response else '(empty)'}...")
                print(f"{'='*80}\n")
                
        except Exception as e:
            print(f"Exception in future: {e}")
            sample = futures[future]
            question, concept, level = sample
            datapoints.append(JudgeDataPoint(
                question=question,
                level=level,
                concept=concept,
                response=f"Error: {e}"
            ))
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)

print(f"\nCompleted processing {len(datapoints)} samples")
            


def serialize_datapoint(dp: JudgeDataPoint) -> dict:
    """Serialize JudgeDataPoint to dict, handling nested Question dataclass."""
    return {
        "question": asdict(dp.question),
        "level": dp.level,
        "concept": dp.concept,
        "response": dp.response
    }

print(f"\nWriting {len(datapoints)} datapoints to judge_input.jsonl...")
with open("judge_input.jsonl", "w") as f:
    f.writelines(json.dumps(serialize_datapoint(dp)) + "\n" for dp in datapoints)
print("Done!")
