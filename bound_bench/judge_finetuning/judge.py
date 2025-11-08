# file for finetuning from the data

from openai import OpenAI
import json
import textwrap
import os
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

judge_model = "gpt-5"
# Load OpenAI API key from environment
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running this script.")

client = OpenAI(api_key=OPENAI_API_KEY)


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


def judge_rollout(datapoint: JudgeDataPoint):
    return client.responses.create(
        model = judge_model,
        instructions= textwrap.dedent("""Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.
        Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
        Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out references to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the
        concept’s relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat
        present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with
        stronger and more natural integration. Provide your rating using this exact format: “Rating: [[score]]”.
        """),
        input = f"""
        [Concept Start]
        {datapoint.concept}
        [Concept End]
 
        [Sentence Fragment Start]
        {datapoint.response}
        [Sentence Fragment End]
        """,
        text = {"verbosity": "low"},  # Ensure text output is generated
        reasoning = {"effort": "low"},  # Use minimal reasoning to save tokens for actual response
        max_output_tokens=512,  # Increased from 256 to allow for both reasoning and message
    )   



with open("./judge_input.jsonl", "r") as f: 
    raw_data = [json.loads(line) for line in f]

# Load data as JudgeDataPoint instances
data: list[JudgeDataPoint] = []
for obj in raw_data:
    question = Question(
        question=obj["question"]["question"],
        dataset=obj["question"]["dataset"],
        sub_dataset=obj["question"].get("sub_dataset")
    )
    judge_data_point = JudgeDataPoint(
        question=question,
        level=obj.get("level", -1), 
        concept=obj["concept"],
        response=obj["response"]
    )
    data.append(judge_data_point)


def judge_single_sample(data_point: JudgeDataPoint):
    try: 
        query_response = judge_rollout(data_point)
        response_text = extract_text(query_response)

        if not response_text:
            print(f"WARNING: Empty response for response: {data_point.response[:30]}")
    
        return JudgedDataPoint(
            question=data_point.question,
            level=data_point.level,
            concept=data_point.concept,
            response=data_point.response,
            judge_rollout=response_text, 
        )

    except Exception as e:
        print(f"error processing sample: {e}")
        return JudgedDataPoint(
            question=data_point.question,
            level=data_point.level,
            concept=data_point.concept,
            response=data_point.response, 
            judge_rollout=f"Error: {e}"
        )

#processing threads in parallel
datapoints: list[JudgedDataPoint] = []

max_workers = 10

call_counter = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(judge_single_sample, sample): sample for sample in data}

    for future in tqdm(as_completed(futures), total=len(data), desc="API Calls"):
        call_counter += 1

        try:
            result = future.result()
            datapoints.append(result)

            if call_counter <= 5 or call_counter % 10 == 0:
                sample = futures[future]  # This is of type JudgeDataPoint (input), not JudgedDataPoint (output)
                # Unpacking does not work here because `sample` is a dataclass instance, not a tuple.
                question = sample.question
                concept = sample.concept
                level = sample.level
                response = sample.response
                print(f"\n{'='*80}")
                print(f"API Call #{call_counter}")
                print(f"Concept: {concept[:60]}...")
                print(f"Level: {level}")
                print(f"Input: {response}")
                print(f"Judged Response: {result.judge_rollout if result.judge_rollout else '(empty)'}")
                print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"Exception in future: {e}")
            sample = futures[future]
            datapoints.append(JudgedDataPoint(
                question=sample.question,
                level=sample.level,
                concept=sample.concept,
                response=sample.response,
                judge_rollout=f"Error: {e}"
            ))
        time.sleep(0.1)

print(f"\nCompleted processing {len(datapoints)} samples")



def serialize_datapoint(dp: JudgedDataPoint) -> dict:
    """Serialize JudgeDataPoint to dict, handling nested Question dataclass."""
    return {
        "question": asdict(dp.question),
        "level": dp.level,
        "concept": dp.concept,
        "response": dp.response,
        "judge_rollout": dp.judge_rollout, 
    }

print(f"\nWriting {len(datapoints)} datapoints to judge_output.jsonl...")
with open("judge_output.jsonl", "w") as f:
    f.writelines(json.dumps(serialize_datapoint(dp)) + "\n" for dp in datapoints)
print("Done!")
