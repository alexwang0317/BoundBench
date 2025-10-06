import json

# Read the tri file
with open('judge_evals/data/questions_10_tri.json', 'r') as f:
    lines = f.readlines()

result = []
current_question_num = None
current_question_title = None
current_concept = None
current_level = None

i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Skip empty lines
    if not line:
        i += 1
        continue
    
    # Check for QUESTION line
    if line.startswith("QUESTION "):
        parts = line.split(":", 1)
        question_part = parts[0].replace("QUESTION ", "").strip()
        current_question_num = int(question_part)
        current_question_title = parts[1].strip() if len(parts) > 1 else ""
        i += 1
        continue
    
    # Check for Concept line
    if line.startswith("Concept: "):
        current_concept = line.replace("Concept: ", "").strip()
        i += 1
        continue
    
    # Check for Level line
    if line.startswith("Level "):
        parts = line.split(":", 1)
        level_part = parts[0].replace("Level ", "").strip()
        current_level = int(level_part)
        response_text = parts[1].strip() if len(parts) > 1 else ""
        
        # Create the object
        obj = {
            "question_num": current_question_num,
            "Concept": current_concept,
            "Response": response_text,
            "true_label": current_level
        }
        result.append(obj)
        i += 1
        continue
    
    i += 1

# Sort by question_num, then by concept, then by true_label for better organization
def sort_key(x):
    concept_order = {
        "Golden Gate Bridge": 0,
        "Stacks Data Structure": 1,
        "Circular Shapes and Rounding": 2,
        "Cooking Actions": 3
    }
    return (x["question_num"], concept_order.get(x["Concept"], 999), x["true_label"])

result.sort(key=sort_key)

# Write to output file
with open('judge_evals/data/questions_120_parsed.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Parsed {len(result)} objects successfully!")
if result:
    print(f"Questions: {len(set(obj['question_num'] for obj in result))}")
    print(f"Total objects: {len(result)}")
else:
    print("No objects parsed. Please check the input file format.")

