import json

# Read both files
with open('/Users/alexwang/BoundBench/offline_judging/data/questions_10_tri.json', 'r') as f:
    tri_data = json.load(f)

with open('/Users/alexwang/BoundBench/offline_judging/data/questions_10_hex.json', 'r') as f:
    hex_data = json.load(f)

# Extract all true_label=0 examples from tri
zero_labels = [item for item in tri_data if item['true_label'] == 0]

print(f"Found {len(zero_labels)} examples with true_label=0 in tri file")

# Create a dictionary to organize hex_data by (question_num, Concept)
hex_dict = {}
for item in hex_data:
    key = (item['question_num'], item['Concept'])
    if key not in hex_dict:
        hex_dict[key] = []
    hex_dict[key].append(item)

# Insert zero_label examples at the beginning of each group
result = []
processed_keys = set()

for item in hex_data:
    key = (item['question_num'], item['Concept'])
    
    # If we haven't processed this key yet, insert the zero_label example first
    if key not in processed_keys:
        # Find the corresponding zero_label example
        zero_example = next((z for z in zero_labels if z['question_num'] == key[0] and z['Concept'] == key[1]), None)
        
        if zero_example:
            result.append(zero_example)
            print(f"Added true_label=0 for question {key[0]}, concept '{key[1]}'")
        
        processed_keys.add(key)
    
    # Add the current item
    result.append(item)

print(f"\nOriginal hex file had {len(hex_data)} examples")
print(f"New file will have {len(result)} examples")

# Verify the structure
print("\nVerifying structure...")
verification = {}
for item in result:
    key = (item['question_num'], item['Concept'])
    if key not in verification:
        verification[key] = []
    verification[key].append(item['true_label'])

for key, labels in sorted(verification.items()):
    print(f"Q{key[0]}, {key[1]}: labels = {labels}")

# Write the result
with open('/Users/alexwang/BoundBench/offline_judging/data/questions_10_hex.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Successfully merged! File saved to questions_10_hex.json")

