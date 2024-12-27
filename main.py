import json
import os
from tabulate import tabulate

def inspect_json_data(json_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    if not data:
        print("No data found in the JSON file.")
        return
    
    print(f"Total number of data entries: {len(data)}")
    
    features = data[0].keys()
    null_counts = {feature: 0 for feature in features}
    
    for entry in data:
        for feature in features:
            if entry[feature] in [None, "", "null"]:
                null_counts[feature] += 1
    
    table = [[feature, count] for feature, count in null_counts.items()]
    print(tabulate(table, headers=["Feature", "Null Values"], tablefmt="grid"))

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_json_file = os.path.join(current_dir, 'final_clean_data.json')
    
    print("Final Cleaned Data:")
    inspect_json_data(final_json_file)
