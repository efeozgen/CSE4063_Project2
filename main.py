import json
import os

def inspect_json_data(json_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    if not data:
        print("No data found in the JSON file.")
        return
    
    print(f"Total number of data entries: {len(data)}")

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_json_file = os.path.join(current_dir, 'final_clean_data.json')
    
    print("Final Cleaned Data:")
    inspect_json_data(final_json_file)
