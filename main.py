import json
import os
from preprocess import apply_one_hot_encoding
import csv

def csv_to_json(csv_file_path, json_file_path):
    data = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

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
    final_csv_file = os.path.join(current_dir, 'final_clean_data.csv')
    encoded_csv_file = os.path.join(current_dir, 'encoded_final_clean_data.csv')
    
    print("Final Cleaned Data:")
    inspect_json_data(final_json_file)
    
    # Apply one-hot encoding to categorical features
    categorical_features = ['tenure', 'propertyType', 'saleEstimate_confidenceLevel']
    apply_one_hot_encoding(final_csv_file, encoded_csv_file, categorical_features)
    print(f"One-hot encoded data saved to {encoded_csv_file}")
    
    encoded_json_file = csv_to_json(encoded_csv_file, 'encoded_final_clean_data.json')
    