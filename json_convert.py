import csv
import json
import os

def csv_to_json(csv_file_path, json_file_path):
    data = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

def drop_features(json_file_path, output_file_path, features_to_drop):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    for entry in data:
        for feature in features_to_drop:
            if feature in entry:
                del entry[feature]
    
    with open(output_file_path, mode='w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=4)

def remove_null_floor_area(json_file_path, output_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    cleaned_data = [entry for entry in data if entry['floorAreaSqM'] not in [None, "", "null"]]
    
    with open(output_file_path, mode='w', encoding='utf-8') as output_file:
        json.dump(cleaned_data, output_file, indent=4)

def replace_null_values(json_file_path, output_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    for entry in data:
        if entry['tenure'] in [None, "", "null"]:
            entry['tenure'] = "Unknown"
        if entry['propertyType'] in [None, "", "null"]:
            entry['propertyType'] = "Unknown"
        if entry['bathrooms'] in [None, "", "null"]:
            entry['bathrooms'] = "0"
        if entry['bedrooms'] in [None, "", "null"]:
            entry['bedrooms'] = "0"
        if entry['livingRooms'] in [None, "", "null"]:
            entry['livingRooms'] = "0"
    
    with open(output_file_path, mode='w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=4)

def remove_null_rent_estimate(json_file_path, output_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    cleaned_data = [entry for entry in data if entry['rentEstimate_lowerPrice'] not in [None, "", "null"]]
    
    with open(output_file_path, mode='w', encoding='utf-8') as output_file:
        json.dump(cleaned_data, output_file, indent=4)

def json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    with open(csv_file_path, mode='w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data[0].keys())  # Write headers
        for entry in data:
            csv_writer.writerow(entry.values())

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, 'kaggle_london_house_price_data.csv')
    json_file = os.path.join(current_dir, 'kaggle_london_house_price_data.json')
    final_json_file = os.path.join(current_dir, 'final_clean_data.json')
    final_csv_file = os.path.join(current_dir, 'final_clean_data.csv')
    
    csv_to_json(csv_file, json_file)
    drop_features(json_file, final_json_file, [
        'fullAddress', 'postcode', 'country', 'outcode', 'history_date', 
        'saleEstimate_ingestedAt', 'saleEstimate_valueChange.saleDate',
        'currentEnergyRating', 'history_percentageChange', 'history_numericChange'
    ])
    remove_null_floor_area(final_json_file, final_json_file)
    replace_null_values(final_json_file, final_json_file)
    remove_null_rent_estimate(final_json_file, final_json_file)
    json_to_csv(final_json_file, final_csv_file)
    
    # Print final data entries count
    with open(final_json_file, mode='r', encoding='utf-8') as json_file:
        final_data = json.load(json_file)
    print(f"Final number of data entries: {len(final_data)}")
