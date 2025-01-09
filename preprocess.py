import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def apply_one_hot_encoding(csv_file_path, output_csv_file_path, categorical_features):
    # Load the dataset
    data = pd.read_csv(csv_file_path)
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_features])
    
    # Convert encoded features to DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    # Drop original categorical features and concatenate encoded features
    data = data.drop(categorical_features, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    
    # Save the transformed dataset
    data.to_csv(output_csv_file_path, index=False)