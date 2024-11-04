#This code shift the emotion labels to remove the bias in the distribution of datapoints. 

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the file path
file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Emotionotopy/emotion_dimension.csv"

# Read the CSV file with no headers
df = pd.read_csv(file_path, header=None)

# Convert all data to numeric, setting errors='coerce' will replace non-numeric values with NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalize each column separately
normalized_data = scaler.fit_transform(df.fillna(0))  # Fill NaN values with 0 before normalization

# Create a new DataFrame with the normalized data
normalized_df = pd.DataFrame(normalized_data)

# Define the file path for saving the normalized DataFrame
output_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Emotionotopy/emotion_dimension_normalized.csv"

# Save the normalized DataFrame to a CSV file
normalized_df.to_csv(output_file_path, index=False, header=False)

print("Normalized DataFrame saved successfully.")
