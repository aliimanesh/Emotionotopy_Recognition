#This code reads the corresponding files of Pupil diameters for all participants, preprocess them and combine them all to create a dataframe.
#To run this code, all Pupil diameter files of all participants should be gathered in a single directory.

#The preprocessing includes linear interpolation of missing values (if missing values are less than 800 datapoints) and normalization of data.
#After preprocessing, every 2000 datapoints (equal to 2 seconds) create a row of the dataframe.

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Directory path for Pupil Diameter data
directory_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0-Pupil/"

# List of participant numbers
participants = [
    "01", "02", "03", "04", "05", "06", "09", 
    "10", "14", "15", "16", "17", "18", "19", "20"
]

# File suffix for Pupil Diameter data
file_suffix = "_recording-eyegaze_physio.tsv"

# List to store individual DataFrames
dfs = []

# Function to process pupil diameter data
def process_pupil_diameter(data, max_zeros=800, pad=100):
    n = len(data)
    zero_indices = np.where(data == 0)[0]
    if len(zero_indices) == 0:
        return data

    i = 0
    while i < len(zero_indices):
        start = zero_indices[i]
        count = 1
        while i + count < len(zero_indices) and zero_indices[i + count] == zero_indices[i] + count:
            count += 1
        end = start + count

        if count < max_zeros:
            start_pad = max(0, start - pad)
            end_pad = min(n, end + pad)
            data[start_pad:end_pad] = 0
        
        i += count

    return data

# Function to perform linear interpolation
def linear_interpolate(data, max_zeros=800):
    n = len(data)
    zero_indices = np.where(data == 0)[0]
    if len(zero_indices) == 0:
        return data
    
    i = 0
    while i < len(zero_indices):
        start = zero_indices[i]
        count = 1
        while i + count < len(zero_indices) and zero_indices[i + count] == zero_indices[i] + count:
            count += 1
        end = start + count

        if count < max_zeros:
            if start == 0 or end == n:  # If zeros are at the start or end, skip interpolation
                i += count
                continue
            non_zero_indices = np.nonzero(data)[0]
            non_zero_values = data[non_zero_indices]
            interpolator = interp1d(non_zero_indices, non_zero_values, kind='linear', fill_value='extrapolate')
            data[start:end] = interpolator(np.arange(start, end))
        
        i += count

    return data

# Function to normalize data between 0 and 1
def normalize_data(data):
    max_val = np.max(data)
    normalized_data = data / max_val  # Normalize between 0 and 1
    return normalized_data

# Number of data points per row
num_datapoints = 2000

# Iterate over participants
for participant in participants:
    # List to store runs of the participant
    participant_dfs = []
    # Iterate over runs (from 1 to 8)
    for i in range(1, 9):
        # Construct file path
        file_path = os.path.join(directory_path, f"sub-{participant}_ses-movie_task-movie_run-{i}{file_suffix}")
        
        # Check if file exists and has at least 3 columns
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist or does not have enough columns. Filling with zeros.")
            # Create rows filled with zeros
            num_rows = 100  # Example number of rows, adjust according to your data structure
            zero_data = [[int(participant), i] + [0] * num_datapoints for _ in range(num_rows)]
            zero_df = pd.DataFrame(zero_data, columns=['Participant_Number', 'Run_Number'] + [f'Data_{k+1}' for k in range(num_datapoints)])
            participant_dfs.append(zero_df)
            continue
        
        # Read the .tsv file into a DataFrame
        df = pd.read_csv(file_path, sep='\t')
        
        # Check if DataFrame has at least 3 columns
        if len(df.columns) < 3:
            print(f"DataFrame for participant {participant}, run {i} does not have enough columns. Filling with zeros.")
            # Create rows filled with zeros
            num_rows = 100  # Example number of rows, adjust according to your data structure
            zero_data = [[int(participant), i] + [0] * num_datapoints for _ in range(num_rows)]
            zero_df = pd.DataFrame(zero_data, columns=['Participant_Number', 'Run_Number'] + [f'Data_{k+1}' for k in range(num_datapoints)])
            participant_dfs.append(zero_df)
            continue
        
        # Extract pupil diameter data (3rd column)
        pupil_diameter = df.iloc[:, 2].values
        
        # Process and interpolate the pupil diameter data
        processed_data = process_pupil_diameter(pupil_diameter.copy())
        interpolated_data = linear_interpolate(processed_data.copy())
        
        # Normalize the interpolated data
        normalized_data = normalize_data(interpolated_data)
        
        # Reshape the data into rows of 2000 subsequent data points
        num_rows = len(normalized_data) // num_datapoints
        reshaped_data = []
        
        for j in range(num_rows):
            start_index = j * num_datapoints
            end_index = start_index + num_datapoints
            
            # Extract pupil diameter data for the specified range
            pupil_data = list(normalized_data[start_index:end_index])  
            
            # Create row data with participant number, run number, and pupil data
            row_data = [int(participant), i] + pupil_data
            reshaped_data.append(row_data)
        
        # Create a new DataFrame with reshaped data
        column_names = ['Participant_Number', 'Run_Number'] + [f'Data_{k+1}' for k in range(num_datapoints)]
        reshaped_df = pd.DataFrame(reshaped_data, columns=column_names)
        
        # Append the DataFrame to the list of participant's DataFrames
        participant_dfs.append(reshaped_df)
    
    # Concatenate all runs of the participant into a single DataFrame
    if participant_dfs:
        participant_combined_df = pd.concat(participant_dfs, ignore_index=True)
        dfs.append(participant_combined_df)

# Concatenate all DataFrames of different participants into a single DataFrame
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Define the file path for saving the combined DataFrame
    output_combined_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0-Pupil/combined_pupil_data_all_participants_normal.csv"
    
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_combined_file_path, index=False)
    
    print("Combined DataFrame for all participants saved successfully.")
else:
    print("No valid data to combine.")