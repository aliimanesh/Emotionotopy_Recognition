#This code reads the corresponding files of PPG signal for all participants, preprocess them and combine them all to create a dataframe.
#To run this code, all files of all participants should be gathered in a single directory.

#The preprocessing includes downsampling, applying Savitzky-Golay filter, and normalization of data.
#After preprocessing, every 200 datapoints (equal to 2 seconds) create a row of the dataframe.

import os
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

# Directory path
directory_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0/"

# List of participant numbers
participants = ["01", "02", "03", "04", "05", "06", "09", "10", "14", "15", "16", "17", "18", "19", "20"]

# File suffix
file_suffix = "_recording-cardresp_physio.tsv"

# List to store individual DataFrames
dfs = []

# Iterate over participants
for participant in participants:
    # List to store runs of the participant
    participant_dfs = []
    # Iterate over runs (from 1 to 8)
    for i in range(1, 9):
        # Construct file path
        file_path = os.path.join(directory_path, f"sub-{participant}_ses-movie_task-movie_run-{i}{file_suffix}")
        
        try:
            # Read the .tsv file into a DataFrame
            df = pd.read_csv(file_path, sep='\t')

            # Downsample the data from 500 Hz to 100 Hz
            df_downsampled = df.iloc[::5, :].copy()  # Select every 5th row (downsampling from 500 Hz to 100 Hz)

            # Preprocess the downsampled data
            # Apply Savitzky-Golay filter
            window_length = 21  # Window length for the filter
            polynomial_order = 3  # Order of the polynomial
            # Assuming the third column contains the PPG signal data
            df_downsampled['Filtered_Signal'] = savgol_filter(df_downsampled.iloc[:, 2], window_length, polynomial_order)

            # Normalize the signal between 0 and 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_downsampled['Normalized_Signal'] = scaler.fit_transform(df_downsampled['Filtered_Signal'].values.reshape(-1, 1))

            # Mark as original data
            df_downsampled['Data_Origin'] = 1

            # Reshape the downsampled data into rows of 200 subsequent data points (2 seconds)
            num_datapoints = 200
            num_rows = len(df_downsampled) // num_datapoints
            reshaped_data = []
            for j in range(num_rows):
                start_index = j * num_datapoints
                end_index = start_index + num_datapoints
                row_data = [int(participant), i, df_downsampled.iloc[start_index]['Data_Origin']] + list(df_downsampled.iloc[start_index:end_index]['Normalized_Signal'])
                reshaped_data.append(row_data)

            # Create a new DataFrame with reshaped data
            column_names = ['Participant_Number', 'Run_Number', 'Data_Origin'] + [f'Data_{k+1}' for k in range(num_datapoints)]
            reshaped_df = pd.DataFrame(reshaped_data, columns=column_names)

            # Append the DataFrame to the list of participant's DataFrames
            participant_dfs.append(reshaped_df)

        except FileNotFoundError:
            # Skip missing files
            continue
    
    # Concatenate all runs of the participant into a single DataFrame
    if participant_dfs:
        participant_combined_df = pd.concat(participant_dfs, ignore_index=True)
        # Append the combined DataFrame of the participant to the list of all DataFrames
        dfs.append(participant_combined_df)

# Concatenate all DataFrames of different participants into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Define the file path for saving the combined DataFrame
output_combined_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0/downsampled_preprocessed_flagged_PPG.csv"

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_combined_file_path, index=False)

print("Preprocessed and combined DataFrame for all participants saved successfully.")
