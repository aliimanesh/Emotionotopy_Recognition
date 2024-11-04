#This code extract some statistical features from the Pupil diameter data.

#The input to this code is the dataframe created in the previous code (1-Pupil_Preprocess_Dataframe.py)

#The output of this code contain participant number and the run number in the 1st and the 2nd columns. In the third column It's been specified that whether
#The row can be used in training the model or not.


#Extract features
#if the number of zeros is more than 1000, write 0 in the third column, otherwise write 1
#mean, median, and variance of the non-zero values 

import pandas as pd
import numpy as np

# Define the file path for reading the combined DataFrame
combined_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0-Pupil/combined_pupil_data_all_participants_normal.csv"

# Read the combined DataFrame
combined_df = pd.read_csv(combined_file_path)

# List to store new DataFrame rows
new_data = []

# Iterate over each row in the combined DataFrame
for index, row in combined_df.iterrows():
    participant_number = row['Participant_Number']
    run_number = row['Run_Number']
    
    # Extract pupil diameter data (excluding the first two columns)
    pupil_data = row[2:].values
    
    # Count the number of zeros
    num_zeros = np.sum(pupil_data == 0)
    
    # Determine the value for the third column based on the number of zeros
    third_column_value = 0 if num_zeros > 1000 else 1
    
    # Extract non-zero values
    non_zero_values = pupil_data[pupil_data != 0]
    
    # Calculate mean, median, and variance of non-zero values
    if len(non_zero_values) > 0:
        mean_value = np.mean(non_zero_values)
        median_value = np.median(non_zero_values)
        variance_value = np.var(non_zero_values)
        max_value = np.max(non_zero_values)  # Calculate maximum value of non-zero values
    else:
        mean_value = np.nan
        median_value = np.nan
        variance_value = np.nan
        max_value = np.nan  # Set max_value to NaN when there are no non-zero values
    
    # Append the new row data (including maximum value)
    new_data.append([participant_number, run_number, third_column_value, mean_value, median_value, variance_value, max_value])

# Create a new DataFrame with the new data
new_column_names = ['Participant_Number', 'Run_Number', 'Zero_Flag', 'Mean', 'Median', 'Variance', 'Max']
new_df = pd.DataFrame(new_data, columns=new_column_names)

# Define the file path for saving the new DataFrame
new_output_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0-Pupil/Pupil_Features.csv"

# Save the new DataFrame to a CSV file
new_df.to_csv(new_output_file_path, index=False)

print("New DataFrame saved successfully.")
