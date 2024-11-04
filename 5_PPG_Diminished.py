#This code remove the last two rows of Run8 for each participant to match the dataframe with emotion labels. 
#The input to this code is the dataframe of PPG Signal . (output of "4-PPG_Preprocess_Dataframe.py")

# diminished

import pandas as pd

# Define the file path for the combined DataFrame
input_combined_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0/downsampled_preprocessed_flagged_PPG.csv"
output_modified_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0/downsampled_preprocessed_flagged_PPG_diminished.csv"

# Read the combined DataFrame from the CSV file
combined_df = pd.read_csv(input_combined_file_path)

# Function to remove the last four rows from run 8 for each participant
def remove_last_four_rows(df):
    # Filter the DataFrame for run 8
    run_8_df = df[df['Run_Number'] == 8]
    # If there are at least four rows, drop the last four rows
    if len(run_8_df) >= 4:
        df = df.iloc[:-4]
    return df

# Group by Participant_Number and apply the function
modified_df = combined_df.groupby('Participant_Number', group_keys=False).apply(remove_last_four_rows)

# Save the modified DataFrame to a new CSV file
modified_df.to_csv(output_modified_file_path, index=False)

print("Modified DataFrame with the last four rows of run 8 removed for each participant saved successfully.")