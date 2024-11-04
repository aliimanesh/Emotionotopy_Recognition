#This code remove the last two rows of Run8 for each participant to match the dataframe with emotion labels. 
#The input to this code is the dataframe of pupil diameter features. (output of "2_Pupil_FeatureExtraction.py")


import pandas as pd

# Define the file path for the combined DataFrame
input_combined_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0-Pupil/Pupil_Features.csv"
output_modified_file_path = "/Users/ali/Desktop/Ali/1.University/01-Cognitive Science/Thesis/Data/Full Data/0-Pupil/Pupil_Features_diminished.csv"
# Read the combined DataFrame from the CSV file
combined_df = pd.read_csv(input_combined_file_path)

# Function to remove the last two rows from run 8 for each participant
def remove_last_two_rows(df):
    # Filter the DataFrame for run 8
    run_8_df = df[df['Run_Number'] == 8]
    # If there are at least two rows, drop the last two rows
    if len(run_8_df) >= 2:
        df = df.iloc[:-2]
    return df

# Group by Participant_Number and apply the function
modified_df = combined_df.groupby('Participant_Number', group_keys=False).apply(remove_last_two_rows)

# Save the modified DataFrame to a new CSV file
modified_df.to_csv(output_modified_file_path, index=False)

print("Modified DataFrame with the last two rows of run 8 removed for each participant saved successfully.")
