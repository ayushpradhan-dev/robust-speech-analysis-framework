# src/opensmile_extractor.py

import os
import subprocess
import tempfile
import pandas as pd
from tqdm.auto import tqdm

def extract_opensmile_features(
    input_df, 
    opensmile_exe_path, 
    config_file_path, 
    audio_file_column='filepath', 
    verbose=True
):
    """
    Extract a feature set for each audio file using the OpenSMILE toolkit.

    Iterate through a DataFrame of audio file paths and execute the external
    OpenSMILE command-line tool for each file. This function manages the
    creation of temporary files for OpenSMILE's output and parses the
    resulting standard CSV files into a single, consolidated DataFrame.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        opensmile_exe_path (str): The full, absolute path to the SMILExtract executable.
                                  (e.g., 'C:/tools/opensmile/bin/SMILExtract.exe').
        config_file_path (str): The path to the OpenSMILE configuration file
                                (e.g., '../data/Androids-Corpus/Androids_fixed.conf').
        audio_file_column (str): The name of the column in input_df that holds the filepaths.
        verbose (bool): If True, print progress bars and error messages.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an audio file,
                      containing a 'filename' column and all extracted feature columns.
                      Returns an empty DataFrame if extraction fails for all files.
    """
    all_features_list = []
    
    # Verify the path to the OpenSMILE executable exists before starting.
    if not os.path.exists(opensmile_exe_path):
        print(f"FATAL ERROR: Could not find SMILExtract at '{opensmile_exe_path}'.")
        return pd.DataFrame()
        
    # Use a temporary directory to store intermediate CSV outputs from OpenSMILE.
    # This directory is automatically created and cleaned up upon exiting the block.
    with tempfile.TemporaryDirectory() as temp_dir:
        if verbose:
            print(f"Using temporary directory for OpenSMILE outputs: {temp_dir}")
        
        # Create an iterator with a progress bar if verbose mode is on.
        iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Extracting OpenSMILE Features") if verbose else input_df.iterrows()

        for index, row in iterator:
            input_audio_path = row[audio_file_column]
            filename = os.path.basename(input_audio_path)
            
            # Define a unique path for this file's feature output within the temp directory.
            output_csv_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.csv")
            
            # Construct the command-line arguments for the subprocess call.
            command = [
                opensmile_exe_path,
                '-C', config_file_path,
                '-I', input_audio_path,
                '-O', output_csv_path,
                # Use the -instname argument to give the first column a consistent name.
                '-instname', filename 
            ]
            
            try:
                # Execute the OpenSMILE command.
                # `check=True` raises a CalledProcessError if the command returns a non-zero exit code.
                # `capture_output=True` captures stdout and stderr for debugging.
                subprocess.run(command, check=True, capture_output=True, text=True)
                
                # Read the standard CSV file produced by the corrected config file.
                features_df = pd.read_csv(output_csv_path, sep=',')
                
                # Parse the features from the single output row.
                # Use iloc to select all columns from the second one onwards, which is a
                # robust way to ignore the first instance name column regardless of its name.
                feature_dict = features_df.iloc[:, 1:].iloc[0].to_dict()
                
                # Add the filename to the dictionary for later merging with metadata.
                feature_dict['filename'] = filename 
                all_features_list.append(feature_dict)
                
            except subprocess.CalledProcessError as e:
                # Catch errors specifically from the SMILExtract executable itself.
                if verbose:
                    print(f"ERROR: OpenSMILE failed for file '{filename}'. Stderr: {e.stderr}")
            except Exception as e:
                # Catch any other unexpected errors during file processing.
                if verbose:
                    print(f"An unexpected error occurred while processing output for '{filename}': {e}")
    
    if not all_features_list:
        print("Warning: No features were successfully extracted. The returned DataFrame is empty.")
        return pd.DataFrame()
        
    # Consolidate the list of feature dictionaries into a final DataFrame.
    return pd.DataFrame(all_features_list)