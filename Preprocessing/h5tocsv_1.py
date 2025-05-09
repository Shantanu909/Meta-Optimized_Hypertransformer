import h5py
import os
import pandas as pd

# Function to convert the tables to CSV
def convert_h5_to_csv(h5_file_path, output_dir):
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Get segments table and convert to DataFrame
            segments_table = h5_file['/segments']
            segments_df = pd.DataFrame(segments_table[:])  # Convert to DataFrame

            # Define the output file name for segments
            segments_csv_path = os.path.join(output_dir, f"{os.path.basename(h5_file_path)}_segments.csv")
            segments_df.to_csv(segments_csv_path, index=False)

            # Get triggers table and convert to DataFrame
            triggers_table = h5_file['/triggers']
            triggers_df = pd.DataFrame(triggers_table[:])  # Convert to DataFrame

            # Define the output file name for triggers
            triggers_csv_path = os.path.join(output_dir, f"{os.path.basename(h5_file_path)}_triggers.csv")
            triggers_df.to_csv(triggers_csv_path, index=False)

            print(f"Converted {h5_file_path} to CSV files.")
    
    except Exception as e:
        print(f"Error processing file {h5_file_path}: {e}")

# Directory containing the H5 files
input_dir = "OMicrons\H1_GWOSC_O3a_4KHZ_R1\H1_GWOSC-4KHZ_R1_STRAIN"
output_dir = "OMicrons\H1_GWOSC_O3a_4KHZ_R1\csvs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all .h5 files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".h5"):
        h5_file_path = os.path.join(input_dir, filename)
        convert_h5_to_csv(h5_file_path, output_dir)

print("Conversion completed.")
