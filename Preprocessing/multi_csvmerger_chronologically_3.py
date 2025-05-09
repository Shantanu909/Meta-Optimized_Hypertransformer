import os
import pandas as pd

def merge_csvs_chronologically(folder_path, output_file='merged.csv'):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_segments = []
    
    for file in csv_files:
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        if not df.empty:
            min_tstart = df['time'].min()
            file_segments.append((min_tstart, path))

    file_segments.sort(key=lambda x: x[0])

    merged_df = pd.concat([pd.read_csv(path) for _, path in file_segments], ignore_index=True)
    merged_df.to_csv(os.path.join(folder_path, output_file), index=False)
    print(f"Merged CSV saved as: {os.path.join(folder_path, output_file)}")

merge_csvs_chronologically('OMicrons\H1_LSC-POP_A_LF_OUT_DQ\csvs')
