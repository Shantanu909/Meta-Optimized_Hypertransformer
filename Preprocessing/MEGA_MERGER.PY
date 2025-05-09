import pandas as pd

# Output file
output_file = "merged_labeled.csv"

# Your files
main_file = "OMicrons\H1_GWOSC_O3a_4KHZ_R1\csvs\merged.csv"  # 0 = signal
aux_files = ["OMicrons\H1_LSC-POP_A_LF_OUT_DQ\csvs\merged.csv", "OMicrons\H1_LSC-REFL_A_LF_OUT_DQ\csvs\merged.csv", "OMicrons\H1_LSC-REFL_A_RF9_Q_ERR_DQOUT\csvs\merged.csv", "OMicrons\H1_PEM-VAULT_MAG_1030X195Y_COIL_X_DQ\csvs\merged.csv", "OMicrons\H1_PEM-VAULT_MAG_1030X195Y_COIL_Y_DQ\csvs\merged.csv"]  

chunk_size = 100_000  

def process_file(file_path, label, header_written):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk['Label'] = label
        chunk.to_csv(output_file, mode='a', index=False, header=not header_written)
        header_written = True
    return header_written

# Step 1: Clear previous output
open(output_file, 'w').close()

# Step 2: Process main file first
header_written = False
header_written = process_file(main_file, 0, header_written)

# Step 3: Process all aux files
for file in aux_files:
    header_written = process_file(file, 1, header_written)

print("✅ Merge complete → saved to:", output_file)
