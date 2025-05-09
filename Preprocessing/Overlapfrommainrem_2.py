import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
MAIN_DIR = "OMicrons\H1_GWOSC_O3a_4KHZ_R1\csvs"  # Folder with main *_triggers.h5 files
AUX_DIR = "OMicrons\H1_LSC-POP_A_LF_OUT_DQ\csvs"   
OUTPUT_DIR = "cleaned_main"
LOG_PATH = os.path.join(OUTPUT_DIR, "match_log.csv")
EXCLUSION_WINDOW = 0.1  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- Load all aux trigger times ---
# print("Collecting all aux trigger times...")
aux_files = glob(os.path.join(AUX_DIR, "*_triggers.csv"))
# print(f"Found {len(aux_files)} aux trigger files.")
# if not aux_files:
#     print(f"⚠️  No aux trigger CSVs found in {AUX_DIR}. Check the path.")
#     exit()

aux_trigger_times = []
for aux_file in tqdm(aux_files):
    df = pd.read_csv(aux_file)
    if 'time' in df.columns:
        aux_trigger_times.append(df['time'].values)

all_aux_times = np.sort(np.concatenate(aux_trigger_times))
print(f"Total aux trigger times loaded: {len(all_aux_times)}")

# --- Prepare log storage ---
log_entries = []

# --- Filter each main trigger file ---
print("Filtering main trigger CSVs...")
main_files = glob(os.path.join(MAIN_DIR, "*_triggers.csv"))
print(f"Found {len(main_files)} main trigger files.")

for main_file in tqdm(main_files):
    df = pd.read_csv(main_file)
    if 'time' not in df.columns:
        continue

    time_array = df['time'].values
    mask = np.ones(len(df), dtype=bool)

    for i, time_val in enumerate(time_array):
        for aux_time in all_aux_times:
            if aux_time - EXCLUSION_WINDOW <= time_val <= aux_time + EXCLUSION_WINDOW:
                mask[i] = False
                # log_entries.append({
                #     "main_file": os.path.basename(main_file),
                #     "main_row_index": i,
                #     "main_row_time": time_val,
                #     "aux_time": aux_time,
                #     "exclusion_window_start": aux_time - EXCLUSION_WINDOW,
                #     "exclusion_window_end": aux_time + EXCLUSION_WINDOW
                # })
                break  # No need to check more aux times for this row

    filtered_df = df[mask]
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(main_file))
    filtered_df.to_csv(out_path, index=False)

# # --- Save log ---
# if log_entries:
#     log_df = pd.DataFrame(log_entries)
#     log_df.to_csv(LOG_PATH, index=False)
#     print(f"\n📝 Log saved to: {LOG_PATH}")
# else:
#     print("\n✅ No matching rows found. All main CSVs were clean.")

print(f"✅ Cleaned main triggers saved to: {OUTPUT_DIR}")
