import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from omiformerv import Omniformer
import os

# -- Parameters --
CSV_PATH = "merged_labeled.csv"  # Replace with actual path
BATCH_SIZE = 32
CONTEXT_DIM = 10
MODEL_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
SEQ_LEN = 100
INPUT_DIM = 11  # Based on your dataset (excluding label and Channel Name)
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Label Encoder for Channel Name --
channel_encoder = LabelEncoder()

# -- Pre-filtering Function --
def filter_events(df, window=5.0):
    df = df.sort_values("time").reset_index(drop=True)
    timestamps = df["time"].values
    labels = df["Label"].values

    # Track invalid indices
    mask = np.ones(len(df), dtype=bool)
    label1_times = timestamps[labels == 1]

    for i, (t, lbl) in enumerate(zip(timestamps, labels)):
        if lbl == 0:
            # Find if there's any label 1 within ±5s
            nearby = ((label1_times > (t - window)) & (label1_times < (t + window))).any()
            if nearby:
                mask[i] = False

    return df[mask]

# -- Custom Dataset with streaming and prefiltering --
class OmniformerCSVDataset(Dataset):
    def __init__(self, csv_path, chunk_size=10000):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.columns = pd.read_csv(csv_path, nrows=1).columns

        # Precompute total rows
        self.total_rows = sum(1 for _ in open(csv_path)) - 1

        # Encode channel names
        all_channels = pd.read_csv(csv_path, usecols=['Channel Name'])['Channel Name'].unique()
        self.encoded_channels = {ch: i for i, ch in enumerate(all_channels)}
        self.context_dim = CONTEXT_DIM

        # Create a filtered index map
        self.valid_indices = self._build_filtered_indices()

    def _build_filtered_indices(self):
        valid_idx = []
        chunk_start = 0
        while chunk_start < self.total_rows:
            chunk = pd.read_csv(
                self.csv_path, skiprows=range(1, chunk_start + 1),
                nrows=self.chunk_size, names=self.columns, header=0
            )
            chunk = filter_events(chunk)
            valid_idx.extend(chunk.index.tolist())
            chunk_start += self.chunk_size
        return valid_idx

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        chunk_start = (actual_idx // self.chunk_size) * self.chunk_size
        skip_rows = range(1, chunk_start + 1)
        chunk = pd.read_csv(self.csv_path, skiprows=skip_rows, nrows=self.chunk_size, names=self.columns, header=0)
        row = chunk.iloc[actual_idx % self.chunk_size]

        features = row[['time', 'frequency', 'tstart', 'tend', 'fstart', 'fend', 'snr', 'q', 'amplitude', 'phase']].astype(np.float32).values
        features_seq = np.tile(features, (SEQ_LEN, 1))

        context_id = self.encoded_channels.get(row['Channel Name'], 0)
        context_vector = np.zeros(self.context_dim, dtype=np.float32)
        context_vector[context_id % self.context_dim] = 1.0

        label = np.array([row['Label']], dtype=np.float32)

        return torch.tensor(features_seq), torch.tensor(context_vector), torch.tensor(label)

# -- Initialize Model --
model = Omniformer(
    input_dim=INPUT_DIM,
    context_dim=CONTEXT_DIM,
    model_dim=MODEL_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    seq_len=SEQ_LEN
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# -- Adaptive Batch Size Wrapper --
def get_dataloader(batch_size):
    dataset = OmniformerCSVDataset(CSV_PATH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

current_batch_size = BATCH_SIZE
loader = get_dataloader(current_batch_size)

# -- Training Loop --
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\n[Epoch {epoch}] Batch size: {current_batch_size} | LR: {scheduler.get_last_lr()[0]:.6f}")

    for batch_idx, (x, ctx, y) in enumerate(loader):
        x, ctx, y = x.to(DEVICE), ctx.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        try:
            output = model(x, ctx).squeeze(1)  # [B]
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"[OOM] Batch {batch_idx} failed. Reducing batch size...")
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                loader = get_dataloader(current_batch_size)
                break  # Restart epoch with smaller batch size
            else:
                raise e

        model.log_loss(loss)
        total_loss += loss.item()

        torch.cuda.empty_cache()  # Prevent memory accumulation

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")
    scheduler.step()

    if epoch % 5 == 0:
        model.save_checkpoint(f"checkpoint_epoch{epoch}.pt")