#!/usr/bin/env python3
"""
Fine-tune existing CNN with horizontal flip augmentation + retrain Draft MLP.
Extracts trained weights from StudentBotImpl.java → fine-tunes → re-exports.
Reuses existing dataset.csv. Generates more draft data for better MLP.

Usage:
  python3 finetune_and_improve.py                   # Full pipeline (default)
  python3 finetune_and_improve.py --cnn-epochs 10    # More CNN fine-tune epochs
  python3 finetune_and_improve.py --draft-matches 2000000  # More draft data
  python3 finetune_and_improve.py --eval-matches 50  # Evaluation match count
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import subprocess
import time
import base64
import platform
import re
import random

# ==========================================
# Configuration
# ==========================================
HC = 1024           # CNN hidden channels (must match existing model)
IC = 14             # CNN input channels (must match existing model)
CNN_EPOCHS = 3      # Fine-tuning epochs (reduced to 3 since 15M dataset is huge)
CNN_LR = 0.0001     # Lower LR for fine-tuning
CNN_BATCH = 16384
CNN_LIMIT = 2000000  # Limit samples per epoch to 2M (about 23 mins/epoch) to save time
MLP_EPOCHS = 30     # More MLP epochs (was 10)
MLP_LR = 0.003
MLP_BATCH = 512
DRAFT_MATCHES = 1000000  # More draft data (was 200K)
CNN_WORKERS = 2      # Safe number of workers for PyTorch to avoid RAM OOM & CUDA issues
SIM_WORKERS = 40      # Run 40 parallel Java simulation processes to leverage the 60 CPU cores
EVAL_MATCHES = 50

# Parse command-line arguments
for i in range(len(sys.argv)):
    if sys.argv[i] == '--cnn-epochs' and i + 1 < len(sys.argv):
        CNN_EPOCHS = int(sys.argv[i + 1])
    elif sys.argv[i] == '--draft-matches' and i + 1 < len(sys.argv):
        DRAFT_MATCHES = int(sys.argv[i + 1])
    elif sys.argv[i] == '--eval-matches' and i + 1 < len(sys.argv):
        EVAL_MATCHES = int(sys.argv[i + 1])

cp_sep = ";" if platform.system() == "Windows" else ":"

# ==========================================
# Step 0: Setup & GPU Check
# ==========================================
print("=" * 60)
print("  FINE-TUNE CNN + IMPROVE DRAFT MLP")
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CNN fine-tune: {CNN_EPOCHS} epochs, LR={CNN_LR}")
print(f"MLP retrain: {MLP_EPOCHS} epochs, LR={MLP_LR}")
print(f"Draft matches: {DRAFT_MATCHES:,}")

# ==========================================
# Step 1: Extract weights from Java file
# ==========================================
print(f"\n=== Step 1: Extracting trained weights from StudentBotImpl.java ===")
java_file = "src/student/StudentBotImpl.java"
if not os.path.exists(java_file):
    print(f"ERROR: {java_file} not found!")
    sys.exit(1)

with open(java_file, 'r', encoding="utf-8") as f:
    java_content = f.read()
print(f"  Java file size: {len(java_content) / 1024 / 1024:.1f} MB")

# --- Extract CNN weights ---
cnn_start = java_content.find('WEIGHT_CHUNKS = {')
cnn_end = java_content.find('};', cnn_start) + 2
cnn_section = java_content[cnn_start:cnn_end]
cnn_chunks = re.findall(r'"([A-Za-z0-9+/=]+)"', cnn_section)
cnn_b64 = ''.join(cnn_chunks)
cnn_bytes = base64.b64decode(cnn_b64)
cnn_flat = np.frombuffer(cnn_bytes, dtype=np.float32).copy()

# Parse layer weights
idx = 0
w1 = cnn_flat[idx:idx + HC * IC * 9].reshape(HC, IC, 3, 3); idx += HC * IC * 9
b1 = cnn_flat[idx:idx + HC].copy(); idx += HC
w2 = cnn_flat[idx:idx + HC * HC * 9].reshape(HC, HC, 3, 3); idx += HC * HC * 9
b2 = cnn_flat[idx:idx + HC].copy(); idx += HC
w3 = cnn_flat[idx:idx + 3 * HC].reshape(3, HC, 1, 1); idx += 3 * HC
b3 = cnn_flat[idx:idx + 3].copy(); idx += 3
print(f"  CNN: {idx:,} floats extracted ({idx * 4 / 1024 / 1024:.1f} MB)")

# --- Extract MLP weights ---
mlp_start = java_content.find('MLP_CHUNKS = {')
mlp_end = java_content.find('};', mlp_start) + 2
mlp_section = java_content[mlp_start:mlp_end]
mlp_chunks = re.findall(r'"([A-Za-z0-9+/=]+)"', mlp_section)
mlp_b64 = ''.join(mlp_chunks)
mlp_bytes = base64.b64decode(mlp_b64)
mlp_flat = np.frombuffer(mlp_bytes, dtype=np.float32).copy()
print(f"  MLP: {len(mlp_flat)} floats extracted")

# ==========================================
# Step 2: Load into PyTorch models
# ==========================================
print(f"\n=== Step 2: Loading weights into PyTorch ===")

class CNNBotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(IC, HC, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(HC, HC, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(HC, 3, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out.view(out.size(0), -1)

model = CNNBotModel().to(device)
model.conv1.weight.data = torch.from_numpy(w1).to(device)
model.conv1.bias.data = torch.from_numpy(b1).to(device)
model.conv2.weight.data = torch.from_numpy(w2).to(device)
model.conv2.bias.data = torch.from_numpy(b2).to(device)
model.conv3.weight.data = torch.from_numpy(w3).to(device)
model.conv3.bias.data = torch.from_numpy(b3).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  CNN loaded: {total_params:,} parameters")

if '--skip-cnn' in sys.argv:
    if os.path.exists('model_checkpoint.pt'):
        print("  Loading CNN weights from model_checkpoint.pt...")
        checkpoint = torch.load('model_checkpoint.pt', map_location=device)
        model.load_state_dict(checkpoint['cnn'])
        print("  CNN weights loaded successfully!")
    else:
        print("  WARNING: model_checkpoint.pt not found! Using Java extracted weights.")

# ==========================================
# Step 3: Fine-tune CNN with data augmentation
# ==========================================
if '--skip-cnn' in sys.argv:
    print("  Skipping CNN fine-tuning because --skip-cnn was passed.")
else:
    CNN_CSV = "dataset.csv"
    if not os.path.exists(CNN_CSV):
        print(f"ERROR: {CNN_CSV} not found! Cannot fine-tune without training data.")
        sys.exit(1)

    class AugmentedStreamingDataset(torch.utils.data.IterableDataset):
        """Streaming dataset with optional horizontal flip augmentation."""
        def __init__(self, csv_file, augment=True):
            self.csv_file = csv_file
            self.augment = augment

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            num_workers = worker_info.num_workers if worker_info is not None else 1
            rng = random.Random(worker_id + int(time.time()) % 10000)

            with open(self.csv_file, 'r', encoding="utf-8") as f:
                header = f.readline()
                for i, line in enumerate(f):
                    if i % num_workers != worker_id:
                        continue
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) != 18:
                        continue

                    active_row = int(parts[0])
                    active_col = int(parts[1])

                    grids = []
                    for p in range(2, 15):
                        grids.append(np.array([float(x) for x in parts[p].split()],
                                              dtype=np.float32).reshape(8, 8))

                    # Build feature grid [14, 8, 8]
                    active_pos = np.zeros((8, 8), dtype=np.float32)
                    active_pos[active_row, active_col] = 1.0

                    feature_grid = np.stack([
                        active_pos, grids[0], grids[1], grids[2], grids[3], grids[4],
                        grids[5], grids[6], grids[7], grids[8], grids[9], grids[10],
                        grids[11], grids[12]
                    ], axis=0)

                    action_class = int(parts[15])
                    target_row = int(parts[16])
                    target_col = int(parts[17])

                    # === DATA AUGMENTATION: random horizontal flip ===
                    if self.augment and rng.random() < 0.5:
                        feature_grid = feature_grid[:, :, ::-1].copy()
                        # Recompute distance map for flipped active position
                        flipped_col = 7 - active_col
                        for r in range(8):
                            for c in range(8):
                                feature_grid[5, r, c] = (abs(r - active_row) + abs(c - flipped_col)) / 16.0
                        # Flip target column
                        target_col = 7 - target_col

                    target_class = action_class * 64 + target_row * 8 + target_col
                    yield torch.tensor(feature_grid), torch.tensor(target_class, dtype=torch.long)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CNN_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)

    start_time = time.time()
    for epoch in range(CNN_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        dataset = AugmentedStreamingDataset(CNN_CSV, augment=True)
        loader = DataLoader(
            dataset, 
            batch_size=CNN_BATCH, 
            num_workers=CNN_WORKERS,
            prefetch_factor=4 if CNN_WORKERS > 0 else None, 
            pin_memory=True
        )

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            if total % CNN_BATCH == 0:
                print(f"  -> {total:,} samples... (Loss: {loss.item():.4f})")

            if total >= CNN_LIMIT:
                print(f"  -> Reached CNN_LIMIT ({CNN_LIMIT:,} samples). Stopping epoch early.")
                break

        scheduler.step()
        acc = correct / max(total, 1) * 100
        lr = optimizer.param_groups[0]['lr']
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"Epoch {epoch + 1}/{CNN_EPOCHS} | Loss: {total_loss / max(total, 1):.4f} | "
              f"Acc: {acc:.2f}% | LR: {lr:.6f} | GPU: {gpu_mem:.1f}GB")

        # Save checkpoint after every epoch to prevent losing progress
        torch.save({'cnn': model.state_dict()}, 'model_checkpoint.pt')
        print(f"  Checkpoint saved: model_checkpoint.pt")

    cnn_time = time.time() - start_time
    print(f"CNN Fine-tuning Complete! ({cnn_time:.0f}s)")

# ==========================================
# Step 4: Generate more draft data
# ==========================================
print(f"\n=== Step 4: Generating {DRAFT_MATCHES:,} draft matches ===")

# First, export new CNN to Java (temp) and recompile so DraftDatasetGenerator uses new CNN
print("Exporting temp CNN weights for draft simulation...")

# Quick export just the weights (replace chunks in existing file)
new_w1 = model.conv1.weight.data.cpu().numpy().astype(np.float32)
new_b1 = model.conv1.bias.data.cpu().numpy().astype(np.float32)
new_w2 = model.conv2.weight.data.cpu().numpy().astype(np.float32)
new_b2 = model.conv2.bias.data.cpu().numpy().astype(np.float32)
new_w3 = model.conv3.weight.data.cpu().numpy().astype(np.float32)
new_b3 = model.conv3.bias.data.cpu().numpy().astype(np.float32)

new_flat_cnn = np.concatenate([
    new_w1.flatten(), new_b1.flatten(),
    new_w2.flatten(), new_b2.flatten(),
    new_w3.flatten(), new_b3.flatten()
])
new_cnn_b64 = base64.b64encode(new_flat_cnn.tobytes()).decode('ascii')
chunk_size = 20000
new_cnn_chunks = [new_cnn_b64[i:i + chunk_size] for i in range(0, len(new_cnn_b64), chunk_size)]
new_cnn_java = ",\n        ".join(f'"{c}"' for c in new_cnn_chunks)

# Replace CNN weights in Java file (keep old MLP for now)
new_content = java_content[:cnn_start] + \
    f'WEIGHT_CHUNKS = {{\n        {new_cnn_java}\n    }};' + \
    java_content[cnn_end:]
with open(java_file, 'w', encoding="utf-8") as f:
    f.write(new_content)
print("  Temp Java file updated with new CNN weights")

# Optimize conv2d in the newly written Java file right now so draft simulation runs fast
if os.path.exists("optimize_and_eval.py"):
    print("  Running optimize_and_eval.py --patch-only to optimize conv2d inference...")
    subprocess.run([sys.executable, "optimize_and_eval.py", "--patch-only"], capture_output=True)

# Recompile for draft simulation
print("  Compiling...")
# Clean up the out directory to prevent "bad class file" wrong version errors
if os.path.exists("out"):
    import shutil
    try:
        shutil.rmtree("out")
    except Exception as e:
        print(f"  WARNING: Could not clean out directory: {e}")
os.makedirs("out", exist_ok=True)

compile_cmd = [
    "javac", "-d", "out",
    "-cp", f"lib/arena-framework.jar{cp_sep}out",
    "src/student/HeuristicBot.java",
    "src/student/StudentBotImpl.java",
    "src/student/DraftDatasetGenerator.java"
]
result = subprocess.run(compile_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  WARNING: Compilation failed, using old bot for draft data")
    print(f"  {result.stderr[:1000]}")
else:
    print("  Compiled with new CNN and optimized inference!")

# Generate draft data in parallel
DRAFT_CSV = "draft_dataset_new.csv"
matches_per_worker = DRAFT_MATCHES // SIM_WORKERS
remainder = DRAFT_MATCHES % SIM_WORKERS
processes = []
temp_files = []

start_time = time.time()
for i in range(SIM_WORKERS):
    worker_matches = matches_per_worker + (remainder if i == SIM_WORKERS - 1 else 0)
    temp_csv = f"draft_new_part_{i}.csv"
    temp_files.append(temp_csv)
    worker_seed = 42 + i * 1337

    run_cmd = [
        "java", "-cp", f"lib/arena-framework.jar{cp_sep}out",
        "student.DraftDatasetGenerator",
        str(worker_matches), temp_csv, str(worker_seed)
    ]
    proc = subprocess.Popen(run_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(proc)

print(f"  Launched {SIM_WORKERS} draft simulators...")
for proc in processes:
    proc.wait()
elapsed = time.time() - start_time
print(f"  Draft simulations completed in {elapsed:.1f}s")

# Merge
print("  Merging draft data...")
with open(DRAFT_CSV, "w", encoding="utf-8") as outfile:
    header_written = False
    for temp_file in temp_files:
        if not os.path.exists(temp_file):
            continue
        with open(temp_file, "r", encoding="utf-8") as infile:
            header = infile.readline()
            if not header_written:
                outfile.write(header)
                header_written = True
            for line in infile:
                outfile.write(line)
        os.remove(temp_file)

sample_count = sum(1 for _ in open(DRAFT_CSV, encoding="utf-8")) - 1
print(f"  Draft dataset: {sample_count:,} samples")

# ==========================================
# Step 5: Train improved Draft MLP
# ==========================================
print(f"\n=== Step 5: Training Draft MLP ({MLP_EPOCHS} epochs) ===")

class DraftDataset(Dataset):
    def __init__(self, csv_file):
        X_list, y_list = [], []
        with open(csv_file, 'r', encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) != 14:
                    continue
                budget = float(parts[0]) / 50.0
                counts = [float(p) / 8.0 for p in parts[1:13]]
                result = float(parts[13])
                X_list.append([budget] + counts)
                y_list.append([result])
        self.X = torch.tensor(X_list, dtype=torch.float32)
        self.y = torch.tensor(y_list, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DraftMLP(nn.Module):
    """Same architecture as original (13->64->64->1) for Java compatibility."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


print("  Loading draft dataset into RAM...")
draft_ds = DraftDataset(DRAFT_CSV)
draft_loader = DataLoader(draft_ds, batch_size=MLP_BATCH, shuffle=True)
print(f"  {len(draft_ds):,} samples loaded")

draft_mlp = DraftMLP().to(device)
criterion_mlp = nn.BCELoss()
optimizer_mlp = optim.Adam(draft_mlp.parameters(), lr=MLP_LR, weight_decay=1e-4)
scheduler_mlp = optim.lr_scheduler.CosineAnnealingLR(optimizer_mlp, T_max=MLP_EPOCHS)

print(f"  Training MLP ({MLP_EPOCHS} epochs, LR={MLP_LR}, weight_decay=1e-4)...")
for epoch in range(MLP_EPOCHS):
    total_loss = 0
    for bx, by in draft_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer_mlp.zero_grad()
        pred = draft_mlp(bx)
        loss = criterion_mlp(pred, by)
        loss.backward()
        optimizer_mlp.step()
        total_loss += loss.item() * bx.size(0)
    scheduler_mlp.step()
    avg_loss = total_loss / len(draft_ds)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1}/{MLP_EPOCHS} | Loss: {avg_loss:.4f}")

print("  Draft MLP Training Complete!")

# Save full checkpoint
torch.save({
    'cnn': model.state_dict(),
    'mlp': draft_mlp.state_dict(),
}, 'model_checkpoint.pt')
print("  Full checkpoint saved: model_checkpoint.pt")

# ==========================================
# Step 6: Export both models to Java
# ==========================================
print(f"\n=== Step 6: Exporting final weights to StudentBotImpl.java ===")

# Re-read the current Java file (which has new CNN weights from step 4)
with open(java_file, 'r', encoding="utf-8") as f:
    final_content = f.read()

# --- Export new MLP weights ---
mlpw1 = draft_mlp.fc1.weight.data.cpu().numpy().astype(np.float32)
mlpb1 = draft_mlp.fc1.bias.data.cpu().numpy().astype(np.float32)
mlpw2 = draft_mlp.fc2.weight.data.cpu().numpy().astype(np.float32)
mlpb2 = draft_mlp.fc2.bias.data.cpu().numpy().astype(np.float32)
mlpw3 = draft_mlp.fc3.weight.data.cpu().numpy().astype(np.float32)
mlpb3 = draft_mlp.fc3.bias.data.cpu().numpy().astype(np.float32)

new_flat_mlp = np.concatenate([
    mlpw1.flatten(), mlpb1.flatten(),
    mlpw2.flatten(), mlpb2.flatten(),
    mlpw3.flatten(), mlpb3.flatten()
])
new_mlp_b64 = base64.b64encode(new_flat_mlp.tobytes()).decode('ascii')
new_mlp_chunks = [new_mlp_b64[i:i + chunk_size] for i in range(0, len(new_mlp_b64), chunk_size)]
new_mlp_java = ",\n        ".join(f'"{c}"' for c in new_mlp_chunks)

# Replace MLP weights in Java file
mlp_start = final_content.find('MLP_CHUNKS = {')
mlp_end = final_content.find('};', mlp_start) + 2
final_content = final_content[:mlp_start] + \
    f'MLP_CHUNKS = {{\n        {new_mlp_java}\n    }};' + \
    final_content[mlp_end:]

with open(java_file, 'w', encoding="utf-8") as f:
    f.write(final_content)
print(f"  Java file updated: {os.path.getsize(java_file) / 1024 / 1024:.1f} MB")

# ==========================================
# Step 7: Optimize conv2d + Compile + Evaluate
# ==========================================
print(f"\n=== Step 7: Optimizing, Compiling, and Evaluating ===")

# Run optimize_and_eval.py for conv2d patching + compilation + evaluation
if os.path.exists("optimize_and_eval.py"):
    print("  Running optimize_and_eval.py...")
    result = subprocess.run(
        [sys.executable, "optimize_and_eval.py", "--matches", str(EVAL_MATCHES)],
        capture_output=False
    )
else:
    # Inline: compile and evaluate without conv2d optimization
    print("  optimize_and_eval.py not found, compiling directly...")
    compile_cmd = [
        "javac", "-d", "out",
        "-cp", f"lib/arena-framework.jar{cp_sep}out",
        "src/student/HeuristicBot.java",
        "src/student/StudentBotImpl.java",
        "src/student/Simulator.java"
    ]
    subprocess.run(compile_cmd, check=True)
    
    eval_cmd = [
        "java", "-Xmx4g", "-server",
        "-cp", f"lib/arena-framework.jar{cp_sep}out",
        "student.Simulator"
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)

print("\n=== ALL DONE ===")
