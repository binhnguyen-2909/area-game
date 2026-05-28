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

# ==========================================
# 1. Setup & GPU Check
# ==========================================
print("=== Step 1: Setup & GPU Check ===")
WORKSPACE_DIR = os.getcwd()
print("Current Workspace Directory:", WORKSPACE_DIR)

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
if device.type == 'cuda':
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Max GPU Memory Limit: 30 GB")
else:
    print("WARNING: Running on CPU. Training might be slow.")

# ==========================================
# 2. Parallel Battle Dataset Generation (CNN)
# ==========================================
print("\n=== Step 2: Generating Expert Battle Dataset (CNN) ===")
TOTAL_MATCHES = 200000       # Total matches for CNN training
NUM_WORKERS = 16          # Run on 16 parallel cores
FINAL_CSV = "dataset.csv"

cp_sep = ";" if platform.system() == "Windows" else ":"

if os.path.exists(FINAL_CSV) and os.path.getsize(FINAL_CSV) > 10 * 1024 * 1024 * 1024:
    print(f"Found existing {FINAL_CSV} ({os.path.getsize(FINAL_CSV)/1024**3:.1f} GB). Skipping Generation!")
else:
    print(f"Parallelizing DatasetGenerator to {TOTAL_MATCHES} matches using {NUM_WORKERS} processes...")
    
    compile_cmd = [
        "javac", "-d", "out", 
        "-cp", f"lib/arena-framework.jar{cp_sep}out", 
        "src/student/StudentBotImpl.java", "src/student/DatasetGenerator.java", "src/student/Simulator.java"
    ]
    print("Compiling Java generator classes...")
    subprocess.run(compile_cmd, check=True)
    
    matches_per_worker = TOTAL_MATCHES // NUM_WORKERS
    remainder = TOTAL_MATCHES % NUM_WORKERS
    
    processes = []
    temp_files = []
    
    start_time = time.time()
    for i in range(NUM_WORKERS):
        worker_matches = matches_per_worker + (remainder if i == NUM_WORKERS - 1 else 0)
        temp_csv = f"dataset_part_{i}.csv"
        temp_files.append(temp_csv)
        
        worker_seed = 1337 + i * 997 
        
        run_cmd = [
            "java", "-cp", f"lib/arena-framework.jar{cp_sep}out",
            "student.DatasetGenerator",
            str(worker_matches),
            temp_csv,
            str(worker_seed)
        ]
        
        proc = subprocess.Popen(run_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)
    
    print(f"Launched {NUM_WORKERS} parallel simulator processes. Simulating matches...")
    for proc in processes:
        proc.wait()
    
    elapsed = time.time() - start_time
    print(f"Simulations completed in {elapsed:.1f} seconds.")
    
    # Merge files
    print("Merging temporary CSV files...")
    with open(FINAL_CSV, "w", encoding="utf-8") as outfile:
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
    
    print(f"Success! CNN Dataset created at {FINAL_CSV} ({sum(1 for _ in open(FINAL_CSV)) - 1:,} samples)")
    
    # ==========================================
    # 3. PyTorch Streaming Dataset Loader
    # ==========================================
print("\n=== Step 3: Setting up Streaming Dataset (14 Channels) ===")
class StreamingArenaDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        with open(self.csv_file, 'r') as f:
            header = f.readline()
            for i, line in enumerate(f):
                if i % num_workers != worker_id: continue

                if not line.strip(): continue
                parts = line.strip().split(',')
                if len(parts) != 18: continue
                
                active_row = int(parts[0])
                active_col = int(parts[1])
                
                ally_pos = np.array([float(x) for x in parts[2].split()], dtype=np.float32).reshape(8, 8)
                enemy_pos = np.array([float(x) for x in parts[3].split()], dtype=np.float32).reshape(8, 8)
                ally_hp = np.array([float(x) for x in parts[4].split()], dtype=np.float32).reshape(8, 8)
                enemy_hp = np.array([float(x) for x in parts[5].split()], dtype=np.float32).reshape(8, 8)
                dist_map = np.array([float(x) for x in parts[6].split()], dtype=np.float32).reshape(8, 8)
                ally_mana = np.array([float(x) for x in parts[7].split()], dtype=np.float32).reshape(8, 8)
                enemy_mana = np.array([float(x) for x in parts[8].split()], dtype=np.float32).reshape(8, 8)
                ally_atk = np.array([float(x) for x in parts[9].split()], dtype=np.float32).reshape(8, 8)
                enemy_atk = np.array([float(x) for x in parts[10].split()], dtype=np.float32).reshape(8, 8)
                ally_def = np.array([float(x) for x in parts[11].split()], dtype=np.float32).reshape(8, 8)
                enemy_def = np.array([float(x) for x in parts[12].split()], dtype=np.float32).reshape(8, 8)
                ally_range = np.array([float(x) for x in parts[13].split()], dtype=np.float32).reshape(8, 8)
                enemy_range = np.array([float(x) for x in parts[14].split()], dtype=np.float32).reshape(8, 8)
                
                active_pos = np.zeros((8, 8), dtype=np.float32)
                active_pos[active_row, active_col] = 1.0
                
                feature_grid = np.stack([
                    active_pos, ally_pos, enemy_pos, ally_hp, enemy_hp, dist_map,
                    ally_mana, enemy_mana, ally_atk, enemy_atk, ally_def, enemy_def, ally_range, enemy_range
                ], axis=0)  # [14, 8, 8]
                
                action_class = int(parts[15])
                target_row = int(parts[16])
                target_col = int(parts[17])
                target_class = action_class * 64 + target_row * 8 + target_col
                
                yield torch.tensor(feature_grid), torch.tensor(target_class, dtype=torch.long)

dataset = StreamingArenaDataset(FINAL_CSV)

# ==========================================
# 4. Define CNN Model Architecture
# ==========================================
print("\n=== Step 4: Initializing CNN Bot Model ===")
class CNNBotModel(nn.Module):
    def __init__(self, in_channels=14, hidden_channels=1024):
        super(CNNBotModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # [Batch, 192]
        return out

model = CNNBotModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: CNN 1024 channels, 14 input features")
print(f"Total parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")

# ==========================================
# 5. Model Training (CNN)
# ==========================================
print("\n=== Step 5: Training CNN Model on GPU ===")
BATCH_SIZE = 32768
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f"Training CNN 1024 on GPU ({NUM_EPOCHS} epochs, batch {BATCH_SIZE})")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    epoch_dataset = StreamingArenaDataset(FINAL_CSV)
    epoch_loader = DataLoader(epoch_dataset, batch_size=BATCH_SIZE, num_workers=16, prefetch_factor=4, pin_memory=True)
    
    for x, y in epoch_loader:
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
        
        # Print progress every 100 batches
        if total // BATCH_SIZE % 100 == 0 and total // BATCH_SIZE > 0 and total % BATCH_SIZE == 0:
            print(f"  -> Processed {total} samples in this epoch... (Loss: {loss.item():.4f})")

    
    scheduler.step()
    train_loss = total_loss / max(total, 1)
    train_acc = correct / max(total, 1) * 100
    lr = optimizer.param_groups[0]['lr']
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | LR: {lr:.6f} | GPU: {gpu_mem:.1f}GB")

print(f"CNN Training Complete! Total samples/epoch: {total}")

# ==========================================
# 6. Parallel Draft Dataset Generation (MLP)
# ==========================================
print("\n=== Step 6: Generating Draft Dataset (MLP) ===")
TOTAL_DRAFT_MATCHES = 200000
FINAL_DRAFT_CSV = "draft_dataset.csv"

print(f"Parallelizing DraftDatasetGenerator to {TOTAL_DRAFT_MATCHES} matches using {NUM_WORKERS} processes...")

compile_draft = [
    "javac", "-d", "out", 
    "-cp", f"lib/arena-framework.jar{cp_sep}out",
    "src/student/StudentBotImpl.java", "src/student/DraftDatasetGenerator.java"
]
print("Compiling DraftDatasetGenerator classes...")
subprocess.run(compile_draft, check=True)

matches_per_worker = TOTAL_DRAFT_MATCHES // NUM_WORKERS
remainder = TOTAL_DRAFT_MATCHES % NUM_WORKERS

processes = []
temp_files = []

start_time = time.time()
for i in range(NUM_WORKERS):
    worker_matches = matches_per_worker + (remainder if i == NUM_WORKERS - 1 else 0)
    temp_csv = f"draft_part_{i}.csv"
    temp_files.append(temp_csv)
    
    worker_seed = 42 + i * 1337 
    
    run_cmd = [
        "java", "-cp", f"lib/arena-framework.jar{cp_sep}out",
        "student.DraftDatasetGenerator",
        str(worker_matches),
        temp_csv,
        str(worker_seed)
    ]
    
    proc = subprocess.Popen(run_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(proc)

print(f"Launched {NUM_WORKERS} parallel draft simulation processes. Simulating selections...")
for proc in processes:
    proc.wait()

elapsed = time.time() - start_time
print(f"Draft simulations completed in {elapsed:.1f} seconds.")

# Merge files
print("Merging draft temp CSV files...")
with open(FINAL_DRAFT_CSV, "w", encoding="utf-8") as outfile:
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

print(f"Success! MLP Draft Dataset created at {FINAL_DRAFT_CSV} ({sum(1 for _ in open(FINAL_DRAFT_CSV)) - 1:,} samples)")

# ==========================================
# 7. Model Training (MLP Draft)
# ==========================================
print("\n=== Step 7: Training Draft MLP Model ===")
class DraftDataset(Dataset):
    def __init__(self, csv_file):
        X_list, y_list = [], []
        # Optimized line-by-line reading to prevent RAM OOM
        with open(csv_file, 'r') as f:
            header = f.readline()
            for line in f:
                if not line.strip(): continue
                parts = line.split(',')
                if len(parts) != 14: continue
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
    def __init__(self):
        super(DraftMLP, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

print("Loading DraftDataset into RAM...")
draft_ds = DraftDataset(FINAL_DRAFT_CSV)
draft_loader = DataLoader(draft_ds, batch_size=512, shuffle=True)

draft_mlp = DraftMLP().to(device)
criterion_mlp = nn.BCELoss()
optimizer_mlp = optim.Adam(draft_mlp.parameters(), lr=0.005)

print("Training Draft MLP...")
for epoch in range(10):
    total_loss = 0
    for bx, by in draft_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer_mlp.zero_grad()
        pred = draft_mlp(bx)
        loss = criterion_mlp(pred, by)
        loss.backward()
        optimizer_mlp.step()
        total_loss += loss.item() * bx.size(0)
    print(f"Epoch {epoch+1}/10 | Loss: {total_loss / len(draft_ds):.4f}")
print("Draft MLP Training Complete!")

# ==========================================
# 8. Export Weights & Generate StudentBotImpl.java
# ==========================================
print("\n=== Step 8: Exporting Model Weights to StudentBotImpl.java ===")
HC = 1024  # hidden channels
IC = 14    # input channels

# --- 1. Export CNN Weights ---
w1 = model.conv1.weight.data.cpu().numpy().astype(np.float32)
b1 = model.conv1.bias.data.cpu().numpy().astype(np.float32)
w2 = model.conv2.weight.data.cpu().numpy().astype(np.float32)
b2 = model.conv2.bias.data.cpu().numpy().astype(np.float32)
w3 = model.conv3.weight.data.cpu().numpy().astype(np.float32)
b3 = model.conv3.bias.data.cpu().numpy().astype(np.float32)

flat_cnn = np.concatenate([
    w1.flatten(), b1.flatten(),
    w2.flatten(), b2.flatten(),
    w3.flatten(), b3.flatten()
])

raw_bytes_cnn = flat_cnn.tobytes()
b64_str_cnn = base64.b64encode(raw_bytes_cnn).decode('ascii')
chunk_size = 20000
chunks_cnn = [b64_str_cnn[i:i+chunk_size] for i in range(0, len(b64_str_cnn), chunk_size)]
java_chunks_cnn = ",\n        ".join(f'"{c}"' for c in chunks_cnn)

# --- 2. Export Draft MLP Weights ---
mlpw1 = draft_mlp.fc1.weight.data.cpu().numpy().astype(np.float32) # [64, 13]
mlpb1 = draft_mlp.fc1.bias.data.cpu().numpy().astype(np.float32)   # [64]
mlpw2 = draft_mlp.fc2.weight.data.cpu().numpy().astype(np.float32) # [64, 64]
mlpb2 = draft_mlp.fc2.bias.data.cpu().numpy().astype(np.float32)   # [64]
mlpw3 = draft_mlp.fc3.weight.data.cpu().numpy().astype(np.float32) # [1, 64]
mlpb3 = draft_mlp.fc3.bias.data.cpu().numpy().astype(np.float32)   # [1]

flat_mlp = np.concatenate([
    mlpw1.flatten(), mlpb1.flatten(),
    mlpw2.flatten(), mlpb2.flatten(),
    mlpw3.flatten(), mlpb3.flatten()
])

raw_bytes_mlp = flat_mlp.tobytes()
b64_str_mlp = base64.b64encode(raw_bytes_mlp).decode('ascii')
chunks_mlp = [b64_str_mlp[i:i+chunk_size] for i in range(0, len(b64_str_mlp), chunk_size)]
java_chunks_mlp = ",\n        ".join(f'"{c}"' for c in chunks_mlp)

java_code = f'''package student;

import arenachallenge.api.*;
import java.util.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class StudentBotImpl implements StudentBot {{
    private static final int LOCAL_MAX_TEAM_SIZE = 8;
    private static final int HC = {HC};
    private static final int IC = {IC};
    
    private static float[][][][] W1;
    private static float[] B1;
    private static float[][][][] W2;
    private static float[] B2;
    private static float[][][][] W3;
    private static float[] B3;
    
    private static float[][] MLPW1;
    private static float[] MLPB1;
    private static float[][] MLPW2;
    private static float[] MLPB2;
    private static float[][] MLPW3;
    private static float[] MLPB3;
    
    private static boolean initialized = false;

    private static final String[] WEIGHT_CHUNKS = {{
        {java_chunks_cnn}
    }};
    
    private static final String[] MLP_CHUNKS = {{
        {java_chunks_mlp}
    }};

    private static synchronized void initWeights() {{
        if (initialized) return;
        
        // Decode MLP
        StringBuilder sbMlp = new StringBuilder();
        for (String chunk : MLP_CHUNKS) sbMlp.append(chunk);
        byte[] decodedMlp = Base64.getDecoder().decode(sbMlp.toString());
        ByteBuffer bufferMlp = ByteBuffer.wrap(decodedMlp).order(ByteOrder.LITTLE_ENDIAN);
        
        MLPW1 = new float[64][13];
        for(int i=0;i<64;i++) for(int j=0;j<13;j++) MLPW1[i][j] = bufferMlp.getFloat();
        MLPB1 = new float[64];
        for(int i=0;i<64;i++) MLPB1[i] = bufferMlp.getFloat();
        
        MLPW2 = new float[64][64];
        for(int i=0;i<64;i++) for(int j=0;j<64;j++) MLPW2[i][j] = bufferMlp.getFloat();
        MLPB2 = new float[64];
        for(int i=0;i<64;i++) MLPB2[i] = bufferMlp.getFloat();
        
        MLPW3 = new float[1][64];
        for(int j=0;j<64;j++) MLPW3[0][j] = bufferMlp.getFloat();
        MLPB3 = new float[1];
        MLPB3[0] = bufferMlp.getFloat();
        
        // Decode CNN
        StringBuilder sb = new StringBuilder();
        for (String chunk : WEIGHT_CHUNKS) sb.append(chunk);
        byte[] decoded = Base64.getDecoder().decode(sb.toString());
        ByteBuffer buffer = ByteBuffer.wrap(decoded).order(ByteOrder.LITTLE_ENDIAN);
        
        W1 = new float[HC][IC][3][3];
        for(int i=0;i<HC;i++) for(int j=0;j<IC;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++) W1[i][j][k][l] = buffer.getFloat();
        B1 = new float[HC];
        for(int i=0;i<HC;i++) B1[i] = buffer.getFloat();
        
        W2 = new float[HC][HC][3][3];
        for(int i=0;i<HC;i++) for(int j=0;j<HC;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++) W2[i][j][k][l] = buffer.getFloat();
        B2 = new float[HC];
        for(int i=0;i<HC;i++) B2[i] = buffer.getFloat();
        
        W3 = new float[3][HC][1][1];
        for(int oc=0;oc<3;oc++) for(int j=0;j<HC;j++) W3[oc][j][0][0] = buffer.getFloat();
        B3 = new float[3];
        for(int oc=0;oc<3;oc++) B3[oc] = buffer.getFloat();
        
        initialized = true;
        System.out.println("CNN and MLP weights loaded!");
    }}

    public StudentBotImpl() {{
        initWeights();
    }}

    @Override
    public String getBotName() {{
        return "CNN_1024_Dual_Bot";
    }}

    private static int getCol(int col, TeamSide side) {{
        return (side == TeamSide.RED) ? (7 - col) : col;
    }}

    private int getChampionIndex(String id) {{
        String[] ALL_IDS = {{"KNIGHT", "ARCHER", "MAGE", "ASSASSIN", "CLERIC", "GUARDIAN", "PALADIN", "WARLOCK", "DRUID", "LANCER", "FROST_WITCH", "BERSERKER"}};
        for (int i=0; i<12; i++) if (ALL_IDS[i].equals(id)) return i;
        return 0;
    }}

    private float evalDraft(int budget, int[] comp) {{
        float[] x = new float[13];
        x[0] = (float) budget / 50.0f;
        for (int i=0; i<12; i++) x[i+1] = (float) comp[i] / 8.0f;
        
        float[] h1 = new float[64];
        for(int i=0; i<64; i++) {{
            float v = MLPB1[i];
            for(int j=0; j<13; j++) v += x[j] * MLPW1[i][j];
            h1[i] = v > 0 ? v : 0;
        }}
        
        float[] h2 = new float[64];
        for(int i=0; i<64; i++) {{
            float v = MLPB2[i];
            for(int j=0; j<64; j++) v += h1[j] * MLPW2[i][j];
            h2[i] = v > 0 ? v : 0;
        }}
        
        float out = MLPB3[0];
        for(int j=0; j<64; j++) out += h2[j] * MLPW3[0][j];
        return out; 
    }}

    @Override
    public List<ChampionPick> draftTeam(DraftView view) {{
        initWeights();
        int budget = view.budget();
        List<ChampionTemplate> available = view.availableChampions();
        
        int[] bestComp = null;
        float bestScore = -999.0f;
        Random rand = new Random();
        
        for (int iter = 0; iter < 1000; iter++) {{
            int spent = 0;
            int count = 0;
            int[] comp = new int[12];
            List<ChampionTemplate> affordable = new ArrayList<>();
            for (ChampionTemplate t : available) if (t.cost() <= budget) affordable.add(t);
            
            while (count < LOCAL_MAX_TEAM_SIZE && !affordable.isEmpty()) {{
                ChampionTemplate t = affordable.get(rand.nextInt(affordable.size()));
                int idx = getChampionIndex(t.id());
                comp[idx]++;
                spent += t.cost();
                count++;
                affordable.clear();
                for (ChampionTemplate a : available) if (spent + a.cost() <= budget) affordable.add(a);
            }}
            
            float score = evalDraft(budget, comp);
            if (score > bestScore) {{
                bestScore = score;
                bestComp = comp;
            }}
        }}
        
        List<ChampionPick> picks = new ArrayList<>();
        if (bestComp != null) {{
            String[] ALL_IDS = {{"KNIGHT", "ARCHER", "MAGE", "ASSASSIN", "CLERIC", "GUARDIAN", "PALADIN", "WARLOCK", "DRUID", "LANCER", "FROST_WITCH", "BERSERKER"}};
            for (int i = 0; i < 12; i++) {{
                for (int k = 0; k < bestComp[i]; k++) {{
                    picks.add(new ChampionPick(ALL_IDS[i]));
                }}
            }}
        }}
        return picks;
    }}

    @Override
    public List<Placement> placeTeam(PlacementView view) {{
        List<Placement> placements = new ArrayList<>();
        List<ChampionSnapshot> team = view.team();
        List<Position> allowed = new ArrayList<>(view.allowedCells());
        TeamSide side = view.yourSide();
        
        // Sort allowed cells: frontline first (closest to enemy)
        if (side == TeamSide.BLUE) {{
            allowed.sort((a, b) -> Integer.compare(b.col(), a.col()));
        }} else {{
            allowed.sort(Comparator.comparingInt(Position::col));
        }}
        
        List<ChampionSnapshot> tanks = new ArrayList<>();
        List<ChampionSnapshot> others = new ArrayList<>();
        for (ChampionSnapshot c : team) {{
            String role = c.displayName();
            if (role.equals("KNIGHT") || role.equals("PALADIN") || role.equals("GUARDIAN") || role.equals("BERSERKER") || role.equals("LANCER") || role.equals("ASSASSIN")) {{
                tanks.add(c);
            }} else {{
                others.add(c);
            }}
        }}
        
        int cellIndex = 0;
        for (ChampionSnapshot tank : tanks) {{
            if (cellIndex < allowed.size()) {{
                placements.add(new Placement(tank.championId(), allowed.get(cellIndex++)));
            }}
        }}
        for (ChampionSnapshot other : others) {{
            if (cellIndex < allowed.size()) {{
                placements.add(new Placement(other.championId(), allowed.get(cellIndex++)));
            }}
        }}
        return placements;
    }}

    @Override
    public List<TurnAction> playTurn(BattleView view) {{
        List<TurnAction> actions = new ArrayList<>();
        TeamSide side = view.yourSide();
        for (ChampionSnapshot ally : view.allies()) {{
            if (!ally.alive()) continue;
            String name = ally.displayName();

            // Build 14-channel input tensor
            float[][][] input = new float[IC][8][8];
            int activeRow = ally.position().row();
            int activeColReal = ally.position().col();
            input[0][activeRow][getCol(activeColReal, side)] = 1.0f;
            
            for (ChampionSnapshot a : view.allies()) {{
                if (a.alive()) {{
                    int r = a.position().row(), c = a.position().col();
                    int mc = getCol(c, side);
                    input[1][r][mc] = 1.0f;
                    input[3][r][mc] = (float) a.hp() / a.maxHp();
                    input[6][r][mc] = (a.maxMana() > 0) ? (float) a.mana() / a.maxMana() : 0.0f;
                    input[8][r][mc] = (float) a.attack() / 10.0f;
                    input[10][r][mc] = (float) a.defense() / 10.0f;
                    input[12][r][mc] = (float) a.range() / 5.0f;
                }}
            }}
            for (ChampionSnapshot e : view.enemies()) {{
                if (e.alive()) {{
                    int r = e.position().row(), c = e.position().col();
                    int mc = getCol(c, side);
                    input[2][r][mc] = 1.0f;
                    input[4][r][mc] = (float) e.hp() / e.maxHp();
                    input[7][r][mc] = (e.maxMana() > 0) ? (float) e.mana() / e.maxMana() : 0.0f;
                    input[9][r][mc] = (float) e.attack() / 10.0f;
                    input[11][r][mc] = (float) e.defense() / 10.0f;
                    input[13][r][mc] = (float) e.range() / 5.0f;
                }}
            }}
            int activeColMirrored = getCol(activeColReal, side);
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    input[5][r][c] = (float)(Math.abs(r - activeRow) + Math.abs(c - activeColMirrored)) / 16.0f;

            // CNN Forward Pass: scores[0]=Move, scores[1]=Attack, scores[2]=Skill
            float[][][] x1 = conv2d_3x3(input, W1, B1);
            relu(x1);
            float[][][] x2 = conv2d_3x3(x1, W2, B2);
            relu(x2);
            float[][][] scores = conv2d_1x1(x2, W3, B3);

            // Find best target across all 3 action channels
            int bestAction = -1;
            Position bestTargetPos = null;
            ChampionSnapshot bestTargetChamp = null;
            float bestVal = -999999.0f;

            // Channel 2: Skill targets (highest priority check)
            if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {{
                // AOE casters: target position
                if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH")) {{
                    for (ChampionSnapshot enemy : view.enemies()) {{
                        if (!enemy.alive()) continue;
                        int dist = ally.position().manhattanDistance(enemy.position());
                        if (dist <= ally.range()) {{
                            float s = scores[2][enemy.position().row()][getCol(enemy.position().col(), side)];
                            if (s > bestVal) {{ bestVal = s; bestAction = 2; bestTargetPos = enemy.position(); bestTargetChamp = null; }}
                        }}
                    }}
                }}
                // Dash skills: Assassin/Lancer (extended range)
                else if (name.equals("ASSASSIN") || name.equals("LANCER")) {{
                    for (ChampionSnapshot enemy : view.enemies()) {{
                        if (!enemy.alive()) continue;
                        int dist = ally.position().manhattanDistance(enemy.position());
                        if (dist <= 3) {{
                            float s = scores[2][enemy.position().row()][getCol(enemy.position().col(), side)];
                            if (s > bestVal) {{ bestVal = s; bestAction = 2; bestTargetPos = enemy.position(); bestTargetChamp = enemy; }}
                        }}
                    }}
                }}
                // Healer: CLERIC/DRUID target allies
                else if (name.equals("CLERIC") || name.equals("DRUID")) {{
                    for (ChampionSnapshot a2 : view.allies()) {{
                        if (!a2.alive()) continue;
                        int dist = ally.position().manhattanDistance(a2.position());
                        if (dist <= ally.range()) {{
                            float s = scores[2][a2.position().row()][getCol(a2.position().col(), side)];
                            if (s > bestVal) {{ bestVal = s; bestAction = 20; bestTargetPos = a2.position(); bestTargetChamp = a2; }}
                        }}
                    }}
                }}
                // Single-target skill: others
                else {{
                    for (ChampionSnapshot enemy : view.enemies()) {{
                        if (!enemy.alive()) continue;
                        int dist = ally.position().manhattanDistance(enemy.position());
                        if (dist <= ally.range()) {{
                            float s = scores[2][enemy.position().row()][getCol(enemy.position().col(), side)];
                            if (s > bestVal) {{ bestVal = s; bestAction = 2; bestTargetPos = enemy.position(); bestTargetChamp = enemy; }}
                        }}
                    }}
                }}
            }}

            // Channel 1: Attack targets
            for (ChampionSnapshot enemy : view.enemies()) {{
                if (!enemy.alive()) continue;
                int dist = ally.position().manhattanDistance(enemy.position());
                if (dist <= ally.range()) {{
                    float s = scores[1][enemy.position().row()][getCol(enemy.position().col(), side)];
                    if (s > bestVal) {{ bestVal = s; bestAction = 1; bestTargetPos = enemy.position(); bestTargetChamp = enemy; }}
                }}
            }}

            // Channel 0: Move targets
            int[] dr = {{-1, 1, 0, 0}};
            int[] dc = {{0, 0, -1, 1}};
            for (int d = 0; d < 4; d++) {{
                int nr = activeRow + dr[d], nc = activeColReal + dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {{
                    boolean occupied = false;
                    for (ChampionSnapshot a2 : view.allies()) if (a2.alive() && a2.position().row() == nr && a2.position().col() == nc) occupied = true;
                    for (ChampionSnapshot e2 : view.enemies()) if (e2.alive() && e2.position().row() == nr && e2.position().col() == nc) occupied = true;
                    if (!occupied) {{
                        float s = scores[0][nr][getCol(nc, side)];
                        if (s > bestVal) {{ bestVal = s; bestAction = 0; bestTargetPos = new Position(nr, nc); bestTargetChamp = null; }}
                    }}
                }}
            }}

            // Execute best action
            if (bestAction == 2) {{
                if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH"))
                    actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestTargetPos, null));
                else
                    actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestTargetPos, bestTargetChamp != null ? bestTargetChamp.championId() : null));
            } else if (bestAction == 20) {{
                actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestTargetPos, bestTargetChamp != null ? bestTargetChamp.championId() : null));
            } else if (bestAction == 1) {{
                actions.add(new TurnAction(ally.championId(), ActionType.ATTACK, null, bestTargetChamp.championId()));
            } else if (bestAction == 0) {{
                actions.add(new TurnAction(ally.championId(), ActionType.MOVE, bestTargetPos, null));
            } else {{
                actions.add(TurnAction.waitAction(ally.championId()));
            }}
        }}
        return actions;
    }}

    private static float[][][] conv2d_3x3(float[][][] input, float[][][][] weights, float[] bias) {{
        int outCh = weights.length, inCh = weights[0].length, h = input[0].length, w = input[0][0].length;
        float[][][] out = new float[outCh][h][w];
        for (int oc = 0; oc < outCh; oc++)
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++) {{
                    float val = bias[oc];
                    for (int ic = 0; ic < inCh; ic++)
                        for (int kr = 0; kr < 3; kr++) {{
                            int nr = r + kr - 1;
                            if (nr < 0 || nr >= h) continue;
                            for (int kc = 0; kc < 3; kc++) {{
                                int nc = c + kc - 1;
                                if (nc < 0 || nc >= w) continue;
                                val += input[ic][nr][nc] * weights[oc][ic][kr][kc];
                            }}
                        }}
                    out[oc][r][c] = val;
                }}
        return out;
    }}

    private static float[][][] conv2d_1x1(float[][][] input, float[][][][] weights, float[] bias) {{
        int outCh = weights.length, inCh = weights[0].length, h = input[0].length, w = input[0][0].length;
        float[][][] out = new float[outCh][h][w];
        for (int oc = 0; oc < outCh; oc++)
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++) {{
                    float val = bias[oc];
                    for (int ic = 0; ic < inCh; ic++) val += input[ic][r][c] * weights[oc][ic][0][0];
                    out[oc][r][c] = val;
                }}
        return out;
    }}

    private static void relu(float[][][] data) {{
        for (int c = 0; c < data.length; c++)
            for (int r = 0; r < data[0].length; r++)
                for (int col = 0; col < data[0][0].length; col++)
                    if (data[c][r][col] < 0.0f) data[c][r][col] = 0.0f;
    }}
}}
'''

print("Writing generated StudentBotImpl.java to disk...")
with open("src/student/StudentBotImpl.java", "w") as f:
    f.write(java_code)
print(f"Java code written! Size: {len(java_code) / 1024 / 1024:.1f} MB")

# ==========================================
# 9. Compile and Run Evaluation Simulator
# ==========================================
print("\n=== Step 9: Compiling CNN Java Bot and Running Simulator ===")
recompile_cmd = [
    "javac", "-cp", f"lib/arena-framework.jar{cp_sep}out",
    "-d", "out",
    "src/student/StudentBotImpl.java", "src/student/Simulator.java"
]
print("Recompiling...")
subprocess.run(recompile_cmd, check=True)

print("Running evaluation simulator (this may take some time depending on your Simulator.java configuration)...")
eval_cmd = [
    "java", "-cp", f"lib/arena-framework.jar{cp_sep}out",
    "student.Simulator"
]
result = subprocess.run(eval_cmd, capture_output=True, text=True)
print("\n=== SIMULATION RESULTS ===")
print(result.stdout)
if result.stderr:
    print("Errors:")
    print(result.stderr)
print("=== Process Finished Successfully ===")
