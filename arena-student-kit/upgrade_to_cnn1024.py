import json

with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def find_cell(query):
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if query in src:
            return i
    return -1

# === CELL 3: Streaming Dataset with 10 channels (13 CSV columns) ===
idx = find_cell("class StreamingArenaDataset")
nb["cells"][idx]["source"] = [
    "class StreamingArenaDataset(torch.utils.data.IterableDataset):\n",
    "    def __init__(self, csv_file):\n",
    "        self.csv_file = csv_file\n",
    "        \n",
    "    def __iter__(self):\n",
    "        with open(self.csv_file, 'r') as f:\n",
    "            header = f.readline()\n",
    "            for line in f:\n",
    "                if not line.strip(): continue\n",
    "                parts = line.strip().split(',')\n",
    "                if len(parts) != 13: continue\n",
    "                \n",
    "                active_row = int(parts[0])\n",
    "                active_col = int(parts[1])\n",
    "                \n",
    "                ally_pos = np.array([float(x) for x in parts[2].split()], dtype=np.float32).reshape(8, 8)\n",
    "                enemy_pos = np.array([float(x) for x in parts[3].split()], dtype=np.float32).reshape(8, 8)\n",
    "                ally_hp = np.array([float(x) for x in parts[4].split()], dtype=np.float32).reshape(8, 8)\n",
    "                enemy_hp = np.array([float(x) for x in parts[5].split()], dtype=np.float32).reshape(8, 8)\n",
    "                dist_map = np.array([float(x) for x in parts[6].split()], dtype=np.float32).reshape(8, 8)\n",
    "                ally_mana = np.array([float(x) for x in parts[7].split()], dtype=np.float32).reshape(8, 8)\n",
    "                enemy_mana = np.array([float(x) for x in parts[8].split()], dtype=np.float32).reshape(8, 8)\n",
    "                ally_atk = np.array([float(x) for x in parts[9].split()], dtype=np.float32).reshape(8, 8)\n",
    "                enemy_atk = np.array([float(x) for x in parts[10].split()], dtype=np.float32).reshape(8, 8)\n",
    "                \n",
    "                active_pos = np.zeros((8, 8), dtype=np.float32)\n",
    "                active_pos[active_row, active_col] = 1.0\n",
    "                \n",
    "                feature_grid = np.stack([\n",
    "                    active_pos, ally_pos, enemy_pos, ally_hp, enemy_hp, dist_map,\n",
    "                    ally_mana, enemy_mana, ally_atk, enemy_atk\n",
    "                ], axis=0)  # [10, 8, 8]\n",
    "                \n",
    "                target_row = int(parts[11])\n",
    "                target_col = int(parts[12])\n",
    "                target_class = target_row * 8 + target_col\n",
    "                \n",
    "                yield torch.tensor(feature_grid), torch.tensor(target_class, dtype=torch.long)\n",
    "\n",
    "print(\"Setting up streaming dataset (10 input channels)...\")\n",
    "dataset = StreamingArenaDataset(\"dataset.csv\")\n"
]

# === CELL 4: CNN 1024 Model ===
idx = find_cell("class CNNBotModel")
nb["cells"][idx]["source"] = [
    "class CNNBotModel(nn.Module):\n",
    "    def __init__(self, in_channels=10, hidden_channels=1024):\n",
    "        super(CNNBotModel, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)\n",
    "        self.relu1 = nn.ReLU()\n",
    "        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)\n",
    "        self.relu2 = nn.ReLU()\n",
    "        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=1)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        out = self.relu1(self.conv1(x))\n",
    "        out = self.relu2(self.conv2(out))\n",
    "        out = self.conv3(out)\n",
    "        out = out.view(out.size(0), -1)  # [Batch, 64]\n",
    "        return out\n",
    "\n",
    "model = CNNBotModel().to(device)\n",
    "total_params = sum(p.numel() for p in model.parameters())\n",
    "print(f\"Model: CNN 1024 channels, 10 input features\")\n",
    "print(f\"Total parameters: {total_params:,}\")\n",
    "print(f\"Model size: {total_params * 4 / 1024 / 1024:.1f} MB\")\n",
    "print(model)\n"
]

# === CELL 5: Training 20 epochs, batch 8192, cosine LR ===
idx = find_cell("Training the model on GPU")
nb["cells"][idx]["source"] = [
    "BATCH_SIZE = 8192\n",
    "NUM_EPOCHS = 20\n",
    "LEARNING_RATE = 0.001\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)\n",
    "scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)\n",
    "\n",
    "print(f\"Training CNN 1024 on GPU ({NUM_EPOCHS} epochs, batch {BATCH_SIZE})\")\n",
    "print(f\"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB\")\n",
    "\n",
    "for epoch in range(NUM_EPOCHS):\n",
    "    model.train()\n",
    "    total_loss = 0.0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    step = 0\n",
    "    \n",
    "    epoch_dataset = StreamingArenaDataset(\"dataset.csv\")\n",
    "    epoch_loader = DataLoader(epoch_dataset, batch_size=BATCH_SIZE)\n",
    "    \n",
    "    for x, y in epoch_loader:\n",
    "        x, y = x.to(device), y.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(x)\n",
    "        loss = criterion(outputs, y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        total_loss += loss.item() * x.size(0)\n",
    "        _, predicted = outputs.max(1)\n",
    "        total += y.size(0)\n",
    "        correct += predicted.eq(y).sum().item()\n",
    "        step += 1\n",
    "    \n",
    "    scheduler.step()\n",
    "    train_loss = total_loss / max(total, 1)\n",
    "    train_acc = correct / max(total, 1) * 100\n",
    "    lr = optimizer.param_groups[0]['lr']\n",
    "    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3\n",
    "    print(f\"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | LR: {lr:.6f} | GPU: {gpu_mem:.1f}GB\")\n",
    "\n",
    "print(f\"Training Complete! Samples/epoch: {total}\")\n"
]

# === CELL 6: Export to Java with Base64, CNN 1024 ===
idx = find_cell("import base64")
# Read the existing java code template and modify for 1024 + 10 channels
# Build the new source for this cell
export_source = r"""import base64

HC = 1024  # hidden channels
IC = 10    # input channels

w1 = model.conv1.weight.data.cpu().numpy().astype(np.float32)  # [HC, IC, 3, 3]
b1 = model.conv1.bias.data.cpu().numpy().astype(np.float32)    # [HC]
w2 = model.conv2.weight.data.cpu().numpy().astype(np.float32)  # [HC, HC, 3, 3]
b2 = model.conv2.bias.data.cpu().numpy().astype(np.float32)    # [HC]
w3 = model.conv3.weight.data.cpu().numpy().astype(np.float32)  # [1, HC, 1, 1]
b3 = model.conv3.bias.data.cpu().numpy().astype(np.float32)    # [1]

flat_weights = np.concatenate([
    w1.flatten(), b1.flatten(),
    w2.flatten(), b2.flatten(),
    w3.flatten(), b3.flatten()
])

print(f"Total parameters: {len(flat_weights):,}")
print(f"Raw bytes: {len(flat_weights) * 4 / 1024 / 1024:.1f} MB")

raw_bytes = flat_weights.tobytes()
b64_str = base64.b64encode(raw_bytes).decode('ascii')
chunk_size = 20000
chunks = [b64_str[i:i+chunk_size] for i in range(0, len(b64_str), chunk_size)]
print(f"Base64 size: {len(b64_str) / 1024 / 1024:.1f} MB in {len(chunks)} chunks")

java_chunks = ",\n        ".join(f'"{c}"' for c in chunks)

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
    private static boolean initialized = false;

    private static final String[] WEIGHT_CHUNKS = {{
        {java_chunks}
    }};

    private static void initWeights() {{
        if (initialized) return;
        StringBuilder sb = new StringBuilder();
        for (String chunk : WEIGHT_CHUNKS) {{
            sb.append(chunk);
        }}
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
        
        W3 = new float[1][HC][1][1];
        for(int j=0;j<HC;j++) W3[0][j][0][0] = buffer.getFloat();
        B3 = new float[1];
        B3[0] = buffer.getFloat();
        
        initialized = true;
        System.out.println("CNN 1024 weights loaded (" + decoded.length + " bytes)");
    }}

    public StudentBotImpl() {{
        initWeights();
    }}

    @Override
    public String getBotName() {{
        return "CNN_1024_Bot";
    }}

    @Override
    public List<ChampionPick> draftTeam(DraftView view) {{
        List<ChampionTemplate> available = view.availableChampions();
        int budget = view.budget();
        List<ChampionPick> picks = new ArrayList<>();
        int spent = 0;

        Map<String, ChampionTemplate> templates = new HashMap<>();
        for (ChampionTemplate t : available) templates.put(t.id(), t);

        List<String> order = Arrays.asList("ARCHER", "KNIGHT", "CLERIC", "MAGE", "WARLOCK");

        while (picks.size() < LOCAL_MAX_TEAM_SIZE) {{
            String pick = null;
            for (String id : order) {{
                ChampionTemplate t = templates.get(id);
                if (t != null && spent + t.cost() <= budget) {{
                    long archerCount = picks.stream().filter(p -> p.templateId().equals("ARCHER")).count();
                    long clericCount = picks.stream().filter(p -> p.templateId().equals("CLERIC")).count();
                    if (id.equals("ARCHER") && archerCount >= 5) continue;
                    if (id.equals("CLERIC") && clericCount >= 1) continue;
                    pick = id;
                    break;
                }}
            }}
            if (pick == null) {{
                for (String id : order) {{
                    ChampionTemplate t = templates.get(id);
                    if (t != null && spent + t.cost() <= budget) {{ pick = id; break; }}
                }}
            }}
            if (pick == null) break;
            picks.add(new ChampionPick(pick));
            spent += templates.get(pick).cost();
        }}
        return picks;
    }}

    @Override
    public List<Placement> placeTeam(PlacementView view) {{
        List<Placement> placements = new ArrayList<>();
        List<ChampionSnapshot> team = view.team();
        List<Position> allowed = new ArrayList<>(view.allowedCells());
        TeamSide side = view.yourSide();
        if (side == TeamSide.BLUE) allowed.sort(Comparator.comparingInt(Position::col));
        else allowed.sort((a, b) -> Integer.compare(b.col(), a.col()));
        for (int i = 0; i < team.size() && i < allowed.size(); i++)
            placements.add(new Placement(team.get(i).championId(), allowed.get(i)));
        return placements;
    }}

    @Override
    public List<TurnAction> playTurn(BattleView view) {{
        List<TurnAction> actions = new ArrayList<>();
        for (ChampionSnapshot ally : view.allies()) {{
            if (!ally.alive()) continue;
            String name = ally.displayName();

            // Healer logic
            if (name.equals("CLERIC") || name.equals("DRUID")) {{
                ChampionSnapshot worstAlly = null;
                double lowestHpRatio = 1.0;
                for (ChampionSnapshot o : view.allies()) {{
                    if (!o.alive()) continue;
                    double r = (double) o.hp() / o.maxHp();
                    if (r < lowestHpRatio) {{ lowestHpRatio = r; worstAlly = o; }}
                }}
                if (worstAlly != null && lowestHpRatio < 0.8 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {{
                    int dist = ally.position().manhattanDistance(worstAlly.position());
                    if (dist <= ally.range()) {{
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, worstAlly.position(), worstAlly.championId()));
                        continue;
                    }}
                }}
            }}

            // Build 10-channel input tensor
            float[][][] input = new float[IC][8][8];
            int activeRow = ally.position().row();
            int activeCol = ally.position().col();
            input[0][activeRow][activeCol] = 1.0f;
            
            for (ChampionSnapshot a : view.allies()) {{
                if (a.alive()) {{
                    int r = a.position().row(), c = a.position().col();
                    input[1][r][c] = 1.0f;
                    input[3][r][c] = (float) a.hp() / a.maxHp();
                    input[6][r][c] = (a.maxMana() > 0) ? (float) a.mana() / a.maxMana() : 0.0f;
                    input[8][r][c] = (float) a.attack() / 10.0f;
                }}
            }}
            for (ChampionSnapshot e : view.enemies()) {{
                if (e.alive()) {{
                    int r = e.position().row(), c = e.position().col();
                    input[2][r][c] = 1.0f;
                    input[4][r][c] = (float) e.hp() / e.maxHp();
                    input[7][r][c] = (e.maxMana() > 0) ? (float) e.mana() / e.maxMana() : 0.0f;
                    input[9][r][c] = (float) e.attack() / 10.0f;
                }}
            }}
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    input[5][r][c] = (float)(Math.abs(r - activeRow) + Math.abs(c - activeCol)) / 16.0f;

            // CNN Forward Pass
            float[][][] x1 = conv2d_3x3(input, W1, B1);
            relu(x1);
            float[][][] x2 = conv2d_3x3(x1, W2, B2);
            relu(x2);
            float[][][] scores = conv2d_1x1(x2, W3, B3);

            // Decision: attack enemy with highest CNN score in range
            ChampionSnapshot bestEnemy = null;
            float bestScore = -999999.0f;
            for (ChampionSnapshot enemy : view.enemies()) {{
                if (!enemy.alive()) continue;
                int dist = ally.position().manhattanDistance(enemy.position());
                if (dist <= ally.range()) {{
                    float s = scores[0][enemy.position().row()][enemy.position().col()];
                    if (s > bestScore) {{ bestScore = s; bestEnemy = enemy; }}
                }}
            }}
            if (bestEnemy != null) {{
                if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {{
                    if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH"))
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestEnemy.position(), null));
                    else
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestEnemy.position(), bestEnemy.championId()));
                }} else {{
                    actions.add(new TurnAction(ally.championId(), ActionType.ATTACK, null, bestEnemy.championId()));
                }}
                continue;
            }}

            // Dash skills (Assassin, Lancer)
            ChampionSnapshot dashEnemy = null;
            float bestDashScore = -999999.0f;
            if (ally.mana() >= 3 && ally.remainingCooldown() == 0 && (name.equals("ASSASSIN") || name.equals("LANCER"))) {{
                for (ChampionSnapshot enemy : view.enemies()) {{
                    if (!enemy.alive()) continue;
                    int dist = ally.position().manhattanDistance(enemy.position());
                    if (dist <= 3) {{
                        float s = scores[0][enemy.position().row()][enemy.position().col()];
                        if (s > bestDashScore) {{ bestDashScore = s; dashEnemy = enemy; }}
                    }}
                }}
            }}
            if (dashEnemy != null) {{
                actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, dashEnemy.position(), dashEnemy.championId()));
                continue;
            }}

            // Move towards best scored cell
            Position bestMove = null;
            float bestMoveScore = -999999.0f;
            int[] dr = {{-1, 1, 0, 0}};
            int[] dc = {{0, 0, -1, 1}};
            for (int d = 0; d < 4; d++) {{
                int nr = activeRow + dr[d], nc = activeCol + dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {{
                    boolean occupied = false;
                    for (ChampionSnapshot a : view.allies()) if (a.alive() && a.position().row() == nr && a.position().col() == nc) occupied = true;
                    for (ChampionSnapshot e : view.enemies()) if (e.alive() && e.position().row() == nr && e.position().col() == nc) occupied = true;
                    if (!occupied) {{
                        float s = scores[0][nr][nc];
                        if (s > bestMoveScore) {{ bestMoveScore = s; bestMove = new Position(nr, nc); }}
                    }}
                }}
            }}
            if (bestMove != null) actions.add(new TurnAction(ally.championId(), ActionType.MOVE, bestMove, null));
            else actions.add(TurnAction.waitAction(ally.championId()));
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
"""

nb["cells"][idx]["source"] = export_source.splitlines(True)

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook upgraded to CNN 1024 + 10 features + 20 epochs + batch 8192")
