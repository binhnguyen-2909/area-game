import json

with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell indices based on partial content
def find_cell(query):
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if query in src:
            return i
    return -1

# 1. Update CNN Architecture
idx_cnn = find_cell("class CNNBotModel(nn.Module):")
nb["cells"][idx_cnn]["source"] = [
    "class CNNBotModel(nn.Module):\n",
    "    def __init__(self, in_channels=6, hidden_channels=256):\n",
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
    "        out = out.view(out.size(0), -1) # Flatten to [Batch, 64] for classification\n",
    "        return out\n",
    "\n",
    "model = CNNBotModel().to(device)\n",
    "print(model)\n"
]

# 2. Update Dataset (Streaming)
idx_ds = find_cell("class ArenaDataset(Dataset):")
nb["cells"][idx_ds]["source"] = [
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
    "                if len(parts) != 9: continue\n",
    "                \n",
    "                active_row = int(parts[0])\n",
    "                active_col = int(parts[1])\n",
    "                \n",
    "                ally_pos = np.array([float(x) for x in parts[2].split()], dtype=np.float32).reshape(8, 8)\n",
    "                enemy_pos = np.array([float(x) for x in parts[3].split()], dtype=np.float32).reshape(8, 8)\n",
    "                ally_hp = np.array([float(x) for x in parts[4].split()], dtype=np.float32).reshape(8, 8)\n",
    "                enemy_hp = np.array([float(x) for x in parts[5].split()], dtype=np.float32).reshape(8, 8)\n",
    "                dist_map = np.array([float(x) for x in parts[6].split()], dtype=np.float32).reshape(8, 8)\n",
    "                \n",
    "                active_pos = np.zeros((8, 8), dtype=np.float32)\n",
    "                active_pos[active_row, active_col] = 1.0\n",
    "                \n",
    "                feature_grid = np.stack([\n",
    "                    active_pos, ally_pos, enemy_pos, ally_hp, enemy_hp, dist_map\n",
    "                ], axis=0)\n",
    "                \n",
    "                target_row = int(parts[7])\n",
    "                target_col = int(parts[8])\n",
    "                target_class = target_row * 8 + target_col\n",
    "                \n",
    "                yield torch.tensor(feature_grid), torch.tensor(target_class, dtype=torch.long)\n",
    "\n",
    "print(\"Setting up streaming dataset...\")\n",
    "dataset = StreamingArenaDataset(\"dataset.csv\")\n"
]

# 3. Update Training Loop
idx_train = find_cell("train_loader = DataLoader(")
nb["cells"][idx_train]["source"] = [
    "train_loader = DataLoader(dataset, batch_size=256)\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.003)\n",
    "\n",
    "print(\"Training the model on GPU (Streaming 1 Epoch)...\")\n",
    "model.train()\n",
    "total_loss = 0.0\n",
    "correct = 0\n",
    "total = 0\n",
    "step = 0\n",
    "\n",
    "for x, y in train_loader:\n",
    "    x, y = x.to(device), y.to(device)\n",
    "    optimizer.zero_grad()\n",
    "    outputs = model(x)\n",
    "    loss = criterion(outputs, y)\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    total_loss += loss.item() * x.size(0)\n",
    "    _, predicted = outputs.max(1)\n",
    "    total += y.size(0)\n",
    "    correct += predicted.eq(y).sum().item()\n",
    "    step += 1\n",
    "    \n",
    "    if step % 100 == 0:\n",
    "        print(f\"Step {step} | Loss: {total_loss/total:.4f} | Acc: {correct/total*100:.2f}%\")\n",
    "\n",
    "print(f\"Training Complete! Total samples processed: {total}\")\n"
]

# 4. Update Java Code Generator
idx_export = find_cell("def format_array_4d")

# Note: Using raw strings or escaping properly for the generated Java code inside Python script
java_export_src = """import base64

w1 = model.conv1.weight.data.cpu().numpy().astype(np.float32)  # [256, 6, 3, 3]
b1 = model.conv1.bias.data.cpu().numpy().astype(np.float32)    # [256]
w2 = model.conv2.weight.data.cpu().numpy().astype(np.float32)  # [256, 256, 3, 3]
b2 = model.conv2.bias.data.cpu().numpy().astype(np.float32)    # [256]
w3 = model.conv3.weight.data.cpu().numpy().astype(np.float32)  # [1, 256, 1, 1]
b3 = model.conv3.bias.data.cpu().numpy().astype(np.float32)    # [1]

flat_weights = np.concatenate([
    w1.flatten(), b1.flatten(),
    w2.flatten(), b2.flatten(),
    w3.flatten(), b3.flatten()
])

raw_bytes = flat_weights.tobytes()
b64_str = base64.b64encode(raw_bytes).decode('ascii')
chunk_size = 20000
chunks = [b64_str[i:i+chunk_size] for i in range(0, len(b64_str), chunk_size)]
java_chunks = ",\\n        ".join(f'"{c}"' for c in chunks)

java_code = f'''package student;

import arenachallenge.api.*;
import java.util.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class StudentBotImpl implements StudentBot {{
    private static final int LOCAL_MAX_TEAM_SIZE = 8;
    
    // CNN 256 Weights
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
        ByteBuffer buffer = ByteBuffer.wrap(decoded).order(ByteOrder.nativeOrder()); // little endian from numpy
        
        W1 = new float[256][6][3][3];
        for(int i=0;i<256;i++) for(int j=0;j<6;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++) W1[i][j][k][l] = buffer.getFloat();
        
        B1 = new float[256];
        for(int i=0;i<256;i++) B1[i] = buffer.getFloat();
        
        W2 = new float[256][256][3][3];
        for(int i=0;i<256;i++) for(int j=0;j<256;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++) W2[i][j][k][l] = buffer.getFloat();
        
        B2 = new float[256];
        for(int i=0;i<256;i++) B2[i] = buffer.getFloat();
        
        W3 = new float[1][256][1][1];
        for(int i=0;i<1;i++) for(int j=0;j<256;j++) for(int k=0;k<1;k++) for(int l=0;l<1;l++) W3[i][j][k][l] = buffer.getFloat();
        
        B3 = new float[1];
        for(int i=0;i<1;i++) B3[i] = buffer.getFloat();
        
        initialized = true;
    }}

    public StudentBotImpl() {{
        initWeights();
    }}

    @Override
    public String getBotName() {{
        return "CNN_Model_Bot_256";
    }}

    @Override
    public List<ChampionPick> draftTeam(DraftView view) {{
        initWeights(); // double check
        List<ChampionTemplate> available = view.availableChampions();
        int budget = view.budget();
        List<ChampionPick> picks = new ArrayList<>();
        int spent = 0;

        Map<String, ChampionTemplate> templates = new HashMap<>();
        for (ChampionTemplate t : available) {{
            templates.put(t.id(), t);
        }}

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
                    if (t != null && spent + t.cost() <= budget) {{
                        pick = id;
                        break;
                    }}
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
        initWeights();
        List<Placement> placements = new ArrayList<>();
        List<ChampionSnapshot> team = view.team();
        List<Position> allowed = new ArrayList<>(view.allowedCells());
        TeamSide side = view.yourSide();

        if (side == TeamSide.BLUE) {{
            allowed.sort(Comparator.comparingInt(Position::col));
        }} else {{
            allowed.sort((a, b) -> Integer.compare(b.col(), a.col()));
        }}

        for (int i = 0; i < team.size() && i < allowed.size(); i++) {{
            placements.add(new Placement(team.get(i).championId(), allowed.get(i)));
        }}

        return placements;
    }}

    @Override
    public List<TurnAction> playTurn(BattleView view) {{
        initWeights();
        List<TurnAction> actions = new ArrayList<>();

        for (ChampionSnapshot ally : view.allies()) {{
            if (!ally.alive()) continue;

            String name = ally.displayName();

            if (name.equals("CLERIC") || name.equals("DRUID")) {{
                ChampionSnapshot worstAlly = null;
                double lowestHpRatio = 1.0;
                for (ChampionSnapshot otherAlly : view.allies()) {{
                    if (!otherAlly.alive()) continue;
                    double hpRatio = (double) otherAlly.hp() / otherAlly.maxHp();
                    if (hpRatio < lowestHpRatio) {{
                        lowestHpRatio = hpRatio;
                        worstAlly = otherAlly;
                    }}
                }}
                if (worstAlly != null && lowestHpRatio < 0.8 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {{
                    int dist = ally.position().manhattanDistance(worstAlly.position());
                    if (dist <= ally.range()) {{
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, worstAlly.position(), worstAlly.championId()));
                        continue;
                    }}
                }}
            }}

            float[][][] input = new float[6][8][8];
            int activeRow = ally.position().row();
            int activeCol = ally.position().col();
            input[0][activeRow][activeCol] = 1.0f;
            
            for (ChampionSnapshot a : view.allies()) {{
                if (a.alive()) {{
                    input[1][a.position().row()][a.position().col()] = 1.0f;
                    input[3][a.position().row()][a.position().col()] = (float) a.hp() / a.maxHp();
                }}
            }}
            for (ChampionSnapshot e : view.enemies()) {{
                if (e.alive()) {{
                    input[2][e.position().row()][e.position().col()] = 1.0f;
                    input[4][e.position().row()][e.position().col()] = (float) e.hp() / e.maxHp();
                }}
            }}
            for (int r = 0; r < 8; r++) {{
                for (int c = 0; c < 8; c++) {{
                    int dist = Math.abs(r - activeRow) + Math.abs(c - activeCol);
                    input[5][r][c] = (float) dist / 16.0f;
                }}
            }}

            float[][][] x1 = conv2d_3x3(input, W1, B1);
            relu(x1);
            float[][][] x2 = conv2d_3x3(x1, W2, B2);
            relu(x2);
            float[][][] scores = conv2d_1x1(x2, W3, B3);

            ChampionSnapshot bestEnemy = null;
            float bestEnemyScore = -999999.0f;
            for (ChampionSnapshot enemy : view.enemies()) {{
                if (!enemy.alive()) continue;
                int dist = ally.position().manhattanDistance(enemy.position());
                if (dist <= ally.range()) {{
                    float s = scores[0][enemy.position().row()][enemy.position().col()];
                    if (s > bestEnemyScore) {{
                        bestEnemyScore = s;
                        bestEnemy = enemy;
                    }}
                }}
            }}

            if (bestEnemy != null) {{
                if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {{
                    if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH")) {{
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestEnemy.position(), null));
                    }} else {{
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestEnemy.position(), bestEnemy.championId()));
                    }}
                }} else {{
                    actions.add(new TurnAction(ally.championId(), ActionType.ATTACK, null, bestEnemy.championId()));
                }}
                continue;
            }}

            ChampionSnapshot dashEnemy = null;
            float bestDashScore = -999999.0f;
            if (ally.mana() >= 3 && ally.remainingCooldown() == 0 && (name.equals("ASSASSIN") || name.equals("LANCER"))) {{
                for (ChampionSnapshot enemy : view.enemies()) {{
                    if (!enemy.alive()) continue;
                    int dist = ally.position().manhattanDistance(enemy.position());
                    if (dist <= 3) {{
                        float s = scores[0][enemy.position().row()][enemy.position().col()];
                        if (s > bestDashScore) {{
                            bestDashScore = s;
                            dashEnemy = enemy;
                        }}
                    }}
                }}
            }}

            if (dashEnemy != null) {{
                actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, dashEnemy.position(), dashEnemy.championId()));
                continue;
            }}

            Position bestMove = null;
            float bestMoveScore = -999999.0f;
            
            int[] dr = {{-1, 1, 0, 0}};
            int[] dc = {{0, 0, -1, 1}};
            
            for (int d = 0; d < 4; d++) {{
                int nr = activeRow + dr[d];
                int nc = activeCol + dc[d];
                
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {{
                    boolean occupied = false;
                    for (ChampionSnapshot a : view.allies()) {{
                        if (a.alive() && a.position().row() == nr && a.position().col() == nc) occupied = true;
                    }}
                    for (ChampionSnapshot e : view.enemies()) {{
                        if (e.alive() && e.position().row() == nr && e.position().col() == nc) occupied = true;
                    }}
                    
                    if (!occupied) {{
                        float s = scores[0][nr][nc];
                        if (s > bestMoveScore) {{
                            bestMoveScore = s;
                            bestMove = new Position(nr, nc);
                        }}
                    }}
                }}
            }}

            if (bestMove != null) {{
                actions.add(new TurnAction(ally.championId(), ActionType.MOVE, bestMove, null));
            }} else {{
                actions.add(TurnAction.waitAction(ally.championId()));
            }}
        }}
        return actions;
    }}

    private static float[][][] conv2d_3x3(float[][][] input, float[][][][] weights, float[] bias) {{
        int outChannels = weights.length;
        int inChannels = weights[0].length;
        int height = input[0].length;
        int width = input[0][0].length;
        float[][][] output = new float[outChannels][height][width];
        
        for (int oc = 0; oc < outChannels; oc++) {{
            for (int r = 0; r < height; r++) {{
                for (int c = 0; c < width; c++) {{
                    float val = bias[oc];
                    for (int ic = 0; ic < inChannels; ic++) {{
                        for (int kr = 0; kr < 3; kr++) {{
                            int nr = r + kr - 1;
                            if (nr < 0 || nr >= height) continue;
                            for (int kc = 0; kc < 3; kc++) {{
                                int nc = c + kc - 1;
                                if (nc < 0 || nc >= width) continue;
                                val += input[ic][nr][nc] * weights[oc][ic][kr][kc];
                            }}
                        }}
                    }}
                    output[oc][r][c] = val;
                }}
            }}
        }}
        return output;
    }}

    private static float[][][] conv2d_1x1(float[][][] input, float[][][][] weights, float[] bias) {{
        int outChannels = weights.length;
        int inChannels = weights[0].length;
        int height = input[0].length;
        int width = input[0][0].length;
        float[][][] output = new float[outChannels][height][width];
        
        for (int oc = 0; oc < outChannels; oc++) {{
            for (int r = 0; r < height; r++) {{
                for (int c = 0; c < width; c++) {{
                    float val = bias[oc];
                    for (int ic = 0; ic < inChannels; ic++) {{
                        val += input[ic][r][c] * weights[oc][ic][0][0];
                    }}
                    output[oc][r][c] = val;
                }}
            }}
        }}
        return output;
    }}

    private static void relu(float[][][] data) {{
        for (int c = 0; c < data.length; c++) {{
            for (int r = 0; r < data[0].length; r++) {{
                for (int col = 0; col < data[0][0].length; col++) {{
                    if (data[c][r][col] < 0.0f) {{
                        data[c][r][col] = 0.0f;
                    }}
                }}
            }}
        }}
    }}
}}
'''

print("Writing generated StudentBotImpl.java to disk...")
with open("src/student/StudentBotImpl.java", "w") as f:
    f.write(java_code)
print("Java code updated! Size of code: ~" + str(len(java_code)) + " bytes")
"""
# Need to format it properly as a list of strings for JSON source
# By splitlines(True) we preserve the newlines exactly
nb["cells"][idx_export]["source"] = java_export_src.splitlines(True)

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
