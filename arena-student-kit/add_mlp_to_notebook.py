import json

def build_notebook_update():
    with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. We need a cell to compile and run DraftDatasetGenerator
    draft_gen_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess\n",
            "import os\n",
            "\n",
            "compile_draft = [\n",
            "    \"javac\", \"-d\", \"out\", \"-cp\", \"lib/arena-framework.jar:out\",\n",
            "    \"src/student/StudentBotImpl.java\", \"src/student/DraftDatasetGenerator.java\"\n",
            "]\n",
            "print(\"Compiling DraftDatasetGenerator...\")\n",
            "subprocess.run(compile_draft, check=True)\n",
            "\n",
            "run_draft = [\n",
            "    \"java\", \"-cp\", \"lib/arena-framework.jar:out\", \"student.DraftDatasetGenerator\"\n",
            "]\n",
            "print(\"Running DraftDatasetGenerator (Simulating 20,000 matches for drafting)... This may take 2-3 minutes.\")\n",
            "subprocess.run(run_draft, check=True)\n",
            "print(\"draft_dataset.csv generated!\")\n"
        ]
    }

    # 2. Cell to train MLP
    mlp_train_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "import numpy as np\n",
            "\n",
            "class DraftDataset(Dataset):\n",
            "    def __init__(self, csv_file):\n",
            "        X_list, y_list = [], []\n",
            "        with open(csv_file, 'r') as f:\n",
            "            header = f.readline()\n",
            "            for line in f:\n",
            "                if not line.strip(): continue\n",
            "                parts = line.split(',')\n",
            "                if len(parts) != 14: continue\n",
            "                budget = float(parts[0]) / 50.0\n",
            "                counts = [float(p) / 8.0 for p in parts[1:13]]\n",
            "                result = float(parts[13])\n",
            "                X_list.append([budget] + counts)\n",
            "                y_list.append([result])\n",
            "        self.X = torch.tensor(X_list, dtype=torch.float32)\n",
            "        self.y = torch.tensor(y_list, dtype=torch.float32)\n",
            "        \n",
            "    def __len__(self):\n",
            "        return len(self.X)\n",
            "        \n",
            "    def __getitem__(self, idx):\n",
            "        return self.X[idx], self.y[idx]\n",
            "\n",
            "class DraftMLP(nn.Module):\n",
            "    def __init__(self):\n",
            "        super(DraftMLP, self).__init__()\n",
            "        self.fc1 = nn.Linear(13, 64)\n",
            "        self.fc2 = nn.Linear(64, 64)\n",
            "        self.fc3 = nn.Linear(64, 1)\n",
            "        \n",
            "    def forward(self, x):\n",
            "        x = torch.relu(self.fc1(x))\n",
            "        x = torch.relu(self.fc2(x))\n",
            "        return torch.sigmoid(self.fc3(x))\n",
            "\n",
            "print(\"Loading DraftDataset...\")\n",
            "draft_ds = DraftDataset(\"draft_dataset.csv\")\n",
            "draft_loader = DataLoader(draft_ds, batch_size=512, shuffle=True)\n",
            "\n",
            "draft_mlp = DraftMLP().to(device)\n",
            "criterion_mlp = nn.BCELoss()\n",
            "optimizer_mlp = optim.Adam(draft_mlp.parameters(), lr=0.005)\n",
            "\n",
            "print(\"Training Draft MLP...\")\n",
            "for epoch in range(10):\n",
            "    total_loss = 0\n",
            "    for bx, by in draft_loader:\n",
            "        bx, by = bx.to(device), by.to(device)\n",
            "        optimizer_mlp.zero_grad()\n",
            "        pred = draft_mlp(bx)\n",
            "        loss = criterion_mlp(pred, by)\n",
            "        loss.backward()\n",
            "        optimizer_mlp.step()\n",
            "        total_loss += loss.item() * bx.size(0)\n",
            "    print(f\"Epoch {epoch+1}/10 | Loss: {total_loss / len(draft_ds):.4f}\")\n",
            "print(\"Draft MLP Training Complete!\")\n"
        ]
    }

    # 3. Modify the Java Export cell to include BOTH CNN and MLP weights
    # We find the base64 export cell
    export_idx = -1
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if "import base64" in src and "CNN" in src:
            export_idx = i
            break
            
    if export_idx == -1:
        print("Could not find export cell!")
        return

    export_source = r"""import base64

# --- 1. Export CNN Weights ---
HC = 1024  # hidden channels
IC = 14    # input channels

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

    private static void initWeights() {{
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
        
        W3 = new float[1][HC][1][1];
        for(int j=0;j<HC;j++) W3[0][j][0][0] = buffer.getFloat();
        B3 = new float[1];
        B3[0] = buffer.getFloat();
        
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
                for (ChampionTemplate a : available) if (spent + a.cost() <= budget - spent) affordable.add(a);
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

            // Build 14-channel input tensor
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
                    input[10][r][c] = (float) a.defense() / 10.0f;
                    input[12][r][c] = (float) a.range() / 5.0f;
                }}
            }}
            for (ChampionSnapshot e : view.enemies()) {{
                if (e.alive()) {{
                    int r = e.position().row(), c = e.position().col();
                    input[2][r][c] = 1.0f;
                    input[4][r][c] = (float) e.hp() / e.maxHp();
                    input[7][r][c] = (e.maxMana() > 0) ? (float) e.mana() / e.maxMana() : 0.0f;
                    input[9][r][c] = (float) e.attack() / 10.0f;
                    input[11][r][c] = (float) e.defense() / 10.0f;
                    input[13][r][c] = (float) e.range() / 5.0f;
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
            float bestScoreAtt = -999999.0f;
            for (ChampionSnapshot enemy : view.enemies()) {{
                if (!enemy.alive()) continue;
                int dist = ally.position().manhattanDistance(enemy.position());
                if (dist <= ally.range()) {{
                    float s = scores[0][enemy.position().row()][enemy.position().col()];
                    if (s > bestScoreAtt) {{ bestScoreAtt = s; bestEnemy = enemy; }}
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
    nb["cells"][export_idx]["source"] = export_source.splitlines(True)

    # Insert draft gen and MLP train BEFORE the export cell
    nb["cells"].insert(export_idx, mlp_train_cell)
    nb["cells"].insert(export_idx, draft_gen_cell)

    with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook successfully updated with MLP architecture.")

if __name__ == "__main__":
    build_notebook_update()
