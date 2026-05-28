import json
import re

# 1. Update DatasetGenerator.java to add Defense, Range and explicit randomized budget
with open("src/student/DatasetGenerator.java", "r") as f:
    java_code = f.read()

# Update budget randomization
java_code = java_code.replace(
    "BattleConfig config = BattleConfig.defaultConfig();",
    "int b = rand.nextInt(7) + 10;\n            BattleConfig config = new BattleConfig(8, 8, b, b, 8, 30, 0, 1);"
)

# Update CSV Header
java_code = java_code.replace(
    "writer.write(\"active_row,active_col,ally_pos,enemy_pos,ally_hp,enemy_hp,dist_map,ally_mana,enemy_mana,ally_atk,enemy_atk,target_row,target_col\\n\");",
    "writer.write(\"active_row,active_col,ally_pos,enemy_pos,ally_hp,enemy_hp,dist_map,ally_mana,enemy_mana,ally_atk,enemy_atk,ally_def,enemy_def,ally_range,enemy_range,target_row,target_col\\n\");"
)

# Add arrays for new features
arrays_target = """            float[] allyAtk = new float[64];
            float[] enemyAtk = new float[64];"""
arrays_repl = """            float[] allyAtk = new float[64];
            float[] enemyAtk = new float[64];
            float[] allyDef = new float[64];
            float[] enemyDef = new float[64];
            float[] allyRange = new float[64];
            float[] enemyRange = new float[64];"""
java_code = java_code.replace(arrays_target, arrays_repl)

# Populate new features for allies
ally_target = """                    allyAtk[idx] = (float) a.attack() / 10.0f;
                }"""
ally_repl = """                    allyAtk[idx] = (float) a.attack() / 10.0f;
                    allyDef[idx] = (float) a.defense() / 10.0f;
                    allyRange[idx] = (float) a.range() / 5.0f;
                }"""
java_code = java_code.replace(ally_target, ally_repl)

# Populate new features for enemies
enemy_target = """                    enemyAtk[idx] = (float) e.attack() / 10.0f;
                }"""
enemy_repl = """                    enemyAtk[idx] = (float) e.attack() / 10.0f;
                    enemyDef[idx] = (float) e.defense() / 10.0f;
                    enemyRange[idx] = (float) e.range() / 5.0f;
                }"""
java_code = java_code.replace(enemy_target, enemy_repl)

# String flatten
str_target = """            String allyAtkStr = arrayToString(allyAtk);
            String enemyAtkStr = arrayToString(enemyAtk);"""
str_repl = """            String allyAtkStr = arrayToString(allyAtk);
            String enemyAtkStr = arrayToString(enemyAtk);
            String allyDefStr = arrayToString(allyDef);
            String enemyDefStr = arrayToString(enemyDef);
            String allyRangeStr = arrayToString(allyRange);
            String enemyRangeStr = arrayToString(enemyRange);"""
java_code = java_code.replace(str_target, str_repl)

# CSV Format string
format_target = """            String csvLine = String.format("%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d\\n",
                    activeRow, activeCol,
                    allyPosStr, enemyPosStr,
                    allyHpStr, enemyHpStr,
                    distMapStr,
                    allyManaStr, enemyManaStr,
                    allyAtkStr, enemyAtkStr,
                    targetPos.row(), targetPos.col());"""
format_repl = """            String csvLine = String.format("%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d\\n",
                    activeRow, activeCol,
                    allyPosStr, enemyPosStr,
                    allyHpStr, enemyHpStr,
                    distMapStr,
                    allyManaStr, enemyManaStr,
                    allyAtkStr, enemyAtkStr,
                    allyDefStr, enemyDefStr,
                    allyRangeStr, enemyRangeStr,
                    targetPos.row(), targetPos.col());"""
java_code = java_code.replace(format_target, format_repl)

with open("src/student/DatasetGenerator.java", "w") as f:
    f.write(java_code)

# 2. Update Notebook
with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def find_cell(query):
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if query in src:
            return i
    return -1

# Update Streaming Dataset Cell
idx = find_cell("class StreamingArenaDataset")
cell_source = "".join(nb["cells"][idx]["source"])
cell_source = cell_source.replace("len(parts) != 13", "len(parts) != 17")
cell_source = cell_source.replace("ally_atk = np.array([float(x) for x in parts[9].split()], dtype=np.float32).reshape(8, 8)\n                enemy_atk = np.array([float(x) for x in parts[10].split()], dtype=np.float32).reshape(8, 8)",
                                  "ally_atk = np.array([float(x) for x in parts[9].split()], dtype=np.float32).reshape(8, 8)\n                enemy_atk = np.array([float(x) for x in parts[10].split()], dtype=np.float32).reshape(8, 8)\n                ally_def = np.array([float(x) for x in parts[11].split()], dtype=np.float32).reshape(8, 8)\n                enemy_def = np.array([float(x) for x in parts[12].split()], dtype=np.float32).reshape(8, 8)\n                ally_range = np.array([float(x) for x in parts[13].split()], dtype=np.float32).reshape(8, 8)\n                enemy_range = np.array([float(x) for x in parts[14].split()], dtype=np.float32).reshape(8, 8)")
cell_source = cell_source.replace("ally_mana, enemy_mana, ally_atk, enemy_atk\n                ], axis=0)  # [10, 8, 8]",
                                  "ally_mana, enemy_mana, ally_atk, enemy_atk, ally_def, enemy_def, ally_range, enemy_range\n                ], axis=0)  # [14, 8, 8]")
cell_source = cell_source.replace("target_row = int(parts[11])\n                target_col = int(parts[12])",
                                  "target_row = int(parts[15])\n                target_col = int(parts[16])")
cell_source = cell_source.replace("10 input channels", "14 input channels")
nb["cells"][idx]["source"] = cell_source.splitlines(True)

# Update CNN Model Definition Cell
idx = find_cell("class CNNBotModel")
cell_source = "".join(nb["cells"][idx]["source"])
cell_source = cell_source.replace("in_channels=10", "in_channels=14")
cell_source = cell_source.replace("10 input features", "14 input features")
nb["cells"][idx]["source"] = cell_source.splitlines(True)

# Update Export to Java Cell
idx = find_cell("import base64")
cell_source = "".join(nb["cells"][idx]["source"])
cell_source = cell_source.replace("IC = 10", "IC = 14")
cell_source = cell_source.replace("input[8][r][c] = (float) a.attack() / 10.0f;\n                }",
                                  "input[8][r][c] = (float) a.attack() / 10.0f;\n                    input[10][r][c] = (float) a.defense() / 10.0f;\n                    input[12][r][c] = (float) a.range() / 5.0f;\n                }")
cell_source = cell_source.replace("input[9][r][c] = (float) e.attack() / 10.0f;\n                }",
                                  "input[9][r][c] = (float) e.attack() / 10.0f;\n                    input[11][r][c] = (float) e.defense() / 10.0f;\n                    input[13][r][c] = (float) e.range() / 5.0f;\n                }")
cell_source = cell_source.replace("float[][][] input = new float[IC][8][8];", "float[][][] input = new float[14][8][8];")
nb["cells"][idx]["source"] = cell_source.splitlines(True)

with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Java and Notebook successfully updated with 14 Channels (Defense, Range included) and Explicit Random Budgets.")
