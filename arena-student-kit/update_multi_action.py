import json

def update_dataset_gen():
    with open("src/student/DatasetGenerator.java", "r") as f:
        code = f.read()
    
    # Extract action class
    # Find the targetPos == null check, and insert actionClass calculation
    target_block = """            // We only record actions that target a specific cell on the board
            if (targetPos == null) continue;

            // Build features"""
            
    repl_block = """            // We only record actions that target a specific cell on the board
            if (targetPos == null) continue;
            
            int actionClass = 0;
            if (action.actionType() == ActionType.ATTACK) actionClass = 1;
            else if (action.actionType() == ActionType.CAST_SKILL) actionClass = 2;

            // Build features"""
    
    if "int actionClass = 0;" not in code:
        code = code.replace(target_block, repl_block)
        
        # Change CSV Header
        code = code.replace(
            'writer.write("active_row,active_col,ally_pos,enemy_pos,ally_hp,enemy_hp,dist_map,ally_mana,enemy_mana,ally_atk,enemy_atk,ally_def,enemy_def,ally_range,enemy_range,target_row,target_col\\n");',
            'writer.write("active_row,active_col,ally_pos,enemy_pos,ally_hp,enemy_hp,dist_map,ally_mana,enemy_mana,ally_atk,enemy_atk,ally_def,enemy_def,ally_range,enemy_range,action_class,target_row,target_col\\n");'
        )
        
        # Change format output
        code = code.replace(
            """String csvLine = String.format("%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d\\n",
                    activeRow, activeCol,
                    allyPosStr, enemyPosStr,
                    allyHpStr, enemyHpStr,
                    distMapStr,
                    allyManaStr, enemyManaStr,
                    allyAtkStr, enemyAtkStr,
                    allyDefStr, enemyDefStr,
                    allyRangeStr, enemyRangeStr,
                    targetPos.row(), targetPos.col());""",
            """String csvLine = String.format("%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d\\n",
                    activeRow, activeCol,
                    allyPosStr, enemyPosStr,
                    allyHpStr, enemyHpStr,
                    distMapStr,
                    allyManaStr, enemyManaStr,
                    allyAtkStr, enemyAtkStr,
                    allyDefStr, enemyDefStr,
                    allyRangeStr, enemyRangeStr,
                    actionClass, targetPos.row(), targetPos.col());"""
        )
        
        with open("src/student/DatasetGenerator.java", "w") as f:
            f.write(code)
        print("DatasetGenerator updated for Multi-Action.")

def update_notebook():
    with open("train_cnn_bot.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        
        # 1. Dataset parsing
        if "class StreamingArenaDataset" in src:
            cell_source = src.replace("len(parts) != 17", "len(parts) != 18")
            cell_source = cell_source.replace(
                "target_row = int(parts[15])\n                target_col = int(parts[16])\n                target_class = target_row * 8 + target_col",
                "action_class = int(parts[15])\n                target_row = int(parts[16])\n                target_col = int(parts[17])\n                target_class = action_class * 64 + target_row * 8 + target_col"
            )
            cell["source"] = cell_source.splitlines(True)
            
        # 2. CNN Model Output channels
        elif "class CNNBotModel" in src:
            cell_source = src.replace("self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=1)", "self.conv3 = nn.Conv2d(hidden_channels, 3, kernel_size=1)")
            cell_source = cell_source.replace("out.size(0), -1)  # [Batch, 64]", "out.size(0), -1)  # [Batch, 192]")
            cell["source"] = cell_source.splitlines(True)
            
        # 3. Java Export logic
        elif "import base64" in src and "CNN" in src:
            # Change Java dimension from 1 to 3
            cell_source = src.replace("W3 = new float[1][HC][1][1];", "W3 = new float[3][HC][1][1];")
            cell_source = cell_source.replace("B3 = new float[1];", "B3 = new float[3];")
            cell_source = cell_source.replace(
                "for(int j=0;j<HC;j++) W3[0][j][0][0] = buffer.getFloat();\n        B3[0] = buffer.getFloat();",
                "for(int i=0;i<3;i++) for(int j=0;j<HC;j++) W3[i][j][0][0] = buffer.getFloat();\n        for(int i=0;i<3;i++) B3[i] = buffer.getFloat();"
            )
            cell_source = cell_source.replace("private static float[][][][] W3;\n    private static float[] B3;", "private static float[][][][] W3;\n    private static float[] B3;")
            
            # Change action logic
            old_logic = """            // Decision: attack enemy with highest CNN score in range
            ChampionSnapshot bestEnemy = null;
            float bestScoreAtt = -999999.0f;
            for (ChampionSnapshot enemy : view.enemies()) {
                if (!enemy.alive()) continue;
                int dist = ally.position().manhattanDistance(enemy.position());
                if (dist <= ally.range()) {
                    float s = scores[0][enemy.position().row()][enemy.position().col()];
                    if (s > bestScoreAtt) { bestScoreAtt = s; bestEnemy = enemy; }
                }
            }
            if (bestEnemy != null) {
                if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                    if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH"))
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestEnemy.position(), null));
                    else
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, bestEnemy.position(), bestEnemy.championId()));
                } else {
                    actions.add(new TurnAction(ally.championId(), ActionType.ATTACK, null, bestEnemy.championId()));
                }
                continue;
            }

            // Dash skills (Assassin, Lancer)
            ChampionSnapshot dashEnemy = null;
            float bestDashScore = -999999.0f;
            if (ally.mana() >= 3 && ally.remainingCooldown() == 0 && (name.equals("ASSASSIN") || name.equals("LANCER"))) {
                for (ChampionSnapshot enemy : view.enemies()) {
                    if (!enemy.alive()) continue;
                    int dist = ally.position().manhattanDistance(enemy.position());
                    if (dist <= 3) {
                        float s = scores[0][enemy.position().row()][enemy.position().col()];
                        if (s > bestDashScore) { bestDashScore = s; dashEnemy = enemy; }
                    }
                }
            }
            if (dashEnemy != null) {
                actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, dashEnemy.position(), dashEnemy.championId()));
                continue;
            }

            // Move towards best scored cell
            Position bestMove = null;
            float bestMoveScore = -999999.0f;
            int[] dr = {-1, 1, 0, 0};
            int[] dc = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nr = activeRow + dr[d], nc = activeCol + dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                    boolean occupied = false;
                    for (ChampionSnapshot a : view.allies()) if (a.alive() && a.position().row() == nr && a.position().col() == nc) occupied = true;
                    for (ChampionSnapshot e : view.enemies()) if (e.alive() && e.position().row() == nr && e.position().col() == nc) occupied = true;
                    if (!occupied) {
                        float s = scores[0][nr][nc];
                        if (s > bestMoveScore) { bestMoveScore = s; bestMove = new Position(nr, nc); }
                    }
                }
            }
            if (bestMove != null) actions.add(new TurnAction(ally.championId(), ActionType.MOVE, bestMove, null));
            else actions.add(TurnAction.waitAction(ally.championId()));
        }"""
            
            # The Healer logic block as well
            healer_logic = """            // Healer logic
            if (name.equals("CLERIC") || name.equals("DRUID")) {
                ChampionSnapshot worstAlly = null;
                double lowestHpRatio = 1.0;
                for (ChampionSnapshot o : view.allies()) {
                    if (!o.alive()) continue;
                    double r = (double) o.hp() / o.maxHp();
                    if (r < lowestHpRatio) { lowestHpRatio = r; worstAlly = o; }
                }
                if (worstAlly != null && lowestHpRatio < 0.8 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                    int dist = ally.position().manhattanDistance(worstAlly.position());
                    if (dist <= ally.range()) {
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, worstAlly.position(), worstAlly.championId()));
                        continue;
                    }
                }
            }"""
            
            # Completely clean up the action code. We define a helper isOccupied inside the loop since we can't add a method easily inside the export string if we just replace. Actually we can define a method but local logic is fine.
            new_logic = """            float bestScore = -999999.0f;
            ActionType bestActionType = ActionType.WAIT;
            Position bestPos = null;
            String bestTargetId = null;

            // 1. Move (Channel 0)
            int[] dr = {-1, 1, 0, 0};
            int[] dc = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nr = activeRow + dr[d], nc = activeCol + dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                    boolean occupied = false;
                    for (ChampionSnapshot a : view.allies()) if (a.alive() && a.position().row() == nr && a.position().col() == nc) occupied = true;
                    for (ChampionSnapshot e : view.enemies()) if (e.alive() && e.position().row() == nr && e.position().col() == nc) occupied = true;
                    if (!occupied) {
                        float s = scores[0][nr][nc];
                        if (s > bestScore) {
                            bestScore = s;
                            bestActionType = ActionType.MOVE;
                            bestPos = new Position(nr, nc);
                            bestTargetId = null;
                        }
                    }
                }
            }

            // 2. Attack (Channel 1)
            for (ChampionSnapshot enemy : view.enemies()) {
                if (!enemy.alive()) continue;
                int dist = ally.position().manhattanDistance(enemy.position());
                if (dist <= ally.range()) {
                    float s = scores[1][enemy.position().row()][enemy.position().col()];
                    if (s > bestScore) {
                        bestScore = s;
                        bestActionType = ActionType.ATTACK;
                        bestPos = enemy.position();
                        bestTargetId = enemy.championId();
                    }
                }
            }

            // 3. Cast Skill (Channel 2)
            if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                for (int r = 0; r < 8; r++) {
                    for (int c = 0; c < 8; c++) {
                        float s = scores[2][r][c];
                        if (s > bestScore) {
                            String tid = null;
                            for (ChampionSnapshot enemy : view.enemies()) if (enemy.alive() && enemy.position().row() == r && enemy.position().col() == c) tid = enemy.championId();
                            for (ChampionSnapshot a : view.allies()) if (a.alive() && a.position().row() == r && a.position().col() == c) tid = a.championId();
                            bestScore = s;
                            bestActionType = ActionType.CAST_SKILL;
                            bestPos = new Position(r, c);
                            bestTargetId = tid;
                        }
                    }
                }
            }

            if (bestActionType == ActionType.WAIT) {
                actions.add(TurnAction.waitAction(ally.championId()));
            } else {
                actions.add(new TurnAction(ally.championId(), bestActionType, bestPos, bestTargetId));
            }
        }"""
            
            cell_source = cell_source.replace(healer_logic, "")
            cell_source = cell_source.replace(old_logic, new_logic)
            cell["source"] = cell_source.splitlines(True)
            
    with open("train_cnn_bot.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated for Multi-Action CNN.")

if __name__ == "__main__":
    update_dataset_gen()
    update_notebook()
