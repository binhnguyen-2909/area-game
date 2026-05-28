package student;

import arenachallenge.api.*;
import arenachallenge.bots.IntermediateBot;
import arenachallenge.bots.SimpleBot;
import arenachallenge.engine.ArenaMatch;
import arenachallenge.engine.BattleConfig;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DatasetGenerator {
    public static boolean recording = false;
    private static BufferedWriter writer;
    private static long recordCount = 0;
    public static List<String> matchBuffer = new ArrayList<>();

    public static void main(String[] args) {
        int totalMatches = 500;
        String outputFilename = "dataset.csv";
        long baseSeed = 1337;

        if (args.length >= 3) {
            try {
                totalMatches = Integer.parseInt(args[0]);
                outputFilename = args[1];
                baseSeed = Long.parseLong(args[2]);
            } catch (Exception e) {
                System.out.println("Invalid arguments. Using defaults.");
            }
        }

        System.out.println("Starting expert dataset generation for " + totalMatches + " matches...");
        recording = true;

        try {
            writer = new BufferedWriter(new FileWriter(outputFilename));
            writer.write("active_row,active_col,ally_pos,enemy_pos,ally_hp,enemy_hp,dist_map,ally_mana,enemy_mana,ally_atk,enemy_atk,ally_def,enemy_def,ally_range,enemy_range,action_class,target_row,target_col\n");
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        StudentBot expert = new StudentBotImpl();
        StudentBot simpleOpponent = new SimpleBot();
        StudentBot intermediateOpponent = new IntermediateBot();

        Random rand = new Random(baseSeed);

        int wins = 0;
        int played = 0;
        while (wins < totalMatches) {
            // Alternate opponent and side
            StudentBot opponent = (played % 2 == 0) ? intermediateOpponent : simpleOpponent;
            boolean expertIsBlue = (played % 4 < 2);

            StudentBot blue = expertIsBlue ? expert : opponent;
            StudentBot red = expertIsBlue ? opponent : expert;

            long seed = rand.nextLong();
            int b = rand.nextInt(96) + 5; // Budget 5 to 100
            BattleConfig config = new BattleConfig(8, 8, b, b, 8, 40, 0, 1);
            ArenaMatch match = new ArenaMatch(config, blue, red, seed);
            
            matchBuffer.clear();

            while (!match.isFinished() && match.getRoundNumber() < config.maxRounds()) {
                match.stepRound();
            }
            
            arenachallenge.engine.MatchResult res = match.buildResult();
            TeamSide expertSide = expertIsBlue ? TeamSide.BLUE : TeamSide.RED;
            if (res.winner() == expertSide) {
                try {
                    for (String line : matchBuffer) {
                        writer.write(line);
                        recordCount++;
                    }
                } catch (IOException e) {}
                wins++;
            }
            played++;
        }

        System.out.printf("Finished playing matches. Collected %d samples.%n", recordCount);

        try {
            writer.close();
            System.out.println("Saved dataset to dataset.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void recordState(BattleView view, List<TurnAction> actions) {
        if (!recording) return;

        TeamSide side = view.yourSide();
        int rows = view.rows();
        int cols = view.cols();

        // Map actions by champion ID
        for (ChampionSnapshot ally : view.allies()) {
            if (!ally.alive()) continue;

            // Find action for this ally
            TurnAction action = null;
            for (TurnAction act : actions) {
                if (act.championId().equals(ally.championId())) {
                    action = act;
                    break;
                }
            }

            if (action == null) continue;

            Position targetPos = null;
            if (action.actionType() == ActionType.MOVE) {
                targetPos = action.targetPosition();
            } else if (action.actionType() == ActionType.ATTACK) {
                String enemyId = action.targetChampionId();
                for (ChampionSnapshot enemy : view.enemies()) {
                    if (enemy.championId().equals(enemyId)) {
                        targetPos = enemy.position();
                        break;
                    }
                }
            } else if (action.actionType() == ActionType.CAST_SKILL) {
                if (action.targetPosition() != null) {
                    targetPos = action.targetPosition();
                } else if (action.targetChampionId() != null) {
                    String targetId = action.targetChampionId();
                    // Could be ally or enemy
                    for (ChampionSnapshot enemy : view.enemies()) {
                        if (enemy.championId().equals(targetId)) {
                            targetPos = enemy.position();
                            break;
                        }
                    }
                    if (targetPos == null) {
                        for (ChampionSnapshot otherAlly : view.allies()) {
                            if (otherAlly.championId().equals(targetId)) {
                                targetPos = otherAlly.position();
                                break;
                            }
                        }
                    }
                }
            }

            // We only record actions that target a specific cell on the board
            if (targetPos == null) continue;
            
            int actionClass = 0;
            if (action.actionType() == ActionType.ATTACK) actionClass = 1;
            else if (action.actionType() == ActionType.CAST_SKILL) actionClass = 2;

            // Build features
            int activeRow = ally.position().row();
            int activeCol = ally.position().col();

            int[] allyPos = new int[64];
            int[] enemyPos = new int[64];
            float[] allyHp = new float[64];
            float[] enemyHp = new float[64];
            float[] distMap = new float[64];
            float[] allyMana = new float[64];
            float[] enemyMana = new float[64];
            float[] allyAtk = new float[64];
            float[] enemyAtk = new float[64];
            float[] allyDef = new float[64];
            float[] enemyDef = new float[64];
            float[] allyRange = new float[64];
            float[] enemyRange = new float[64];

            for (ChampionSnapshot a : view.allies()) {
                if (a.alive()) {
                    int idx = a.position().row() * 8 + a.position().col();
                    allyPos[idx] = 1;
                    allyHp[idx] = (float) a.hp() / a.maxHp();
                    allyMana[idx] = (a.maxMana() > 0) ? (float) a.mana() / a.maxMana() : 0.0f;
                    allyAtk[idx] = (float) a.attack() / 10.0f;
                    allyDef[idx] = (float) a.defense() / 10.0f;
                    allyRange[idx] = (float) a.range() / 5.0f;
                }
            }

            for (ChampionSnapshot e : view.enemies()) {
                if (e.alive()) {
                    int idx = e.position().row() * 8 + e.position().col();
                    enemyPos[idx] = 1;
                    enemyHp[idx] = (float) e.hp() / e.maxHp();
                    enemyMana[idx] = (e.maxMana() > 0) ? (float) e.mana() / e.maxMana() : 0.0f;
                    enemyAtk[idx] = (float) e.attack() / 10.0f;
                    enemyDef[idx] = (float) e.defense() / 10.0f;
                    enemyRange[idx] = (float) e.range() / 5.0f;
                }
            }

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int idx = r * 8 + c;
                    int manhattan = Math.abs(r - activeRow) + Math.abs(c - activeCol);
                    distMap[idx] = (float) manhattan / 16.0f;
                }
            }

            // Flatten features to space-separated strings
            String allyPosStr = arrayToString(allyPos);
            String enemyPosStr = arrayToString(enemyPos);
            String allyHpStr = arrayToString(allyHp);
            String enemyHpStr = arrayToString(enemyHp);
            String distMapStr = arrayToString(distMap);
            String allyManaStr = arrayToString(allyMana);
            String enemyManaStr = arrayToString(enemyMana);
            String allyAtkStr = arrayToString(allyAtk);
            String enemyAtkStr = arrayToString(enemyAtk);
            String allyDefStr = arrayToString(allyDef);
            String enemyDefStr = arrayToString(enemyDef);
            String allyRangeStr = arrayToString(allyRange);
            String enemyRangeStr = arrayToString(enemyRange);

            String csvLine = String.format("%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d\n",
                    activeRow, activeCol,
                    allyPosStr, enemyPosStr,
                    allyHpStr, enemyHpStr,
                    distMapStr,
                    allyManaStr, enemyManaStr,
                    allyAtkStr, enemyAtkStr,
                    allyDefStr, enemyDefStr,
                    allyRangeStr, enemyRangeStr,
                    actionClass, targetPos.row(), targetPos.col());

            matchBuffer.add(csvLine);
        }
    }

    private static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(" ");
        }
        return sb.toString();
    }

    private static String arrayToString(float[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            sb.append(String.format(java.util.Locale.US, "%.4f", arr[i]));
            if (i < arr.length - 1) sb.append(" ");
        }
        return sb.toString();
    }
}
