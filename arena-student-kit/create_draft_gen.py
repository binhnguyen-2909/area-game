import os

draft_gen_code = """package student;

import arenachallenge.api.*;
import arenachallenge.engine.ArenaMatch;
import arenachallenge.engine.BattleConfig;
import arenachallenge.engine.MatchResult;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DraftDatasetGenerator {
    static class RandomDraftCNNBot extends StudentBotImpl {
        private Random rand;
        public RandomDraftCNNBot(long seed) {
            super();
            this.rand = new Random(seed);
        }
        @Override
        public List<ChampionPick> draftTeam(DraftView view) {
            List<ChampionTemplate> available = view.availableChampions();
            int budget = view.budget();
            List<ChampionPick> picks = new ArrayList<>();
            int spent = 0;
            List<ChampionTemplate> affordable = new ArrayList<>();
            for (ChampionTemplate t : available) if (t.cost() <= budget) affordable.add(t);
            
            while (picks.size() < 8 && !affordable.isEmpty()) {
                ChampionTemplate t = affordable.get(rand.nextInt(affordable.size()));
                picks.add(new ChampionPick(t.id()));
                spent += t.cost();
                affordable.clear();
                for (ChampionTemplate a : available) if (spent + a.cost() <= budget) affordable.add(a);
            }
            return picks;
        }
    }

    public static void main(String[] args) throws IOException {
        int totalMatches = 20000;
        String outputFilename = "draft_dataset.csv";
        long baseSeed = 42;

        if (args.length >= 3) {
            try {
                totalMatches = Integer.parseInt(args[0]);
                outputFilename = args[1];
                baseSeed = Long.parseLong(args[2]);
            } catch (Exception e) {
                System.out.println("Invalid arguments. Using defaults.");
            }
        }

        System.out.println("Starting draft dataset generation for " + totalMatches + " matches...");
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilename));
        
        String[] ALL_IDS = {"KNIGHT", "ARCHER", "MAGE", "ASSASSIN", "CLERIC", "GUARDIAN", "PALADIN", "WARLOCK", "DRUID", "LANCER", "FROST_WITCH", "BERSERKER"};
        writer.write("budget," + String.join(",", ALL_IDS) + ",result\\n");
        
        Random rand = new Random(baseSeed);
        
        for (int i = 0; i < totalMatches; i++) {
            int budget = rand.nextInt(51); // 0 to 50
            BattleConfig config = new BattleConfig(8, 8, budget, budget, 8, 40, 0, 1);
            
            StudentBot blue = new RandomDraftCNNBot(rand.nextLong());
            StudentBot red = new RandomDraftCNNBot(rand.nextLong());
            
            ArenaMatch match = new ArenaMatch(config, blue, red, rand.nextLong());
            while (!match.isFinished() && match.getRoundNumber() < config.maxRounds()) {
                match.stepRound();
            }
            MatchResult result = match.buildResult();
            
            writeTeam(writer, budget, match.getBlueDrafted(), result.winner() == TeamSide.BLUE ? 1.0f : (result.winner() == null ? 0.5f : 0.0f), ALL_IDS);
            writeTeam(writer, budget, match.getRedDrafted(), result.winner() == TeamSide.RED ? 1.0f : (result.winner() == null ? 0.5f : 0.0f), ALL_IDS);
            
            if ((i + 1) % 5000 == 0) {
                System.out.println("Played " + (i + 1) + " matches...");
            }
        }
        writer.close();
        System.out.println("Finished generating draft dataset.");
    }
    
    private static void writeTeam(BufferedWriter writer, int budget, List<ChampionTemplate> drafted, float result, String[] ALL_IDS) throws IOException {
        int[] counts = new int[ALL_IDS.length];
        for (ChampionTemplate t : drafted) {
            for (int i = 0; i < ALL_IDS.length; i++) {
                if (t.id().equals(ALL_IDS[i])) counts[i]++;
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append(budget);
        for (int c : counts) sb.append(",").append(c);
        sb.append(",").append(result).append("\\n");
        writer.write(sb.toString());
    }
}
"""

with open("src/student/DraftDatasetGenerator.java", "w") as f:
    f.write(draft_gen_code)
print("DraftDatasetGenerator.java created.")
