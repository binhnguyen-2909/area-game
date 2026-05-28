package student;

import arenachallenge.api.StudentBot;
import arenachallenge.api.TeamSide;
import arenachallenge.bots.IntermediateBot;
import arenachallenge.bots.SimpleBot;
import arenachallenge.engine.ArenaMatch;
import arenachallenge.engine.BattleConfig;
import arenachallenge.engine.MatchResult;

import java.util.Random;

public class Simulator {
    public static void main(String[] args) {
        String botClassname = "student.StudentBotImpl";
        if (args.length > 0) {
            botClassname = args[0];
        }

        StudentBot candidate;
        try {
            candidate = (StudentBot) Class.forName(botClassname).getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            System.err.println("Could not instantiate bot " + botClassname);
            e.printStackTrace();
            return;
        }

        System.out.println("Testing bot: " + candidate.getBotName() + " (" + botClassname + ")");

        // Test against SimpleBot
        testMatchup(candidate, new SimpleBot(), "SimpleBot", 50);

        // Test against IntermediateBot
        testMatchup(candidate, new IntermediateBot(), "IntermediateBot", 50);
    }

    private static void testMatchup(StudentBot candidate, StudentBot opponent, String opponentName, int matches) {
        int candidateWins = 0;
        int opponentWins = 0;
        int draws = 0;

        int candidateAsBlueWins = 0;
        int candidateAsBlueMatches = 0;
        int candidateAsRedWins = 0;
        int candidateAsRedMatches = 0;

        Random rand = new Random(42);

        for (int i = 0; i < matches; i++) {
            // Alternate sides
            boolean candidateIsBlue = (i % 2 == 0);
            StudentBot blue = candidateIsBlue ? candidate : opponent;
            StudentBot red = candidateIsBlue ? opponent : candidate;

            long seed = rand.nextLong();
            BattleConfig config = BattleConfig.defaultConfig();

            // ArenaMatch constructor: BattleConfig, blueBot, redBot, seed
            ArenaMatch match = new ArenaMatch(config, blue, red, seed);

            // Run match
            while (!match.isFinished() && match.getRoundNumber() < config.maxRounds()) {
                match.stepRound();
            }

            MatchResult result = match.buildResult();
            TeamSide winner = result.winner();

            if (candidateIsBlue) {
                candidateAsBlueMatches++;
                if (winner == TeamSide.BLUE) {
                    candidateWins++;
                    candidateAsBlueWins++;
                } else if (winner == TeamSide.RED) {
                    opponentWins++;
                } else {
                    draws++;
                }
            } else {
                candidateAsRedMatches++;
                if (winner == TeamSide.RED) {
                    candidateWins++;
                    candidateAsRedWins++;
                } else if (winner == TeamSide.BLUE) {
                    opponentWins++;
                } else {
                    draws++;
                }
            }
        }

        double winRate = (double) candidateWins / matches * 100;
        double blueWinRate = (double) candidateAsBlueWins / candidateAsBlueMatches * 100;
        double redWinRate = (double) candidateAsRedWins / candidateAsRedMatches * 100;

        System.out.printf("Matchup vs %s over %d matches:%n", opponentName, matches);
        System.out.printf("  Total Wins: %d (%.2f%%)%n", candidateWins, winRate);
        System.out.printf("  As BLUE (Goes first): %d/%d (%.2f%%)%n", candidateAsBlueWins, candidateAsBlueMatches,
                blueWinRate);
        System.out.printf("  As RED (Goes second): %d/%d (%.2f%%)%n", candidateAsRedWins, candidateAsRedMatches,
                redWinRate);
        System.out.printf("  Opponent Wins: %d, Draws: %d%n", opponentWins, draws);
    }
}
