package student;

import arenachallenge.api.*;
import arenachallenge.bots.IntermediateBot;
import arenachallenge.bots.SimpleBot;
import arenachallenge.engine.ArenaMatch;
import arenachallenge.engine.BattleConfig;
import arenachallenge.engine.MatchResult;

import java.util.*;

public class Optimizer {
    // We will define a few draft strategy IDs and placement strategy IDs, and evaluate them.
    
    static class StrategyBot implements StudentBot {
        int draftMode; // 0: archer-heavy, 1: tank-heavy, 2: balanced
        int placeMode; // 0: tanks front, others back, 1: all back, 2: all front
        boolean useFocusFire; // focus weakest or nearest
        boolean keepDistance; // ranged units step back if enemies too close

        public StrategyBot(int draftMode, int placeMode, boolean useFocusFire, boolean keepDistance) {
            this.draftMode = draftMode;
            this.placeMode = placeMode;
            this.useFocusFire = useFocusFire;
            this.keepDistance = keepDistance;
        }

        @Override
        public String getBotName() {
            return String.format("Bot_D%d_P%d_F%b_K%b", draftMode, placeMode, useFocusFire, keepDistance);
        }

        @Override
        public List<ChampionPick> draftTeam(DraftView view) {
            List<ChampionTemplate> available = view.availableChampions();
            int budget = view.budget();
            List<ChampionPick> picks = new ArrayList<>();
            int spent = 0;

            // Map available
            Map<String, ChampionTemplate> templates = new HashMap<>();
            for (ChampionTemplate t : available) {
                templates.put(t.id(), t);
            }

            if (draftMode == 0) {
                // Archer-heavy
                // Buy as many Archers as possible.
                // If we have budget, buy Healer (Cleric) or Tank (Knight).
                List<String> order = Arrays.asList("ARCHER", "KNIGHT", "CLERIC", "MAGE", "WARLOCK");
                while (picks.size() < 8) {
                    String pick = null;
                    for (String id : order) {
                        ChampionTemplate t = templates.get(id);
                        if (t != null && spent + t.cost() <= budget) {
                            // Constraints
                            long archerCount = picks.stream().filter(p -> p.templateId().equals("ARCHER")).count();
                            long clericCount = picks.stream().filter(p -> p.templateId().equals("CLERIC")).count();
                            if (id.equals("ARCHER") && archerCount >= 5) continue;
                            if (id.equals("CLERIC") && clericCount >= 1) continue;
                            pick = id;
                            break;
                        }
                    }
                    if (pick == null) {
                        // Relax
                        for (String id : order) {
                            ChampionTemplate t = templates.get(id);
                            if (t != null && spent + t.cost() <= budget) {
                                pick = id;
                                break;
                            }
                        }
                    }
                    if (pick == null) break;
                    picks.add(new ChampionPick(pick));
                    spent += templates.get(pick).cost();
                }
            } else if (draftMode == 1) {
                // Tank-heavy
                List<String> order = Arrays.asList("KNIGHT", "PALADIN", "CLERIC", "ARCHER", "GUARDIAN");
                while (picks.size() < 8) {
                    String pick = null;
                    for (String id : order) {
                        ChampionTemplate t = templates.get(id);
                        if (t != null && spent + t.cost() <= budget) {
                            pick = id;
                            break;
                        }
                    }
                    if (pick == null) break;
                    picks.add(new ChampionPick(pick));
                    spent += templates.get(pick).cost();
                }
            } else {
                // Balanced
                List<String> order = Arrays.asList("ARCHER", "KNIGHT", "CLERIC", "PALADIN", "MAGE", "WARLOCK");
                int healerCount = 0;
                int tankCount = 0;
                int archerCount = 0;

                while (picks.size() < 8) {
                    String pick = null;
                    for (String id : order) {
                        ChampionTemplate t = templates.get(id);
                        if (t == null || spent + t.cost() > budget) continue;

                        if (id.equals("CLERIC") && healerCount >= 1) continue;
                        if ((id.equals("KNIGHT") || id.equals("PALADIN")) && tankCount >= 2) continue;
                        if (id.equals("ARCHER") && archerCount >= 3) continue;

                        pick = id;
                        break;
                    }
                    if (pick == null) {
                        for (String id : order) {
                            ChampionTemplate t = templates.get(id);
                            if (t != null && spent + t.cost() <= budget) {
                                pick = id;
                                break;
                            }
                        }
                    }
                    if (pick == null) break;
                    picks.add(new ChampionPick(pick));
                    spent += templates.get(pick).cost();

                    if (pick.equals("CLERIC")) healerCount++;
                    if (pick.equals("KNIGHT") || pick.equals("PALADIN")) tankCount++;
                    if (pick.equals("ARCHER")) archerCount++;
                }
            }

            return picks;
        }

        @Override
        public List<Placement> placeTeam(PlacementView view) {
            List<Placement> placements = new ArrayList<>();
            List<ChampionSnapshot> team = view.team();
            List<Position> allowed = new ArrayList<>(view.allowedCells());
            TeamSide side = view.yourSide();

            if (placeMode == 0) {
                // Tanks front, others back
                if (side == TeamSide.BLUE) {
                    allowed.sort((a, b) -> Integer.compare(b.col(), a.col()));
                } else {
                    allowed.sort(Comparator.comparingInt(Position::col));
                }
                List<ChampionSnapshot> tanks = new ArrayList<>();
                List<ChampionSnapshot> others = new ArrayList<>();
                for (ChampionSnapshot c : team) {
                    if (c.displayName().equals("KNIGHT") || c.displayName().equals("PALADIN") || c.displayName().equals("GUARDIAN")) {
                        tanks.add(c);
                    } else {
                        others.add(c);
                    }
                }
                int idx = 0;
                for (ChampionSnapshot t : tanks) {
                    if (idx < allowed.size()) placements.add(new Placement(t.championId(), allowed.get(idx++)));
                }
                for (ChampionSnapshot o : others) {
                    if (idx < allowed.size()) placements.add(new Placement(o.championId(), allowed.get(idx++)));
                }
            } else if (placeMode == 1) {
                // All back (furthest from enemy)
                if (side == TeamSide.BLUE) {
                    allowed.sort(Comparator.comparingInt(Position::col)); // col 0 first
                } else {
                    allowed.sort((a, b) -> Integer.compare(b.col(), a.col())); // col 7 first
                }
                for (int i = 0; i < team.size() && i < allowed.size(); i++) {
                    placements.add(new Placement(team.get(i).championId(), allowed.get(i)));
                }
            } else {
                // Default (as in allowed order)
                for (int i = 0; i < team.size() && i < allowed.size(); i++) {
                    placements.add(new Placement(team.get(i).championId(), allowed.get(i)));
                }
            }
            return placements;
        }

        @Override
        public List<TurnAction> playTurn(BattleView view) {
            List<TurnAction> actions = new ArrayList<>();
            for (ChampionSnapshot ally : view.allies()) {
                if (!ally.alive()) continue;

                String name = ally.displayName();

                // Healer skill logic
                if (name.equals("CLERIC") || name.equals("DRUID")) {
                    ChampionSnapshot worstAlly = null;
                    double lowestHpRatio = 1.0;
                    for (ChampionSnapshot otherAlly : view.allies()) {
                        if (!otherAlly.alive()) continue;
                        double hpRatio = (double) otherAlly.hp() / otherAlly.maxHp();
                        if (hpRatio < lowestHpRatio) {
                            lowestHpRatio = hpRatio;
                            worstAlly = otherAlly;
                        }
                    }
                    if (worstAlly != null && lowestHpRatio < 0.8 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                        int dist = ally.position().manhattanDistance(worstAlly.position());
                        if (dist <= ally.range()) {
                            actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, worstAlly.position(), worstAlly.championId()));
                            continue;
                        }
                    }
                }

                // Offensive action selection
                ChampionSnapshot target = null;
                if (useFocusFire) {
                    // target weakest enemy in range
                    target = weakestEnemyInRange(ally, view.enemies(), ally.range());
                } else {
                    // target nearest enemy in range
                    target = nearestEnemyInRange(ally, view.enemies(), ally.range());
                }

                if (target != null) {
                    if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                        if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH")) {
                            actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, target.position(), null));
                        } else {
                            actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, target.position(), target.championId()));
                        }
                    } else {
                        actions.add(new TurnAction(ally.championId(), ActionType.ATTACK, null, target.championId()));
                    }
                    continue;
                }

                // No enemy in range: move
                ChampionSnapshot nearest = nearestEnemy(ally, view.enemies());
                if (nearest != null) {
                    int dist = ally.position().manhattanDistance(nearest.position());
                    
                    // Skill check for dash/charge
                    if (name.equals("ASSASSIN") && dist <= 3 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, nearest.position(), nearest.championId()));
                    } else if (name.equals("LANCER") && dist <= 3 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, nearest.position(), nearest.championId()));
                    } else {
                        // Regular movement
                        Position step = stepToward(ally.position(), nearest.position());
                        actions.add(new TurnAction(ally.championId(), ActionType.MOVE, step, null));
                    }
                } else {
                    actions.add(TurnAction.waitAction(ally.championId()));
                }
            }
            return actions;
        }

        private ChampionSnapshot weakestEnemyInRange(ChampionSnapshot actor, List<ChampionSnapshot> enemies, int range) {
            ChampionSnapshot best = null;
            int lowestHp = Integer.MAX_VALUE;
            for (ChampionSnapshot enemy : enemies) {
                if (!enemy.alive()) continue;
                int dist = actor.position().manhattanDistance(enemy.position());
                if (dist <= range) {
                    if (enemy.hp() < lowestHp) {
                        lowestHp = enemy.hp();
                        best = enemy;
                    }
                }
            }
            return best;
        }

        private ChampionSnapshot nearestEnemyInRange(ChampionSnapshot actor, List<ChampionSnapshot> enemies, int range) {
            ChampionSnapshot best = null;
            int bestDist = Integer.MAX_VALUE;
            for (ChampionSnapshot enemy : enemies) {
                if (!enemy.alive()) continue;
                int dist = actor.position().manhattanDistance(enemy.position());
                if (dist <= range) {
                    if (dist < bestDist) {
                        bestDist = dist;
                        best = enemy;
                    }
                }
            }
            return best;
        }

        private ChampionSnapshot nearestEnemy(ChampionSnapshot actor, List<ChampionSnapshot> enemies) {
            ChampionSnapshot best = null;
            int bestDistance = Integer.MAX_VALUE;
            for (ChampionSnapshot enemy : enemies) {
                if (!enemy.alive()) continue;
                int distance = actor.position().manhattanDistance(enemy.position());
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = enemy;
                }
            }
            return best;
        }

        private Position stepToward(Position from, Position to) {
            if (from.row() < to.row()) return new Position(from.row() + 1, from.col());
            if (from.row() > to.row()) return new Position(from.row() - 1, from.col());
            if (from.col() < to.col()) return new Position(from.row(), from.col() + 1);
            if (from.col() > to.col()) return new Position(from.row(), from.col() - 1);
            return from;
        }
    }

    public static void main(String[] args) {
        StudentBot opponent = new IntermediateBot();
        
        // Define hyperparameter grid
        int[] draftModes = {0, 1, 2};
        int[] placeModes = {0, 1};
        boolean[] focusFireModes = {true, false};
        boolean[] keepDistanceModes = {true, false};

        System.out.println("Starting search...");
        
        double bestWinRate = -1.0;
        String bestConfig = "";

        for (int dm : draftModes) {
            for (int pm : placeModes) {
                for (boolean ff : focusFireModes) {
                    for (boolean kd : keepDistanceModes) {
                        StrategyBot bot = new StrategyBot(dm, pm, ff, kd);
                        double winRate = evaluate(bot, opponent, 100); // 100 matches for speed
                        System.out.printf("Config: D%d P%d F%b K%b | Win Rate: %.2f%%%n", dm, pm, ff, kd, winRate);
                        if (winRate > bestWinRate) {
                            bestWinRate = winRate;
                            bestConfig = String.format("D%d P%d F%b K%b", dm, pm, ff, kd);
                        }
                    }
                }
            }
        }
        System.out.printf("Best Config: %s | Best Win Rate: %.2f%%%n", bestConfig, bestWinRate);
    }

    private static double evaluate(StudentBot bot, StudentBot opponent, int matches) {
        int wins = 0;
        Random rand = new Random(42);
        for (int i = 0; i < matches; i++) {
            boolean isBlue = (i % 2 == 0);
            StudentBot blue = isBlue ? bot : opponent;
            StudentBot red = isBlue ? opponent : bot;

            ArenaMatch match = new ArenaMatch(BattleConfig.defaultConfig(), blue, red, rand.nextLong());
            while (!match.isFinished()) {
                match.stepRound();
            }

            TeamSide winner = match.buildResult().winner();
            if (isBlue && winner == TeamSide.BLUE) wins++;
            else if (!isBlue && winner == TeamSide.RED) wins++;
        }
        return (double) wins / matches * 100;
    }
}
