package student;

import arenachallenge.api.*;
import java.util.*;
import java.util.stream.Collectors;

public class HeuristicBot implements StudentBot {
    private static final int LOCAL_MAX_TEAM_SIZE = 8;

    @Override
    public String getBotName() {
        return "Heuristic Expert Bot";
    }

    @Override
    public List<ChampionPick> draftTeam(DraftView view) {
        List<ChampionTemplate> available = view.availableChampions();
        int budget = view.budget();

        // Count current picks
        List<ChampionPick> picks = new ArrayList<>();
        int spent = 0;

        // Categorize templates
        Map<String, ChampionTemplate> templates = available.stream()
                .collect(Collectors.toMap(ChampionTemplate::id, t -> t));

        // We want a balanced team:
        // Priority order of templates to pick:
        // 1. ARCHER (ranged, cheap, good speed)
        // 2. KNIGHT (great tank, cheap)
        // 3. CLERIC (healer, essential)
        // 4. PALADIN (tank/atk hybrid)
        // 5. MAGE (ranged splash)
        // 6. WARLOCK / FROST_WITCH (ranged)
        
        List<String> priorityList = Arrays.asList(
            "ARCHER", "KNIGHT", "CLERIC", "PALADIN", "MAGE", "WARLOCK", "FROST_WITCH", "LANCER", "ASSASSIN", "DRUID", "BERSERKER", "GUARDIAN"
        );

        // Keep drafting while we have budget and space
        int healerCount = 0;
        int tankCount = 0;
        int archerCount = 0;

        while (picks.size() < LOCAL_MAX_TEAM_SIZE) {
            String bestChoice = null;
            int bestCost = 999;

            for (String id : priorityList) {
                ChampionTemplate t = templates.get(id);
                if (t == null) continue;
                if (spent + t.cost() > budget) continue;

                // Constraints:
                if ((id.equals("CLERIC") || id.equals("DRUID")) && healerCount >= 1) continue;
                if ((id.equals("KNIGHT") || id.equals("PALADIN") || id.equals("GUARDIAN")) && tankCount >= 2) continue;
                if (id.equals("ARCHER") && archerCount >= 3) continue;

                bestChoice = id;
                bestCost = t.cost();
                break;
            }

            // If we couldn't find a choice satisfying constraints, relax them
            if (bestChoice == null) {
                for (String id : priorityList) {
                    ChampionTemplate t = templates.get(id);
                    if (t == null) continue;
                    if (spent + t.cost() > budget) continue;

                    bestChoice = id;
                    bestCost = t.cost();
                    break;
                }
            }

            if (bestChoice == null) {
                // If still null, just find the absolute cheapest available
                final int remaining = budget - spent;
                ChampionTemplate cheapest = available.stream()
                        .filter(t -> t.cost() <= remaining)
                        .min(Comparator.comparingInt(ChampionTemplate::cost))
                        .orElse(null);
                if (cheapest != null) {
                    bestChoice = cheapest.id();
                    bestCost = cheapest.cost();
                }
            }

            if (bestChoice == null) {
                break; // No more money or no templates fit budget
            }

            picks.add(new ChampionPick(bestChoice));
            spent += bestCost;

            if (bestChoice.equals("CLERIC") || bestChoice.equals("DRUID")) healerCount++;
            if (bestChoice.equals("KNIGHT") || bestChoice.equals("PALADIN") || bestChoice.equals("GUARDIAN")) tankCount++;
            if (bestChoice.equals("ARCHER")) archerCount++;
        }

        return picks;
    }

    @Override
    public List<Placement> placeTeam(PlacementView view) {
        List<Placement> placements = new ArrayList<>();
        List<ChampionSnapshot> team = view.team();
        List<Position> allowed = new ArrayList<>(view.allowedCells());
        TeamSide side = view.yourSide();

        // Sort allowed cells based on proximity to enemy:
        // BLUE team starts on the left. So cells closer to enemy have higher column indices.
        // RED team starts on the right. So cells closer to enemy have lower column indices.
        if (side == TeamSide.BLUE) {
            // Frontline has high col, backline has low col
            allowed.sort((a, b) -> Integer.compare(b.col(), a.col()));
        } else {
            // Frontline has low col, backline has high col
            allowed.sort(Comparator.comparingInt(Position::col));
        }

        // Separate tanks and non-tanks
        List<ChampionSnapshot> tanks = new ArrayList<>();
        List<ChampionSnapshot> others = new ArrayList<>();

        for (ChampionSnapshot c : team) {
            String role = c.displayName();
            if (role.equals("KNIGHT") || role.equals("PALADIN") || role.equals("GUARDIAN") || role.equals("BERSERKER") || role.equals("LANCER") || role.equals("ASSASSIN")) {
                tanks.add(c);
            } else {
                others.add(c);
            }
        }

        // Place tanks first (closer to enemy)
        int cellIndex = 0;
        for (ChampionSnapshot tank : tanks) {
            if (cellIndex < allowed.size()) {
                placements.add(new Placement(tank.championId(), allowed.get(cellIndex++)));
            }
        }
        // Place others next
        for (ChampionSnapshot other : others) {
            if (cellIndex < allowed.size()) {
                placements.add(new Placement(other.championId(), allowed.get(cellIndex++)));
            }
        }

        return placements;
    }

    @Override
    public List<TurnAction> playTurn(BattleView view) {
        List<TurnAction> actions = new ArrayList<>();
        TeamSide side = view.yourSide();

        for (ChampionSnapshot ally : view.allies()) {
            if (!ally.alive()) continue;

            String name = ally.displayName();

            // Healer logic (Cleric / Druid)
            if (name.equals("CLERIC") || name.equals("DRUID")) {
                // Find lowest HP percentage ally
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

                // If an ally is below 85% health, try to heal them
                if (worstAlly != null && lowestHpRatio < 0.85 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                    int dist = ally.position().manhattanDistance(worstAlly.position());
                    if (dist <= ally.range()) {
                        // Cast healing skill
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, worstAlly.position(), worstAlly.championId()));
                        continue;
                    } else {
                        // Move closer to heal
                        Position step = stepToward(ally.position(), worstAlly.position());
                        actions.add(new TurnAction(ally.championId(), ActionType.MOVE, step, null));
                        continue;
                    }
                }
            }

            // Offensive skill logic
            ChampionSnapshot target = weakestEnemyInRange(ally, view.enemies(), ally.range());
            if (target != null) {
                // We have an enemy in range!
                if (ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                    // Use skill
                    // Mage, Warlock, Frost Witch skills target a position on board
                    if (name.equals("MAGE") || name.equals("WARLOCK") || name.equals("FROST_WITCH")) {
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, target.position(), null));
                    } else {
                        actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, target.position(), target.championId()));
                    }
                } else {
                    // Attack
                    actions.add(new TurnAction(ally.championId(), ActionType.ATTACK, null, target.championId()));
                }
                continue;
            }

            // If no enemy in range, move towards the nearest enemy
            ChampionSnapshot nearest = nearestEnemy(ally, view.enemies());
            if (nearest != null) {
                // If Assassin or Lancer, check if skill can cast (high range or dash)
                int dist = ally.position().manhattanDistance(nearest.position());
                if (name.equals("ASSASSIN") && dist <= 3 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                    // Dash Strike targets champion
                    actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, nearest.position(), nearest.championId()));
                } else if (name.equals("LANCER") && dist <= 3 && ally.mana() >= 3 && ally.remainingCooldown() == 0) {
                    // Piercing Charge targets champion
                    actions.add(new TurnAction(ally.championId(), ActionType.CAST_SKILL, nearest.position(), nearest.championId()));
                } else {
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
        if (from.row() < to.row()) {
            return new Position(from.row() + 1, from.col());
        }
        if (from.row() > to.row()) {
            return new Position(from.row() - 1, from.col());
        }
        if (from.col() < to.col()) {
            return new Position(from.row(), from.col() + 1);
        }
        if (from.col() > to.col()) {
            return new Position(from.row(), from.col() - 1);
        }
        return from;
    }
}
