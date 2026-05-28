# Arena Bot Challenge Requirements

## Introduction

In this challenge, each student group will design a Java bot for a turn-based tactical arena game.

Each bot must:

1. build a team under a gold budget
2. place champions on the board
3. choose actions every round
4. defeat the opposing team

This challenge is not only about battle. It also tests:

- team composition
- placement strategy
- target selection
- resource management
- skill usage
- round-by-round decision making

## What You Receive

You are given:

- a compiled framework
- a starter bot file
- a local GUI to test your bot
- built-in `SimpleBot` and `IntermediateBot` opponents

You are **not** given:

- the hidden teacher tournament framework
- the final knockout bracket
- any private teacher reference bot

## What You Must Implement

You must implement your solution inside:

- `src/student/StudentBotImpl.java`

Your bot must implement 3 phases:

1. `draftTeam(DraftView view)`
2. `placeTeam(PlacementView view)`
3. `playTurn(BattleView view)`

## Game Flow

### Phase 1: Draft

Each side receives the same random gold budget.

Your bot must choose champions from the available pool without exceeding the budget.

Your bot can also inspect which side it is playing on through:

- `view.yourSide()`

This means you may use a different strategy when your bot is `BLUE` or `RED`.

### Phase 2: Placement

After drafting, your bot chooses legal starting cells for the drafted champions.

### Phase 3: Battle

The battle then proceeds round by round.

During each round:

- each alive champion may act once
- your bot may return actions for multiple champions
- each champion may choose to move, attack, defend, cast a skill, or wait

## Current Local Practice Rules

- board size: `8x8`
- budget range: `10` to `15`
- max team size: `8`
- max rounds: `40`

Available action types:

- `MOVE`
- `ATTACK`
- `DEFEND`
- `CAST_SKILL`
- `WAIT`

## Champion Pool

The local practice framework currently includes:

- `KNIGHT`
- `ARCHER`
- `MAGE`
- `ASSASSIN`
- `CLERIC`
- `GUARDIAN`
- `PALADIN`
- `WARLOCK`
- `DRUID`
- `LANCER`
- `FROST_WITCH`
- `BERSERKER`

Each champion has:

- cost
- HP
- mana
- attack
- defense
- attack range
- move range
- speed
- one skill
- mana cost of the skill
- skill cooldown

Some skills are:

- single-target attacks
- area-of-effect attacks
- self-cast defensive skills
- ally healing or support skills

## Available API

Your bot will mainly use:

- `StudentBot`
- `DraftView`
- `PlacementView`
- `BattleView`
- `ChampionTemplate`
- `ChampionSnapshot`
- `ChampionPick`
- `Placement`
- `TurnAction`
- `ActionType`
- `Position`
- `TeamSide`

### `DraftView`

Gives your bot:

- the budget
- your side
- the available champion templates

### `PlacementView`

Gives your bot:

- board size
- your side
- your drafted team
- allowed starting cells

### `BattleView`

Gives your bot:

- current round number
- board size
- your side
- ally snapshots
- enemy snapshots
- recent battle log

### `TurnAction`

Represents one champion action.

Depending on the action type, your bot may need:

- a target champion
- a target board position
- or no target

### `ChampionPick`

Represents one draft choice.

Your bot returns `List<ChampionPick>` from `draftTeam(...)`.

### `Placement`

Represents one starting position assignment.

Your bot returns `List<Placement>` from `placeTeam(...)`.

### `Position`

Represents one board cell using:

- row
- col

### `TeamSide`

Represents the team side:

- `BLUE`
- `RED`

## What Your Bot Should Do In Each Move

When writing `playTurn(...)`, your bot should think about each alive champion and decide:

- is an enemy already in range?
- is it better to use a normal attack or a skill?
- is there enough mana?
- is the skill cooldown ready?
- should this champion move closer?
- should this champion protect itself or allies?
- who is the best target?
- should the skill target one champion or one position on the board?

## Local Testing Suggestion

Before submission, your bot should be able to:

- compile successfully
- run in the GUI
- play valid battles
- beat or compete with `SimpleBot`
- compete with `IntermediateBot`
- test both as `BLUE` and as `RED`

## Deliverables

Each team submits:

- `StudentBotImpl.java`

Optional:

- a short explanation of the algorithm or strategy
- any design notes or known weaknesses

## Important Advice

- do not hard-code for only one budget or one matchup
- think about both drafting and battle
- use mana and cooldown carefully
- try to balance offense, defense, mobility, and support
- design for many different battles, not just one seed

## Note

This student kit is mainly for local practice. The instructor may adjust tournament settings, bracket format, or evaluation rules for the final competition.
