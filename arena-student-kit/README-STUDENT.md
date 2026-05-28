# Arena Bot Student Starter

This package contains the student-facing starter for the turn-based arena challenge.

Also read:

- `CHALLENGE.md`

## Overview

Your team will implement one Java bot that plays a tactical arena game.

Your bot must handle 3 phases:

1. `draftTeam(...)`
2. `placeTeam(...)`
3. `playTurn(...)`

So your job is not only to fight well. You must also:

- choose a good team under budget
- place champions well on the board
- decide actions every round

## What You Need To Edit

Only edit:

- `src/student/StudentBotImpl.java`

Do not rename the package or class unless your instructor tells you to.

## What Is Hidden

The framework is already compiled into:

- `lib/arena-framework.jar`

That JAR contains:

- the GUI
- the arena engine
- the public API
- the built-in `SimpleBot`
- the built-in `IntermediateBot`

## Current Local Champion Pool

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

- gold cost
- maximum HP
- attack
- defense
- attack range
- move range
- speed
- maximum mana
- one skill
- mana cost of the skill
- skill cooldown

## Compile Your Bot

macOS / Linux:

```bash
./compile.sh
```

Windows:

```bat
compile.bat
```

This creates:

- `out/student/StudentBotImpl.class`

## Run The GUI

macOS / Linux:

```bash
./run_gui.sh
```

Windows:

```bat
run_gui.bat
```

The GUI lets you:

- test your bot
- choose whether your bot plays as `BLUE` or `RED`
- choose the opponent bot
- change the random seed
- view the shared champion pool
- step round by round
- run or pause the battle

## Test Against Built-In Bots

Inside the GUI, use the `Opponent` dropdown to select:

- `arenachallenge.bots.SimpleBot`
- `arenachallenge.bots.IntermediateBot`

Recommended workflow:

1. first make your bot reliably beat `SimpleBot`
2. then improve it to compete with `IntermediateBot`

## VS Code

Recommended:

- install a JDK 17 or newer
- open the `arena-student-kit` folder in VS Code
- use `Terminal` to run the scripts above

This package also includes:

- `.vscode/tasks.json`
- `.vscode/launch.json`

So in VS Code you can:

- run the build task: `Compile Arena Bot`
- run the launch configuration: `Run Arena GUI`

## Bot API

Look at these API types:

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

### 1. `draftTeam(DraftView view)`

This method decides which champions your team buys.

Useful data:

- `view.budget()`
- `view.yourSide()`
- `view.availableChampions()`

Your bot should:

- choose a legal team
- keep total cost within the budget
- optionally choose a different strategy for `BLUE` and `RED`
- think about team balance

To return a draft choice, create:

- `new ChampionPick(templateId)`

Example:

```java
new ChampionPick("KNIGHT")
```

### 2. `placeTeam(PlacementView view)`

This method decides where your drafted champions start on the board.

Useful data:

- `view.rows()`
- `view.cols()`
- `view.yourSide()`
- `view.team()`
- `view.allowedCells()`

Your bot should:

- place tanks, damage dealers, and support units sensibly
- use only the allowed starting cells

To place one drafted champion, create:

- `new Placement(championId, position)`

Example:

```java
new Placement("BLUE-1", new Position(2, 1))
```

### 3. `playTurn(BattleView view)`

This method decides actions during battle.

Useful data:

- `view.roundNumber()`
- `view.rows()`
- `view.cols()`
- `view.yourSide()`
- `view.allies()`
- `view.enemies()`
- `view.recentLog()`

Important:

- `playTurn(...)` returns `List<TurnAction>`
- each alive champion may act once in the round
- your bot may choose actions for multiple champions in the same round
- if your bot gives no action for a champion, that champion effectively does nothing

`BattleView` does not directly move or attack for you. It only gives the current state. Your bot must inspect that state and then build the `TurnAction` list itself.

## Actions

Your bot can return these action types:

- `ActionType.MOVE`
- `ActionType.ATTACK`
- `ActionType.DEFEND`
- `ActionType.CAST_SKILL`
- `ActionType.WAIT`

Useful helper:

- `TurnAction.waitAction(championId)`

## API And Helper Types

### `ChampionPick`

`ChampionPick` is used only in the draft phase.

Field:

- `templateId`

Meaning:

- the ID of the champion type your bot wants to buy

### `Placement`

`Placement` is used only in the placement phase.

Fields:

- `championId`
- `position`

Meaning:

- which drafted champion is being placed
- which starting cell that champion should use

### `Position`

`Position` stores a board cell:

- `row`
- `col`

Useful helper:

- `manhattanDistance(other)`

This is useful for:

- checking how close enemies are
- choosing targets
- moving toward or away from a cell

### `TeamSide`

Possible values:

- `TeamSide.BLUE`
- `TeamSide.RED`

Useful helper:

- `yourSide.opponent()`

This is useful when your bot wants to reason about the other team.

## How To Build A `TurnAction`

`TurnAction` has 4 parts:

- `championId`
- `actionType`
- `targetPosition`
- `targetChampionId`

Different action types use different fields:

- `MOVE`
  - use `targetPosition`
  - `targetChampionId` should be `null`

- `ATTACK`
  - use `targetChampionId`
  - `targetPosition` should be `null`

- `DEFEND`
  - both target fields can be `null`

- `CAST_SKILL`
  - some skills use `targetChampionId`
  - some skills use `targetPosition`
  - some self-cast skills use neither

- `WAIT`
  - both target fields can be `null`

Example:

```java
new TurnAction(championId, ActionType.ATTACK, null, enemyId)
```

## What Your Bot Must Decide Each Round

For each alive champion, think about:

- should this champion move?
- should this champion attack?
- should this champion defend?
- should this champion cast a skill?
- if casting, who or where is the target?

Good round logic often includes:

- choosing the best target
- checking whether an enemy is in range
- checking whether mana is enough
- checking whether skill cooldown is ready
- deciding whether to protect weak allies

## Skill Targeting

Skills are not all the same.

Some are:

- single-target enemy skills
- single-target ally skills
- area-of-effect skills using a board position
- self-cast or self-centered skills

So your bot must know whether a skill expects:

- a champion target
- a board cell target
- or no explicit target

## Useful Champion Data

From `ChampionTemplate` or `ChampionSnapshot`, your bot can inspect:

- `displayName`
- `role`
- `cost`
- `hp` / `maxHp`
- `mana` / `maxMana`
- `attack`
- `defense`
- `range`
- `moveRange`
- `speed`
- `remainingCooldown`
- `skillName`
- `skillDescription`
- `position`
- `alive`
- `defending`

## Function Summary

The 3 main bot functions behave like this:

- `getBotName()`
  - returns your team name shown in the simulator

- `draftTeam(DraftView view)`
  - returns `List<ChampionPick>`
  - decides which champion types to buy

- `placeTeam(PlacementView view)`
  - returns `List<Placement>`
  - decides where drafted champions start

- `playTurn(BattleView view)`
  - returns `List<TurnAction>`
  - decides what each alive champion does in the current round

## Local Rules Summary

- both teams receive the same random budget
- the budget is usually between `10` and `15`
- each side can draft up to `8` champions
- the board size is `8x8`
- each alive champion may act once per round
- battles end when one side loses all champions or the round limit is reached

## Example Seeds

Try these seeds:

- `1000`
- `1001`
- `1002`
- `1003`
- `1004`

Different seeds change the battle budget.

## Submission Suggestion

Submit only:

- `src/student/StudentBotImpl.java`
