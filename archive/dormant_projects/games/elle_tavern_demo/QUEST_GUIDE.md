# Elle Tavern Demo - Quest Guide

**Complete walkthrough of all quests with emotion-based progression**

Created: 2025-11-16

---

## Table of Contents

1. [Quest System Overview](#quest-system-overview)
2. [Quest Dependency Graph](#quest-dependency-graph)
3. [Quest Walkthroughs](#quest-walkthroughs)
4. [Emotion Requirements](#emotion-requirements)
5. [Reward Tables](#reward-tables)
6. [Tips and Strategies](#tips-and-strategies)

---

## Quest System Overview

The Elle Tavern Demo features **5 interconnected quests** that showcase Elle Game Engine's emotion-based quest generation system.

### Key Features

- **Emotion-Aware**: Quest availability depends on NPC emotional state (valence, trust)
- **Progressive Difficulty**: Easy → Medium → Hard → Epic
- **Interconnected**: Completing quests unlocks others
- **Multiple Outcomes**: Success, partial completion, failure
- **Dynamic Rewards**: Rewards adapt to quest difficulty and NPC emotion

### Quest Philosophy

> "Trust is earned through deeds, not words."

The quest system requires you to build relationships with NPCs through consistent actions. Higher trust unlocks more challenging and rewarding quests.

---

## Quest Dependency Graph

```
                    START
                      |
        +-------------+-------------+
        |                           |
   [Rat Problem]               [Lost Cat]
   (Innkeeper)                  (Child)
   Trust: 0.3                  Trust: 0.2
        |                           |
        |                           |
        +---------> [Lost Shipment] <+
                    (Merchant)
                    Trust: 0.4
              Prerequisites: rat_problem
                        |
                        |
                  [Bandit Trouble]
                     (Guard)
                   Trust: 0.5
          Prerequisites: lost_shipment + lost_cat
                        |
                        |
                  [Hidden Path]
                (Mysterious Stranger)
                 Trust: 0.7 (multi-NPC)
            Prerequisites: bandit_trouble
                        |
                        v
                      END
```

### Quest Unlocking Chain

| Quest | Unlocks | Requires Completed |
|-------|---------|-------------------|
| Rat Problem | Lost Shipment | None |
| Lost Cat | Bandit Trouble | None |
| Lost Shipment | Bandit Trouble | Rat Problem |
| Bandit Trouble | Hidden Path | Lost Shipment + Lost Cat |
| Hidden Path | (Final Quest) | Bandit Trouble |

---

## Quest Walkthroughs

### Quest 1: The Cellar Rat Problem

**Giver**: Bob the Innkeeper
**Location**: Rusty Tankard Inn
**Difficulty**: Trivial
**Recommended Level**: 1
**Estimated Time**: 5-10 minutes

#### Overview

Bob's tavern cellar is infested with rats eating all the food stores. Clear them out to earn his trust.

#### Objectives

1. **Talk to Bob about the rats**
   - Location: Rusty Tankard Inn (main room)
   - Dialogue triggers quest acceptance

2. **Clear 5 rats from cellar**
   - Location: Tavern cellar (behind the bar)
   - Strategy: Rats hide in corners and behind barrels
   - Combat: Weak enemies, single strike defeats them
   - Progress: 0/5 → 5/5

3. **Report back to Bob**
   - Return to Bob in main room
   - Complete quest and receive reward

#### Rewards

- **XP**: 50
- **Gold**: 10
- **Reputation**: Innkeeper +20
- **Unlocks**: Lost Shipment quest (merchant now trusts you)

#### Emotion Changes

- Innkeeper valence: +0.3 (worried → relieved)
- Innkeeper trust: +0.2 (0.3 → 0.5)

#### Sample Dialogue

**Offer** (Bob is worried):
> "Oh thank goodness you're here! The cellar is infested with rats - they're destroying everything! Please, I need your help!"

**Complete** (Bob is grateful):
> "You did it! The cellar is finally clear! I can't thank you enough. Here, take this gold - you've more than earned it. You're a true friend!"

---

### Quest 2: Lily's Lost Cat

**Giver**: Lily (Child)
**Location**: Town Square
**Difficulty**: Easy
**Recommended Level**: 1
**Estimated Time**: 10-15 minutes

#### Overview

Little Lily's cat Whiskers has gone missing. Search the town to find her beloved pet.

#### Objectives

1. **Talk to Lily about Whiskers**
   - Location: Town Square
   - Lily is crying, very sad

2. **Search the tavern**
   - Location: Rusty Tankard Inn
   - Talk to Bob - he saw an orange cat earlier

3. **Search the market square**
   - Location: Market Square
   - Check around the bakery stall

4. **Find Whiskers**
   - **Hint**: Check behind the barrels near the bakery
   - Location: Market Square (hidden behind barrels)
   - Interaction: Pick up cat

5. **Return Whiskers to Lily**
   - Return to Town Square
   - Give cat to Lily

#### Rewards

- **XP**: 75
- **Gold**: 5
- **Items**: Lucky Charm (minor luck boost)
- **Reputation**: Child +50
- **Unlocks**: Bandit Trouble quest (Lily tells guard about you)

#### Emotion Changes

- Child valence: +0.5 (sad → happy)
- Child trust: +0.3 (0.2 → 0.5)

#### Sample Dialogue

**Offer** (Lily is sad/crying):
> "*sobbing* Whiskers is missing! She's orange with white paws and she's my best friend! *wipes tears* Can you help me find her?"

**Complete** (Lily is overjoyed):
> "*hugs cat* WHISKERS! You found her! You found her! *bouncing with joy* Thank you thank you thank you! You're my hero!"

---

### Quest 3: Marcus's Lost Shipment

**Giver**: Marcus the Merchant
**Location**: Market Square
**Difficulty**: Normal
**Recommended Level**: 2
**Estimated Time**: 15-20 minutes

#### Prerequisites

- **Must complete**: Rat Problem quest
- **Innkeeper trust**: ≥0.5 (Bob vouches for you)
- **Merchant trust**: ≥0.4

#### Overview

Bandits stole Marcus's valuable shipment. Recover the goods from their camp.

#### Objectives

1. **Talk to Marcus about the stolen goods**
   - Location: Market Square
   - Marcus is worried about his business

2. **Ask Captain Sarah about the bandits**
   - Location: Town Square (guard post)
   - Sarah provides intel on bandit camp location

3. **Find the bandit camp**
   - Location: Follow forest path north from Town Square
   - Travel: Town Square → Forest Path
   - Discovery: Bandit camp clearing

4. **Recover the stolen goods**
   - Location: Bandit camp (main tent)
   - Strategy: Goods are in main tent, lightly guarded
   - Combat: 2-3 bandits (moderate difficulty)

5. **Return goods to Marcus**
   - Return to Market Square
   - Give goods to Marcus

#### Rewards

- **XP**: 150
- **Gold**: 50
- **Items**: Silver Ring (minor magic item)
- **Reputation**: Merchant +30
- **Unlocks**: Bandit Trouble quest

#### Emotion Changes

- Merchant valence: +0.4 (worried → grateful)
- Merchant trust: +0.4 (0.4 → 0.8)

#### Sample Dialogue

**Offer** (Marcus is worried):
> "You there! You look capable. I've had a valuable shipment stolen by bandits on the forest road. Are you brave enough to recover it?"

**Complete** (Marcus is grateful):
> "My goods! You actually recovered them! I was starting to lose hope. Here - take this silver ring and gold as thanks. You've saved my business!"

---

### Quest 4: Clear the Bandit Camp

**Giver**: Captain Sarah (Guard)
**Location**: Town Square (Guard Post)
**Difficulty**: Hard
**Recommended Level**: 3
**Estimated Time**: 20-30 minutes

#### Prerequisites

- **Must complete**: Lost Shipment + Lost Cat
- **Guard trust**: ≥0.5
- **Merchant trust**: ≥0.6 (vouches for you)
- **Child trust**: ≥0.4 (mentions you to guard)

#### Overview

Permanently clear the bandit camp that's been terrorizing travelers. This is a dangerous combat mission.

#### Objectives

1. **Talk to Captain Sarah**
   - Location: Town Square (Guard Post)
   - Sarah briefs you on the mission

2. **Scout the bandit camp**
   - Location: Forest Path → Bandit Camp
   - Stealth: Observe camp layout
   - Intelligence: Count enemies (8 bandits + 1 leader)

3. **Defeat 8 bandits**
   - Combat: Multiple waves of enemies
   - Strategy: Pull small groups, don't fight all at once
   - Progress: 0/8 → 8/8

4. **Defeat the bandit leader**
   - Boss Fight: Scarred leader (high health, strong attacks)
   - Strategy: Dodge his charge attack, counterattack
   - Reward: Leader drops Iron Sword

5. **Report back to Captain Sarah**
   - Return to Town Square
   - Report mission success

#### Rewards

- **XP**: 300
- **Gold**: 100
- **Items**: Guard Badge, Iron Sword
- **Reputation**: Guard +50
- **World Flags**: bandits_cleared, hero_status
- **Unlocks**: Hidden Path quest

#### Emotion Changes

- Guard valence: +0.5 (serious → proud)
- Guard trust: +0.5 (0.5 → 1.0)

#### Sample Dialogue

**Offer** (Captain Sarah is serious):
> "Adventurer. Those bandits you fought? They're part of a larger problem. I need someone capable to clear out their entire camp permanently. Interested?"

**Complete** (Sarah is impressed):
> "You did it! The bandit camp is destroyed and their leader defeated. I'm impressed - that took real skill and courage. Here's your reward and this guard's badge. You've earned it."

---

### Quest 5: Secrets of the Forest

**Giver**: Mysterious Stranger
**Location**: Forest Path (appears after Bandit Trouble)
**Difficulty**: Epic
**Recommended Level**: 5
**Estimated Time**: 30-45 minutes

#### Prerequisites

- **Must complete**: Bandit Trouble
- **Guard trust**: ≥0.7
- **Innkeeper trust**: ≥0.6
- **Merchant trust**: ≥0.5
- **Mysterious Stranger trust**: ≥0.3

#### Overview

A mysterious stranger reveals ancient secrets hidden in the forest, accessible only to those proven worthy.

#### Objectives

1. **Speak with the mysterious stranger**
   - Location: Forest Path (near bandit camp)
   - Stranger reveals ancient secret location

2. **Gather testimonials from townspeople**
   - Talk to 3 NPCs who vouch for you:
     - **Bob (Innkeeper)**: "Saved my inn from rats! Trustworthy and brave."
     - **Marcus (Merchant)**: "Recovered my shipment! Honest and skilled."
     - **Captain Sarah (Guard)**: "Cleared bandits! Courage and honor personified."
   - Progress: 0/3 → 3/3

3. **Unlock the hidden forest path**
   - Return to stranger with testimonials
   - Stranger performs ritual to reveal hidden path
   - Cutscene: Ancient runes glow on trees

4. **Explore the ancient forest area**
   - Location: Hidden Forest (new area)
   - Discovery: Ancient stone monuments
   - Atmosphere: Magic hums in the air

5. **Solve the ancient puzzle**
   - Puzzle: Align glyphs representing seasons and trust
   - Solution: Cycle pattern based on NPC relationships
   - Hint: Trust flows like seasons - it cycles and grows

6. **Discover the ancient secret**
   - Chamber opens revealing:
     - Ancient texts (lore)
     - Ancient Amulet (powerful magic item)
     - Knowledge of ages past

#### Rewards

- **XP**: 500
- **Gold**: 200
- **Items**: Ancient Amulet, Mystery Scroll
- **Reputation**: Mysterious Stranger +100
- **World Flags**: forest_unlocked, ancient_secret_found

#### Time Limit

**1 hour** (3600 seconds) - Creates tension and urgency

#### Emotion Changes

- Mysterious Stranger valence: +0.3 (cryptic → satisfied)
- Mysterious Stranger trust: +0.6 (0.3 → 0.9)

#### Sample Dialogue

**Offer** (Stranger is mysterious):
> "*A hooded figure steps from the shadows* You have proven yourself worthy through deeds, not words. There are... ancient secrets in the forest. Secrets only the worthy may find. Are you interested?"

**Complete** (Stranger is reverent):
> "*The ancient puzzle unlocks, revealing a hidden chamber* You have proven yourself worthy in every way. This knowledge is now yours. Use it wisely."

---

## Emotion Requirements

### Trust Thresholds by Quest

| Quest | Innkeeper | Merchant | Guard | Child | Stranger |
|-------|-----------|----------|-------|-------|----------|
| Rat Problem | 0.3 | - | - | - | - |
| Lost Cat | - | - | - | 0.2 | - |
| Lost Shipment | 0.5 | 0.4 | - | - | - |
| Bandit Trouble | - | 0.6 | 0.5 | 0.4 | - |
| Hidden Path | 0.6 | 0.5 | 0.7 | - | 0.3 |

### Valence Requirements

| Quest | NPC | Valence | Meaning |
|-------|-----|---------|---------|
| Rat Problem | Innkeeper | -0.2 | Worried about rats |
| Lost Cat | Child | -0.4 | Sad about lost cat |
| Lost Shipment | Merchant | -0.3 | Worried about theft |
| Bandit Trouble | Guard | -0.2 | Concerned about bandits |

### Emotion Progression

As you complete quests, NPCs' emotions evolve:

```
Innkeeper: Worried → Relieved → Trusting
  Quest 1: valence -0.2 → +0.1 (Δ +0.3)
  Trust: 0.3 → 0.5 → 0.6

Merchant: Worried → Grateful → Business Partner
  Quest 3: valence -0.3 → +0.1 (Δ +0.4)
  Trust: 0.4 → 0.8

Guard: Serious → Impressed → Respectful
  Quest 4: valence -0.2 → +0.3 (Δ +0.5)
  Trust: 0.5 → 1.0

Child: Sad → Happy → Adoring
  Quest 2: valence -0.4 → +0.1 (Δ +0.5)
  Trust: 0.2 → 0.5

Stranger: Cryptic → Satisfied → Reverent
  Quest 5: valence 0.0 → +0.3 (Δ +0.3)
  Trust: 0.3 → 0.9
```

---

## Reward Tables

### XP Rewards

| Quest | XP | Difficulty Ratio |
|-------|----|--------------------|
| Rat Problem | 50 | Trivial (1x) |
| Lost Cat | 75 | Easy (1.5x) |
| Lost Shipment | 150 | Normal (3x) |
| Bandit Trouble | 300 | Hard (6x) |
| Hidden Path | 500 | Epic (10x) |
| **Total** | **1,075** | Average 3x per quest |

### Gold Rewards

| Quest | Gold | Notes |
|-------|------|-------|
| Rat Problem | 10 | Small tavern reward |
| Lost Cat | 5 | Child's limited means |
| Lost Shipment | 50 | Merchant's gratitude |
| Bandit Trouble | 100 | Guard official payment |
| Hidden Path | 200 | Ancient treasure |
| **Total** | **365** | Average 73g per quest |

### Item Rewards

| Quest | Items | Description |
|-------|-------|-------------|
| Rat Problem | - | No items |
| Lost Cat | Lucky Charm | +5% luck (minor trinket) |
| Lost Shipment | Silver Ring | +10% charisma (social boost) |
| Bandit Trouble | Guard Badge, Iron Sword | Official recognition + weapon |
| Hidden Path | Ancient Amulet, Mystery Scroll | +20% magic, ancient lore |

### Reputation Rewards

| Quest | NPC | Rep Change | New Total (approx) |
|-------|-----|------------|--------------------|
| Rat Problem | Innkeeper | +20 | 20 |
| Lost Cat | Child | +50 | 50 |
| Lost Shipment | Merchant | +30 | 30 |
| Bandit Trouble | Guard | +50 | 50 |
| Hidden Path | Stranger | +100 | 100 |

### World Flag Changes

| Quest | Flags Set | Impact |
|-------|-----------|--------|
| Rat Problem | rats_cleared | Tavern cellar accessible |
| Lost Cat | cat_found | Child is happy |
| Lost Shipment | shipment_recovered | Merchant has goods |
| Bandit Trouble | bandits_cleared, hero_status | Roads safe, NPC reactions change |
| Hidden Path | forest_unlocked, ancient_secret_found | New area, endgame content |

---

## Tips and Strategies

### Building Trust Quickly

1. **Start with easy quests**: Rat Problem and Lost Cat require low trust
2. **Complete side objectives**: Help NPCs beyond quest requirements
3. **Be consistent**: Multiple successful quests build trust faster
4. **Avoid failure**: Failed quests reduce trust

### Quest Order Optimization

**Fastest Route**:
```
1. Rat Problem (5 min)
2. Lost Cat (10 min)
3. Lost Shipment (15 min)
4. Bandit Trouble (25 min)
5. Hidden Path (40 min)
Total: ~95 minutes
```

**Exploration Route** (alternate):
```
1. Lost Cat (explore town)
2. Rat Problem (while at tavern)
3. Lost Shipment (after trust built)
4. Bandit Trouble (late game)
5. Hidden Path (finale)
```

### Combat Tips

**Rat Problem**:
- Rats are weak, single strike kills
- Corner them to prevent escape
- Watch for group attacks

**Lost Shipment**:
- 2-3 bandits guard the goods
- Use hit-and-run tactics
- Stealth approach possible

**Bandit Trouble**:
- Don't fight all 8 at once!
- Pull small groups (2-3 bandits)
- Save health items for leader fight
- Leader telegraphs charge attack - dodge and counter

### Emotion Management

**Increasing NPC Valence** (happiness):
- Complete their quests
- Bring gifts (flowers, food)
- Engage in friendly conversation
- Help other NPCs they care about

**Increasing NPC Trust**:
- Keep promises
- Complete quests successfully
- Don't fail their quests
- Demonstrate competence repeatedly

**Watch for Decay**:
- Emotions decay toward baseline over time
- Trust decays slower than valence
- Re-engage NPCs periodically to maintain relationships

### Hidden Path Puzzle Solution

**Glyph Alignment Pattern**:
```
Season Cycle: Spring → Summer → Fall → Winter
Trust Cycle: Stranger → Acquaintance → Friend → Ally

Solution:
  Top: Spring (new growth, trust forms)
  Right: Summer (trust strengthens)
  Bottom: Fall (trust matures)
  Left: Winter (trust endures)

Rotate glyphs clockwise to match seasons.
```

### Speedrun Notes

**World Record Potential**: ~45 minutes (all quests)

**Skip Strategies**:
- Lost Cat can be skipped if you find shipment first (harder)
- Bandit Trouble requires either Lost Cat OR Lost Shipment + high guard trust
- Hidden Path strictly requires all previous quests

**Optimal Path**:
1. Rat Problem (3 min speedrun)
2. Lost Cat (5 min if you know barrel location)
3. Lost Shipment (8 min with good combat)
4. Bandit Trouble (15 min boss rush)
5. Hidden Path (14 min puzzle memorized)

---

## Quest Graph Visualization

### Full Dependency Tree

```
Level 1 (Starter Quests)
├─ Rat Problem ────────────┐
│  Innkeeper               │
│  Trust: 0.3              ├──> Level 2 (Medium Quest)
│  XP: 50                  │    Lost Shipment
│                          │    Merchant
└─ Lost Cat ───────────────┤    Trust: 0.4 (+ prereq)
   Child                   │    XP: 150
   Trust: 0.2              │         │
   XP: 75                  │         │
                           │         v
                           │    Level 3 (Hard Quest)
                           │    Bandit Trouble
                           │    Guard
                           │    Trust: 0.5 (multi-NPC)
                           │    XP: 300
                           │         │
                           │         │
                           └────────>v
                                Level 4 (Epic Quest)
                                Hidden Path
                                Mysterious Stranger
                                Trust: 0.7 (multi-NPC)
                                XP: 500
```

### Parallel Progression Paths

Players can choose different paths:

**Combat Path**:
```
Rat Problem → Lost Shipment → Bandit Trouble → Hidden Path
(Focus: Combat skills, guard reputation)
```

**Social Path**:
```
Lost Cat → Rat Problem → Lost Shipment → Bandit Trouble → Hidden Path
(Focus: Trust building, social connections)
```

**Completionist Path**:
```
Do everything, explore all dialogue, maximize rewards
```

---

## Conclusion

The Elle Tavern Demo quest system demonstrates:

✅ **Emotion-based progression** - Trust unlocks opportunities
✅ **Interconnected storytelling** - NPCs reference each other
✅ **Progressive difficulty** - Easy → Epic scaling
✅ **Meaningful rewards** - Items, reputation, world changes
✅ **Replayability** - Different paths, dynamic quests

**Estimated Total Playtime**: 60-90 minutes (all quests)
**Total Rewards**: 1,075 XP, 365 Gold, 5 unique items

Enjoy your adventure in the Elle Tavern Demo!
