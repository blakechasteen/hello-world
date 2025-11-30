# EduVerse CLI - Quick Start Guide

**Ready to play?** This guide gets you up and running in 2 minutes.

## Start the Game

```bash
cd c:/Users/blake/OneDrive/Documents/mythRL
python EduVerse/eduverse_cli.py
```

You'll see:
```
======================================================================
EduVerse - AI-Powered K-12 Learning Platform
Week 3 Interactive Demo
======================================================================

Initializing world...
[OK] World ready with 16 locations and 9 NPCs

Type 'help' for commands or 'quit' to exit.

School Lobby
----------------------------------------------------------------------
A bright, welcoming entrance with student art on the walls...

>
```

## Essential Commands

### Getting Your Bearings

```
> help           # Show all commands
> look           # See where you are
> status         # Check your stats
> locations      # List all places
```

### Exploring the World

```
> locations                      # See all 16 locations
> travel school_library          # Go to the library
> travel fantasy_wizard_tower    # Visit a fantasy world
> travel space_research_lab      # Explore a space station
```

### Meeting NPCs

```
> npcs                  # See who's at your location
> talk Dr. Martinez     # Start a conversation
> quest                 # Ask for a learning quest
> accept                # Accept the offered quest
> complete              # Complete your quest (simulated)
```

### Tracking Progress

```
> status    # See level, XP, skills, learning style
> story     # View narrative history and threads
```

## Example Session

Here's a complete 5-minute gameplay session:

```
> look
School Lobby
----------------------------------------------------------------------
A bright, welcoming entrance...

NPCs here (1):
  - Sam (peer)

> talk Sam
Sam: "Hey! Want to work together on something?"

> quest
Sam offers you a quest:
  Title: Introduction to Reading Comprehension
  Type: PRACTICE
  Difficulty: EASY
  Rewards: 100 XP

> accept
You accepted the quest: Introduction to Reading Comprehension

> complete
Quest complete!
Rewards earned: XP: 100.0

> status
Player: Student
----------------------------------------------------------------------
Level: 1
Total XP: 100.0
Grade: 8

Quests Completed: 1/1

> travel fantasy_wizard_tower
You travel to Tower of Numerical Wizardry.

> look
Tower of Numerical Wizardry
----------------------------------------------------------------------
A towering spire where mathematical magic is studied...

NPCs here (1):
  - Professor Cipher (teacher)

> talk Professor Cipher
Professor Cipher: "Welcome, young mathematician. Ready to discover..."

> story
Narrative Summary
----------------------------------------------------------------------
Total Events: 5
Active Threads: 6/6

Subject Progress:
  ela: 5 events

Recent Significant Events:
  [MEDIUM] Started quest: Introduction to Reading Comprehension
  [HIGH] Completed quest: Introduction to Reading Comprehension
  [LOW] Traveled to Tower of Numerical Wizardry
  [MEDIUM] Talked to Professor Cipher

> quit
Thanks for playing EduVerse!

Session Summary:
  Total XP: 100.0
  Events: 5
  Locations Visited: 2
```

## The Three Worlds

### School World (6 locations)
Realistic K-12 learning environment:
- School Lobby
- Mathematics Classroom
- Science Laboratory
- School Library
- Computer Lab & AI Studio
- Cafeteria

### Fantasy World (5 locations)
Gamified learning with medieval theme:
- Mystical Town Square
- Tower of Numerical Wizardry
- Alchemist's Laboratory
- Library of Eternal Stories
- Adventurer's Guild Hall

### Space World (5 locations)
Sci-fi themed STEM learning:
- Command Deck
- Scientific Research Laboratory
- Engineering Bay
- AI Core Chamber
- Observation Lounge

## The 9 NPCs

Each NPC has a unique teaching style:

**School World:**
- **Sam** (peer) - Collaborative
- **Ms. Thompson** (teacher) - Supportive
- **Dr. Martinez** (teacher) - Demonstrative

**Fantasy World:**
- **Professor Cipher** (teacher) - Challenging
- **Sage Storyteller** (mentor) - Supportive
- **Master Alchemist** (specialist) - Demonstrative

**Space World:**
- **Captain Nova** (mentor) - Challenging
- **Dr. Quantum** (specialist) - Socratic
- **Engineer Hayes** (guide) - Collaborative

## Tips for Grant Demos

When demonstrating for stakeholders:

1. **Start with 'locations'** - Shows scope (16 locations, 3 worlds)
2. **Talk to an NPC** - Demonstrates adaptive dialogue
3. **Accept and complete a quest** - Shows curriculum alignment
4. **Check 'status'** - Displays XP and learning style tracking
5. **Run 'story'** - Proves narrative coherence and event tracking

## What Makes This Special

- **Adaptive Learning**: Thompson Sampling chooses optimal difficulty
- **Narrative Coherence**: Events form causal chains
- **Teaching Styles**: 5 different pedagogical approaches
- **Curriculum Aligned**: Common Core State Standards (K-12)
- **Multi-World**: Same content, different themes (engagement)

## Getting Help

- Type `help` in the game for command reference
- See [EDUVERSE_CLI_COMPLETE.md](EDUVERSE_CLI_COMPLETE.md) for full documentation
- See [EDUVERSE_WEEK_3_COMPLETE.md](EDUVERSE_WEEK_3_COMPLETE.md) for architecture

## Next Steps

After exploring:

1. Check out the code in `EduVerse/eduverse_cli.py`
2. Explore the Week 3 systems:
   - `EduVerse/game/world_generator.py` - Location generation
   - `EduVerse/game/npc_system.py` - NPC dialogue & teaching
   - `EduVerse/game/narrative_memory.py` - Event tracking
   - `EduVerse/game/world_integration.py` - System integration

3. Run tests: `python test_eduverse_cli.py`

---

**Have fun exploring EduVerse!** 🎓✨
