"""
Quest dialogue system for Elle Tavern Demo.

Provides rich, emotion-aware dialogue for each quest state:
- offer: Initial quest offering
- accept: When player accepts
- decline: When player declines
- progress: During quest progression
- hint: If player seems stuck
- complete: When quest is completed
- already_complete: If player talks to giver after completion

Dialogues adapt to NPC emotional state for immersion.

Created: 2025-11-16
"""

from typing import Dict, List, Optional


# ============================================================================
# Quest Dialogues
# ============================================================================

QUEST_DIALOGUES = {
    # ========================================================================
    # Quest 1: Rat Problem
    # ========================================================================
    "rat_problem": {
        "offer": {
            "happy": "Hey there, friend! I hate to ask, but we've got a bit of a rat problem in the cellar. Would you mind helping out? I'll make it worth your while!",
            "neutral": "Traveler, I need help with something. We've got rats in the cellar eating all the food stores. Can you clear them out?",
            "worried": "Oh thank goodness you're here! The cellar is infested with rats - they're destroying everything! Please, I need your help!",
            "desperate": "Please, you have to help me! The rats are out of control down there! I can't run an inn with rodents eating all my supplies!"
        },
        "accept": "Oh, thank you so much! The cellar door is right behind the bar. Be careful down there - those rats are bigger than you'd think! Here, take this candle.",
        "decline": {
            "understanding": "I understand, traveler. It's not the most glamorous work. If you change your mind, I'll still be here.",
            "disappointed": "Oh... I see. Well, I suppose I'll have to find someone else. The offer stands if you reconsider.",
            "desperate": "Please reconsider! I really need help here. No one else has been willing to help me..."
        },
        "progress": {
            "early": "How's it going down there? Those rats giving you trouble? Remember, there should be about 5 of the nasty creatures.",
            "midway": "I can hear some commotion down there! You're making progress, I can tell. Keep at it!",
            "almost_done": "Just a few more, I think! You're doing great work!"
        },
        "hint": "The rats tend to hide in the corners and behind the barrels. A swift strike should do the trick - they're not very brave creatures.",
        "complete": {
            "grateful": "You did it! The cellar is finally clear! I can't thank you enough. Here, take this gold - you've more than earned it. You're a true friend!",
            "relieved": "Thank the gods! I was at my wit's end. You're a lifesaver! Please, take this as a token of my gratitude.",
            "happy": "Excellent work! The Rusty Tankard is rat-free thanks to you! Come by anytime for a free drink!"
        },
        "already_complete": "Thanks again for dealing with those rats! The cellar's never been cleaner. You're always welcome here, friend.",
        "objectives": {
            "talk_to_bob": "So, you want to know about the rats? Well, they appeared about a week ago. Started small, but now there must be 5 or 6 of them down there!",
            "clear_rats": "Good luck down there! Give those rats what for!",
            "report_to_bob": "All clear? Wonderful! Let me give you your reward."
        }
    },

    # ========================================================================
    # Quest 2: Lost Cat
    # ========================================================================
    "lost_cat": {
        "offer": {
            "sad": "*sniffling* M-Mister... have you seen my cat? Her name is Whiskers and she's been gone since this morning...",
            "crying": "*sobbing* Whiskers is missing! She's orange with white paws and she's my best friend! *wipes tears* Can you help me find her?",
            "worried": "Excuse me! I can't find my cat anywhere! I've looked and looked but she's nowhere! Will you help me look for Whiskers?"
        },
        "accept": "*brightens up* Oh thank you! Thank you so much! Whiskers is orange and white, and she likes to hide in small spaces. She's probably scared...",
        "decline": {
            "sad": "*lip trembles* Oh... okay... *looks down* I'll keep looking by myself then...",
            "crying": "*starts crying harder* But... but I need help! Whiskers might be scared!",
            "understanding": "That's okay... I know you're busy. If you see an orange cat, though, will you tell me?"
        },
        "progress": {
            "searching": "Did you find her yet? I'm so worried! She doesn't like being outside for long...",
            "hopeful": "Any sign of her? Maybe check behind things - she loves to squeeze into tight spots!",
            "anxious": "It's getting late... I hope Whiskers is okay. Please keep looking!"
        },
        "hint": "Whiskers really likes the market square - there's a bakery there and she loves the smell of fresh bread. Maybe check around the barrels behind the stalls?",
        "complete": {
            "overjoyed": "*hugs cat* WHISKERS! You found her! You found her! *bouncing with joy* Thank you thank you thank you! You're my hero!",
            "crying_happy": "*tears of joy* I thought I'd never see her again! *pets cat* Thank you so much! Here, this is my lucky charm - I want you to have it!",
            "grateful": "You brought her back! *hugs the player* You're the best! Whiskers and I will never forget this!"
        },
        "already_complete": "*playing with Whiskers* Hi! Look, Whiskers is happy and safe! Thank you again for finding her!",
        "objectives": {
            "talk_to_lily": "Whiskers went missing this morning! I think she chased a butterfly and got lost...",
            "search_tavern": "The innkeeper said he saw an orange cat earlier... but I looked everywhere inside!",
            "search_market": "There's a bakery in the market square - Whiskers loves the smell of bread!",
            "find_cat": "*You hear a faint meow from behind the barrels...*",
            "return_cat": "*Lily sees you carrying Whiskers* Is that... WHISKERS?!"
        }
    },

    # ========================================================================
    # Quest 3: Lost Shipment
    # ========================================================================
    "lost_shipment": {
        "offer": {
            "worried": "You there! You look capable. I've had a valuable shipment stolen by bandits on the forest road. Are you brave enough to recover it?",
            "angry": "Those cursed bandits! They stole my goods - an entire wagonload! I need someone skilled to get it back. Can I count on you?",
            "desperate": "Please, I need help! Bandits took everything - my entire shipment! Without those goods, my business is ruined! I'll pay well!"
        },
        "accept": "Excellent! The bandits ambushed my wagon on the forest path north of here. Captain Sarah at the town square might know where their camp is. Be careful - they're dangerous!",
        "decline": {
            "disappointed": "I see. Well, I can't force you to risk your life. But if you change your mind, I'll make it worth your while.",
            "angry": "Figures! No one wants to help an honest merchant anymore. Fine, I'll find someone else!",
            "resigned": "I understand - it's dangerous work. If you reconsider, the offer stands."
        },
        "progress": {
            "early": "Any luck finding the bandits? Captain Sarah should know where they're hiding out.",
            "midway": "You found the camp? Be careful - bandits fight dirty! My goods should be in their main tent.",
            "almost_done": "You've got my goods? Fantastic! Bring them back quickly - those bandits might come looking for you!"
        },
        "hint": "Captain Sarah patrols the town square. She's been tracking those bandits for weeks - she'll know where to find them. The forest path leads to their camp.",
        "complete": {
            "grateful": "My goods! You actually recovered them! I was starting to lose hope. Here - take this silver ring and gold as thanks. You've saved my business!",
            "relieved": "Thank the gods! Everything's here! You're a skilled adventurer indeed. Please, accept this reward - you've more than earned it!",
            "happy": "Excellent work! Those bandits won't trouble me again, I hope. You have my gratitude and my business - any time you need supplies, I'll give you a fair price!"
        },
        "already_complete": "Business is booming again, thanks to you! If you ever need supplies, I'll give you my best prices. You're a true friend to merchants!",
        "objectives": {
            "talk_to_marcus": "The bandits hit me three days ago. Took everything - silks, spices, tools. I'm ruined without those goods!",
            "talk_to_guard": "Captain Sarah here. Those bandits have been a menace for months. Their camp is hidden on the old forest path - follow the road north from the square.",
            "find_bandits": "*You spot smoke rising from a clearing ahead... the bandit camp!*",
            "recover_goods": "*The merchant's goods are piled in the main tent, poorly guarded.*",
            "return_goods": "You're back! And with my goods! I can't believe it!"
        }
    },

    # ========================================================================
    # Quest 4: Bandit Trouble
    # ========================================================================
    "bandit_trouble": {
        "offer": {
            "serious": "Adventurer. Those bandits you fought? They're part of a larger problem. I need someone capable to clear out their entire camp permanently. Interested?",
            "urgent": "I'll get straight to the point - the bandit camp needs to be eliminated. They've terrorized travelers for too long. You've proven yourself capable. Will you help?",
            "respectful": "I've heard about your work with the merchant. That took courage. I have a proposition: help me clear the bandit camp entirely, and you'll have the guard's full support."
        },
        "accept": "Good. This is dangerous work - there are at least 8 bandits, plus their leader. Scout the camp first, find their weaknesses. Strike hard and fast. I'll have reinforcements ready if you succeed.",
        "decline": {
            "disappointed": "I understand - it's a tough assignment. But those bandits won't stop on their own. If you change your mind, find me at the town square.",
            "respectful": "Fair enough. It's not work for everyone. The offer stands if you reconsider.",
            "urgent": "Think it over carefully. Every day we wait, more travelers suffer. I need someone with your skills."
        },
        "progress": {
            "early": "Scout the camp thoroughly before engaging. Know your enemy's positions and numbers.",
            "combat": "Watch yourself out there! Bandits travel in groups and they know the forest well.",
            "almost_done": "The leader should be in the main tent. Take him down and the rest will scatter!"
        },
        "hint": "The bandits change guard shifts at dusk - that's your best opportunity. Their leader is a brute with a scar across his left eye. He won't go down easy.",
        "complete": {
            "impressed": "You did it! The bandit camp is destroyed and their leader defeated. I'm impressed - that took real skill and courage. Here's your reward and this guard's badge. You've earned it.",
            "grateful": "Outstanding work! The roads are safe again thanks to you. Take this badge - it marks you as a friend of the guard. You have our respect and gratitude.",
            "proud": "I knew you could do it! The bandits are scattered and their leader is finished. You're a true hero. This town owes you a great debt."
        },
        "already_complete": "The roads have been peaceful since you cleared that camp. Travelers can move freely again. You have the eternal gratitude of the town guard.",
        "objectives": {
            "talk_to_sarah": "Those bandits have been plaguing us for months. Hit-and-run attacks, stolen goods, terrorized travelers. It ends now.",
            "scout_camp": "*You count 8 bandits around the camp, plus one large figure in the main tent - likely the leader.*",
            "defeat_bandits": "*The bandits fight fiercely, but your skill prevails!*",
            "defeat_leader": "*The scarred leader roars a challenge: 'You'll regret crossing us!'*",
            "report_to_sarah": "You're back! Tell me everything - is the camp destroyed?"
        }
    },

    # ========================================================================
    # Quest 5: Hidden Path
    # ========================================================================
    "hidden_path": {
        "offer": {
            "mysterious": "*A hooded figure steps from the shadows* You have proven yourself worthy through deeds, not words. There are... ancient secrets in the forest. Secrets only the worthy may find. Are you interested?",
            "cryptic": "I've been watching you, hero. The townspeople speak highly of your courage and honor. This makes you... eligible. For knowledge long hidden. Do you seek truth?",
            "serious": "*Lowers hood, revealing ancient eyes* The old paths are opening. I can guide you - but only if you are trusted by those you've helped. Are you ready to discover what lies beyond?"
        },
        "accept": "Good. But first, you must prove the townspeople truly trust you. Bring me their words - testimonials from the innkeeper, the merchant, and the guard captain. Then, and only then, will the path reveal itself.",
        "decline": {
            "mysterious": "*Nods slowly* Not all are ready for what lies hidden. Perhaps another time, when fate brings us together again.",
            "understanding": "The secrets have waited centuries. They can wait longer. Seek me when you are ready.",
            "cryptic": "The path chooses the traveler, not the reverse. If you are meant to walk it, you will return."
        },
        "progress": {
            "early": "The townspeople's trust is key. Seek out those you've helped - the innkeeper, merchant, and guard. Their words carry weight.",
            "gathering": "Good progress. Each testimonial opens a lock. You're nearly there.",
            "ready": "You have their trust. Now, follow me. The path awaits, but time is short - ancient magic fades with the light."
        },
        "hint": "Trust is earned through consistent deeds, not grand gestures. Those you've helped will vouch for you - but only if you truly earned their respect.",
        "complete": {
            "reverent": "*The ancient puzzle unlocks, revealing a hidden chamber* You have proven yourself worthy in every way. This knowledge is now yours. Use it wisely.",
            "satisfied": "The ancient ones chose well. You possess courage, skill, and the trust of the people. Take this amulet - it holds power beyond measure. You've earned it.",
            "mysterious": "*Smiles knowingly* The secrets are revealed. You are now a keeper of ancient knowledge. May you use it to protect, never to harm."
        },
        "already_complete": "*The stranger nods in acknowledgment* You walk a path few have trodden. Use your knowledge well, keeper of secrets.",
        "objectives": {
            "talk_to_stranger": "You seek answers? Then first, prove your worth. The townspeople must vouch for you.",
            "gather_trust": {
                "innkeeper": "That adventurer? Saved my inn from rats! Trustworthy and brave. You have my word!",
                "merchant": "Recovered my entire shipment from bandits! Honest and skilled. I vouch for them completely.",
                "guard": "Cleared the bandit camp single-handedly. Courage and honor personified. I trust them with my life."
            },
            "unlock_forest": "*The stranger traces a symbol in the air. Ancient runes glow on nearby trees, revealing a hidden path.*",
            "explore_forest": "*The forest path leads to a clearing with strange stone monuments. Magic hums in the air.*",
            "solve_puzzle": "*Ancient glyphs must be aligned. The pattern represents the cycle of seasons and trust.*",
            "discover_secret": "*The chamber opens. Inside: ancient texts, a glowing amulet, and knowledge of ages past.*"
        }
    }
}


# ============================================================================
# Dialogue Helper Functions
# ============================================================================

def get_quest_dialogue(
    quest_id: str,
    dialogue_type: str,
    npc_emotion: Optional[str] = None,
    objective_id: Optional[str] = None
) -> str:
    """
    Get appropriate dialogue for quest and NPC emotion.

    Args:
        quest_id: Quest identifier
        dialogue_type: Type of dialogue (offer, accept, decline, etc.)
        npc_emotion: NPC's emotional state (happy, sad, angry, etc.)
        objective_id: Specific objective for objective-specific dialogue

    Returns:
        Dialogue text
    """
    if quest_id not in QUEST_DIALOGUES:
        return "..."

    quest_dialogues = QUEST_DIALOGUES[quest_id]

    # Handle objective-specific dialogue
    if dialogue_type == "objective" and objective_id:
        objectives = quest_dialogues.get("objectives", {})
        if objective_id in objectives:
            obj_dialogue = objectives[objective_id]
            # If objective dialogue is a dict (multiple NPCs), return first
            if isinstance(obj_dialogue, dict):
                return next(iter(obj_dialogue.values()))
            return obj_dialogue

    # Get dialogue for type
    dialogue = quest_dialogues.get(dialogue_type)

    if dialogue is None:
        return "..."

    # If dialogue is emotion-aware (dict), select based on emotion
    if isinstance(dialogue, dict):
        if npc_emotion and npc_emotion in dialogue:
            return dialogue[npc_emotion]
        # Default to first option if emotion not found
        return next(iter(dialogue.values()))

    return dialogue


def get_npc_emotion_label(npc_emotion_data: Dict[str, float]) -> str:
    """
    Get emotion label from emotion data.

    Args:
        npc_emotion_data: Dictionary with valence, arousal, trust, etc.

    Returns:
        Emotion label (happy, sad, angry, etc.)
    """
    valence = npc_emotion_data.get("valence", 0.0)
    arousal = npc_emotion_data.get("arousal", 0.5)
    trust = npc_emotion_data.get("trust", 0.5)

    # Determine emotion based on valence primarily
    if valence > 0.5:
        if arousal > 0.6:
            return "happy"
        else:
            return "content"
    elif valence < -0.5:
        if arousal > 0.6:
            return "angry"
        else:
            return "sad"
    elif valence < -0.2:
        if trust < 0.4:
            return "suspicious"
        else:
            return "worried"
    else:
        return "neutral"


def get_progress_dialogue(
    quest_id: str,
    completion_percentage: float,
    npc_emotion: Optional[str] = None
) -> str:
    """
    Get progress-appropriate dialogue based on quest completion.

    Args:
        quest_id: Quest identifier
        completion_percentage: 0.0 to 1.0 completion
        npc_emotion: NPC emotional state

    Returns:
        Progress dialogue text
    """
    if quest_id not in QUEST_DIALOGUES:
        return "How's it going?"

    progress_dialogues = QUEST_DIALOGUES[quest_id].get("progress", {})

    # If progress is a simple string, return it
    if isinstance(progress_dialogues, str):
        return progress_dialogues

    # If progress is a dict with stages
    if isinstance(progress_dialogues, dict):
        if completion_percentage < 0.33:
            stage = "early"
        elif completion_percentage < 0.66:
            stage = "midway"
        else:
            stage = "almost_done"

        return progress_dialogues.get(stage, "How's it going?")

    return "How's it going?"


# ============================================================================
# Dialogue System Examples
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=== Quest Dialogue System Demo ===\n")

    # Example 1: Offering rat quest with different emotions
    print("1. Rat Problem Quest - Different NPC emotions:")
    print(f"  Happy: {get_quest_dialogue('rat_problem', 'offer', 'happy')}")
    print(f"  Worried: {get_quest_dialogue('rat_problem', 'offer', 'worried')}\n")

    # Example 2: Lost cat quest progression
    print("2. Lost Cat Quest - Progress dialogue:")
    print(f"  Early (20%): {get_progress_dialogue('lost_cat', 0.2)}")
    print(f"  Midway (50%): {get_progress_dialogue('lost_cat', 0.5)}")
    print(f"  Almost done (80%): {get_progress_dialogue('lost_cat', 0.8)}\n")

    # Example 3: Emotion label conversion
    print("3. Emotion label conversion:")
    emotion_data = {"valence": -0.6, "arousal": 0.7, "trust": 0.3}
    label = get_npc_emotion_label(emotion_data)
    print(f"  Emotion data: {emotion_data}")
    print(f"  Label: {label}\n")

    # Example 4: Quest completion with different emotions
    print("4. Lost Shipment completion - Different emotions:")
    print(f"  Grateful: {get_quest_dialogue('lost_shipment', 'complete', 'grateful')}")
    print(f"  Happy: {get_quest_dialogue('lost_shipment', 'complete', 'happy')}\n")

    # Example 5: Mysterious stranger dialogue
    print("5. Hidden Path quest - Mysterious offering:")
    print(f"  {get_quest_dialogue('hidden_path', 'offer', 'mysterious')}\n")
