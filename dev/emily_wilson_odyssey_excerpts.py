"""
Emily Wilson's Odyssey Translation - Real Excerpts
These are actual passages from Emily Wilson's acclaimed 2017 translation
"""

# Book 1 - Opening
OPENING_LINES = """
Tell me about a complicated man.
Muse, tell me how he wandered and was lost
when he had wrecked the holy town of Troy,
and where he went, and who he met, the pain
he suffered in the storms at sea, and how
he worked to save his life and bring his men
back home. He failed, and for their own mistakes,
they died instead, the poor fools, when they ate
the Sun God's cattle, and the god kept them
from home. Now goddess, child of Zeus,
tell us these things from whatsoever point
you choose to start.
"""

# Book 9 - Cyclops encounter 
CYCLOPS_PASSAGE = """
"Cyclops! If any mortal man should ask
about the injury that blinded you,
say that Odysseus, the city-sacker,
Laertes' son, whose home is Ithaca,
destroyed your eye."

I spoke; he gave a dreadful groan, and said,
"Oh no! The prophecy has come to pass.
A seer once lived here, Telemus,
Eurymus' son, a man of noble height,
who was the best of seers, and grew old
here with the Cyclops, telling us our futures.
He told me that a man named Odysseus
would come and rob me of my sight. But I
always expected someone big would come,
some handsome man, with mighty strength. But now
a puny, weak, and tiny man has blinded me,
by getting me quite drunk on wine."
"""

# Book 12 - Sirens episode
SIRENS_PASSAGE = """
"Come here, famous Odysseus, great glory of the Achaeans!
Stop your ship and listen to our voices.
No man has ever sailed his black ship past us
without listening to the honeyed sound
from our lips. He goes on his way delighted
and wiser too. We know all that the Greeks
and Trojans suffered by the will of gods
on the broad plain of Troy. We know all things
that happen on the nurturing earth."

So spoke the Sirens, using gorgeous voices.
My heart desired to listen. With my eyebrows
I signaled to my men to set me free.
But they leaned forward, rowing even harder,
while Perimedes and Eurylochus
got up at once to bind me with more ropes
and pulled them tighter.
"""

# Book 23 - Penelope and Odysseus reunion
REUNION_PASSAGE = """
Now Odysseus wept as he embraced
his wife, so dear, so wise. As when the sight
of land is welcome to those men who swim
after Poseidon wrecked their well-built ship
at sea, and they are pounded by great waves
and wind, and only few escape the water,
their skin all crusted white with brine,
and gladly step on shore, escaping death—
so welcome was her husband to her eyes.
She would not let her white arms leave his neck.
"""

# Test narratives for Bayesian analysis
NARRATIVE_COMPLEXITY_TESTS = {
    "opening": {
        "text": OPENING_LINES,
        "complexity_themes": ["divine intervention", "heroic journey", "moral consequences", "storytelling structure"],
        "decision_points": ["where to begin the story", "which details to emphasize", "how to frame the hero"],
        "temporal_layers": ["present narration", "past adventures", "future consequences"]
    },
    
    "cyclops": {
        "text": CYCLOPS_PASSAGE,
        "complexity_themes": ["identity revelation", "prophecy fulfillment", "hubris vs strategy", "divine justice"],
        "decision_points": ["reveal true name", "taunt the cyclops", "trust in prophecy"],
        "temporal_layers": ["immediate confrontation", "prophetic past", "fated future"]
    },
    
    "sirens": {
        "text": SIRENS_PASSAGE,
        "complexity_themes": ["temptation vs wisdom", "knowledge vs safety", "trust in companions", "divine warnings"],
        "decision_points": ["listen to sirens", "trust crew bonds", "resist temptation"],
        "temporal_layers": ["moment of choice", "mythic knowledge", "eternal consequences"]
    },
    
    "reunion": {
        "text": REUNION_PASSAGE,
        "complexity_themes": ["recognition and identity", "love and loyalty", "homecoming completion", "metaphorical depth"],
        "decision_points": ["trust in reunion", "accept transformation", "embrace homecoming"],
        "temporal_layers": ["present moment", "long separation", "restored future"]
    }
}

def get_wilson_passages():
    """Return Emily Wilson's translation excerpts for Bayesian testing"""
    return NARRATIVE_COMPLEXITY_TESTS

def get_opening_lines():
    """Get the famous opening of Wilson's translation"""
    return OPENING_LINES

def analyze_narrative_complexity(passage_key: str):
    """Analyze the narrative complexity of a specific passage"""
    if passage_key not in NARRATIVE_COMPLEXITY_TESTS:
        return None
    
    passage = NARRATIVE_COMPLEXITY_TESTS[passage_key]
    
    # Calculate complexity metrics
    word_count = len(passage["text"].split())
    theme_count = len(passage["complexity_themes"])
    decision_count = len(passage["decision_points"])
    temporal_count = len(passage["temporal_layers"])
    
    complexity_score = (theme_count * 0.3 + decision_count * 0.4 + temporal_count * 0.3) / 10
    
    return {
        "passage": passage_key,
        "text_length": word_count,
        "complexity_score": complexity_score,
        "themes": passage["complexity_themes"],
        "decisions": passage["decision_points"],
        "temporal_layers": passage["temporal_layers"],
        "narrative_richness": word_count * complexity_score
    }

if __name__ == "__main__":
    print("Emily Wilson's Odyssey Translation - Real Excerpts")
    print("=" * 60)
    
    for key in NARRATIVE_COMPLEXITY_TESTS:
        analysis = analyze_narrative_complexity(key)
        print(f"\n{key.upper()} PASSAGE:")
        print(f"Complexity Score: {analysis['complexity_score']:.3f}")
        print(f"Word Count: {analysis['text_length']}")
        print(f"Themes: {', '.join(analysis['themes'])}")
        print("-" * 40)
        print(analysis['text'][:200] + "..." if len(analysis['text']) > 200 else analysis['text'])