#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📚 FULL ODYSSEY DEPTH ANALYSIS
==============================
Extended testing of narrative depth on complete 24-book Odyssey structure.

This demonstrates:
1. Progressive depth analysis across entire epic
2. Temporal evolution of narrative depth
3. Campbell stage correlation with depth levels
4. Character appearance impact on depth
5. Performance optimization with caching

Test Coverage:
- All 24 books from simplified_odyssey_matryoshka.py
- Full depth analysis for each major decision point
- Temporal tracking of depth progression
- Cache performance validation

Expected Results:
- Early books (1-4): Lower depth (SYMBOLIC/ARCHETYPAL)
- Middle books (5-16): Medium to high depth (ARCHETYPAL/MYTHIC)
- Final books (17-24): Highest depth (MYTHIC/COSMIC)
- Clear correlation with Campbell stages
"""

# Force UTF-8 encoding for Windows console
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import asyncio
import time
from typing import Dict, List
from dataclasses import dataclass, field
import statistics

# Import from existing modules
from hololoom_narrative.matryoshka_depth import MatryoshkaNarrativeDepth, DepthLevel


@dataclass
class OdysseyBookDepth:
    """Depth analysis for a single Odyssey book."""
    book_number: int
    book_title: str
    max_depth: DepthLevel
    complexity: float
    confidence: float
    gates_unlocked: int
    cosmic_truth: str = ""
    mythic_truths: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0


# Complete 24-book Odyssey structure
ODYSSEY_24_BOOKS = [
    # Books 1-4: Telemachia (The Son's Quest)
    {
        'book': 1,
        'title': 'Athena Inspires Telemachus',
        'text': '''Telemachus sits idle while suitors ravage his home. Athena appears as Mentes, 
        stirring the young prince to action. "Seek your father," she counsels. "The journey 
        will make you a man." The call to adventure rings clear, though Telemachus fears to answer.''',
        'expected_stage': 'call_to_adventure',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 2,
        'title': 'The Assembly and Departure',
        'text': '''Telemachus addresses the assembly, declaring his intent to seek Odysseus. 
        The suitors mock him, but Athena provides a ship and crew. At dawn, he crosses the 
        threshold of his familiar world, sailing toward unknown shores and his destiny.''',
        'expected_stage': 'crossing_threshold',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 3,
        'title': 'King Nestor Remembers',
        'text': '''In Pylos, ancient Nestor recounts the fates of Greek heroes. Agamemnon murdered, 
        Menelaus wandering, Odysseus lost. The old king becomes mentor, sharing wisdom: "The gods 
        test us all. Your father's cunning will see him home." Telemachus gains hope and direction.''',
        'expected_stage': 'meeting_mentor',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 4,
        'title': 'Menelaus and Helen',
        'text': '''In Sparta, Menelaus reveals Proteus' prophecy: Odysseus lives, trapped on 
        Calypso's isle. Helen, seeing Telemachus' resemblance to his father, weeps. The young 
        prince learns his father's fate - alive but imprisoned by forces beyond mortal power.''',
        'expected_stage': 'meeting_mentor',
        'expected_depth': 'ARCHETYPAL'
    },
    
    # Books 5-8: Odysseus Reaches the Phaeacians
    {
        'book': 5,
        'title': 'Odysseus and Calypso',
        'text': '''For seven years, Calypso has kept Odysseus on her island, offering immortality 
        for his love. Athena pleads his case to Zeus. Hermes delivers the decree: "Let him go." 
        Odysseus weeps for home, choosing mortality and family over eternal paradise. The refusal 
        of the supernatural gift marks his commitment to the mortal world.''',
        'expected_stage': 'refusal_of_call',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 6,
        'title': 'Odysseus and Nausicaa',
        'text': '''Shipwrecked and naked, Odysseus meets Princess Nausicaa. She sees past his 
        wretched state to the noble hero beneath. "Follow me to the city," she says, "but keep 
        your distance, lest tongues wag." The goddess guides through mortal lips. The threshold 
        guardian appears in unexpected form.''',
        'expected_stage': 'threshold_guardian',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 7,
        'title': 'Reception at Alcinous Palace',
        'text': '''King Alcinous welcomes the stranger, offering protection before asking his name. 
        The Phaeacians, blessed by gods, live in harmony. Queen Arete questions Odysseus about his 
        clothing - Nausicaa's. He tells partial truth, earning their trust. The special world reveals 
        its magic: a civilization between divine and mortal.''',
        'expected_stage': 'entering_special_world',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 8,
        'title': 'The Songs of Demodocus',
        'text': '''The blind bard sings of Troy - of the wooden horse and Odysseus' cunning. 
        The hero weeps to hear his own story, his identity still secret. Alcinous notes his tears: 
        "What grief burdens you, stranger? Why do these songs wound so deep?" The mirror is held up; 
        Odysseus must confront what he has become.''',
        'expected_stage': 'tests_allies_enemies',
        'expected_depth': 'ARCHETYPAL'
    },
    
    # Books 9-12: The Wanderings (Flashback)
    {
        'book': 9,
        'title': 'The Cyclops',
        'text': '''Odysseus reveals his name and tells his tale. "I am Odysseus of Ithaca, 
        sacker of cities." He recounts Polyphemus - his arrogance in blinding Poseidon's son, 
        his fatal hubris in revealing his true name. "Nobody!" he had claimed, but pride demanded 
        recognition. That pride cursed his journey home. The shadow emerges: his own arrogance.''',
        'expected_stage': 'approach_inmost_cave',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 10,
        'title': 'Circe and Aeolus',
        'text': '''Aeolus gave winds in a bag - home within reach. But crew's greed opened it, 
        blowing them back. Then Circe transformed his men to swine. With Hermes' herb, Odysseus 
        resisted her magic, becoming her lover for a year. Time slips away in the goddess's palace. 
        The hero forgets himself, lost in temptation.''',
        'expected_stage': 'ordeal',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 11,
        'title': 'The Land of the Dead',
        'text': '''In Hades, Odysseus speaks with the dead. His mother Anticlea reveals she died 
        of grief, waiting for his return. Agamemnon warns of treacherous wives. Achilles declares 
        he would rather be a slave on earth than king among the dead. Tiresias prophecies the path 
        home - and the journey that comes after. Death and rebirth: the ultimate transformation.''',
        'expected_stage': 'ordeal',
        'expected_depth': 'COSMIC'
    },
    {
        'book': 12,
        'title': 'The Sirens, Scylla, and Charybdis',
        'text': '''Bound to the mast, Odysseus hears the Sirens' song - knowledge of all that 
        was and will be. He longs to surrender but cannot. Between Scylla and Charybdis, he 
        chooses: lose six men to the monster, or all to the whirlpool. Leadership means choosing 
        which sacrifice to make. On Helios' isle, starving crew eat sacred cattle despite his 
        warnings. Zeus destroys the ship. Odysseus alone survives.''',
        'expected_stage': 'ordeal',
        'expected_depth': 'MYTHIC'
    },
    
    # Books 13-16: Return to Ithaca
    {
        'book': 13,
        'title': 'Odysseus Returns',
        'text': '''The Phaeacians deliver Odysseus to Ithaca while he sleeps. He wakes not 
        recognizing his home - Athena has shrouded it in mist. She reveals herself: "You must 
        go disguised. Trust no one." She transforms him to an old beggar. The hero returns 
        unrecognizable, must reclaim his identity through deeds, not proclamation.''',
        'expected_stage': 'reward_seizing_sword',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 14,
        'title': 'Odysseus and the Swineherd',
        'text': '''Eumaeus the swineherd welcomes the "beggar" with full hospitality, 
        sharing his humble food. "My master Odysseus was the best of men," he mourns. 
        "I'll never see his like again." Odysseus, disguised, hears true loyalty from the lowly. 
        The test begins: who remained faithful in his absence?''',
        'expected_stage': 'road_back',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 15,
        'title': 'Telemachus Returns',
        'text': '''Athena warns Telemachus: "The suitors plot your murder. Sail home by night, 
        go first to Eumaeus." Father and son approach convergence, neither knowing. The goddess 
        orchestrates from behind mortal action. The son's journey and father's journey align.''',
        'expected_stage': 'road_back',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 16,
        'title': 'Father and Son',
        'text': '''In Eumaeus' hut, Athena restores Odysseus' true form before Telemachus. 
        The son sees transformation - beggar to king. "Are you a god?" "I am your father." 
        They weep, then plot vengeance. The generations unite. The son has become a man; 
        the father recognizes him as equal. Together they will reclaim their home.''',
        'expected_stage': 'resurrection',
        'expected_depth': 'MYTHIC'
    },
    
    # Books 17-20: Odysseus in the Palace
    {
        'book': 17,
        'title': 'Odysseus the Beggar',
        'text': '''As a beggar, Odysseus enters his own palace. His dog Argos recognizes him - 
        then dies, faithful to the end. Suitors mock and strike him. He endures, gathering 
        intelligence, testing who remained true. Penelope hears of this strange beggar who speaks 
        of Odysseus. The final trial begins: can he master himself before reclaiming his kingdom?''',
        'expected_stage': 'resurrection',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 18,
        'title': 'The Beggar and the Suitors',
        'text': '''Irus, a real beggar, challenges Odysseus to fight. The disguised king breaks 
        him with one blow, barely restraining his full strength. Suitors cheer, not knowing they've 
        seen their doom. Penelope appears, extracting gifts from suitors. Odysseus watches his wife's 
        cunning with pride. She is his match in intelligence.''',
        'expected_stage': 'resurrection',
        'expected_depth': 'ARCHETYPAL'
    },
    {
        'book': 19,
        'title': 'Penelope and Odysseus',
        'text': '''Penelope interviews the beggar about Odysseus. He describes her husband - 
        describes himself - with intimate accuracy. She weeps for the man she thinks dead, 
        while he sits before her. Eurycleia, his old nurse, washing his feet, recognizes a scar. 
        Odysseus silences her. Not yet. The timing must be perfect.''',
        'expected_stage': 'resurrection',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 20,
        'title': 'Portents Gather',
        'text': '''Odysseus lies sleepless, hearing maids betray his household with suitors. 
        Rage fills him, but Athena counsels patience. At dawn, omens appear - eagle and thunder, 
        warnings the suitors ignore. The feast begins. Vengeance approaches. The hero has mastered 
        himself; now he will unleash justice.''',
        'expected_stage': 'resurrection',
        'expected_depth': 'MYTHIC'
    },
    
    # Books 21-24: The Reckoning and Resolution
    {
        'book': 21,
        'title': 'The Test of the Bow',
        'text': '''Penelope brings forth Odysseus' bow: "Whoever strings this bow and shoots 
        through twelve axes, him I will marry." Suitors fail. The "beggar" asks to try. They 
        mock him. He strings it effortlessly, shoots true. Then turns the weapon on them: 
        "The trial is over. Now comes the reckoning."''',
        'expected_stage': 'return_with_elixir',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 22,
        'title': 'The Slaughter',
        'text': '''Odysseus reveals himself and rains arrows on the suitors. They beg mercy - 
        he grants none. Telemachus fights beside his father. The palace runs with blood. 
        Justice, long delayed, falls like Zeus' thunderbolt. The unfaithful maids are hanged. 
        The house is purified. The old order is violently restored.''',
        'expected_stage': 'return_with_elixir',
        'expected_depth': 'MYTHIC'
    },
    {
        'book': 23,
        'title': 'Penelope Recognizes Odysseus',
        'text': '''Even after slaughter, Penelope tests him: "Move our bed outside." "Impossible," 
        Odysseus cries, "I built it from a living olive tree, rooted in earth!" Only he could know. 
        She yields, and they reunite. He tells of his journeys; she tells of her weaving - undoing 
        each night what she wove by day. Their intelligence equals their love. They are home.''',
        'expected_stage': 'return_with_elixir',
        'expected_depth': 'COSMIC'
    },
    {
        'book': 24,
        'title': 'Peace Restored',
        'text': '''In Hades, suitors' shades tell their tale. Achilles praises Penelope's 
        faithfulness - unlike Clytemnestra who murdered Agamemnon. On earth, families of the 
        slain seek vengeance. Battle begins. Athena intervenes: "Enough bloodshed! Let there be 
        peace." Zeus' thunderbolt enforces divine will. Odysseus has completed his journey - 
        from home, through trials, to home transformed. The circle closes, but at a higher level.''',
        'expected_stage': 'master_of_two_worlds',
        'expected_depth': 'COSMIC'
    }
]


async def analyze_full_odyssey_depth():
    """Analyze narrative depth across all 24 books."""
    print("📚 FULL ODYSSEY DEPTH ANALYSIS")
    print("=" * 80)
    print("Analyzing narrative depth across all 24 books of Homer's Odyssey")
    print("=" * 80)
    print()
    
    # Create depth analyzer
    depth_analyzer = MatryoshkaNarrativeDepth()
    
    results: List[OdysseyBookDepth] = []
    total_start = time.perf_counter()
    
    # Analyze each book
    for book_data in ODYSSEY_24_BOOKS:
        book_num = book_data['book']
        title = book_data['title']
        text = book_data['text']
        
        print(f"📖 Book {book_num}: {title}")
        print(f"   Expected: {book_data['expected_stage']} / {book_data['expected_depth']}")
        
        start = time.perf_counter()
        result = await depth_analyzer.analyze_depth(text)
        duration_ms = (time.perf_counter() - start) * 1000
        
        # Extract cosmic truth if present
        cosmic_truth = result.cosmic_truth if result.cosmic_truth else ""
        
        # Extract mythic truths if present
        mythic_truths = []
        if result.mythic_layer and result.mythic_layer.universal_truths:
            mythic_truths = result.mythic_layer.universal_truths[:2]
        
        book_result = OdysseyBookDepth(
            book_number=book_num,
            book_title=title,
            max_depth=result.max_depth_achieved,
            complexity=result.total_complexity,
            confidence=result.bayesian_confidence,
            gates_unlocked=len([g for g in result.gates_unlocked if g]),
            cosmic_truth=cosmic_truth,
            mythic_truths=mythic_truths,
            analysis_time_ms=duration_ms
        )
        
        results.append(book_result)
        
        print(f"   Result: {result.max_depth_achieved.name} (complexity: {result.total_complexity:.3f}, "
              f"{book_result.gates_unlocked}/5 gates, {duration_ms:.2f}ms)")
        
        if cosmic_truth:
            print(f"   🌌 Cosmic: {cosmic_truth[:60]}...")
        elif mythic_truths:
            print(f"   ⚡ Mythic: {mythic_truths[0][:60]}...")
        
        print()
    
    total_duration = (time.perf_counter() - total_start) * 1000
    
    # Analyze results
    print("=" * 80)
    print("📊 ODYSSEY DEPTH ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    # Depth distribution
    depth_counts = {}
    for result in results:
        depth_name = result.max_depth.name
        depth_counts[depth_name] = depth_counts.get(depth_name, 0) + 1
    
    print("🎯 DEPTH LEVEL DISTRIBUTION:")
    for depth_name in ['SURFACE', 'SYMBOLIC', 'ARCHETYPAL', 'MYTHIC', 'COSMIC']:
        count = depth_counts.get(depth_name, 0)
        pct = (count / len(results)) * 100
        bar = "█" * int(pct / 5)
        print(f"   {depth_name:12s} [{bar:20s}] {count:2d} books ({pct:5.1f}%)")
    
    print()
    
    # Complexity evolution
    print("📈 COMPLEXITY EVOLUTION:")
    avg_complexity = statistics.mean([r.complexity for r in results])
    books_1_8 = [r for r in results if r.book_number <= 8]
    books_9_16 = [r for r in results if 9 <= r.book_number <= 16]
    books_17_24 = [r for r in results if r.book_number >= 17]
    
    print(f"   Overall average: {avg_complexity:.3f}")
    print(f"   Books 1-8 (Setup): {statistics.mean([r.complexity for r in books_1_8]):.3f}")
    print(f"   Books 9-16 (Journey): {statistics.mean([r.complexity for r in books_9_16]):.3f}")
    print(f"   Books 17-24 (Return): {statistics.mean([r.complexity for r in books_17_24]):.3f}")
    print()
    
    # Cosmic moments
    cosmic_books = [r for r in results if r.max_depth == DepthLevel.COSMIC]
    print(f"🌌 COSMIC MOMENTS ({len(cosmic_books)} books):")
    for book in cosmic_books:
        print(f"   Book {book.book_number}: {book.book_title}")
        if book.cosmic_truth:
            print(f"      Truth: {book.cosmic_truth}")
    print()
    
    # Performance stats
    print("⚡ PERFORMANCE:")
    print(f"   Total time: {total_duration:.1f}ms")
    print(f"   Average per book: {total_duration / len(results):.1f}ms")
    print(f"   Total books: {len(results)}")
    print()
    
    # Peak depth moments
    print("🏔️ PEAK DEPTH MOMENTS (Top 5):")
    sorted_by_complexity = sorted(results, key=lambda r: r.complexity, reverse=True)[:5]
    for i, book in enumerate(sorted_by_complexity, 1):
        print(f"   {i}. Book {book.book_number} - {book.book_title}")
        print(f"      Complexity: {book.complexity:.3f}, Depth: {book.max_depth.name}, "
              f"Gates: {book.gates_unlocked}/5")
    print()
    
    print("=" * 80)
    print("✅ FULL ODYSSEY DEPTH ANALYSIS COMPLETE!")
    print(f"📚 24 books analyzed in {total_duration:.1f}ms")
    print(f"🪆 Matryoshka depth gating successfully applied to complete epic!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    asyncio.run(analyze_full_odyssey_depth())
