# 🚀 Running the Demos

All three demo scripts are now ready to run from their directory!

## Quick Start

```bash
cd C:\Users\blake\Documents\mythRL\HoloLoom\semantic_calculus

# Demo 1: True Matryoshka nesting (4x speedup)
python matryoshka_streaming.py

# Demo 2: Recursive composition (inheritance chains)
python recursive_matryoshka.py

# Demo 3: Multi-projection spaces (multiple perspectives)
python multi_projection.py
```

## What Each Demo Shows

### 1. matryoshka_streaming.py
**True Matryoshka nesting with matched temporal + dimensional scales**

Output shows:
- Word-level: 96D → 16D (fastest)
- Phrase-level: 192D → 64D
- Sentence-level: 384D → 128D
- Paragraph-level: 384D → 244D (full richness)
- Performance report showing 4x speedup

**Look for:** Speedup metrics at end

### 2. recursive_matryoshka.py
**Recursive composition where each level inherits previous understanding**

Output shows:
- Word-level: inherits from []
- Phrase-level: inherits from [word]
- Sentence-level: inherits from [phrase, word]
- Paragraph-level: inherits from [sentence, phrase, word]

**Look for:** Inheritance chains in snapshots

### 3. multi_projection.py
**Multiple projection spaces running simultaneously**

Output shows:
- Semantic projection (244D narrative dimensions)
- Emotion projection (48D affective dimensions)
- Archetype projection (32D Jungian archetypes)
- Raw embedding (96D pass-through)
- Agreement metric (how much projections align)

**Look for:** Cross-projection agreement scores

## Troubleshooting

### If you see "No module named 'HoloLoom'"
- Make sure you're running from: `C:\Users\blake\Documents\mythRL\HoloLoom\semantic_calculus`
- The scripts add the repo root to sys.path automatically

### If you see "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### If you see warnings about missing dependencies
The system gracefully degrades - demos will still run with fallback implementations

## Expected Runtime

Each demo takes ~10-30 seconds to complete:
- Initialization: 5-10s (loading embeddings, learning axes)
- Analysis: 5-10s (streaming word-by-word)
- Summary: instant

## What To Verify

After running all three demos, check:

✅ **matryoshka_streaming.py**
- Shows performance report
- Word-level is 3-5x faster than paragraph-level
- All scales produce valid semantic projections

✅ **recursive_matryoshka.py**
- Shows inheritance chains correctly
- Paragraph inherits from all three lower levels
- Inheritance depth increases at each level

✅ **multi_projection.py**
- All four projections run successfully
- Agreement metric is between 0-1
- Interpretations make semantic sense

## Next Steps

Once all demos run successfully:

1. Review [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) - Full testing checklist
2. Test on your own texts
3. Experiment with different parameters
4. See [MEANING_AS_FEATURE.md](MEANING_AS_FEATURE.md) for integration roadmap

## Common Questions

**Q: Why do demos take time to initialize?**
A: Learning 244D semantic axes from embeddings takes 5-10 seconds (one-time cost per run)

**Q: Can I speed up the demos?**
A: Yes! Reduce `snapshot_interval` or `words_per_second` parameters in the code

**Q: Where's the visualization?**
A: Run `visualize_streaming.py` for live 6-panel graphs (requires matplotlib)

**Q: Can I use real sentence-transformers?**
A: Yes! Install with `pip install sentence-transformers` - will auto-detect and use

---

**Ready to verify!** 🧪✨
