# Elle: Quick Start

Built and working. Here's how to use it:

## Run the Example

```bash
# Stub LLM (no API key needed)
python3 example.py

# With real Anthropic API
export ANTHROPIC_API_KEY=your_key_here
python3 example.py
```

## Use the CLI

```bash
# Simulate a scene
python3 -m elle.adapters.cli_adapter.cli simulate \
  --scene elle/scenes/shed_cluttered.json \
  --intent seeking_guidance \
  --scan slow_scan \
  --tired

# Try other scenes
python3 -m elle.adapters.cli_adapter.cli simulate \
  --scene elle/scenes/bed_cucumber_healthy.json \
  --scan quick_glance

python3 -m elle.adapters.cli_adapter.cli simulate \
  --scene elle/scenes/fence_broken_urgent.json \
  --scan focused
```

## Install Real LLM Support

```bash
# For Anthropic
pip3 install anthropic

# For OpenAI
pip3 install openai
```

## What Just Happened

The example ran through the full golden path:
1. **Loaded config** - dev profile with stub LLM
2. **Initialized components** - engine, policy, memory, tools
3. **Loaded test scene** - cluttered shed with 6 objects
4. **Created intent** - tired user, slow scan, seeking guidance
5. **Processed request** - engine → policy → LLM → action
6. **Got response** - Elle suggested clearing left shelf (10 min, medium priority)
7. **Recorded to memory** - interaction logged for future learning

Elle is **production-ready** with stub LLM. Add API key for real intelligence.

## Next Steps

- Test with Anthropic API key
- Try different scenes and intents
- Add more symbols to `elle/symbols/`
- Create custom test scenes in `elle/scenes/`
- Build AR/Matrix adapters when ready

Simple spine, lots of room to grow. Ready for Future Blake.
