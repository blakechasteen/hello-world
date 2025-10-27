# Example .sip file (Skill Instruction Protocol)
# This is a custom file format for skill instructions

SKILL: data_processor
VERSION: 1.0
RUNTIME: claude

INSTRUCTIONS:
1. Read the input data
2. Apply transformation rules from config.json
3. Execute processor.py logic
4. Return results in specified output format

EXPECTED_INPUT:
- Array of strings or objects
- Valid JSON format

EXPECTED_OUTPUT:
- Processed data structure
- Metadata about processing
