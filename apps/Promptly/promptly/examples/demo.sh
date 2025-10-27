#!/bin/bash

# Promptly Demo Script
# This script demonstrates all major features of Promptly

set -e

echo "=================================="
echo "Promptly Feature Demo"
echo "=================================="
echo ""

# Clean up any existing demo
if [ -d ".promptly" ]; then
    echo "Cleaning up existing repository..."
    rm -rf .promptly
    echo ""
fi

# 1. Initialize
echo "1. Initializing Promptly repository..."
python3 promptly.py init
echo ""

# 2. Add prompts
echo "2. Adding prompts..."
python3 promptly.py add summarizer "Summarize the following text in 2-3 sentences: {text}"
python3 promptly.py add translator "Translate the following text to {language}: {text}"
python3 promptly.py add analyzer "Analyze the sentiment and key themes in: {text}"
echo ""

# 3. List prompts
echo "3. Listing all prompts..."
python3 promptly.py list
echo ""

# 4. Get a specific prompt
echo "4. Getting the 'summarizer' prompt..."
python3 promptly.py get summarizer
echo ""

# 5. Update a prompt (creates new version)
echo "5. Updating 'summarizer' to create version 2..."
python3 promptly.py add summarizer "Provide a concise 2-3 sentence summary of the following text, focusing on the main ideas: {text}"
echo ""

# 6. Update again for version 3
echo "6. Creating version 3 with more specific instructions..."
python3 promptly.py add summarizer "Provide a clear and concise summary (2-3 sentences) of the following text. Focus on: main ideas, key facts, and actionable insights. Text: {text}"
echo ""

# 7. View history
echo "7. Viewing commit history for 'summarizer'..."
python3 promptly.py log --name summarizer
echo ""

# 8. Create a branch
echo "8. Creating experimental branch..."
python3 promptly.py branch experimental
echo ""

# 9. Switch to experimental branch
echo "9. Switching to experimental branch..."
python3 promptly.py checkout experimental
echo ""

# 10. Make changes on experimental branch
echo "10. Adding experimental version on new branch..."
python3 promptly.py add summarizer "🚀 EXPERIMENTAL: Summarize this with bullet points: {text}"
python3 promptly.py add creative-writer "Write a creative story about: {topic}"
echo ""

# 11. List prompts on experimental branch
echo "11. Listing prompts on experimental branch..."
python3 promptly.py list
echo ""

# 12. Switch back to main
echo "12. Switching back to main branch..."
python3 promptly.py checkout main
echo ""

# 13. Create another branch for A/B testing
echo "13. Creating A/B test branches..."
python3 promptly.py branch variant-a
python3 promptly.py branch variant-b
echo ""

# 14. Setup variant-a
echo "14. Setting up variant-a..."
python3 promptly.py checkout variant-a
python3 promptly.py add ad-copy "Create compelling ad copy with emotional appeal for: {product}. Target audience: {audience}"
echo ""

# 15. Setup variant-b
echo "15. Setting up variant-b..."
python3 promptly.py checkout variant-b
python3 promptly.py add ad-copy "Create data-driven ad copy with statistics and facts for: {product}. Target audience: {audience}"
echo ""

# 16. Back to main and create chain
echo "16. Back to main - creating content pipeline..."
python3 promptly.py checkout main
python3 promptly.py add outline "Create a detailed outline for an article about: {topic}"
python3 promptly.py add expand "Expand the following outline into full paragraphs: {output}"
python3 promptly.py add polish "Polish and improve the following text for clarity and engagement: {output}"
echo ""

# 17. Create the chain
echo "17. Creating content creation chain..."
python3 promptly.py chain create content-pipeline outline expand polish --description "Complete content creation workflow"
echo ""

# 18. View all branches
echo "18. Summary of all branches:"
echo "   - main (original prompts)"
echo "   - experimental (experimental features)"
echo "   - variant-a (emotional ad copy)"
echo "   - variant-b (data-driven ad copy)"
echo ""

# 19. Final log
echo "19. Viewing recent commits on main..."
python3 promptly.py log --limit 5
echo ""

# 20. Show structure
echo "20. Repository structure:"
echo ""
tree -L 3 .promptly/ 2>/dev/null || find .promptly -type f | head -20
echo ""

echo "=================================="
echo "Demo Complete!"
echo "=================================="
echo ""
echo "What was demonstrated:"
echo "  ✓ Repository initialization"
echo "  ✓ Adding and updating prompts (versioning)"
echo "  ✓ Viewing prompts and history"
echo "  ✓ Creating and switching branches"
echo "  ✓ Setting up A/B testing variants"
echo "  ✓ Creating prompt chains"
echo ""
echo "Try these commands yourself:"
echo "  promptly get summarizer"
echo "  promptly log --name summarizer"
echo "  promptly checkout experimental"
echo "  promptly list"
echo ""
echo "For evaluation and chain execution, see:"
echo "  examples/test_cases.json"
echo "  examples/chain_input.yaml"
echo "  examples/advanced_integration.py"
echo ""
