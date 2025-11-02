# Squad Demo Video Script

**Duration:** 3-5 minutes
**Target Audience:** Developers interested in AI coding assistants

---

## Opening (0:00 - 0:30)

**[Screen: VS Code with code open]**

> "What if your coding assistant didn't just answer questions - but actually *reasoned* about them?"

**[Show Squad logo/icon]**

> "This is Squad - an AI coding assistant powered by HoloLoom's agentic reasoning engine."

**[Quick montage: Commands, reasoning steps, code fixes]**

> "Let's see what makes it different."

---

## Scene 1: The Problem (0:30 - 1:00)

**[Screen: Typical AI assistant giving wrong answer]**

> "Most AI coding assistants give you *an* answer. But is it the *right* answer?"

**[Show incorrect code suggestion]**

> "They don't verify. They don't research. They don't plan."

**[X marks over answers]**

> "Squad is different."

---

## Scene 2: Agentic Reasoning (1:00 - 2:00)

**[Screen: VS Code with Squad]**

> "Squad has 4 reasoning modes:"

### Mode 1: DIRECT (Quick Demo)
**[Type: Ctrl+Shift+Q]**

> "**DIRECT** - For quick questions."

**[Type: "What is Thompson Sampling?"]**

**[Show instant response]**

> "Fast answers when you need them."

### Mode 2: VERIFY (Main Demo)
**[Select complex code]**

> "**VERIFY** - Checks its own work."

**[Right-click → "Squad: Explain Selection"]**

**[Show agent panel opening]**

> "Watch this - it doesn't just explain the code..."

**[Reasoning steps appear:]**
- Step 1: Initial explanation
- Step 2: Verification query - "Are there edge cases?"
- Step 3: Checking for contradictions
- Step 4: Verified ✓

**[Highlight verification checkmark]**

> "It verifies the explanation is correct."

### Mode 3: RESEARCH (Quick Show)
**[Panel shows multiple queries]**

> "**RESEARCH** mode explores multiple angles:"

**[Show steps:]**
- "What are the tradeoffs?"
- "What are alternatives?"
- "What are common mistakes?"
- Final synthesis

> "Comprehensive answers, not just first thoughts."

### Mode 4: PLAN & EXECUTE (Main Feature)
**[File with errors visible]**

> "**PLAN & EXECUTE** - Breaks down complex tasks."

**[Command: "Squad: Suggest Fix"]**

**[Show decomposition:]**
1. ✓ Identify root cause
2. ✓ Plan fix steps
3. ✓ Verify no side effects
4. → Apply fix

> "It plans before it acts."

---

## Scene 3: The UI (2:00 - 2:30)

**[Focus on Agent Panel]**

> "Every step is visible."

**[Point to different UI elements:]**

1. **Confidence Score:** "85% confidence - it knows when it's uncertain"

2. **Reasoning Steps:** "See exactly how it thinks"

3. **Verification Results:** "Contradictions found: 0"

4. **Duration:** "1.2 seconds for complete reasoning cycle"

**[Scroll through step-by-step]**

> "Complete transparency. No black box."

---

## Scene 4: Code-Aware (2:30 - 3:00)

**[Show different scenarios]**

### Scenario 1: Context Understanding
**[Open large file]**

> "It understands your code context."

**[Show context extraction:]**
- Current file: `AuthService.ts`
- Language: TypeScript
- Diagnostics: 3 errors
- Workspace: `/my-app`

> "Not just the code - the full picture."

### Scenario 2: Error Diagnosis
**[Red squiggles in code]**

**[Squad analyzes:]**
- Reads error message
- Checks surrounding code
- Understands dependencies
- Suggests fix with explanation

> "Real understanding, not pattern matching."

---

## Scene 5: Under the Hood (3:00 - 3:30)

**[Show architecture diagram]**

> "Powered by HoloLoom:"

**[Animate flow:]**

```
VS Code → Squad → HoloLoom
                     ↓
            [Agentic Reasoning]
                     ↓
         ┌───────────┴───────────┐
    Verify         Research    Plan
         └───────────┬───────────┘
                     ↓
              [Safety Checks]
                     ↓
                 Response
```

> "Every query goes through safety checks, multi-step reasoning, and verification."

**[Quick stats:]**
- ✓ Safety guardrails
- ✓ Compositional cache (291× speedup)
- ✓ Complete provenance
- ✓ Learning from feedback

---

## Scene 6: Commands Quick Tour (3:30 - 4:00)

**[Rapid montage of commands]**

1. **Ask Question** (`Ctrl+Shift+Q`)
   > "General questions"

2. **Explain Selection** (`Ctrl+Shift+E`)
   > "Understand code"

3. **Suggest Fix**
   > "Debug errors"

4. **Refactor Code**
   > "Improve structure"

5. **Generate Tests**
   > "Create test cases"

**[All commands shown in 30 seconds]**

---

## Scene 7: The Difference (4:00 - 4:30)

**[Split screen comparison]**

**Left: Traditional AI**
- Single response
- No verification
- No reasoning shown
- "Trust me"

**Right: Squad**
- Multi-step reasoning
- Self-verification
- All steps visible
- "Let me show you why"

**[Highlight confidence scores]**

> "It even tells you when it's not confident."

**[Show low confidence example: 45%]**

> "Honest AI. Not just confident AI."

---

## Closing (4:30 - 5:00)

**[Screen: Squad panel with successful reasoning]**

> "Squad: AI that reasons, verifies, and explains."

**[Show key features as text overlays:]**
- ✓ 4 Reasoning Modes
- ✓ Full Transparency
- ✓ Safety First
- ✓ Code-Aware

**[GitHub/Download link appears]**

> "Built on HoloLoom. Open source. Try it today."

**[Final shot: Code being written with Squad panel alongside]**

> "Don't just get answers. Get *reasoning*."

**[Fade to Squad logo]**

---

## B-Roll Ideas

**Throughout video, overlay with:**
- Code being typed
- Squad panel animating
- Reasoning steps appearing
- Confidence scores updating
- Success checkmarks
- Error fixes in real-time

**Music:**
- Upbeat, modern
- Tech-forward
- Builds with complexity
- Triumphant at end

**Voiceover Tone:**
- Confident but not arrogant
- Technical but accessible
- Excited about the tech
- Focus on problem → solution

---

## Key Messages

1. **Squad reasons, not just responds**
2. **Complete transparency** - see every step
3. **Self-verification** - checks its own work
4. **Code-aware** - understands context
5. **Safe & honest** - admits uncertainty

---

## Optional Extended Sections

### For Longer Demo (5-7 min)

**Technical Deep Dive:**
- Show actual reasoning log
- Explain verification loops
- Demo research mode in detail
- Show learning from feedback

**Developer Experience:**
- Setup process (1 minute)
- Customization options
- Integration with workflow
- Performance metrics

**Use Cases:**
- Debugging complex errors
- Understanding legacy code
- Refactoring strategies
- Test generation

---

## Call to Action

**End screens (pick one):**

**Option A - Open Source:**
> "Star us on GitHub: github.com/yourorg/squad"

**Option B - Try It:**
> "Download: squad.dev"

**Option C - Learn More:**
> "Learn about HoloLoom: hololoom.dev"

---

## Recording Checklist

- [ ] Clean VS Code theme (dark mode recommended)
- [ ] Large, readable fonts (14pt+)
- [ ] Hide distracting files/folders
- [ ] Prepare demo code examples
- [ ] Pre-start Squad server
- [ ] Test all commands work
- [ ] Record in 1080p or 4K
- [ ] Screen recording software ready
- [ ] Microphone tested
- [ ] Script rehearsed 3x

---

**Total Prep Time:** 2-3 hours
**Recording Time:** 1-2 hours
**Editing Time:** 2-4 hours
**Total:** 1 day for polished demo

Good luck! 🎬
