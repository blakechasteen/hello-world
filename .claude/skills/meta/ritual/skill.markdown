# Skill: Coding Ritual Orchestrator

## Metadata

- **Name**: `ritual`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-12-30`
- **Last Updated**: `2025-12-30`
- **Category**: `meta`
- **Tags**: `workflow, session, productivity, coding-ritual, structured-development`

## Description

**Short Description** (1-2 sentences):
A structured coding ritual system that provides ceremony and discipline for development sessions, integrating protocol-first design, wave validation, and safety-first principles.

**Detailed Description**:
The Coding Ritual skill implements a complete workflow for structured development sessions:

1. **Session Opening** (`/ritual open`) - Initialize a focused coding session with clear goals
2. **Protocol-First Design** (`/ritual design`) - Define interfaces before implementation
3. **Wave Validation Build** (`/ritual build`) - Three-wave validation: Core → Edge cases → Integration
4. **Safety Checklist** (`/ritual check`) - Systematic safety and quality review
5. **Session Closing** (`/ritual close`) - Capture lessons learned and archive session

The ritual is designed around key development principles:
- **"Protocol-first design"** - Define interfaces before implementation
- **"Wave validation"** - Validate in progressive waves (does it work? what breaks it? does it play nice?)
- **"Safety-first"** - Every session includes safety review
- **"Learn from every session"** - Capture and preserve lessons learned

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read) - Read session state, past sessions
- [x] File system access (write) - Persist session state, create archives
- [ ] Code execution (bash) - Optional: git diff metrics
- [ ] Code execution (python) - Optional: run tests
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [x] User interaction (questions) - Ask for success criteria, lessons learned

## Dependencies

**Required Skills** (if this skill builds on others):
- None (standalone, but integrates with others when available)

**Optional Integration Skills**:
- `skill_tester` - Invoked by `/ritual check` when available
- `skill_security_analyzer` - Invoked by `/ritual check` when available
- `continuous_learning_capture` - Subscribes to ritual events

**External Dependencies**:
- None required

**HoloLoom Integration** (optional):
- [ ] Uses HoloLoom memory system - Store session context
- [ ] Uses HoloLoom RAG - Search past sessions
- [ ] Uses HoloLoom alignment framework - Safety checks
- [ ] Uses HoloLoom learning systems - Phase recommendations

## Input Schema

**Expected Input Format**:
```json
{
  "command": "open | design | build | check | close | status | help",
  "args": {
    "task": "string (required for 'open')",
    "complexity": "lite | fast | full | research (optional, default: fast)",
    "component": "string (for 'design')",
    "wave": "1 | 2 | 3 (for 'build')",
    "criteria": ["array of strings (for 'open')"],
    "lessons": ["array of strings (for 'close')"]
  }
}
```

**Example Input (Open Session)**:
```json
{
  "command": "open",
  "args": {
    "task": "Implement user authentication",
    "complexity": "full",
    "criteria": [
      "Users can log in with email/password",
      "Passwords are hashed securely",
      "Session tokens expire after 24 hours"
    ]
  }
}
```

**Example Input (Check Safety)**:
```json
{
  "command": "check",
  "args": {}
}
```

## Output Schema

**Expected Output Format**:
```json
{
  "phase": "current phase name",
  "session_id": "session identifier",
  "status": "success | in_progress | blocked | error",
  "message": "human-readable status message",
  "next_steps": ["array of recommended actions"],
  "data": {
    "specific": "phase-specific data"
  },
  "metadata": {
    "execution_time_ms": 0,
    "events_emitted": ["event IDs if EventBus available"],
    "correlation_id": "session correlation ID"
  }
}
```

**Example Output (Session Opened)**:
```json
{
  "phase": "open",
  "session_id": "2025-12-30_implement-user-authentication",
  "status": "success",
  "message": "Ritual session opened. Ready to begin.",
  "next_steps": [
    "Use /ritual design to define interfaces",
    "Use /ritual build to start implementation",
    "Use /ritual status to check progress"
  ],
  "data": {
    "task": "Implement user authentication",
    "complexity": "full",
    "success_criteria": [
      "Users can log in with email/password",
      "Passwords are hashed securely",
      "Session tokens expire after 24 hours"
    ]
  },
  "metadata": {
    "execution_time_ms": 15,
    "events_emitted": ["ritual.session.opened"],
    "correlation_id": "ritual-abc12345"
  }
}
```

## Prompt Template

```markdown
You are executing the Coding Ritual skill, a structured development workflow system.

**Current Command**: {command}
**Session State**: {session_state}

## Commands

### /ritual open
Initialize a new coding session with:
- Task description
- Complexity level (lite/fast/full/research)
- Success criteria (what defines "done")

Ask for success criteria if not provided. Create session state and emit opening event.

### /ritual design
Guide through protocol-first interface design:
1. Ask clarifying questions about the component
2. Generate interface/protocol definition
3. Identify dependencies and potential fallbacks
4. Document the decision with rationale

### /ritual build
Execute wave validation build process:
- **Wave 1**: Core functionality - "Does it work?"
- **Wave 2**: Edge cases - "What breaks it?"
- **Wave 3**: Integration - "Does it play nice?"

Track which wave we're on. Block progression if wave fails.

### /ritual check
Run through safety-first checklist:
- [ ] Graceful degradation (fails safely?)
- [ ] Input validation (rejects bad input?)
- [ ] Error messages (clear and helpful?)
- [ ] No hardcoded secrets
- [ ] Timeout handling
- [ ] Resource cleanup

If skill_tester or skill_security_analyzer are available, invoke them.

### /ritual close
Wrap up the session:
1. Gather git diff metrics if available
2. Ask for lessons learned
3. Generate session summary
4. Archive to .ritual/sessions/
5. Update lessons.md

### /ritual status
Show current session status:
- Phase
- Progress through waves
- Success criteria status
- Time elapsed

### /ritual help
Display available commands and their usage.

**Requirements**:
1. Always check for active session before operations (except open/help)
2. Persist state to .ritual/current_session.json
3. Emit events when EventBus is available
4. Be encouraging but honest about progress
5. Guide the user through the ritual, don't just execute commands

**Quality Standards**:
- Clear, actionable feedback at each step
- Preserve all session data for future learning
- Integrate with other skills when available
- Fail gracefully if dependencies missing
```

## Commands Reference

### `/ritual open [task]`
Start a new coding session.

**Arguments**:
- `task` (required): Description of what you're building
- `--complexity`: lite, fast (default), full, or research
- `--criteria`: Success criteria (will prompt if not provided)

**Example**:
```
/ritual open "Add dark mode toggle" --complexity fast
```

### `/ritual design [component]`
Design a component with protocol-first approach.

**Arguments**:
- `component` (optional): Name of component to design

**Example**:
```
/ritual design ThemeProvider
```

### `/ritual build [wave]`
Progress through build waves.

**Arguments**:
- `wave` (optional): 1, 2, or 3 (defaults to current + 1)

**Example**:
```
/ritual build 2
```

### `/ritual check`
Run safety checklist review.

**Example**:
```
/ritual check
```

### `/ritual close`
End the session and archive.

**Example**:
```
/ritual close
```

### `/ritual status`
Show current session status.

**Example**:
```
/ritual status
```

### `/ritual help`
Show help for ritual commands.

**Example**:
```
/ritual help
```

## Examples

### Example 1: Basic Session

**Input**: User says "/ritual open Add user avatar upload"

**Expected Behavior**:
1. Create session with task "Add user avatar upload"
2. Prompt for success criteria
3. Create .ritual/current_session.json
4. Display session opened message with next steps

**Expected Output**:
```
✨ Ritual Session Opened

**Task**: Add user avatar upload
**Complexity**: fast
**Session ID**: 2025-12-30_add-user-avatar-upload

Let's define success criteria. What needs to be true when you're done?

Suggestions:
- Users can upload images
- Images are validated and resized
- Avatar displays in profile

Enter your criteria (one per line, empty line when done):
```

### Example 2: Design Phase

**Input**: User says "/ritual design ImageUploader"

**Expected Behavior**:
1. Ask clarifying questions about the component
2. Generate interface definition
3. Identify dependencies
4. Document decision

**Expected Output**:
```
📐 Protocol-First Design: ImageUploader

**Clarifying Questions**:
1. What image formats should be supported?
2. What's the maximum file size?
3. Should we support crop/resize?
4. Where are images stored (local, S3, etc.)?

[After user answers]

**Proposed Interface**:
```python
class ImageUploader(Protocol):
    async def upload(self, file: UploadFile) -> ImageResult: ...
    async def validate(self, file: UploadFile) -> ValidationResult: ...
    async def resize(self, image_id: str, dimensions: tuple) -> ImageResult: ...
```

**Dependencies**:
- PIL/Pillow for image processing
- Storage backend (S3 or local)

**Fallbacks**:
- If resize fails, keep original
- If storage fails, return clear error
```

### Example 3: Wave Validation

**Input**: User says "/ritual build 1"

**Expected Behavior**:
1. Guide through Wave 1 (core functionality)
2. Ask if core functionality works
3. Record result

**Expected Output**:
```
🌊 Wave 1: Core Functionality

"Does it work?"

This wave validates the happy path. Check:
- [ ] Basic upload works
- [ ] Image appears in storage
- [ ] Response returns image URL

When you've verified these, mark as passed or note issues.

[P]ass / [F]ail / [N]otes:
```

### Example 4: Safety Check

**Input**: User says "/ritual check"

**Expected Behavior**:
1. Run through safety checklist
2. Invoke skill_tester if available
3. Invoke skill_security_analyzer if available
4. Record results

**Expected Output**:
```
🛡️ Safety-First Checklist

**Graceful Degradation**:
- [ ] What happens if storage is unavailable?
- [ ] What happens if image is corrupted?
→ Does it fail safely? [Y/N]

**Input Validation**:
- [ ] File size limits enforced?
- [ ] File type validated (not just extension)?
- [ ] Malicious content checked?
→ Rejects bad input? [Y/N]

**Error Messages**:
- [ ] Clear error for invalid file type?
- [ ] Clear error for size exceeded?
→ Helpful and safe? [Y/N]

**Security**:
- [ ] No hardcoded credentials?
- [ ] Secure file naming (no path traversal)?
→ Secure? [Y/N]

[skill_security_analyzer available - running automated checks...]
```

### Example 5: Session Close

**Input**: User says "/ritual close"

**Expected Behavior**:
1. Gather git diff if available
2. Prompt for lessons learned
3. Archive session
4. Update lessons.md

**Expected Output**:
```
🏁 Closing Ritual Session

**Session Summary**:
- Task: Add user avatar upload
- Duration: 2h 15m
- Waves completed: 3/3
- Safety checks: 6/6 passed

**Git Metrics**:
- Files changed: 8
- Lines added: 245
- Lines removed: 12

**Lessons Learned**:
What did you learn from this session?

> Always validate file content, not just extension
> S3 pre-signed URLs simplify upload flow

Session archived to: .ritual/sessions/2025-12-30_add-user-avatar-upload.md
Lessons added to: .ritual/lessons.md

✨ Ritual complete. Well done!
```

## Testing Checklist

Before deploying this skill, verify:

- [x] **Functionality**: All commands execute correctly
- [x] **State Persistence**: Session state survives restarts
- [x] **Error Handling**: Graceful handling of missing session, corrupted state
- [ ] **Security**: No prompt injection vulnerabilities (run `skill_security_analyzer`)
- [x] **Performance**: Commands execute quickly (<100ms)
- [x] **Token Efficiency**: Prompts are concise
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: All dependencies documented
- [x] **Edge Cases**: Handles edge cases (no session, double open, etc.)
- [x] **Output Consistency**: Returns consistent format

## Security Considerations

**Potential Risks**:
- Session data written to disk (mitigated: .gitignore for current_session.json)
- User-provided task names in filenames (mitigated: sanitization in session_id generation)

**Data Privacy**:
- [x] Does not log sensitive user data
- [x] Does not expose internal system details
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Does not attempt privilege escalation
- [x] Only writes to .ritual/ directory

## Performance Characteristics

- **Expected Latency**: <100ms for all commands
- **Token Usage**: ~500-1000 tokens per interaction
- **Resource Requirements**: Minimal (file I/O only)
- **Scalability**: Single-user (one session at a time)

## EventBus Integration

When EventBus is available, the ritual skill emits events for integration with other skills:

**Events Emitted**:
- `ritual.session.opened` - Session started
- `ritual.design.started` - Design phase begun
- `ritual.design.completed` - Design recorded
- `ritual.build.wave_N` - Build wave N started/completed
- `ritual.check.completed` - Safety check finished
- `ritual.session.closed` - Session ended

**Event Payload**:
```json
{
  "session_id": "2025-12-30_task-name",
  "correlation_id": "ritual-abc12345",
  "causation_id": "previous-event-id",
  "sequence": 1,
  "payload": {
    "phase": "open",
    "task": "Task description",
    "complexity": "fast"
  }
}
```

**Subscribers**:
- `continuous_learning_capture` - Captures session lessons
- `skill_tester` - Responds to check events
- `skill_security_analyzer` - Responds to check events

## Maintenance Notes

**Known Limitations**:
- Single session at a time
- Local file storage only (no cloud sync)
- Manual success criteria tracking

**Future Enhancements**:
- Thompson Sampling for phase recommendations
- Automatic success criteria from task description
- Integration with project management tools
- Team ritual support (shared sessions)

**Changelog**:
- **v1.0.0** (2025-12-30): Initial release with core functionality

## License

MIT License - Part of HoloLoom project

## Support

**Issues**: https://github.com/HoloLoom/HoloLoom/issues
**Documentation**: See .claude/skills/meta/ritual/README.md
**Contributors**: HoloLoom Team