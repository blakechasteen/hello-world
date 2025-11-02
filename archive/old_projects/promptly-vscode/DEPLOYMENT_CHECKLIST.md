# Promptly v0.2.0 - Deployment Checklist

## 🚢 Pre-Deployment Verification

### ✅ Code Compilation
- [x] TypeScript compiles without errors
- [x] All source files present
- [x] Output files generated:
  - [x] extension.js (3.8KB)
  - [x] ExecutionClient.js (7.4KB)
  - [x] ExecutionPanel.js (29.7KB)
  - [x] PromptlyBridge.js (7.9KB)

### ✅ Documentation
- [x] README.md updated with v0.2.0 features
- [x] EXECUTION_GUIDE.md (comprehensive guide)
- [x] EXECUTION_QUICKSTART.md (5-minute tutorial)
- [x] CHAINS_AND_LOOPS_COMPLETE.md (implementation summary)
- [x] DEPLOYMENT_CHECKLIST.md (this file)

### ✅ Code Quality
- [x] TypeScript strict mode enabled
- [x] Type-safe interfaces throughout
- [x] Error handling implemented
- [x] Resource cleanup (WebSocket, event handlers)
- [x] No console errors during compilation

### ✅ Dependencies
- [x] axios: ^1.6.2
- [x] @types/node: ^20.x
- [x] @types/vscode: ^1.85.0
- [x] typescript: ^5.3.3
- [x] No security vulnerabilities

---

## 🧪 Testing Requirements

### Manual Testing (Required Before Release)

#### 1. Environment Setup
- [ ] Python 3.8+ installed
- [ ] Ollama installed and running
- [ ] Model pulled: `ollama pull llama3.2:3b`
- [ ] Python bridge starts without errors

#### 2. Basic Extension Functionality
- [ ] Extension activates in VS Code
- [ ] Promptly sidebar appears
- [ ] Prompt library loads
- [ ] Refresh command works
- [ ] View prompt command works

#### 3. Execution Panel
- [ ] Panel opens via Play button
- [ ] Panel opens via command palette
- [ ] All three mode buttons work
- [ ] UI renders correctly
- [ ] VS Code theme colors applied

#### 4. Skill Execution
- [ ] Skill name input works
- [ ] User input textarea works
- [ ] Execute button triggers execution
- [ ] Progress bar updates
- [ ] Status indicator shows running (blue pulse)
- [ ] Output appears in output box
- [ ] Status changes to completed (green)
- [ ] Error handling works (invalid skill name)

#### 5. Chain Execution
- [ ] Chain builder appears
- [ ] Can add skills (+ Add Skill)
- [ ] Can remove skills (✕ button)
- [ ] Skills renumber correctly
- [ ] Initial input textarea works
- [ ] Execute button triggers chain
- [ ] Progress shows "Step N/Total"
- [ ] Each step broadcasts update
- [ ] Final output appears
- [ ] Intermediate results tracked

#### 6. Loop Execution
- [ ] All 6 loop types selectable:
  - [ ] ⚡ Refine
  - [ ] 🔍 Critique
  - [ ] 🧩 Decompose
  - [ ] ✓ Verify
  - [ ] 🌟 Explore
  - [ ] ∞ Hofstadter
- [ ] Max iterations slider works (1-10)
- [ ] Quality threshold slider works (0.5-1.0)
- [ ] Values display correctly
- [ ] Iteration progress shows
- [ ] Quality scores update
- [ ] Early stopping works (threshold reached)
- [ ] Max iterations limit works
- [ ] Output appears with metadata

#### 7. Real-Time Streaming
- [ ] WebSocket connects on panel open
- [ ] Progress updates in real-time
- [ ] No lag between backend and UI
- [ ] Auto-reconnect works (restart bridge test)
- [ ] Ping/pong keeps connection alive
- [ ] Multiple executions don't interfere

#### 8. Error Handling
- [ ] Bridge not running: Shows clear error
- [ ] Invalid skill name: Shows error message
- [ ] Empty input: Button disabled
- [ ] Ollama not running: Shows error
- [ ] Network timeout: Graceful failure
- [ ] WebSocket disconnect: Auto-reconnect

---

## 🔧 Python Bridge Testing

### Bridge Startup
```bash
cd Promptly
python promptly/vscode_bridge.py
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765
```

### Health Check
```bash
curl http://localhost:8765/health
```

Expected response:
```json
{
  "status": "healthy",
  "promptly_available": true,
  "timestamp": "2025-10-27T..."
}
```

### Endpoint Testing

#### Test Skill Execution
```bash
curl -X POST http://localhost:8765/execute/skill \
  -H "Content-Type: application/json" \
  -d '{
    "skill_name": "test_prompt",
    "user_input": "Hello, world!"
  }'
```

Expected:
```json
{
  "execution_id": "uuid-here",
  "status": "queued",
  "message": "Skill execution ... queued"
}
```

#### Test Status Endpoint
```bash
curl http://localhost:8765/execute/status/{execution_id}
```

Expected:
```json
{
  "execution_id": "uuid",
  "status": "running" | "completed" | "failed",
  "progress": 0.0-1.0,
  "current_step": "...",
  "output": "..." (when complete)
}
```

---

## 📦 Packaging

### Create .vsix Package

#### Option 1: vsce (Recommended)
```bash
# Install vsce if not already installed
npm install -g @vscode/vsce

# Package extension
cd promptly-vscode
vsce package
```

Output: `promptly-0.2.0.vsix`

#### Option 2: Manual ZIP
```bash
cd promptly-vscode
zip -r promptly-0.2.0.vsix \
  package.json \
  README.md \
  EXECUTION_GUIDE.md \
  EXECUTION_QUICKSTART.md \
  out/ \
  media/ \
  -x "*.ts" "*.map"
```

### Package Contents Verification
- [ ] package.json (with v0.2.0)
- [ ] All JavaScript files (out/)
- [ ] Documentation files
- [ ] Media assets (if any)
- [ ] No source .ts files
- [ ] No node_modules

---

## 🚀 Deployment Steps

### 1. Local Installation Testing
```bash
# Install from .vsix
code --install-extension promptly-0.2.0.vsix

# Restart VS Code
# Test all functionality
```

### 2. Internal Beta Testing
- [ ] Share .vsix with 2-3 beta testers
- [ ] Collect feedback on:
  - Installation process
  - Feature usability
  - Performance
  - Documentation clarity
  - Bug reports

### 3. Marketplace Publishing (Optional)
```bash
# Login to Azure DevOps
vsce login <publisher-name>

# Publish to VS Code Marketplace
vsce publish
```

Or publish manually:
1. Go to https://marketplace.visualstudio.com/manage
2. Upload .vsix file
3. Add description, screenshots, tags
4. Publish

---

## 📊 Performance Benchmarks

### Expected Performance
Run these tests and verify metrics:

| Operation | Expected Time | Pass/Fail |
|-----------|--------------|-----------|
| Extension activation | <2s | [ ] |
| Panel open | <500ms | [ ] |
| Skill execution (3B model) | 2-5s | [ ] |
| Chain (3 skills) | 6-15s | [ ] |
| Loop (5 iterations) | 10-25s | [ ] |
| WebSocket latency | <100ms | [ ] |
| UI responsiveness | <16ms/frame | [ ] |

### Resource Usage
Monitor and verify acceptable limits:

| Resource | Expected | Actual | Pass/Fail |
|----------|----------|--------|-----------|
| Extension RAM | ~50MB | _____ MB | [ ] |
| Python bridge RAM | ~100MB | _____ MB | [ ] |
| Ollama RAM | 2-4GB | _____ GB | [ ] |
| WebSocket traffic | <1KB/s | _____ KB/s | [ ] |

---

## 🔒 Security Checklist

- [x] No API keys hardcoded
- [x] No credentials in source
- [x] CORS properly configured
- [x] localhost-only binding (127.0.0.1)
- [x] No eval() or unsafe code
- [x] Input validation on all endpoints
- [x] Proper error messages (no stack traces to user)
- [ ] Dependencies scanned for vulnerabilities

### Security Scan
```bash
npm audit
# Should show 0 vulnerabilities
```

---

## 📝 Release Checklist

### Documentation
- [ ] README.md reflects current features
- [ ] All guides reviewed for accuracy
- [ ] Screenshots/GIFs added (future)
- [ ] Video walkthrough recorded (future)
- [ ] Changelog updated

### Code Quality
- [ ] No TODO comments in shipped code
- [ ] No console.log() statements (or guarded)
- [ ] All TypeScript strict checks pass
- [ ] No ESLint errors
- [ ] Proper error handling throughout

### User Experience
- [ ] Loading states implemented
- [ ] Error messages are clear
- [ ] Success feedback provided
- [ ] Keyboard shortcuts documented
- [ ] Tooltips/help text added where needed

### Integration
- [ ] Works with Promptly core
- [ ] Compatible with latest Ollama
- [ ] No conflicts with other extensions
- [ ] Proper cleanup on disable/uninstall

---

## 🐛 Known Issues (Document)

List any known issues/limitations for release notes:

1. **No execution cancellation** (in progress)
   - Workaround: Wait for completion or restart bridge
2. **Claude API requires API key** (not configured by default)
   - Workaround: Use Ollama backend
3. **Large models may timeout** (>13B parameters)
   - Workaround: Adjust timeout or use smaller models
4. **No execution history** (v1.1 planned)
   - Workaround: Manual note-taking

---

## 🎯 Success Criteria

### Must Have (Blocking)
- [ ] Extension activates without errors
- [ ] All three execution modes work
- [ ] WebSocket streaming functional
- [ ] Documentation complete
- [ ] No critical bugs

### Should Have (Nice to Have)
- [ ] Performance benchmarks met
- [ ] Beta tester feedback positive
- [ ] Screenshots/video ready
- [ ] Marketplace listing prepared

### Could Have (Future)
- Execution history viewer
- Visual chain composer
- Keyboard shortcuts
- Analytics dashboard

---

## 📢 Launch Checklist

### Pre-Launch
- [ ] Final testing complete
- [ ] Documentation reviewed
- [ ] Release notes written
- [ ] .vsix package created
- [ ] Backup of all files

### Launch Day
- [ ] Publish to marketplace (if applicable)
- [ ] Update GitHub repository
- [ ] Post announcement
- [ ] Monitor for issues
- [ ] Respond to feedback

### Post-Launch (Week 1)
- [ ] Monitor install count
- [ ] Track bug reports
- [ ] Gather user feedback
- [ ] Plan v0.2.1 fixes

---

## 🎉 Ready to Ship When...

**All Must-Have criteria met:**
- [x] Code compiles ✅
- [x] Documentation complete ✅
- [ ] Manual testing passed (pending)
- [ ] Performance acceptable (pending)
- [ ] No critical bugs (pending)

**Ship Status**: 🟡 **READY FOR TESTING**

Next step: **Run manual testing checklist above**

---

**Created**: 2025-10-27
**Version**: 0.2.0
**Status**: Pre-release verification
