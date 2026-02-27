# Template Gallery Integration - Testing Guide

## Quick Test (5 minutes)

### Test 1: Modal Opens ✅
```
1. Open workflow_builder.html in browser
2. Click "📚 Templates" button in toolbar
3. Verify:
   - Modal appears with gallery loaded
   - Close button (×) visible
   - Templates display properly
   - Search bar visible
```

### Test 2: Template Selection ✅
```
1. In templates modal, find "Research Pipeline" card
2. Click "Use" button on card
3. Verify:
   - Preview modal opens
   - Shows template details (complexity, agents, time, category)
   - "Use Template" button present
```

### Test 3: Template Loading ✅
```
1. In preview modal, click "Use Template"
2. Verify:
   - Modal closes automatically
   - Green success notification appears
   - Notification shows template name
   - Canvas appears to update (if workflow_builder.js functions available)
   - Workflow title updates to template name
```

### Test 4: Modal Closes ✅
```
1. Click close button (×) or outside modal
2. Verify:
   - Modal disappears
   - Workflow builder canvas visible
   - No error messages in console
```

---

## Complete Test (15 minutes)

### Test 5: Error Handling ✅
```
1. Open browser DevTools (F12)
2. Go to Console tab
3. Manually call: loadWorkflowFromTemplate('nonexistent.json')
4. Verify:
   - Red error notification appears
   - Error message: "Failed to load template: HTTP 404"
   - Console shows error details
   - Canvas remains functional
```

### Test 6: URL Parameter Loading ✅
```
1. Paste this URL in address bar:
   http://localhost:8000/workflow_builder.html?template=research_pipeline.json
2. Verify:
   - Page loads
   - Template loads automatically
   - No modal appears
   - Workflow visible in canvas
   - Title updated to "Research Pipeline"
   - URL changes to just workflow_builder.html (params removed)
```

### Test 7: Standalone Gallery ✅
```
1. Open template_gallery.html directly
2. Click "Use" on any template
3. Click "Use Template" in preview
4. Verify:
   - Redirects to workflow_builder.html?template=...
   - Builder loads template automatically
5. (Alternatively) Manually set searchParams and test redirect
```

### Test 8: sessionStorage Fallback ✅
```
1. In browser console, manually:
   sessionStorage.setItem('selectedTemplate', 'research_pipeline.json')
2. Reload workflow_builder.html
3. Verify:
   - Template loads automatically
   - sessionStorage cleared
```

### Test 9: postMessage Communication ✅
```
1. Open DevTools Network tab
2. Open templates modal
3. Open DevTools Console tab in template_gallery.html (iframe)
4. Click "Use Template"
5. Verify in parent console:
   - "Sent template selection to parent:" message logged
   - Parent receives correct filename
   - Modal closes automatically
```

### Test 10: Notification Animations ✅
```
1. Load any template
2. Watch green notification appear
3. Verify:
   - Slides in from right
   - Shows for ~3 seconds
   - Slides out to right
   - Disappears cleanly (no leftover DOM elements)
```

---

## Advanced Testing (30 minutes)

### Test 11: Multiple Rapid Clicks ✅
```
1. Open templates modal
2. Rapidly click "Use" on different templates
3. Quickly click different template cards
4. Verify:
   - No errors occur
   - Only one preview modal at a time
   - Correct template loads each time
   - No memory leaks (check DevTools heap)
```

### Test 12: Search & Filter + Template Load ✅
```
1. Open templates modal
2. Search for "research"
3. Verify only Research Pipeline shows
4. Click "Use" on it
5. Load successfully
6. Repeat with different searches
```

### Test 13: Cross-Origin Security ✅
```
1. In browser console (in iframe context):
   window.parent.postMessage({type: 'test'}, 'http://different-origin.com')
2. Verify in parent console:
   - "Rejected message from untrusted origin" warning
   - Message NOT processed
   - Canvas unchanged
```

### Test 14: Large Workflow Handling ✅
```
1. If complex template available, load it
2. Verify:
   - Notification shows without error
   - Title updates correctly
   - No timeout errors
   - Page responsive (no freezing)
```

### Test 15: Browser Compatibility ✅
```
Test in multiple browsers:
- Chrome/Chromium ✅
- Firefox ✅
- Safari ✅
- Edge ✅
- Mobile Chrome ✅
- Mobile Safari ✅

For each browser, verify:
- Modal opens properly
- postMessage works
- Notifications display correctly
- Animations smooth
```

---

## DevTools Debugging

### Enable Detailed Logging
```javascript
// In Console, add logging to postMessage handler:
window.addEventListener('message', function(event) {
    console.log('🔵 postMessage received:', {
        origin: event.origin,
        type: event.data?.type,
        filename: event.data?.filename,
        trusted: event.isTrusted
    });
    // ... rest of handler
});
```

### Monitor Template Loading
```javascript
// Wrap fetch to monitor requests:
const originalFetch = window.fetch;
window.fetch = function(...args) {
    console.log('📥 Fetch:', args[0]);
    return originalFetch(...args)
        .then(r => {
            console.log('✅ Response:', r.status, r.statusText);
            return r;
        })
        .catch(e => {
            console.error('❌ Fetch error:', e);
            throw e;
        });
};
```

### Check Memory
```javascript
// In DevTools Console:
performance.memory  // Shows heap usage
// Repeat tests and check for memory growth
```

---

## Expected Behavior

### Success Path
```
📚 Templates button
    ↓ [Modal opens, iframe loads]
Click template card
    ↓ [Preview modal opens]
Click "Use Template"
    ↓ [postMessage sent]
✅ Success notification
    ↓ [Modal closes]
Canvas ready with workflow
```

### Error Path
```
Load nonexistent template
    ↓ [fetch returns 404]
❌ Error notification
    ↓ [Shows specific error message]
Modal remains visible
    ↓ [User can try another template]
```

### Fallback Path (Standalone Mode)
```
template_gallery.html opens
    ↓ [Detects window.parent === window]
Click "Use Template"
    ↓ [sessionStorage + redirect]
workflow_builder.html?template=...
    ↓ [DOMContentLoaded loads from URL]
✅ Template loaded
```

---

## Console Warnings (Normal)

These warnings can appear and are expected:

```
⚠️ (if no populateCanvas function)
"No workflow loading function found"
→ Fallback used successfully

⚠️ (if template already in sessionStorage)
Nothing - sessionStorage cleared properly

⚠️ (if URL params present on page load)
History state updated (params removed)
→ Browser address bar cleaned up
```

## Console Errors (Should NOT appear)

```
❌ "Templates modal not found"
→ Modal div missing from HTML

❌ "Rejected message from untrusted origin"
→ postMessage from wrong origin (security feature, not error)

❌ "Failed to load template: HTTP 404"
→ Template file missing (expected error, shows notification)

❌ Uncaught TypeError in loadWorkflow functions
→ workflow_builder.js functions not loading properly
```

---

## Checklist for Sign-Off

- [ ] Modal opens/closes smoothly
- [ ] Template selection triggers load
- [ ] Success notifications appear correctly
- [ ] Error notifications show helpful messages
- [ ] postMessage communication works
- [ ] URL parameter loading works
- [ ] sessionStorage fallback works
- [ ] No JavaScript errors in console
- [ ] No memory leaks after multiple loads
- [ ] Works in Chrome, Firefox, Safari
- [ ] Works on mobile (responsive)
- [ ] Modal closes with × button
- [ ] Modal closes after successful load
- [ ] Canvas remains editable after load
- [ ] Workflow title updates correctly
- [ ] Notifications auto-dismiss
- [ ] Animations smooth
- [ ] No cross-origin errors (expected security warnings OK)

---

## Performance Baselines

Document actual performance on your system:

| Metric | Baseline | Actual | Status |
|--------|----------|--------|--------|
| Modal open | <100ms | ___ | |
| iframe load | <200ms | ___ | |
| Template select | <50ms | ___ | |
| Fetch JSON | <100ms | ___ | |
| Canvas populate | <500ms | ___ | |
| Notification display | <300ms | ___ | |
| Total end-to-end | <1000ms | ___ | |

---

## Report Template

```markdown
## Template Integration Test Report

**Date**: [DATE]
**Tester**: [NAME]
**Browser**: [BROWSER + VERSION]
**OS**: [WINDOWS/MAC/LINUX]

### Tests Completed
- [ ] Test 1: Modal Opens
- [ ] Test 2: Template Selection
- [ ] Test 3: Template Loading
- [ ] Test 4: Modal Closes
- [ ] Test 5: Error Handling
- [ ] Test 6: URL Parameters
- [ ] Test 7: Standalone Mode
- [ ] Test 8: sessionStorage
- [ ] Test 9: postMessage
- [ ] Test 10: Notifications

### Issues Found
(List any bugs or unexpected behavior)

### Performance
(Document actual timings)

### Sign-Off
- [ ] All tests passed
- [ ] No critical issues
- [ ] Ready for production

**Tester Signature**: _______________
```

---

## Troubleshooting Common Issues

### Issue: "Templates modal not found" error
**Cause**: Modal element not in HTML
**Solution**:
```html
<!-- Add this if missing from workflow_builder.html -->
<div class="modal" id="templatesModal">
    <!-- Will be populated dynamically -->
</div>
```

### Issue: postMessage not working
**Cause**: iframe sandbox restrictions
**Solution**: Verify iframe has correct attributes:
```html
<iframe
    src="template_gallery.html"
    sandbox="allow-same-origin allow-scripts allow-popups"
></iframe>
```

### Issue: Template file 404 error
**Cause**: File path incorrect or file missing
**Solution**:
1. Verify file exists: `hololoom/web_dashboard/example_workflows/research_pipeline.json`
2. Check filename matches exactly (case-sensitive)
3. Verify JSON is valid: `https://jsonlint.com/`

### Issue: Notification doesn't appear
**Cause**: Toast element missing or showNotification function error
**Solution**:
1. Verify `<div class="toast" id="toast"></div>` exists
2. Check console for JavaScript errors
3. Test manually: `showNotification('Test', 'success')`

### Issue: Canvas doesn't update after loading
**Cause**: No workflow loading function found
**Solution**:
1. Verify `workflow_builder.js` is loaded
2. Check if `populateCanvas()` or `loadWorkflow()` functions exist
3. Check console for errors in those functions

---

End of Testing Guide
