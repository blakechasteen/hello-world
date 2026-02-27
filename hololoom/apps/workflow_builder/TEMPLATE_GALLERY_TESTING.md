# Template Gallery - Testing & Demo Guide

**Date**: December 9, 2025
**Purpose**: Verify all features work correctly before deployment

## How to Test

### 1. Open the Gallery
```bash
# Option A: Direct file
open hololoom/web_dashboard/template_gallery.html

# Option B: Local server (recommended)
cd hololoom/web_dashboard
python -m http.server 8000
# Visit: http://localhost:8000/template_gallery.html

# Option C: VS Code Live Server
# Right-click template_gallery.html → "Open with Live Server"
```

### 2. Visual Inspection

**Header Section**:
- [ ] Title "Workflow Templates" visible
- [ ] Subtitle "Pre-built workflows..." visible
- [ ] Search bar present with 🔍 icon
- [ ] All 6 category tabs visible (All, Research, CRM, Support, Content, Safety)
- [ ] "All Templates" tab is active (blue background)

**Template Grid**:
- [ ] 8 template cards visible in grid
- [ ] Cards have visual icons (🔍, 👥, etc.)
- [ ] Cards have names and descriptions
- [ ] Complexity dots visible (1-3 dots per card)
- [ ] "Use" buttons visible on each card
- [ ] Template count shows "8 templates"

**Card Details**:
- [ ] Each card shows icon at top
- [ ] Template name visible (bold text)
- [ ] Description shows (2 lines max)
- [ ] Complexity badge with dots
- [ ] Agent count (e.g., "6 agents")
- [ ] Time estimate (e.g., "2-5 min")
- [ ] Tags shown at bottom
- [ ] Use button is blue and clickable

### 3. Functional Testing

#### Test Category Filtering
```
Test: Click "Research" tab
Expected: Only 1 template (Research Pipeline) shows
Count: "1 template"

Test: Click "CRM" tab
Expected: 4 templates show (Lead Scoring variants, Daily Action, BDR)
Count: "4 templates"

Test: Click "Support" tab
Expected: 1 template (Support Triage)
Count: "1 template"

Test: Click "Content" tab
Expected: 1 template (Content Creation)
Count: "1 template"

Test: Click "Safety" tab
Expected: 1 template (Safety-Gated Query)
Count: "1 template"

Test: Click "All Templates" tab
Expected: All 8 templates show
Count: "8 templates"
```

#### Test Search Functionality
```
Test: Type "research" in search
Expected: Shows only Research Pipeline
Count: "1 template"

Test: Clear search, type "lead"
Expected: Shows both lead scoring templates
Count: "2 templates"

Test: Type "safety"
Expected: Shows Safety-Gated Query
Count: "1 template"

Test: Type "asdfghjkl" (nonsense)
Expected: "No templates found" message
Count: "0 templates"

Test: Clear search completely
Expected: Returns to showing all templates
Count: "8 templates"

Test: Type "crmm" (with typo)
Expected: Still finds CRM templates (fuzzy matching)
Count: "4 templates"
```

#### Test Template Cards
```
For each card, verify:
- [ ] Hover effect (card lifts up, shadow expands)
- [ ] Colors readable
- [ ] Text not cut off
- [ ] Use button is clickable
- [ ] All metadata visible

Test clicking "Use" on "Research Pipeline":
- [ ] Modal appears
- [ ] Title shows "Research Pipeline"
- [ ] Preview area shows workflow info
- [ ] Complexity shows "Complex"
- [ ] Agent count shows "6"
- [ ] Time shows "2-5 min"
- [ ] Category shows "Research"
- [ ] Close (X) button present
- [ ] "Close" button present
- [ ] "Use Template" button present
```

#### Test Modal Functionality
```
With modal open:

Test: Click X button (top right)
Expected: Modal closes, returns to gallery

Test: Click "Close" button
Expected: Modal closes, returns to gallery

Test: Press Escape key
Expected: Modal closes, returns to gallery

Test: Click background outside modal
Expected: Modal closes (if implemented)

Test: Click "Use Template" button
Expected: Redirects to workflow_builder.html with template loaded
```

#### Test Responsive Design
```
Desktop (1920×1080):
- [ ] Grid shows 4 columns
- [ ] Cards are properly sized
- [ ] No horizontal scrolling

Tablet (768×1024):
- [ ] Grid shows 2-3 columns
- [ ] Cards fit nicely
- [ ] No overflow

Mobile (375×667):
- [ ] Grid shows 1 column (full width)
- [ ] Cards responsive
- [ ] Touch targets large enough
- [ ] No side scrolling

To test:
1. Open DevTools (F12)
2. Click device toolbar icon
3. Select different devices
4. Refresh page
5. Verify layout on each
```

#### Test Keyboard Navigation
```
Test: Press Tab multiple times
Expected: Focus moves through interactive elements (tabs, buttons)
Visual: Dotted outline appears around focused element

Test: Press Enter on focused tab
Expected: Tab becomes active, filters update

Test: Press Enter on focused Use button
Expected: Modal opens for that template

Test: With modal open, press Escape
Expected: Modal closes

Test: Tab through filter tabs
Expected: Can tab between all 6 tabs
```

### 4. Browser Testing

Test in each browser:
```
Chrome:
- [ ] Opens correctly
- [ ] All animations smooth
- [ ] Search works
- [ ] Modal functions
- [ ] No console errors

Firefox:
- [ ] Opens correctly
- [ ] All animations smooth
- [ ] Search works
- [ ] Modal functions
- [ ] No console errors

Safari:
- [ ] Opens correctly
- [ ] All animations smooth
- [ ] Search works
- [ ] Modal functions
- [ ] No console errors

Edge:
- [ ] Opens correctly
- [ ] All animations smooth
- [ ] Search works
- [ ] Modal functions
- [ ] No console errors
```

### 5. Console Testing

Open DevTools (F12) → Console tab:

```javascript
// Test: Access gallery object
gallery
// Expected: TemplateGallery object with methods

// Test: Get all templates
gallery.templates.length
// Expected: 8

// Test: Get filtered templates
gallery.currentCategory = 'crm'
gallery.getFilteredTemplates().length
// Expected: 4

// Test: Search
gallery.searchQuery = 'lead'
gallery.getFilteredTemplates().length
// Expected: 2

// Test: Get recommendations
gallery.getRecommendations('sales')
// Expected: Array with lead scoring templates

// Test: Get statistics
gallery.getStatistics()
// Expected: { totalTemplates: 8, categories: [...], ... }

// Check for errors
// Expected: No red error messages
```

### 6. Performance Testing

Measure load time:
```javascript
// In browser console, go to Performance tab
// Click record
// Refresh page
// Stop recording

// Check:
- [ ] Page load < 2 seconds
- [ ] First paint < 1 second
- [ ] Template render < 100ms
- [ ] No jank (smooth scrolling)
```

## Test Scenarios

### Scenario 1: New User Discovery
```
Flow:
1. User opens template_gallery.html
2. Sees all 8 templates at once
3. Reads descriptions
4. Identifies "Research Pipeline" as interesting
5. Clicks "Use" button
6. Sees preview modal
7. Reviews details
8. Clicks "Use Template"
9. Redirected to workflow builder with template loaded

Expected Result: ✅ Template loads in builder, ready to customize
```

### Scenario 2: CRM User
```
Flow:
1. User opens gallery
2. Clicks "CRM" category tab
3. Sees 4 CRM-related templates
4. Clicks different templates to preview
5. Reads complexities and metadata
6. Chooses "Lead Scoring (Simple)" for quick start
7. Uses template

Expected Result: ✅ Simple lead scoring workflow loads
```

### Scenario 3: Search User
```
Flow:
1. User wants to find safety-related workflows
2. Types "safety" in search bar
3. Sees "Safety-Gated Query" template
4. Types "score" to find lead scoring
5. Sees 2 lead scoring templates
6. Clears search to see all again

Expected Result: ✅ Search works correctly
```

### Scenario 4: Mobile User
```
Flow:
1. User opens gallery on phone (375×667)
2. See single column layout
3. Scroll through templates
4. Tap on template
5. Modal opens (full screen or near-full)
6. Tap "Use Template"
7. Loads in builder

Expected Result: ✅ Mobile experience is smooth and functional
```

### Scenario 5: Power User
```
Flow:
1. User knows they want a specific template
2. Types name in search (e.g., "research")
3. Instantly sees filtered result
4. Click use
5. Verifies in console that analytics tracked

Expected Result: ✅ Fast workflow for returning users
```

## Quality Checklist

### Visual Quality
- [ ] No layout breaks
- [ ] Colors are consistent
- [ ] Typography is readable
- [ ] Spacing looks balanced
- [ ] Hover states are visible
- [ ] No overlapping elements
- [ ] Icons align properly
- [ ] Status badges look good

### Functional Quality
- [ ] Search returns correct results
- [ ] Filtering works for all categories
- [ ] Modal opens/closes smoothly
- [ ] Use button redirects correctly
- [ ] No JavaScript errors
- [ ] Keyboard navigation works
- [ ] Mobile responsive
- [ ] Fast performance

### Usability Quality
- [ ] First-time user understands purpose
- [ ] Navigation is intuitive
- [ ] Controls are obvious
- [ ] Feedback is clear (e.g., counts update)
- [ ] No confusing text
- [ ] Status clear (active tab, etc.)
- [ ] Easy to find templates
- [ ] Information hierarchy is clear

### Accessibility Quality
- [ ] Keyboard accessible
- [ ] Screen reader friendly
- [ ] Color contrast sufficient
- [ ] Text is scalable
- [ ] Focus indicators visible
- [ ] Semantic HTML used
- [ ] ARIA labels present
- [ ] Mobile accessible

## Demo Script

Use this to demonstrate the gallery to others:

### 30-Second Demo
```
1. Open gallery (show title + search)
2. Point out all 8 templates
3. Type "safety" in search (show filtering)
4. Click Use on Safety template
5. Show modal with details
6. Click "Use Template"
7. Show it loads in builder
```

### 2-Minute Demo
```
1. Open gallery
2. Explain the 6 categories
3. Click each tab (Research → CRM → Support → Content → Safety)
4. Explain template cards (icon, name, desc, complexity, time, tags)
5. Do a search demo (type "lead", show fuzzy matching)
6. Click Use on a template
7. Show modal and explain fields (complexity, agents, time, category)
8. Click "Use Template"
9. Show template loads
10. Explain you can now edit/customize in builder
```

### 5-Minute Demo + Q&A
```
1. Full 2-minute demo
2. Show keyboard navigation (Tab, Enter, Escape)
3. Show responsive design (use DevTools)
4. Show console (gallery object, methods)
5. Explain analytics (localStorage usage stats)
6. Demo adding a new template (edit TEMPLATE_METADATA)
7. Field questions about Wave 2 features
```

## Known Behaviors

### Expected When Testing

1. **Status Badges**: Some templates have "NEW" or "POPULAR" badges
   - Research Pipeline: POPULAR
   - Safety-Gated Query: NEW
   - Content Creation: BETA
   - Others: No badge

2. **Complexity Ratings**: Vary by template
   - Simple (1 dot): Lead Scoring
   - Medium (2 dots): BDR, Daily Action, Support Triage, Safety-Gated
   - Complex (3 dots): Research Pipeline, Multi-Factor Scoring, Content Creation

3. **Time Estimates**: Based on number of agents
   - <1 min: Lead Scoring
   - 1-2 min: Simple templates
   - 2-5 min: Complex templates

4. **Search is Case-Insensitive**: "LEAD" = "lead" = "Lead"

5. **Fuzzy Matching**: "leding" might find "lead" (typo tolerance)

## Troubleshooting During Testing

| Issue | Solution |
|-------|----------|
| Templates don't show | Check `example_workflows/` directory exists |
| Modal won't open | Try refreshing page, check console for errors |
| Search doesn't work | Check search query matches template name/tags |
| Layout looks broken | Zoom to 100%, clear browser cache, refresh |
| Styling looks wrong | Try different browser, check system display settings |
| Can't see status badges | They only show on certain templates (NEW, POPULAR, BETA) |

## Test Results Template

Copy and fill out:

```markdown
# Test Results - Template Gallery Wave 1

**Date**: [TODAY]
**Tester**: [YOUR NAME]
**Browser**: [Chrome/Firefox/Safari/Edge]
**Device**: [Desktop/Tablet/Mobile]

## Visual Tests
- [ ] Header visible and correct
- [ ] All 8 templates show
- [ ] Cards look good
- [ ] Icons display
- [ ] Text readable

## Functional Tests
- [ ] Search works (test: "research")
- [ ] Category filter works (test: CRM tab)
- [ ] Modal opens on Use button
- [ ] Modal closes on X button
- [ ] Modal closes on Escape key
- [ ] Use Template redirects

## Responsive Tests
- [ ] Desktop layout: ✅/❌
- [ ] Tablet layout: ✅/❌
- [ ] Mobile layout: ✅/❌

## Browser Tests
- [ ] Chrome: ✅/❌
- [ ] Firefox: ✅/❌
- [ ] Safari: ✅/❌
- [ ] Edge: ✅/❌

## Overall
- [ ] Gallery is production-ready
- [ ] No major issues found
- [ ] Recommend deployment: ✅/❌

## Notes
[Any other observations or issues]
```

## Performance Benchmarks

Expected results:

| Metric | Target | Actual |
|--------|--------|--------|
| Page Load | <1s | _____ |
| First Paint | <500ms | _____ |
| Template Render | <100ms | _____ |
| Search Filter | <10ms | _____ |
| Modal Open | <200ms | _____ |
| Memory Usage | <5MB | _____ |

## Next Steps After Testing

1. **If all tests pass**: Deploy to production ✅
2. **If minor issues**: Fix and re-test
3. **If major issues**: File bug reports and schedule fixes
4. **Planning Wave 2**: Schedule template variants feature
5. **Gather feedback**: Ask users for improvement ideas

---

**Testing Complete When**:
- All visual checks pass
- All functional tests pass
- Works on major browsers
- Responsive on all screen sizes
- No console errors
- Documentation is accurate

Happy testing! 🧪
