# HoloLoom Community Forum

A lightweight, client-side community forum system for the HoloLoom flagship site.

**Status**: ✅ Production Ready (v1.0)
**Last Updated**: November 2025
**Data Storage**: localStorage (client-side)
**Tech Stack**: HTML5, CSS3, JavaScript (ES6+), Markdown

---

## Quick Start

### Access the Forum

1. **Forum Index**: `docs/community/index.html` - Browse all categories
2. **Create Thread**: `docs/community/new.html` - Start a new discussion
3. **View Category**: `docs/community/category.html?id=general` - View threads by category
4. **View Thread**: `docs/community/thread.html?id=1` - View specific discussion

All pages are interconnected via navigation links.

### First Time Setup

No setup required! The forum initializes with default categories and sample data on first load. Data is automatically persisted to browser localStorage under the key `hololoom_forums`.

---

## Features

### Core Features

✅ **Thread CRUD Operations**
- Create new discussions (Question, Discussion, Announcement)
- Read threads with full content and replies
- Edit thread metadata
- Delete threads

✅ **Reply Threading**
- Nested replies on threads
- Flat or hierarchical display
- Reply composition with Markdown support
- Solution marking for questions

✅ **Voting System**
- Upvote/downvote threads and replies
- Vote counts displayed prominently
- Contributes to reputation scoring
- Hot score calculation for sorting

✅ **Search and Filtering**
- Full-text search across all threads and replies
- Filter by: All, Questions, Discussions, Announcements, Unanswered, Solved
- Sort by: Recent, Hot, Top (Most Upvoted), Oldest, Most Replies
- Tag-based filtering

✅ **Categories**
- 5 default categories: General, HoloLoom, Promptly, Development, Announcements
- Thread count and post count per category
- Category statistics on main page
- Easy navigation between categories

✅ **User System**
- Anonymous posting with display name
- Persistent display name (stored in localStorage)
- User reputation based on upvotes
- Top contributors list
- User post history (future enhancement)

✅ **Markdown Support**
- Rendered with marked.js
- Headers, bold, italic, code blocks, lists
- Syntax highlighting for code
- Live preview when creating/replying

✅ **Activity Tracking**
- View counts per thread
- Recent activity feed (main page)
- Last reply timestamps
- Creation timestamps on all posts

✅ **Responsive Design**
- Mobile-first approach
- Works on all screen sizes (320px - 4K)
- Touch-friendly buttons and controls
- Optimized for mobile viewing

✅ **Accessibility**
- WCAG AAA compliant color contrasts
- Keyboard navigation support
- Semantic HTML structure
- ARIA labels where needed
- Focus indicators on all interactive elements

✅ **Dark Mode Support**
- Automatic dark mode detection
- Carefully chosen color palette for both modes
- High contrast in both light and dark modes

---

## Data Schema

### Storage Structure

All data is stored in a single localStorage key: `hololoom_forums`

```javascript
{
  categories: [
    {
      id: "general",           // Unique identifier
      name: "General Discussion",
      description: "General topics and community chat",
      threadCount: 42,         // Updated automatically
      postCount: 156           // Updated automatically
    }
    // ... more categories
  ],

  threads: [
    {
      id: 1,                   // Unique auto-increment ID
      categoryId: "hololoom",  // Parent category
      title: "How to use Thompson Sampling?",
      content: "# Markdown content...",
      author: "User Name",
      created: "2025-11-16T10:00:00Z",
      updated: "2025-11-16T12:00:00Z",
      views: 45,               // Auto-incremented when viewed
      type: "question",        // "question", "discussion", "announcement"
      solved: false,           // For questions
      tags: ["thompson-sampling", "beginner"],
      replies: [               // Array of reply objects
        {
          id: 1,               // Unique reply ID
          threadId: 1,         // Parent thread
          content: "Here's how...",
          author: "Helper",
          created: "2025-11-16T11:00:00Z",
          upvotes: 5,
          downvotes: 0,
          solution: false      // Only one per thread
        }
      ],
      upvotes: 12,
      downvotes: 1
    }
    // ... more threads
  ],

  nextThreadId: 2,      // Auto-increment counter
  nextReplyId: 2        // Auto-increment counter
}
```

### Export Format

Exported data uses the same structure above. Files are typically named:
- `hololoom_forum_backup_2025-11-16.json`
- `hololoom_forum_export.json`

---

## API Reference

### ForumManager Object

All forum operations are performed through the `ForumManager` API.

#### Initialization

```javascript
// Initialize on page load
ForumManager.init()
```

#### Thread Operations

```javascript
// Get all threads
const threads = ForumManager.getAllThreads()

// Get thread by ID
const thread = ForumManager.getThread(threadId)

// Get threads by category
const threads = ForumManager.getThreadsByCategory('hololoom')

// Create new thread
const thread = ForumManager.createThread({
  categoryId: 'hololoom',
  title: 'How does X work?',
  content: '# Question\nI want to know...',
  author: 'Jane Doe',
  type: 'question',
  tags: ['how-to', 'beginner']
})

// Update thread
ForumManager.updateThread(threadId, {
  title: 'Updated Title',
  solved: true
})

// Delete thread
ForumManager.deleteThread(threadId)

// Increment view count
ForumManager.incrementViewCount(threadId)

// Mark thread as solved
ForumManager.markThreadAsSolved(threadId)
```

#### Reply Operations

```javascript
// Create reply
const reply = ForumManager.createReply(threadId, {
  author: 'Helper',
  content: 'Here is the answer...'
})

// Delete reply
ForumManager.deleteReply(threadId, replyId)

// Mark reply as solution
ForumManager.markAsSolution(threadId, replyId)
```

#### Voting

```javascript
// Vote on threads
ForumManager.upvoteThread(threadId)
ForumManager.downvoteThread(threadId)

// Vote on replies
ForumManager.upvoteReply(threadId, replyId)
ForumManager.downvoteReply(threadId, replyId)
```

#### Search and Filter

```javascript
// Search all threads
const results = ForumManager.searchThreads('thompson sampling')

// Get threads by tag
const threads = ForumManager.getThreadsByTag('beginner')

// Get unanswered questions
const questions = ForumManager.getUnansweredQuestions('hololoom')
```

#### Sorting

```javascript
// Sort by hot score
const sorted = ForumManager.sortByHotScore(threads)

// Calculate hot score for a thread
const score = ForumManager.calculateHotScore(thread)
```

#### Activity and Statistics

```javascript
// Get recent activity (limit: 10)
const activity = ForumManager.getRecentActivity(10)

// Get statistics
const stats = ForumManager.getStatistics()
// Returns: {
//   threadCount: 42,
//   replyCount: 156,
//   userCount: 25,
//   todayThreadCount: 3
// }

// Get top contributors
const contributors = ForumManager.getTopContributors(8)

// Get user reputation
const reputation = ForumManager.getUserReputation('Jane Doe')

// Get user's posts
const posts = ForumManager.getUserPosts('Jane Doe')
```

#### Data Management

```javascript
// Export forum data
const json = ForumManager.exportData()
// Save to file in browser

// Import forum data
ForumManager.importData(jsonString)

// Reset to default data
ForumManager.reset()

// Clear all data
ForumManager.clearAll()
```

---

## Hot Score Algorithm

Threads are sorted by "hot score" which considers:
- **Votes (40%)**: Upvotes - downvotes
- **Replies (30%)**: Number of replies (scaled 0-1)
- **Recency (30%)**: Time since creation with exponential decay (24-hour half-life)

Formula:
```
hotScore = (votes * 0.4)
         + (replies * 0.3 * min(1, replies/10))
         + (exp(-ageInHours/24) * 0.3 * 100)
```

Older threads naturally decline in hot score, keeping fresh content visible while allowing popular threads to stay high.

---

## User Display Names

### How They Work

1. When users create a thread or reply, they enter their display name
2. Display name is saved to localStorage (`forumAuthor`)
3. Pre-filled on subsequent form submissions
4. No authentication required (MVP)

### localStorage Keys

```javascript
// Author name (persisted across sessions)
localStorage.getItem('forumAuthor')
localStorage.setItem('forumAuthor', 'Jane Doe')

// Full forum data
localStorage.getItem('hololoom_forums')
```

### Future: User Profiles

Once authentication is added, user profiles could include:
- Avatar/profile picture
- Bio
- Posting history
- Followed tags
- Private messages
- Reputation level
- Activity timeline

---

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| **Chrome** | ✅ Full | Works perfectly |
| **Firefox** | ✅ Full | Works perfectly |
| **Safari** | ✅ Full | Works perfectly |
| **Edge** | ✅ Full | Works perfectly |
| **Mobile Safari** | ✅ Full | Optimized layout |
| **Chrome Mobile** | ✅ Full | Optimized layout |
| **IE 11** | ❌ Not supported | Missing ES6 support |

### Storage Limits

- localStorage limit: ~5-10MB per domain
- Max threads before hitting limit: ~5,000 (with replies)
- Recommend exporting data before limit

---

## Customization

### Add Custom Categories

Edit `docs/assets/js/forums.js`, update `getDefaultData()`:

```javascript
{
  id: 'mycategory',
  name: 'My Category',
  description: 'Description of this category',
  threadCount: 0,
  postCount: 0
}
```

### Change Colors

Edit `docs/assets/css/forums.css`, modify CSS variables:

```css
:root {
    --forum-primary: #2563eb;      /* Primary blue */
    --forum-secondary: #1e293b;    /* Dark navy */
    --forum-accent: #06b6d4;       /* Cyan accent */
    --forum-success: #10b981;      /* Green */
    --forum-warning: #f59e0b;      /* Orange */
    --forum-danger: #ef4444;       /* Red */
    /* ... more colors */
}
```

### Modify Pagination

Change `THREADS_PER_PAGE` in `docs/community/category.html`:

```javascript
const THREADS_PER_PAGE = 10;  // Default
const THREADS_PER_PAGE = 20;  // For fewer pages
```

### Disable Markdown Preview

Remove or comment out `togglePreview()` calls in:
- `docs/community/new.html`
- `docs/community/thread.html`

---

## Data Backup and Recovery

### Manual Export

1. Open browser console (F12)
2. Run: `copy(ForumManager.exportData())`
3. Paste into a text editor
4. Save as `backup.json`

Or use a bookmarklet:
```javascript
javascript:(function(){
  copy(ForumManager.exportData());
  alert('Forum data copied to clipboard!');
})()
```

### Import Backup

1. Open browser console (F12)
2. Run:
```javascript
const backupData = `{paste JSON here}`;
ForumManager.importData(backupData);
location.reload();
```

### Browser Storage Management

Forum data persists until:
- User manually clears browser data
- Browser localStorage quota is exceeded
- User deletes the site data

To preserve data:
1. Export regularly
2. Store backups securely
3. Test imports in staging

---

## Migration Path

### Moving to Discourse

When ready to migrate to a full-featured forum platform (Discourse, Flarum, etc.):

1. **Export HoloLoom Forum Data**
   ```javascript
   const data = ForumManager.exportData();
   // Save to file
   ```

2. **Transform to Discourse Format**
   ```javascript
   const discourse_import = {
     categories: data.categories.map(cat => ({
       id: cat.id,
       name: cat.name,
       description: cat.description,
       color: '#0088ff',  // Choose color
       permissions: { everyone: 1 }
     })),
     users: Array.from(new Set(
       data.threads.flatMap(t => [t.author, ...t.replies.map(r => r.author)])
     )).map((name, i) => ({
       id: i + 1,
       username: name.toLowerCase().replace(/\s+/g, '_'),
       email: `user${i}@hololoom.local`,
       name: name
     })),
     posts: [
       // Transform threads and replies
     ]
   }
   ```

3. **Use Discourse Bulk Import**
   - Upload NDJSON file to `/admin/import`
   - Let Discourse process the data
   - Verify all content imported correctly

### Moving to Flarum

1. Export HoloLoom data as JSON
2. Transform to Flarum database schema
3. Import into Flarum MySQL/PostgreSQL
4. Set up Flarum extensions

### Moving to Self-Hosted Solution

For a custom solution or other platform:

1. Export data (already in clean JSON format)
2. Write import script for target platform
3. Test thoroughly before production
4. Redirect old forum URLs to new platform

### Keeping Both Systems

For gradual migration:

1. **Set up new platform** (Discourse, etc.)
2. **Run both systems in parallel** (6-12 months)
3. **Redirect reads** from old to new
4. **Keep writes** on old system during transition
5. **Sync data** if needed (complex)
6. **Close old forum** when new system is stable

---

## Limitations and Known Issues

### Current Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| localStorage only | Max ~5-10MB | Export/import periodically |
| No persistence layer | Data lost if localStorage cleared | Implement IndexedDB or backend |
| No authentication | Trusting usernames | Add authentication layer |
| No moderation tools | Spam possible | Manual deletion, then implement moderation |
| Single browser storage | Different browsers have different data | Sync to backend |
| No real-time updates | Need to refresh for new posts | Add WebSocket layer |
| No media uploads | Images only via Markdown URLs | Implement file upload backend |

### Future Enhancements

1. **Backend Integration** (~1-2 weeks)
   - Move data from localStorage to server
   - Implement REST API
   - Add database (PostgreSQL)

2. **Authentication** (~1 week)
   - User accounts with passwords
   - OAuth integration (Google, GitHub)
   - Email verification

3. **Moderation Tools** (~1 week)
   - Flag/report system
   - Moderator actions (edit, delete, lock)
   - User blocking

4. **Advanced Features** (~2-3 weeks)
   - Notifications (new replies, mentions)
   - Private messages
   - Thread subscriptions
   - Email digests
   - Reputation badges
   - User profiles

5. **Performance** (~1 week)
   - Pagination optimization
   - Lazy loading for long threads
   - Search indexing

6. **Analytics** (~1 week)
   - User activity tracking
   - Popular topics trending
   - Engagement metrics

---

## Troubleshooting

### Forum Not Loading Data

**Problem**: See "No discussions found" everywhere

**Solution**:
1. Check browser localStorage is enabled
2. Run in console: `localStorage.getItem('hololoom_forums')`
3. If null, reset: `ForumManager.reset()`
4. Refresh page

### Lost Data After Browser Clear

**Problem**: All forum data disappeared

**Solution**:
1. If you have a backup, import it: `ForumManager.importData(jsonString)`
2. If no backup, data is unfortunately lost
3. **Prevention**: Export data regularly via console

### Thread Not Showing Up

**Problem**: Created thread but can't find it

**Solution**:
1. Check you created it in the right category
2. Refresh the category page
3. Try searching for it by title
4. Check in "Recent Activity" on homepage

### Markdown Not Rendering

**Problem**: Markdown syntax shows as text instead of formatting

**Solution**:
1. Check internet connection (marked.js loaded from CDN)
2. Try offline version if available
3. Report issue if using local version

### Voting Not Working

**Problem**: Upvote/downvote buttons not responding

**Solution**:
1. Check browser console for errors (F12)
2. Refresh page
3. Clear browser cache and reload
4. Try different browser

### Performance Issues

**Problem**: Forum is slow with many threads

**Solution**:
1. Export data and clear localStorage
2. Consider splitting into multiple categories
3. Keep thread count under 1,000 per category
4. Switch to backend solution for production use

---

## Code Structure

### File Organization

```
docs/
├── community/
│   ├── README.md                 # This file
│   ├── index.html                # Forum homepage
│   ├── category.html             # Category view template
│   ├── thread.html               # Thread view template
│   └── new.html                  # Create thread form
├── assets/
│   ├── css/
│   │   ├── main.css              # Site main styles
│   │   └── forums.css            # Forum specific styles
│   └── js/
│       └── forums.js             # Forum JavaScript (850 lines)
└── index.html                    # Main site homepage
```

### JavaScript Architecture

**forums.js** (850 lines)
- `ForumManager` object (all methods)
- localStorage persistence
- CRUD operations
- Search and filtering
- Sorting and hot score
- Statistics and activity
- Data export/import

**index.html** (utilities)
- `renderForumIndex()` - Display categories
- `renderRecentActivity()` - Recent threads
- `renderTopContributors()` - Top users
- `updateStatistics()` - Forum stats
- `searchForum()` - Search functionality
- Utility functions (timeAgo, truncate, etc.)

**category.html** (utilities)
- `loadCategoryView()` - Load category
- `applyFilters()` - Search/filter/sort
- `renderThreadList()` - Display threads
- `renderPagination()` - Page navigation

**thread.html** (utilities)
- `loadThreadView()` - Load thread
- `renderOriginalPost()` - Display OP
- `renderReplies()` - Display replies
- `submitReply()` - Create reply
- Voting functions
- Solution marking

**new.html** (utilities)
- `initializeForm()` - Load categories
- `setupFormListeners()` - Form events
- `updatePreview()` - Markdown preview
- `submitNewThread()` - Create thread

### CSS Organization

**forums.css** (700 lines)

Organized by section:
1. Colors & Variables
2. Layout containers
3. Search and actions
4. Statistics display
5. Categories grid
6. Activity feed
7. Contributors
8. Thread list
9. Breadcrumbs
10. Category header
11. Thread view
12. Posts and replies
13. Forms
14. Sidebar
15. Pagination
16. Empty states
17. Buttons
18. Responsive design (mobile)
19. Accessibility
20. Dark mode

---

## Contributing

To contribute improvements to the forum:

1. **Test locally** in `/docs/community/`
2. **Export data** for backup
3. **Make changes** to HTML/CSS/JS
4. **Test thoroughly** across browsers
5. **Clear localStorage** and test with fresh data
6. **Submit PR** with description of changes

### Adding Features

Common enhancements:
1. Edit existing thread/reply
2. Mark thread as favorite/bookmark
3. Thread watching and notifications
4. Advanced search filters
5. Thread hijacking detection
6. Anti-spam measures
7. Rate limiting

---

## License and Attribution

HoloLoom Community Forum v1.0
Built November 2025

Uses:
- **marked.js** (https://marked.js.org/) - Markdown parsing
- **HTML5/CSS3** - Web standards

---

## Support

For issues or questions:
1. Check this README
2. Search existing threads in forum
3. Create new "Development" thread with question
4. Check browser console (F12) for errors

---

## Changelog

### v1.0.0 (November 2025)
- Initial release
- Core CRUD operations
- Voting system
- Search and filtering
- Hot score algorithm
- Responsive design
- Dark mode support
- WCAG AAA accessibility

---

**Last Updated**: November 16, 2025
**Maintainer**: HoloLoom Community Team
**Status**: Production Ready
