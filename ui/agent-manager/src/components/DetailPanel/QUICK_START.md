# DetailPanel - Quick Start Guide

**Get up and running in 5 minutes!**

---

## 1. Copy the Component

```bash
# Files you need:
- DetailPanel.tsx (main component)
- types.ts (type definitions)
- index.ts (exports)
```

---

## 2. Configure Tailwind CSS

Add to `tailwind.config.js`:

```javascript
theme: {
  extend: {
    animation: {
      'slide-in': 'slide-in 0.3s ease-out',
    },
    keyframes: {
      'slide-in': {
        '0%': { transform: 'translateX(100%)', opacity: '0' },
        '100%': { transform: 'translateX(0)', opacity: '1' },
      },
    },
  },
}
```

---

## 3. Import & Use

```typescript
import { DetailPanel } from './components/DetailPanel';
import { useState } from 'react';

function MyApp() {
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);

  return (
    <div className="flex h-screen">
      {/* Main content */}
      <div className="flex-1">
        <button onClick={() => setSelectedThreadId('thread-1')}>
          Select Thread
        </button>
      </div>

      {/* Detail panel */}
      {selectedThreadId && (
        <div className="w-96">
          <DetailPanel
            threadId={selectedThreadId}
            onClose={() => setSelectedThreadId(null)}
          />
        </div>
      )}
    </div>
  );
}
```

---

## 4. Verify Store

Make sure your Zustand store has these methods:

```typescript
// Already in useAgentManagerStore
- getThreadById(threadId)
- updateThread(threadId, updates)
- getThreadDependencies(threadId)
```

---

## 5. Test It

```bash
npm start
# Click "Select Thread" button
# DetailPanel should slide in from the right
```

---

## Done! 🎉

You now have a fully functional detail panel showing:
- ✅ Thread information
- ✅ Confidence tracking
- ✅ Progress bars
- ✅ Dependencies
- ✅ Tabbed interface (History, Memory, Files)

---

## Common Tasks

### Change Panel Width
```tsx
<div className="w-96">  {/* Change 96 to desired width */}
  <DetailPanel ... />
</div>
```

### Add Custom Styling
```tsx
<div className="w-96 custom-class">
  <DetailPanel ... />
</div>
```

### Close on Selection
```tsx
{selectedThreadId && (
  <div className="w-96" onClick={()=> setSelectedThreadId(null)}>
    <DetailPanel
      threadId={selectedThreadId}
      onClose={() => setSelectedThreadId(null)}
    />
  </div>
)}
```

### Integrate with OutlineView
```tsx
<OutlineView onSelectThread={(id) => setSelectedThreadId(id)} />
{selectedThreadId && <DetailPanel ... />}
```

---

## Keyboard Shortcuts

When editing thread name:
- **Enter**: Save
- **Escape**: Cancel
- **Tab**: Next field

---

## Next Steps

1. ✅ Basic setup complete
2. 📚 Read `DETAIL_PANEL_README.md` for all features
3. 🧩 Add child components when needed:
   - `StepHistory` (History tab)
   - `MemoryNodes` (Memory tab)
   - `FileTreeViewer` (Files tab)
4. 🧪 Write tests using examples in `DetailPanel.example.tsx`
5. 🚀 Deploy to production

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Panel doesn't appear | Check Zustand store has data |
| Styling looks off | Verify Tailwind config updated |
| Animation doesn't work | Check `animate-slide-in` in Tailwind config |
| Name editing broken | Verify store has `updateThread` method |

---

## Need Help?

- **Features**: See `DETAIL_PANEL_README.md`
- **Integration**: See `INTEGRATION_GUIDE.md`
- **Styling**: See `TAILWIND_CONFIG.md`
- **Examples**: See `DetailPanel.example.tsx`

---

**That's it! You're ready to use DetailPanel. Happy coding! 🚀**
