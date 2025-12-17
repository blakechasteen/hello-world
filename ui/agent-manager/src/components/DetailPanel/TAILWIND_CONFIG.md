# Tailwind CSS Configuration for DetailPanel

## Required Animation: `animate-slide-in`

The DetailPanel component uses a custom Tailwind animation `animate-slide-in` for smooth entrance effects. To enable this animation, add the following configuration to your `tailwind.config.js`:

## Installation

### Step 1: Update `tailwind.config.js`

Add the custom animation to your Tailwind configuration:

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      // Add custom animations
      animation: {
        'slide-in': 'slide-in 0.3s ease-out',
      },
      // Add keyframe definitions
      keyframes: {
        'slide-in': {
          '0%': {
            transform: 'translateX(100%)',
            opacity: '0',
          },
          '100%': {
            transform: 'translateX(0)',
            opacity: '1',
          },
        },
      },
    },
  },
  plugins: [],
};
```

### Step 2: Verify Installation

After updating your Tailwind config, rebuild your CSS:

```bash
# If using Vite
npm run dev

# If using Create React App
npm start

# If using custom build script
npm run build:css
```

## Animation Details

### `animate-slide-in`

**Purpose**: Smooth slide-in animation from the right side of the screen

**Properties**:
- **Duration**: 300ms (0.3s)
- **Timing**: ease-out (faster at start, slower at end)
- **Direction**: Right to left (translateX: 100% → 0%)
- **Opacity**: Fade in simultaneously (0 → 1)

**Effect**: Creates a smooth, professional entrance as the detail panel appears

### Example Usage

```tsx
// Applied in DetailPanel.tsx
<div className="... animate-slide-in">
  {/* DetailPanel content */}
</div>
```

## Alternative Animations

If you prefer different entrance animations, you can customize:

### Slide-in Faster (200ms)
```javascript
'slide-in-fast': 'slide-in-fast 0.2s ease-out',
```

### Slide-in Slower (500ms)
```javascript
'slide-in-slow': 'slide-in-slow 0.5s ease-out',
```

### Fade-in Only (no slide)
```javascript
theme: {
  extend: {
    animation: {
      'fade-in': 'fade-in 0.3s ease-out',
    },
    keyframes: {
      'fade-in': {
        '0%': { opacity: '0' },
        '100%': { opacity: '1' },
      },
    },
  },
},
```

Then use in DetailPanel:
```tsx
<div className="... animate-fade-in">
  {/* DetailPanel content */}
</div>
```

### Slide-in from Top
```javascript
'slide-in-top': 'slide-in-top 0.3s ease-out',

// Keyframes
'slide-in-top': {
  '0%': {
    transform: 'translateY(-100%)',
    opacity: '0',
  },
  '100%': {
    transform: 'translateY(0)',
    opacity: '1',
  },
},
```

## Verification Checklist

- [ ] Added `animate` configuration to `theme.extend`
- [ ] Added `keyframes` configuration to `theme.extend`
- [ ] Rebuilt CSS after updating config
- [ ] DetailPanel animation works smoothly
- [ ] Animation respects system preference for `prefers-reduced-motion`

## Accessibility: Respecting `prefers-reduced-motion`

To respect user preferences for reduced motion, you can modify the animation:

```javascript
// tailwind.config.js
theme: {
  extend: {
    animation: {
      'slide-in': 'slide-in 0.3s ease-out',
    },
    keyframes: {
      'slide-in': {
        '0%': {
          transform: 'translateX(100%)',
          opacity: '0',
        },
        '100%': {
          transform: 'translateX(0)',
          opacity: '1',
        },
      },
    },
  },
},
```

Then add a media query override in your global CSS:

```css
/* globals.css or index.css */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

Or create a Tailwind plugin:

```javascript
// tailwind.config.js
plugins: [
  function({ addBase }) {
    addBase({
      '@media (prefers-reduced-motion: reduce)': {
        '*': {
          animationDuration: '0.01ms !important',
          animationIterationCount: '1 !important',
          transitionDuration: '0.01ms !important',
        },
      },
    });
  },
],
```

## CSS Variables (Optional)

For more flexibility, you can use CSS variables:

```javascript
// tailwind.config.js
theme: {
  extend: {
    animation: {
      'slide-in': 'slide-in var(--animation-duration, 0.3s) ease-out',
    },
  },
},
```

Then set the variable dynamically:

```css
.detail-panel {
  --animation-duration: 0.5s;
}
```

## Performance Tips

1. **Use `transform` and `opacity`**: These properties are GPU-accelerated and performant
2. **Keep duration short**: 200-400ms feels responsive to users
3. **Use `ease-out`**: Feels more natural than `ease-in-out`
4. **Test on low-end devices**: Ensure smooth animation on older hardware

## Browser Compatibility

The slide-in animation uses:
- `transform: translateX()` - Supported in all modern browsers
- `opacity` - Supported in all modern browsers
- `ease-out` timing function - Supported in all modern browsers

**Minimum browsers**:
- Chrome 26+
- Firefox 16+
- Safari 9+
- Edge 12+
- IE 10+ (with fallback to instant display)

## Troubleshooting

### Animation doesn't appear
- [ ] Verify `animate-slide-in` class is applied
- [ ] Check that Tailwind CSS is built (look for `animate-slide-in` in your CSS file)
- [ ] Ensure animation definition is in `tailwind.config.js`
- [ ] Check browser console for CSS errors

### Animation is too slow/fast
- Adjust duration in config: change `0.3s` to desired value
- For faster: use `0.2s`
- For slower: use `0.5s`

### Animation doesn't work on mobile
- Check `prefers-reduced-motion` setting
- Verify CSS is being applied
- Test in browser DevTools animation inspector

### Conflicts with other animations
- Ensure no other rules define `@keyframes slide-in`
- Use unique animation names if extending
- Check for animation overwrites in custom CSS

## Complete Example Config

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          900: '#0f172a',
          800: '#1e293b',
          700: '#334155',
          // ... other colors
        },
      },
      animation: {
        'slide-in': 'slide-in 0.3s ease-out',
      },
      keyframes: {
        'slide-in': {
          '0%': {
            transform: 'translateX(100%)',
            opacity: '0',
          },
          '100%': {
            transform: 'translateX(0)',
            opacity: '1',
          },
        },
      },
    },
  },
  plugins: [
    // Respect prefers-reduced-motion
    function({ addBase }) {
      addBase({
        '@media (prefers-reduced-motion: reduce)': {
          '*': {
            animationDuration: '0.01ms !important',
            animationIterationCount: '1 !important',
            transitionDuration: '0.01ms !important',
          },
        },
      });
    },
  ],
};
```

## Testing

Test the animation in different scenarios:

```bash
# Test on different screen sizes
npx tailwindcss -w

# Test with reduced motion enabled
# In browser DevTools: Rendering → Emulate CSS media feature prefers-reduced-motion → reduce

# Test performance
# In Chrome DevTools: Performance tab → record → trigger DetailPanel
```

---

**Version**: 1.0
**Last Updated**: December 2025
**Tailwind CSS**: 3.0+
