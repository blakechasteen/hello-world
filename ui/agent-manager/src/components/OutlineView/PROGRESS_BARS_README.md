# ProgressBars Component

Multi-dimensional progress tracking component for HoloLoom Agent Manager UI. Displays three progress metrics simultaneously:

1. **Step Progress** (Blue) - Current step / total steps
2. **Time Progress** (Amber) - Elapsed time / time budget
3. **Token Progress** (Purple) - Tokens used / token budget

## Philosophy

> "Tufte-style: Maximum data, minimum ink"

The component follows Edward Tufte's data visualization principles:
- Maximize the data-ink ratio
- Minimize decorative elements
- Show meaning prominently
- Color-blind friendly palette
- Responsive and accessible

## Features

- ✅ Three layout variants (stacked, inline, detailed)
- ✅ Three size presets (sm, md, lg)
- ✅ Graceful handling of undefined budgets
- ✅ Overflow detection (shows red when over budget)
- ✅ Smooth CSS transitions
- ✅ Optional shimmer animation
- ✅ Accessibility support (ARIA roles and attributes)
- ✅ TypeScript support with full type safety

## Components

### ProgressBar (Single Bar)

Reusable single progress bar with configurable appearance.

```typescript
interface ProgressBarProps {
  value: number;                           // Current value
  max?: number;                            // Max value (default: 100 for %)
  color: 'blue' | 'amber' | 'purple' | 'green' | 'red';
  size?: 'sm' | 'md' | 'lg';               // Bar height
  showLabel?: boolean;                     // Show percentage/value
  label?: string;                          // Custom label
  formatValue?: (value: number, max?: number) => string;
  animated?: boolean;                      // Shimmer animation
  className?: string;
}
```

#### Basic Usage

```tsx
import { ProgressBar } from '@components';

<ProgressBar
  value={45}
  max={100}
  color="blue"
  size="md"
  showLabel={true}
/>
```

#### All Size Options

```tsx
<ProgressBar value={50} max={100} color="blue" size="sm" /> {/* 4px */}
<ProgressBar value={50} max={100} color="blue" size="md" /> {/* 6px */}
<ProgressBar value={50} max={100} color="blue" size="lg" /> {/* 8px */}
```

#### Color Meanings

- **Blue** - Primary progress (steps, iterations, primary metric)
- **Amber** - Time-related progress (with optional warning on overage)
- **Purple** - Token/resource usage
- **Green** - Success or positive progress
- **Red** - Over-budget or error state

### ProgressBars (Multi-Dimensional)

Main component showing three progress metrics simultaneously.

```typescript
interface ProgressBarsProps {
  currentStep: number;
  totalSteps: number;
  elapsedTimeMs: number;
  timeBudgetMs?: number;                   // Optional
  tokensUsed: number;
  tokenBudget?: number;                    // Optional
  variant?: 'stacked' | 'inline' | 'detailed';
  size?: 'sm' | 'md' | 'lg';
  showPercentages?: boolean;
  showValues?: boolean;
  className?: string;
}
```

## Usage Examples

### Example 1: Basic Stacked (Default)

Vertical layout with three bars stacked on top of each other. Most compact.

```tsx
import { ProgressBars } from '@components';

<ProgressBars
  currentStep={3}
  totalSteps={7}
  elapsedTimeMs={4500}
  timeBudgetMs={10500}
  tokensUsed={750}
  tokenBudget={1750}
/>
```

### Example 2: Inline Variant

Horizontal layout with bars side by side. Best for wide containers.

```tsx
<ProgressBars
  currentStep={5}
  totalSteps={10}
  elapsedTimeMs={6000}
  timeBudgetMs={12000}
  tokensUsed={1400}
  tokenBudget={3500}
  variant="inline"
  size="sm"
/>
```

### Example 3: Detailed Variant

Stacked with labels, headers, and value display. Most informative.

```tsx
<ProgressBars
  currentStep={2}
  totalSteps={5}
  elapsedTimeMs={4000}
  timeBudgetMs={10000}
  tokensUsed={1000}
  tokenBudget={2500}
  variant="detailed"
  size="lg"
  showValues={true}
/>
```

### Example 4: Without Budgets

Only show step progress if time and token budgets are not available.

```tsx
<ProgressBars
  currentStep={3}
  totalSteps={6}
  elapsedTimeMs={3000}
  tokensUsed={600}
  // No timeBudgetMs or tokenBudget - will only show step progress
/>
```

### Example 5: Integration with AgentThread

Use with thread data from HoloLoom backend:

```tsx
interface AgentThread {
  currentStep: number;
  totalSteps: number;
  elapsedTimeMs: number;
  tokensUsed: number;
  tokenBudget?: number;
}

const thread: AgentThread = {
  currentStep: 3,
  totalSteps: 7,
  elapsedTimeMs: 4500,
  tokensUsed: 750,
  tokenBudget: 1750,
};

<ProgressBars
  currentStep={thread.currentStep}
  totalSteps={thread.totalSteps}
  elapsedTimeMs={thread.elapsedTimeMs}
  timeBudgetMs={10500}
  tokensUsed={thread.tokensUsed}
  tokenBudget={thread.tokenBudget}
  variant="detailed"
/>
```

## Layout Variants

### Stacked (Default)

Vertical layout - most compact, works well in narrow containers.

```
█████░░░░░░░░░  Step 3/7
██████░░░░░░░░░  Time 4.5s/10.5s
███░░░░░░░░░░░░░  Tokens 750/1750
```

**Best for:**
- Sidebar views
- Narrow containers
- Minimal space requirements
- Default/fallback layout

### Inline

Horizontal layout - bars side by side for direct comparison.

```
█████░░░░░░░░░  ██████░░░░░░░░░  ███░░░░░░░░░░░░░
```

**Best for:**
- Wide containers
- Dashboard views
- Comparing progress rates
- Responsive layouts

### Detailed

Stacked with labels, headers, and optional value display.

```
STEPS
█████░░░░░░░░░ 3/7

TIME
██████░░░░░░░░░ 4.5s / 10.5s

TOKENS
███░░░░░░░░░░░░░ 750 / 1750
```

**Best for:**
- Detailed monitoring views
- Primary progress display
- When space permits
- Full context needed

## Size Presets

| Size | Height | Use Case |
|------|--------|----------|
| `sm` | 4px | Compact layouts, lists, dense info |
| `md` | 6px | Default, general purpose |
| `lg` | 8px | Primary display, focus areas |

## Color Handling

### Budget Exceeded

When a value exceeds its budget, the bar automatically turns red:

```tsx
// Time over budget
<ProgressBars
  elapsedTimeMs={12000}
  timeBudgetMs={10000}  // Over! → Bar turns red
  // ...
/>

// Token over budget
<ProgressBars
  tokensUsed={2500}
  tokenBudget={2000}  // Over! → Bar turns red
  // ...
/>
```

### Optional Budgets

If a budget is not provided (undefined), that progress bar is not shown:

```tsx
// Only step progress shown
<ProgressBars
  currentStep={3}
  totalSteps={7}
  elapsedTimeMs={4500}
  tokensUsed={750}
  // No timeBudgetMs → Time bar not shown
  // No tokenBudget → Token bar not shown
/>
```

## Display Options

### Minimal (Default)

No labels, just the bars.

```tsx
<ProgressBars {...props} showPercentages={false} showValues={false} />
```

### With Percentages

Shows completion percentage on each bar.

```tsx
<ProgressBars {...props} showPercentages={true} />
// Output: "50%", "45%", "30%"
```

### With Values

Shows actual values (e.g., "3/7", "4.5s / 10.5s", "750 / 1750").

```tsx
<ProgressBars {...props} showValues={true} />
// Output: "3/7", "4.5s / 10.5s", "750 / 1750"
```

### Detailed Variant

Automatically shows formatted labels with headers:

```tsx
<ProgressBars {...props} variant="detailed" />
// Shows headers ("STEPS", "TIME", "TOKENS") and formatted values
```

## Accessibility

The component includes full accessibility support:

- ✅ ARIA `role="region"` on container
- ✅ ARIA `aria-label` describing purpose
- ✅ Individual progress bars have `role="progressbar"`
- ✅ ARIA `aria-valuenow`, `aria-valuemin`, `aria-valuemax` on bars
- ✅ Keyboard navigation support (inherited from Tailwind)
- ✅ Screen reader friendly labels
- ✅ Color not the only indicator (gradients + structure)

## Animations

### Smooth Transitions

All value changes animate smoothly over 300ms:

```tsx
// When currentStep changes, bar width animates
<ProgressBars currentStep={3} {...} /> // → step goes 2 → 3
// Bar width transitions smoothly (transition-all duration-300)
```

### Optional Shimmer

Enable pulse animation for "in progress" states:

```tsx
<ProgressBar
  value={50}
  max={100}
  animated={true}  // Enable shimmer
/>
```

The animation is disabled automatically when progress is complete:
- Step progress: `animated={currentStep < totalSteps}`
- Time progress: `animated={elapsedTimeMs < timeBudgetMs}`
- Token progress: `animated={tokensUsed < tokenBudget}`

## Time Formatting

Times are automatically formatted:

```
0-999ms    → "125ms"
1000+ms    → "2.5s"
```

## Token Formatting

Token counts are automatically formatted:

```
0-999      → "750"
1000+      → "1.2k"
```

## Customization

### Custom Styling

Use the `className` prop to add custom styles:

```tsx
<ProgressBars
  {...props}
  className="p-2 bg-slate-800/50 rounded border border-slate-700"
/>
```

### Custom Formatters

Provide custom formatting functions:

```tsx
<ProgressBar
  value={4500}
  max={10000}
  color="amber"
  formatValue={(value, max) => {
    const pct = ((value / max) * 100).toFixed(1);
    return `${pct}% done`;
  }}
/>
```

## Integration with OutlineView

The ProgressBars component is designed for the OutlineView feature in Phase 3:

```tsx
import { ProgressBars, StepRow } from '@components';

export const OutlineViewExample = ({ threads }) => {
  return (
    <div>
      {threads.map(thread => (
        <div key={thread.id}>
          {/* Thread-level progress */}
          <ProgressBars
            currentStep={thread.currentStep}
            totalSteps={thread.totalSteps}
            elapsedTimeMs={thread.elapsedTimeMs}
            timeBudgetMs={thread.timeBudgetMs}
            tokensUsed={thread.tokensUsed}
            tokenBudget={thread.tokenBudget}
            variant="detailed"
          />

          {/* Individual steps */}
          {thread.steps.map(step => (
            <StepRow key={step.id} step={step} />
          ))}
        </div>
      ))}
    </div>
  );
};
```

## Performance

- ⚡ **Lightweight**: ~8kb minified (both components combined)
- ⚡ **No external dependencies**: Uses only React and Tailwind
- ⚡ **Optimized renders**: useMemo for expensive calculations
- ⚡ **CSS transitions**: Hardware-accelerated animations
- ⚡ **Accessible**: Full ARIA support without JS overhead

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Type Safety

Full TypeScript support with exported types:

```typescript
import type {
  ProgressBarProps,
  ProgressBarsProps,
  ProgressColor,
  ProgressSize,
  ProgressVariant,
} from '@components';
```

## Testing

Example test cases:

```typescript
describe('ProgressBars', () => {
  it('shows only step progress when no budgets provided', () => {
    render(
      <ProgressBars
        currentStep={2}
        totalSteps={5}
        elapsedTimeMs={2000}
        tokensUsed={400}
      />
    );
    expect(screen.getByRole('region')).toBeInTheDocument();
    // Time and token bars should not be present
  });

  it('turns red when over budget', () => {
    const { container } = render(
      <ProgressBars
        currentStep={5}
        totalSteps={5}
        elapsedTimeMs={12000}
        timeBudgetMs={10000}  // Over!
        tokensUsed={1000}
        tokenBudget={2000}
      />
    );
    // Check for red gradient in time bar
  });

  it('formats tokens with k suffix', () => {
    render(
      <ProgressBars
        currentStep={5}
        totalSteps={5}
        elapsedTimeMs={5000}
        tokensUsed={1500}
        tokenBudget={3000}
        variant="detailed"
        showValues={true}
      />
    );
    expect(screen.getByText(/1.5k/)).toBeInTheDocument();
  });
});
```

## Examples

See `ProgressBars.examples.tsx` for complete working examples including:

- ✅ Stacked variant with animation
- ✅ Inline variant
- ✅ Detailed variant with labels
- ✅ All size options
- ✅ Over-budget states
- ✅ Optional budgets
- ✅ Display options comparison

## Related Components

- **StepRow** - Individual task step display
- **ProgressBar** - Single progress bar (reusable)
- **StatusIcon** - Step status indicators
- **ConfidenceIndicator** - Confidence score display

## Future Enhancements

- [ ] Custom gradient colors
- [ ] Animated speed control
- [ ] Milestone markers
- [ ] Historical trend sparklines
- [ ] Touch gesture support
- [ ] Dark/light mode theming
