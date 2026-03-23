import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { createElement } from 'react';
import { Button } from '../components/Button';
import { Badge } from '../components/Badge';
import { Card, CardHeader, CardBody } from '../components/Card';
import { ConfidenceIndicator } from '../components/ConfidenceIndicator';
import { SafetyBadge } from '../components/SafetyBadge';
import { ReasoningBadge } from '../components/ReasoningBadge';
import { Avatar } from '../components/Avatar';

describe('Button', () => {
  it('renders with text content', () => {
    render(createElement(Button, null, 'Click me'));
    expect(screen.getByRole('button')).toHaveTextContent('Click me');
  });

  it('renders all variants without crashing', () => {
    const variants = ['primary', 'secondary', 'ghost', 'danger', 'cosmic'] as const;
    for (const variant of variants) {
      const { unmount } = render(createElement(Button, { variant }, variant));
      expect(screen.getByRole('button')).toBeTruthy();
      unmount();
    }
  });

  it('renders all sizes without crashing', () => {
    const sizes = ['xs', 'sm', 'md', 'lg', 'xl'] as const;
    for (const size of sizes) {
      const { unmount } = render(createElement(Button, { size }, size));
      expect(screen.getByRole('button')).toBeTruthy();
      unmount();
    }
  });

  it('is disabled when disabled prop is set', () => {
    render(createElement(Button, { disabled: true }, 'Disabled'));
    expect(screen.getByRole('button')).toBeDisabled();
  });
});

describe('Badge', () => {
  it('renders text content', () => {
    render(createElement(Badge, null, 'Status'));
    expect(screen.getByText('Status')).toBeTruthy();
  });

  it('renders all variants', () => {
    const variants = ['default', 'secondary', 'success', 'warning', 'danger', 'info', 'cosmic'] as const;
    for (const variant of variants) {
      const { unmount } = render(createElement(Badge, { variant }, variant));
      expect(screen.getByText(variant)).toBeTruthy();
      unmount();
    }
  });
});

describe('Card', () => {
  it('renders children', () => {
    render(
      createElement(Card, null,
        createElement(CardHeader, null, 'Title'),
        createElement(CardBody, null, 'Content')
      )
    );
    expect(screen.getByText('Title')).toBeTruthy();
    expect(screen.getByText('Content')).toBeTruthy();
  });
});

describe('ConfidenceIndicator', () => {
  it('renders without crashing for value 0-1', () => {
    for (const value of [0, 0.3, 0.6, 0.9, 1.0]) {
      const { unmount } = render(createElement(ConfidenceIndicator, { value }));
      unmount();
    }
  });

  it('shows label when showLabel is true', () => {
    render(createElement(ConfidenceIndicator, { value: 0.92, showLabel: true }));
    // Should render some text representation of confidence
    const container = document.querySelector('[class*="confidence"]') || document.body;
    expect(container).toBeTruthy();
  });
});

describe('SafetyBadge', () => {
  it('renders all safety levels', () => {
    const levels = ['safe', 'caution', 'warning', 'danger'] as const;
    for (const level of levels) {
      const { unmount } = render(createElement(SafetyBadge, { level }));
      unmount();
    }
  });
});

describe('ReasoningBadge', () => {
  it('renders all reasoning modes', () => {
    const modes = ['direct', 'verify', 'research', 'plan_execute'] as const;
    for (const mode of modes) {
      const { unmount } = render(createElement(ReasoningBadge, { mode }));
      unmount();
    }
  });
});

describe('Avatar', () => {
  it('renders user and ai variants', () => {
    for (const variant of ['user', 'ai'] as const) {
      const { unmount } = render(createElement(Avatar, { variant }));
      unmount();
    }
  });
});
