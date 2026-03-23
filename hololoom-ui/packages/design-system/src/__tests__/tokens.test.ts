import { describe, it, expect } from 'vitest';
import { glass, effectsCSSVariables, shadows, duration, easing, animation } from '../tokens/effects';
import { cosmic, confidence, safety } from '../tokens/colors';
import { spacing } from '../tokens/spacing';

describe('design tokens', () => {
  describe('glass morphism tokens', () => {
    it('defines three intensity levels', () => {
      expect(glass.bg.low).toContain('rgba');
      expect(glass.bg.medium).toContain('rgba');
      expect(glass.bg.high).toContain('rgba');
    });

    it('defines blur levels', () => {
      expect(glass.blur.low).toContain('blur');
      expect(glass.blur.medium).toContain('blur');
      expect(glass.blur.high).toContain('blur');
    });

    it('exports CSS variables', () => {
      expect(effectsCSSVariables['--glass-bg']).toBe(glass.bg.medium);
      expect(effectsCSSVariables['--glass-blur']).toBe(glass.blur.medium);
      expect(effectsCSSVariables['--glass-border']).toBe(glass.border.default);
    });
  });

  describe('effects tokens', () => {
    it('has cosmic glow shadows', () => {
      expect(shadows.glow.nebula).toContain('rgba');
      expect(shadows.glow.aurora).toContain('rgba');
    });

    it('has HoloLoom-specific easings', () => {
      expect(easing.weaveIn).toContain('cubic-bezier');
      expect(easing.cosmic).toContain('cubic-bezier');
      expect(easing.spring).toContain('cubic-bezier');
    });

    it('has cosmic animations', () => {
      expect(animation.glow).toContain('glow');
      expect(animation.weave).toContain('weave');
      expect(animation.aurora).toContain('aurora');
    });

    it('defines standard durations', () => {
      expect(duration.fast).toBe('100ms');
      expect(duration.normal).toBe('200ms');
      expect(duration.slow).toBe('300ms');
    });
  });

  describe('colors', () => {
    it('exports cosmic color palette', () => {
      expect(cosmic).toBeTruthy();
      expect(cosmic.nebula).toBeTruthy();
    });

    it('exports confidence colors', () => {
      expect(confidence.high).toBeTruthy();
      expect(confidence.low).toBeTruthy();
    });

    it('exports safety colors', () => {
      expect(safety.safe).toBeTruthy();
      expect(safety.danger).toBeTruthy();
    });
  });

  describe('spacing', () => {
    it('exports spacing tokens', () => {
      expect(spacing).toBeTruthy();
      expect(typeof spacing).toBe('object');
    });
  });
});
