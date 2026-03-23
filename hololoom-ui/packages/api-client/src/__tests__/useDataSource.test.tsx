import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { createElement, type ReactNode } from 'react';
import { HoloLoomProvider } from '../hooks';
import { useDataSource } from '../hooks';

// Minimal provider wrapper
function wrapper({ children }: { children: ReactNode }) {
  return createElement(
    HoloLoomProvider,
    { config: { baseUrl: 'http://localhost:8000', enableWebSocket: false } },
    children
  );
}

describe('useDataSource', () => {
  it('returns live data on successful fetch', async () => {
    const fetcher = vi.fn().mockResolvedValue({ count: 42 });

    const { result } = renderHook(
      () => useDataSource(fetcher, {}),
      { wrapper }
    );

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toEqual({ count: 42 });
    expect(result.current.source).toBe('live');
    expect(result.current.error).toBeNull();
  });

  it('falls to mock tier when fetcher throws', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('network down'));
    const mock = vi.fn().mockReturnValue({ count: 0, fallback: true });

    const { result } = renderHook(
      () => useDataSource(fetcher, { mock }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toEqual({ count: 0, fallback: true });
    expect(result.current.source).toBe('mock');
    expect(result.current.error).toBeTruthy();
  });

  it('falls to empty tier when no mock provided and fetcher throws', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('fail'));

    const { result } = renderHook(
      () => useDataSource(fetcher, {}),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toBeNull();
    expect(result.current.source).toBe('empty');
  });

  it('caches to sessionStorage and serves stale on subsequent failure', async () => {
    // First call succeeds and caches
    const fetcher = vi.fn().mockResolvedValueOnce({ cached: true });

    const { result, rerender } = renderHook(
      () => useDataSource(fetcher, { cacheKey: 'test:stale' }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.source).toBe('live'));
    expect(result.current.data).toEqual({ cached: true });

    // Second call fails — should fall to stale cache
    fetcher.mockRejectedValueOnce(new Error('down'));

    // Trigger refetch
    await result.current.refetch();

    // After refetch with failure, should get stale data
    await waitFor(() => {
      expect(result.current.source).toBe('stale');
    });
    expect(result.current.data).toEqual({ cached: true });
  });

  it('auto-refreshes at specified interval', async () => {
    let callCount = 0;
    const fetcher = vi.fn().mockImplementation(() => {
      callCount++;
      return Promise.resolve({ count: callCount });
    });

    const { result } = renderHook(
      () => useDataSource(fetcher, { refreshInterval: 100 }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.source).toBe('live'));
    expect(result.current.data).toEqual({ count: 1 });

    // Wait for at least one refresh
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2), { timeout: 500 });
  });
});
