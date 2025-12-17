'use client';

import { ThemeProvider } from '@hololoom/design-system';
import { HoloLoomProvider } from '@hololoom/api-client';
import type { ReactNode } from 'react';

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  // API configuration - can be overridden via environment variables
  const apiConfig = {
    baseUrl: process.env.NEXT_PUBLIC_HOLOLOOM_API_URL || 'http://localhost:8000',
    enableWebSocket: true,
  };

  return (
    <ThemeProvider defaultTheme="modern" storageKey="hololoom-theme">
      <HoloLoomProvider config={apiConfig}>
        {children}
      </HoloLoomProvider>
    </ThemeProvider>
  );
}
