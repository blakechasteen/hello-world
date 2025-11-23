/**
 * Main App Component
 * ==================
 *
 * **Created**: November 22, 2025
 * **Purpose**: Root application component
 */

import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SkillBrowser } from './components/SkillBrowser';
import './styles.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: false,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        <SkillBrowser />
      </div>
    </QueryClientProvider>
  );
};
