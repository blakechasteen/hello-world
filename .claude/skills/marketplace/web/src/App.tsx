/**
 * Main App Component
 * ==================
 *
 * **Created**: November 22, 2025
 * **Updated**: December 3, 2025 - Added skill detail routing
 * **Purpose**: Root application component with simple routing
 */

import React, { useState, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SkillBrowser } from './components/SkillBrowser';
import { SkillDetail } from './components/SkillDetail';
import './styles.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * Main application with simple state-based routing.
 *
 * Views:
 * - null selectedSkillId: Show SkillBrowser (main grid view)
 * - string selectedSkillId: Show SkillDetail (full skill view)
 */
export const App: React.FC = () => {
  const [selectedSkillId, setSelectedSkillId] = useState<string | null>(null);

  const handleSelectSkill = useCallback((skillId: string) => {
    setSelectedSkillId(skillId);
  }, []);

  const handleBackToBrowser = useCallback(() => {
    setSelectedSkillId(null);
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        {selectedSkillId ? (
          <SkillDetail
            skillId={selectedSkillId}
            onBack={handleBackToBrowser}
          />
        ) : (
          <SkillBrowser onSelectSkill={handleSelectSkill} />
        )}
      </div>
    </QueryClientProvider>
  );
};
