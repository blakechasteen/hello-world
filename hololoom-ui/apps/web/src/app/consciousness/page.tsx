'use client';

import { WeavingProvider } from '@/contexts/WeavingContext';
import { CognitiveShell } from '@/components/consciousness/CognitiveShell';
import { Navigation } from '@/components/Navigation';

export default function ConsciousnessPage() {
  return (
    <WeavingProvider>
      <div className="flex flex-col h-screen bg-bg-primary">
        <Navigation />
        <CognitiveShell />
      </div>
    </WeavingProvider>
  );
}
