'use client';

import { useState, useCallback } from 'react';
import { Navigation } from '@/components/Navigation';
import { Button, Card, CardBody, Input, Badge } from '@hololoom/design-system';
import { useHoloLoom, useDataSource } from '@hololoom/api-client';
import type { JennySpecDTO } from '@hololoom/api-client';
import { JennyPanel } from '@/components/jenny/JennyPanel';

export default function JennyPage() {
  const { client } = useHoloLoom();
  const [query, setQuery] = useState('');
  const [panels, setPanels] = useState<JennySpecDTO[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  // Fetch existing active panels via DataSource protocol
  const { data: activePanels, source: panelsSource } = useDataSource(
    () => client.jennyPanels().then((r) => r.panels),
    { cacheKey: 'hololoom:jenny-panels' }
  );

  const handleAsk = useCallback(async () => {
    if (!query.trim() || isAsking) return;
    setIsAsking(true);

    try {
      const result = await client.jennyAsk(query.trim());
      setPanels((prev) => [...result.panels, ...prev]);
      setQuery('');
    } catch {
      // Jenny API unavailable — graceful degradation, just show empty
    } finally {
      setIsAsking(false);
    }
  }, [client, query, isAsking]);

  const allPanels = [...panels, ...(activePanels ?? [])];

  return (
    <div className="min-h-screen bg-bg-primary">
      <Navigation />

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-fg-primary mb-1">
            Jenny — Generative Visualization
          </h1>
          <p className="text-fg-tertiary">
            Ask a question and Jenny computes the right panels to show.
            {panelsSource !== 'live' && panelsSource !== 'empty' && (
              <Badge variant="secondary" size="sm" className="ml-2">{panelsSource}</Badge>
            )}
          </p>
        </div>

        {/* Query input */}
        <Card className="mb-8">
          <CardBody>
            <form
              onSubmit={(e) => { e.preventDefault(); handleAsk(); }}
              className="flex gap-3"
            >
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to visualize?"
                className="flex-1"
                disabled={isAsking}
              />
              <Button type="submit" disabled={!query.trim() || isAsking}>
                {isAsking ? 'Generating...' : 'Ask Jenny'}
              </Button>
            </form>
          </CardBody>
        </Card>

        {/* Panel grid */}
        {allPanels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {allPanels
              .filter((p) => p.lifecycle !== 'archived')
              .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))
              .map((spec) => (
                <JennyPanel key={spec.spec_id} spec={spec} />
              ))}
          </div>
        ) : (
          <div className="text-center py-20">
            <p className="text-fg-tertiary text-lg">
              No panels yet. Ask Jenny a question to generate visualizations.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
