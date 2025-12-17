'use client';

import { useState } from 'react';
import { Input, Button, Badge } from '@hololoom/design-system';

interface MemorySearchProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: (query: string) => void;
}

type FilterType = 'all' | 'episodic' | 'semantic' | 'procedural';
type TimeRange = 'all' | '24h' | '7d' | '30d';

export function MemorySearch({ value, onChange, onSearch }: MemorySearchProps) {
  const [filterType, setFilterType] = useState<FilterType>('all');
  const [timeRange, setTimeRange] = useState<TimeRange>('all');
  const [showFilters, setShowFilters] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(value);
  };

  const filterTypes: { value: FilterType; label: string }[] = [
    { value: 'all', label: 'All Types' },
    { value: 'episodic', label: 'Episodic' },
    { value: 'semantic', label: 'Semantic' },
    { value: 'procedural', label: 'Procedural' },
  ];

  const timeRanges: { value: TimeRange; label: string }[] = [
    { value: 'all', label: 'All Time' },
    { value: '24h', label: 'Last 24h' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' },
  ];

  return (
    <div className="bg-bg-primary border border-border-primary rounded-xl p-4">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-tertiary" />
          <Input
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Search memories by content, entities, or concepts..."
            className="pl-10 w-full"
          />
        </div>
        <Button
          type="button"
          variant="secondary"
          onClick={() => setShowFilters(!showFilters)}
        >
          <FilterIcon />
          <span className="hidden sm:inline ml-2">Filters</span>
          {(filterType !== 'all' || timeRange !== 'all') && (
            <Badge variant="accent" size="sm" className="ml-2">
              {(filterType !== 'all' ? 1 : 0) + (timeRange !== 'all' ? 1 : 0)}
            </Badge>
          )}
        </Button>
        <Button type="submit">
          Search
        </Button>
      </form>

      {/* Expanded Filters */}
      {showFilters && (
        <div className="mt-4 pt-4 border-t border-border-primary">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Memory Type Filter */}
            <div>
              <label className="block text-sm font-medium text-fg-secondary mb-2">
                Memory Type
              </label>
              <div className="flex flex-wrap gap-2">
                {filterTypes.map(({ value: v, label }) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setFilterType(v)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      filterType === v
                        ? 'bg-cosmic-nebula/20 text-cosmic-nebula border border-cosmic-nebula/30'
                        : 'bg-bg-secondary text-fg-tertiary hover:text-fg-secondary'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Time Range Filter */}
            <div>
              <label className="block text-sm font-medium text-fg-secondary mb-2">
                Time Range
              </label>
              <div className="flex flex-wrap gap-2">
                {timeRanges.map(({ value: v, label }) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setTimeRange(v)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      timeRange === v
                        ? 'bg-cosmic-nebula/20 text-cosmic-nebula border border-cosmic-nebula/30'
                        : 'bg-bg-secondary text-fg-tertiary hover:text-fg-secondary'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-4 mt-4">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                setFilterType('all');
                setTimeRange('all');
              }}
            >
              Clear Filters
            </Button>
            <span className="text-xs text-fg-tertiary">
              Tip: Use @entity to search by entity, #type for memory type
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function FilterIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}
