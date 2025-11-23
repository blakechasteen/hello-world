/**
 * Search Bar Component
 * ====================
 *
 * **Created**: November 22, 2025
 * **Purpose**: Search and filter skills in the marketplace
 *
 * Provides text search, category filter, and sort options.
 */

import React, { useState } from 'react';
import { Search, Filter, SortAsc } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string, category?: string, sortBy?: string) => void;
  categories: string[];
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, categories }) => {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [showFilters, setShowFilters] = useState(false);

  const handleSearch = () => {
    onSearch(query, category || undefined, sortBy);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="search-bar">
      <div className="search-input-wrapper">
        <Search size={20} />
        <input
          type="text"
          className="search-input"
          placeholder="Search skills..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button className="btn btn-primary" onClick={handleSearch}>
          Search
        </button>
        <button
          className="btn btn-outline"
          onClick={() => setShowFilters(!showFilters)}
        >
          <Filter size={16} />
          Filters
        </button>
      </div>

      {showFilters && (
        <div className="search-filters">
          <div className="filter-group">
            <label>
              Category:
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                <option value="">All Categories</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat.charAt(0).toUpperCase() + cat.slice(1)}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="filter-group">
            <label>
              <SortAsc size={16} /> Sort by:
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                <option value="name">Name</option>
                <option value="popularity">Popularity</option>
                <option value="rating">Rating</option>
                <option value="recency">Recently Updated</option>
              </select>
            </label>
          </div>

          <div className="filter-actions">
            <button className="btn btn-sm" onClick={handleSearch}>
              Apply Filters
            </button>
            <button
              className="btn btn-sm btn-outline"
              onClick={() => {
                setCategory('');
                setSortBy('name');
                setQuery('');
              }}
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
