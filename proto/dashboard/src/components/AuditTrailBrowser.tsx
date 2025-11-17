/**
 * Audit Trail Browser Component
 *
 * Searchable UI for audit logs with:
 * - Real-time event stream
 * - Advanced filtering (date, user, type, outcome)
 * - CSV/JSON export
 * - Event detail modal
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import {
  Search,
  Filter,
  Download,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  Clock,
  AlertCircle,
  Calendar,
  User,
  FileText
} from 'lucide-react';
import { AuditEvent, EventType, Outcome } from '../types';

interface AuditTrailBrowserProps {
  onEventClick?: (event: AuditEvent) => void;
}

const EVENT_TYPE_COLORS = {
  [EventType.COMMAND]: 'bg-blue-100 text-blue-800',
  [EventType.DECISION]: 'bg-purple-100 text-purple-800',
  [EventType.APPROVAL]: 'bg-green-100 text-green-800',
  [EventType.WORKFLOW]: 'bg-indigo-100 text-indigo-800',
  [EventType.ACCESS]: 'bg-yellow-100 text-yellow-800',
  [EventType.ERROR]: 'bg-red-100 text-red-800',
  [EventType.CONFIG_CHANGE]: 'bg-orange-100 text-orange-800',
  [EventType.SYSTEM]: 'bg-gray-100 text-gray-800',
};

const OUTCOME_ICONS = {
  [Outcome.SUCCESS]: <CheckCircle2 className="w-4 h-4 text-green-600" />,
  [Outcome.FAILURE]: <XCircle className="w-4 h-4 text-red-600" />,
  [Outcome.PENDING]: <Clock className="w-4 h-4 text-yellow-600" />,
  [Outcome.CANCELLED]: <AlertCircle className="w-4 h-4 text-gray-600" />,
};

export const AuditTrailBrowser: React.FC<AuditTrailBrowserProps> = ({
  onEventClick,
}) => {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEventTypes, setSelectedEventTypes] = useState<Set<EventType>>(new Set());
  const [selectedOutcomes, setSelectedOutcomes] = useState<Set<Outcome>>(new Set());
  const [selectedUser, setSelectedUser] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  // Pagination
  const [limit, setLimit] = useState(50);
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    fetchEvents();
  }, [limit]);

  useEffect(() => {
    applyFilters();
  }, [events, searchQuery, selectedEventTypes, selectedOutcomes, selectedUser, dateFrom, dateTo]);

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`http://localhost:8000/api/audit`, {
        params: { limit }
      });

      if (response.data.success) {
        setEvents(response.data.data.events);
      }
    } catch (error) {
      console.error('Failed to fetch audit events:', error);
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...events];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(e =>
        e.action.toLowerCase().includes(query) ||
        e.user.toLowerCase().includes(query) ||
        (e.room && e.room.toLowerCase().includes(query)) ||
        JSON.stringify(e.context).toLowerCase().includes(query)
      );
    }

    // Event type filter
    if (selectedEventTypes.size > 0) {
      filtered = filtered.filter(e => selectedEventTypes.has(e.event_type));
    }

    // Outcome filter
    if (selectedOutcomes.size > 0) {
      filtered = filtered.filter(e => selectedOutcomes.has(e.outcome));
    }

    // User filter
    if (selectedUser) {
      filtered = filtered.filter(e => e.user === selectedUser);
    }

    // Date range filter
    if (dateFrom) {
      const fromDate = new Date(dateFrom);
      filtered = filtered.filter(e => new Date(e.timestamp) >= fromDate);
    }
    if (dateTo) {
      const toDate = new Date(dateTo);
      toDate.setHours(23, 59, 59, 999); // End of day
      filtered = filtered.filter(e => new Date(e.timestamp) <= toDate);
    }

    setFilteredEvents(filtered);
  };

  const handleEventTypeToggle = (type: EventType) => {
    const newTypes = new Set(selectedEventTypes);
    if (newTypes.has(type)) {
      newTypes.delete(type);
    } else {
      newTypes.add(type);
    }
    setSelectedEventTypes(newTypes);
  };

  const handleOutcomeToggle = (outcome: Outcome) => {
    const newOutcomes = new Set(selectedOutcomes);
    if (newOutcomes.has(outcome)) {
      newOutcomes.delete(outcome);
    } else {
      newOutcomes.add(outcome);
    }
    setSelectedOutcomes(newOutcomes);
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setSelectedEventTypes(new Set());
    setSelectedOutcomes(new Set());
    setSelectedUser('');
    setDateFrom('');
    setDateTo('');
  };

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const response = await axios.get(`http://localhost:8000/api/audit`, {
        params: { limit: 1000 },
        responseType: 'blob'
      });

      if (format === 'csv') {
        // Convert to CSV
        const csv = eventsToCSV(filteredEvents);
        downloadFile(csv, 'audit-trail.csv', 'text/csv');
      } else {
        // Download as JSON
        const json = JSON.stringify(filteredEvents, null, 2);
        downloadFile(json, 'audit-trail.json', 'application/json');
      }
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const eventsToCSV = (events: AuditEvent[]): string => {
    const headers = ['Timestamp', 'Event Type', 'User', 'Room', 'Action', 'Outcome', 'Error'];
    const rows = events.map(e => [
      e.timestamp,
      e.event_type,
      e.user,
      e.room || '',
      e.action,
      e.outcome,
      e.error_message || ''
    ]);

    return [
      headers.join(','),
      ...rows.map(r => r.map(cell => `"${cell}"`).join(','))
    ].join('\n');
  };

  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const uniqueUsers = Array.from(new Set(events.map(e => e.user)));

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Audit Trail Browser
        </h2>
        <p className="text-gray-600">
          Searchable event log with filtering and export
        </p>
      </div>

      {/* Search and Controls */}
      <div className="mb-6 space-y-4">
        {/* Search Bar */}
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search events (action, user, room, context)..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            <Filter className="w-5 h-5" />
            Filters
            {showFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('csv')}
              className="flex items-center gap-2 px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors"
              title="Export to CSV"
            >
              <Download className="w-5 h-5" />
              CSV
            </button>
            <button
              onClick={() => handleExport('json')}
              className="flex items-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
              title="Export to JSON"
            >
              <Download className="w-5 h-5" />
              JSON
            </button>
          </div>
        </div>

        {/* Advanced Filters */}
        {showFilters && (
          <div className="p-4 bg-gray-50 rounded-lg space-y-4">
            {/* Date Range */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Calendar className="inline w-4 h-4 mr-1" />
                  From Date
                </label>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Calendar className="inline w-4 h-4 mr-1" />
                  To Date
                </label>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* User Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <User className="inline w-4 h-4 mr-1" />
                User
              </label>
              <select
                value={selectedUser}
                onChange={(e) => setSelectedUser(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Users</option>
                {uniqueUsers.map(user => (
                  <option key={user} value={user}>{user}</option>
                ))}
              </select>
            </div>

            {/* Event Type Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Event Types
              </label>
              <div className="flex flex-wrap gap-2">
                {Object.values(EventType).map(type => (
                  <button
                    key={type}
                    onClick={() => handleEventTypeToggle(type)}
                    className={`
                      px-3 py-1 rounded text-xs font-medium transition-all
                      ${selectedEventTypes.has(type) || selectedEventTypes.size === 0
                        ? EVENT_TYPE_COLORS[type]
                        : 'bg-gray-200 text-gray-500 opacity-50'
                      }
                    `}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Outcome Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Outcomes
              </label>
              <div className="flex flex-wrap gap-2">
                {Object.values(Outcome).map(outcome => (
                  <button
                    key={outcome}
                    onClick={() => handleOutcomeToggle(outcome)}
                    className={`
                      px-3 py-1 rounded text-xs font-medium transition-all
                      ${selectedOutcomes.has(outcome) || selectedOutcomes.size === 0
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-gray-200 text-gray-500 opacity-50'
                      }
                    `}
                  >
                    {outcome}
                  </button>
                ))}
              </div>
            </div>

            {/* Clear Filters */}
            <div className="flex items-center justify-between pt-2 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                Showing {filteredEvents.length} of {events.length} events
              </p>
              <button
                onClick={handleClearFilters}
                className="text-sm text-blue-600 hover:text-blue-800 font-medium"
              >
                Clear All Filters
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Events List */}
      <div className="space-y-2">
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="mt-2 text-gray-600">Loading events...</p>
          </div>
        ) : filteredEvents.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-600">No events found</p>
            <p className="text-sm text-gray-500 mt-1">Try adjusting your filters</p>
          </div>
        ) : (
          filteredEvents.map((event) => (
            <div
              key={event.event_id}
              onClick={() => {
                setSelectedEvent(event);
                if (onEventClick) onEventClick(event);
              }}
              className={`
                p-4 border rounded-lg cursor-pointer transition-all
                ${selectedEvent?.event_id === event.event_id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }
              `}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    {OUTCOME_ICONS[event.outcome]}
                    <span className={`px-2 py-1 rounded text-xs font-medium ${EVENT_TYPE_COLORS[event.event_type]}`}>
                      {event.event_type}
                    </span>
                    <span className="text-sm font-medium text-gray-900">
                      {event.action}
                    </span>
                  </div>

                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span>
                      <User className="inline w-3 h-3 mr-1" />
                      {event.user}
                    </span>
                    {event.room && (
                      <span className="font-mono text-xs">{event.room}</span>
                    )}
                    <span className="text-gray-400">
                      {format(new Date(event.timestamp), 'MMM dd, yyyy HH:mm:ss')}
                    </span>
                  </div>

                  {event.error_message && (
                    <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                      <strong>Error:</strong> {event.error_message}
                    </div>
                  )}
                </div>

                <span className={`
                  px-2 py-1 rounded text-xs font-medium
                  ${event.outcome === Outcome.SUCCESS ? 'bg-green-100 text-green-800' : ''}
                  ${event.outcome === Outcome.FAILURE ? 'bg-red-100 text-red-800' : ''}
                  ${event.outcome === Outcome.PENDING ? 'bg-yellow-100 text-yellow-800' : ''}
                  ${event.outcome === Outcome.CANCELLED ? 'bg-gray-100 text-gray-800' : ''}
                `}>
                  {event.outcome}
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Selected Event Details */}
      {selectedEvent && (
        <div className="mt-6 p-4 bg-indigo-50 border-2 border-indigo-500 rounded-lg">
          <h3 className="font-medium text-indigo-900 mb-3">Event Details</h3>

          <div className="grid grid-cols-2 gap-3 text-sm mb-3">
            <div>
              <span className="text-indigo-600 font-medium">Event ID:</span>{' '}
              <span className="font-mono text-xs">{selectedEvent.event_id}</span>
            </div>
            <div>
              <span className="text-indigo-600 font-medium">Timestamp:</span>{' '}
              {format(new Date(selectedEvent.timestamp), 'PPpp')}
            </div>
            <div>
              <span className="text-indigo-600 font-medium">Type:</span>{' '}
              {selectedEvent.event_type}
            </div>
            <div>
              <span className="text-indigo-600 font-medium">Outcome:</span>{' '}
              {selectedEvent.outcome}
            </div>
          </div>

          {Object.keys(selectedEvent.context).length > 0 && (
            <div className="mt-3 pt-3 border-t border-indigo-300">
              <span className="text-indigo-600 font-medium text-sm">Context:</span>
              <pre className="mt-1 text-xs bg-indigo-100 p-2 rounded overflow-auto max-h-40">
                {JSON.stringify(selectedEvent.context, null, 2)}
              </pre>
            </div>
          )}

          {selectedEvent.metadata && Object.keys(selectedEvent.metadata).length > 0 && (
            <div className="mt-3 pt-3 border-t border-indigo-300">
              <span className="text-indigo-600 font-medium text-sm">Metadata:</span>
              <pre className="mt-1 text-xs bg-indigo-100 p-2 rounded overflow-auto max-h-40">
                {JSON.stringify(selectedEvent.metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Load More */}
      {events.length >= limit && (
        <div className="mt-4 text-center">
          <button
            onClick={() => setLimit(limit + 50)}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Load More Events
          </button>
        </div>
      )}
    </div>
  );
};
