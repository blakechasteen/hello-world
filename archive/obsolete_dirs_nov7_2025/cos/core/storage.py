"""
HoloLoom COS - Event Storage & Retrieval API

Philosophy: Simple, fast, reliable SQLite-based storage
Following HoloLoom conventions for async operations
"""

import sqlite3
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import json

from .types import (
    Event, EventType, EventSource, Category, ProductLine,
    DailySummary, WeeklySummary, ProductPerformance
)


class EventStore:
    """
    Event storage and retrieval.

    Philosophy:
    - Append-only event stream (immutability)
    - Views are derived (not stored)
    - Full-text search support
    - Fast retrieval by common patterns
    """

    def __init__(self, db_path: str = "cos_events.db"):
        self.db_path = Path(db_path)
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database exists and schema is current"""
        # Read schema file
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema = f.read()

        # Create/update database
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-safe database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        return conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def store(self, event: Event) -> int:
        """
        Store event in database.

        Returns:
            event_id (int)
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._store_sync, event)

    def _store_sync(self, event: Event) -> int:
        """Synchronous store implementation"""
        conn = self._get_connection()

        data = event.to_dict()

        # Insert event
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (
                type, raw_input, timestamp, source, confidence, verified,
                parsed_data, amount, quantity, unit, item, category,
                product_line, related_to, parent_goal, tags, location,
                verification_note
            ) VALUES (
                :type, :raw_input, :timestamp, :source, :confidence, :verified,
                :parsed_data, :amount, :quantity, :unit, :item, :category,
                :product_line, :related_to, :parent_goal, :tags, :location,
                :verification_note
            )
        ''', data)

        conn.commit()
        event_id = cursor.lastrowid
        conn.close()

        return event_id

    async def get_by_id(self, event_id: int) -> Optional[Event]:
        """Retrieve event by ID"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_by_id_sync, event_id)

    def _get_by_id_sync(self, event_id: int) -> Optional[Event]:
        """Synchronous get by ID"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('SELECT * FROM events WHERE id = ?', (event_id,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return Event.from_dict(dict(row))
        return None

    async def get_unverified(self) -> List[Event]:
        """
        Get all unverified events (for HITL queue).

        Priority: purchases, sales, tasks with confidence < 0.85
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_unverified_sync)

    def _get_unverified_sync(self) -> List[Event]:
        """Synchronous get unverified"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM events
            WHERE verified = FALSE
              AND type IN ('purchase', 'sale', 'task')
              AND (confidence < 0.85 OR confidence IS NULL)
            ORDER BY timestamp DESC
        ''')

        events = []
        for row in cursor.fetchall():
            events.append(Event.from_dict(dict(row)))

        conn.close()
        return events

    async def verify(self, event_id: int, verified: bool = True, note: Optional[str] = None):
        """Mark event as verified (HITL confirmation)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._verify_sync, event_id, verified, note)

    def _verify_sync(self, event_id: int, verified: bool, note: Optional[str]):
        """Synchronous verify"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('''
            UPDATE events
            SET verified = ?, verification_note = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (verified, note, event_id))

        conn.commit()
        conn.close()

    async def update(self, event_id: int, updates: Dict[str, Any]):
        """Update event fields (for corrections)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._update_sync, event_id, updates)

    def _update_sync(self, event_id: int, updates: Dict[str, Any]):
        """Synchronous update"""
        conn = self._get_connection()

        # Build UPDATE query dynamically
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [event_id]

        cursor = conn.cursor()
        cursor.execute(f'''
            UPDATE events
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', values)

        conn.commit()
        conn.close()

    async def query(
        self,
        event_type: Optional[EventType] = None,
        category: Optional[Category] = None,
        product_line: Optional[ProductLine] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        verified_only: bool = False,
        limit: int = 100
    ) -> List[Event]:
        """
        Query events with filters.

        Common queries:
        - All tasks from last week
        - All sales for a product line
        - All unverified purchases
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._query_sync,
            event_type, category, product_line, start_date, end_date, verified_only, limit
        )

    def _query_sync(
        self,
        event_type: Optional[EventType],
        category: Optional[Category],
        product_line: Optional[ProductLine],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        verified_only: bool,
        limit: int
    ) -> List[Event]:
        """Synchronous query"""
        conn = self._get_connection()

        # Build WHERE clause
        conditions = []
        params = []

        if event_type:
            conditions.append("type = ?")
            params.append(event_type.value)

        if category:
            conditions.append("category = ?")
            params.append(category.value)

        if product_line:
            conditions.append("product_line = ?")
            params.append(product_line.value)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        if verified_only:
            conditions.append("verified = TRUE")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Execute query
        cursor = conn.cursor()
        cursor.execute(f'''
            SELECT * FROM events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', params + [limit])

        events = []
        for row in cursor.fetchall():
            events.append(Event.from_dict(dict(row)))

        conn.close()
        return events

    async def search(self, query: str, limit: int = 50) -> List[Event]:
        """Full-text search on raw_input and tags"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> List[Event]:
        """Synchronous search"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('''
            SELECT e.*
            FROM events e
            JOIN events_fts fts ON e.id = fts.rowid
            WHERE events_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (query, limit))

        events = []
        for row in cursor.fetchall():
            events.append(Event.from_dict(dict(row)))

        conn.close()
        return events

    async def get_daily_summary(self, date: datetime) -> DailySummary:
        """Get summary for a specific day"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_daily_summary_sync, date)

    def _get_daily_summary_sync(self, date: datetime) -> DailySummary:
        """Synchronous daily summary"""
        conn = self._get_connection()

        date_str = date.strftime('%Y-%m-%d')

        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM daily_summary
            WHERE date = ?
        ''', (date_str,))

        rows = cursor.fetchall()

        # Aggregate across product lines
        summary = DailySummary(date=date)

        for row in rows:
            row_dict = dict(row)
            summary.revenue += Decimal(str(row_dict.get('revenue', 0) or 0))
            summary.cogs += Decimal(str(row_dict.get('cogs', 0) or 0))
            summary.labor_cost += Decimal(str(row_dict.get('labor_cost', 0) or 0))
            summary.hours_worked += Decimal(str(row_dict.get('hours_worked', 0) or 0))

            # Track by event type
            event_type = EventType(row_dict['type'])
            summary.events_by_type[event_type] = row_dict['event_count']

            # Track revenue by product
            if row_dict.get('product_line'):
                product = ProductLine(row_dict['product_line'])
                revenue = Decimal(str(row_dict.get('revenue', 0) or 0))
                summary.revenue_by_product[product] = revenue

        summary.profit = summary.revenue - summary.cogs - summary.labor_cost

        conn.close()
        return summary

    async def get_weekly_summary(self, week: str) -> WeeklySummary:
        """
        Get summary for a week.

        Args:
            week: ISO week format "2025-W45"
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_weekly_summary_sync, week)

    def _get_weekly_summary_sync(self, week: str) -> WeeklySummary:
        """Synchronous weekly summary"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM weekly_summary
            WHERE week = ?
        ''', (week,))

        row = cursor.fetchone()

        if row:
            row_dict = dict(row)
            return WeeklySummary(
                week=week,
                revenue=Decimal(str(row_dict.get('revenue', 0) or 0)),
                cogs=Decimal(str(row_dict.get('cogs', 0) or 0)),
                labor_cost=Decimal(str(row_dict.get('labor_cost', 0) or 0)),
                overhead=Decimal(str(row_dict.get('overhead', 0) or 0)),
                hours_worked=Decimal(str(row_dict.get('hours_worked', 0) or 0)),
                active_product_lines=row_dict.get('active_product_lines', 0),
                profit=Decimal(str(row_dict.get('revenue', 0) or 0))
                    - Decimal(str(row_dict.get('cogs', 0) or 0))
                    - Decimal(str(row_dict.get('labor_cost', 0) or 0))
                    - Decimal(str(row_dict.get('overhead', 0) or 0)),
            )

        conn.close()
        return WeeklySummary(week=week)

    async def get_product_performance(
        self,
        product_line: ProductLine,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ProductPerformance:
        """Get performance metrics for a product line"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._get_product_performance_sync,
            product_line, start_date, end_date
        )

    def _get_product_performance_sync(
        self,
        product_line: ProductLine,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> ProductPerformance:
        """Synchronous product performance"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM product_performance
            WHERE product_line = ?
        ''', (product_line.value,))

        row = cursor.fetchone()

        if row:
            row_dict = dict(row)
            return ProductPerformance(
                product_line=product_line,
                revenue=Decimal(str(row_dict.get('revenue', 0) or 0)),
                material_cost=Decimal(str(row_dict.get('material_cost', 0) or 0)),
                labor_cost=Decimal(str(row_dict.get('labor_cost', 0) or 0)),
                hours_worked=Decimal(str(row_dict.get('hours_worked', 0) or 0)),
            )

        conn.close()
        return ProductPerformance(product_line=product_line)

    async def get_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Get current inventory levels"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_inventory_sync)

    def _get_inventory_sync(self) -> Dict[str, Dict[str, Any]]:
        """Synchronous inventory"""
        conn = self._get_connection()

        cursor = conn.cursor()
        cursor.execute('SELECT * FROM inventory_current')

        inventory = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            item = row_dict['item']
            inventory[item] = {
                'quantity': float(row_dict['current_quantity']),
                'unit': row_dict['unit'],
                'avg_cost': float(row_dict['avg_unit_cost']) if row_dict['avg_unit_cost'] else 0,
                'last_updated': row_dict['last_updated'],
            }

        conn.close()
        return inventory


# Singleton instance
_store_instance: Optional[EventStore] = None


def get_store(db_path: str = "cos_events.db") -> EventStore:
    """Get singleton store instance"""
    global _store_instance
    if _store_instance is None:
        _store_instance = EventStore(db_path)
    return _store_instance
