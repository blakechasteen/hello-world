"""
Governance Engine - Democratic decision-making for O2

Implements:
- Proposal system (anyone can propose)
- Transparent voting (reaction-based)
- Automatic execution (approved proposals auto-execute)
- Complete audit trail
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psycopg2
import psycopg2.extras
import os

logger = logging.getLogger('o2-bot.governance')


class GovernanceEngine:
    """
    Democratic governance system for O2 communities.

    Features:
    - Proposal creation (anyone can propose changes)
    - Voting (yes/no/abstain via reactions)
    - Configurable thresholds (majority, 2/3, consensus)
    - Time-boxed voting periods
    - Automatic execution on approval
    """

    def __init__(self, matrix_client):
        self.client = matrix_client
        self.db_conn = None

    async def initialize(self):
        """Initialize database connection."""
        # Connect to PostgreSQL
        self.db_conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'o2'),
            user=os.getenv('POSTGRES_USER', 'o2'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        logger.info("Governance engine connected to database")

    async def create_proposal(
        self,
        room_id: str,
        author: str,
        title: str,
        description: str = "",
        proposal_type: str = "policy",
        threshold: float = 0.67,
        duration_hours: int = 72
    ) -> int:
        """
        Create a new proposal.

        Args:
            room_id: Matrix room ID
            author: User ID of proposal author
            title: Proposal title
            description: Detailed description (optional)
            proposal_type: Type (policy, feature, governance, technical)
            threshold: Approval threshold (0.0-1.0)
            duration_hours: Voting duration in hours

        Returns:
            Proposal ID
        """
        ends_at = datetime.now() + timedelta(hours=duration_hours)

        with self.db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO proposals
                (room_id, author, title, description, proposal_type, threshold, duration_hours, ends_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (room_id, author, title, description, proposal_type, threshold, duration_hours, ends_at)
            )
            proposal_id = cur.fetchone()[0]
            self.db_conn.commit()

        logger.info(f"Created proposal #{proposal_id} in room {room_id} by {author}")
        return proposal_id

    async def record_vote(self, proposal_id: int, user_id: str, vote: str):
        """
        Record a vote on a proposal.

        Args:
            proposal_id: Proposal ID
            user_id: User ID
            vote: Vote (yes, no, abstain)
        """
        if vote not in ['yes', 'no', 'abstain']:
            raise ValueError(f"Invalid vote: {vote}")

        with self.db_conn.cursor() as cur:
            # Upsert vote (update if exists, insert if not)
            cur.execute(
                """
                INSERT INTO votes (proposal_id, user_id, vote)
                VALUES (%s, %s, %s)
                ON CONFLICT (proposal_id, user_id)
                DO UPDATE SET vote = EXCLUDED.vote, voted_at = CURRENT_TIMESTAMP
                """,
                (proposal_id, user_id, vote)
            )
            self.db_conn.commit()

        logger.info(f"Recorded vote: {user_id} voted {vote} on proposal #{proposal_id}")

    async def tally_votes(self, proposal_id: int) -> Dict:
        """
        Tally votes for a proposal.

        Args:
            proposal_id: Proposal ID

        Returns:
            Dictionary with vote counts and outcome
        """
        with self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Get proposal
            cur.execute(
                "SELECT * FROM proposals WHERE id = %s",
                (proposal_id,)
            )
            proposal = cur.fetchone()

            if not proposal:
                raise ValueError(f"Proposal #{proposal_id} not found")

            # Get vote counts
            cur.execute(
                """
                SELECT vote, COUNT(*) as count
                FROM votes
                WHERE proposal_id = %s
                GROUP BY vote
                """,
                (proposal_id,)
            )
            votes = {row['vote']: row['count'] for row in cur.fetchall()}

        yes_count = votes.get('yes', 0)
        no_count = votes.get('no', 0)
        abstain_count = votes.get('abstain', 0)
        total = yes_count + no_count  # Abstentions don't count toward total

        # Calculate percentages
        yes_pct = yes_count / total if total > 0 else 0.0
        no_pct = no_count / total if total > 0 else 0.0

        # Determine outcome
        threshold = proposal['threshold']
        if yes_pct >= threshold:
            outcome = "PASSED"
            status = "approved"
        elif datetime.now() > proposal['ends_at']:
            outcome = "REJECTED (voting period ended)"
            status = "rejected"
        else:
            outcome = "PENDING (voting still open)"
            status = "active"

        # Update proposal status
        with self.db_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE proposals
                SET status = %s,
                    result = %s::jsonb
                WHERE id = %s
                """,
                (
                    status,
                    psycopg2.extras.Json({
                        'yes': yes_count,
                        'no': no_count,
                        'abstain': abstain_count,
                        'total': total,
                        'outcome': outcome
                    }),
                    proposal_id
                )
            )
            self.db_conn.commit()

        return {
            'proposal_id': proposal_id,
            'yes': yes_count,
            'no': no_count,
            'abstain': abstain_count,
            'total': total,
            'yes_pct': yes_pct,
            'no_pct': no_pct,
            'threshold': threshold,
            'status': status,
            'outcome': outcome,
            'ends_at': proposal['ends_at']
        }

    async def get_latest_proposal(self, room_id: str) -> Optional[int]:
        """
        Get the latest active proposal in a room.

        Args:
            room_id: Matrix room ID

        Returns:
            Proposal ID or None
        """
        with self.db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM proposals
                WHERE room_id = %s AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (room_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    async def get_active_proposals(self, room_id: str) -> List[Dict]:
        """
        Get all active proposals in a room.

        Args:
            room_id: Matrix room ID

        Returns:
            List of proposal dictionaries
        """
        with self.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM proposals
                WHERE room_id = %s AND status = 'active'
                ORDER BY created_at DESC
                """,
                (room_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    async def execute_proposal(self, proposal_id: int):
        """
        Execute an approved proposal.

        Args:
            proposal_id: Proposal ID
        """
        # TODO: Implement proposal execution logic
        # This will depend on proposal type:
        # - policy: Update room policies
        # - feature: Enable/disable features
        # - governance: Update voting rules
        # - technical: Update configuration

        logger.info(f"Executing proposal #{proposal_id} (not implemented yet)")

    async def close(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Governance engine database connection closed")
