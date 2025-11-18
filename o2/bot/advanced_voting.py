"""
Advanced Voting Systems - Ranked Choice & Liquid Democracy

Implements:
- Ranked Choice Voting (instant runoff)
- Liquid Democracy (vote delegation)
- Quadratic Voting (prevent plutocracy)
- Approval Voting (multiple choices)
- Score Voting (rate candidates 0-10)
"""

import asyncio
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
import math

logger = logging.getLogger('o2-bot.advanced_voting')


class VotingMethod(Enum):
    """Supported voting methods."""
    SIMPLE_MAJORITY = "simple_majority"  # >50%
    SUPERMAJORITY = "supermajority"  # 2/3 or 3/4
    RANKED_CHOICE = "ranked_choice"  # Instant runoff
    APPROVAL = "approval"  # Vote for multiple options
    SCORE = "score"  # Rate each option 0-10
    QUADRATIC = "quadratic"  # Voice credits, sqrt weighting
    LIQUID = "liquid"  # Delegate votes to others


class RankedVote:
    """
    Represents a ranked choice vote.

    Example: User ranks options [A, B, C] in order of preference
    """

    def __init__(self, user_id: str, rankings: List[str]):
        self.user_id = user_id
        self.rankings = rankings  # Ordered list of option IDs
        self.voted_at = datetime.now()

    def get_choice_at_rank(self, rank: int) -> Optional[str]:
        """Get choice at specific rank (0-indexed)."""
        if rank < len(self.rankings):
            return self.rankings[rank]
        return None

    def remove_eliminated(self, eliminated: Set[str]):
        """Remove eliminated options from rankings."""
        self.rankings = [r for r in self.rankings if r not in eliminated]


class DelegatedVote:
    """
    Represents a delegated vote in liquid democracy.

    User delegates their voting power to another user (proxy).
    """

    def __init__(self, delegator: str, delegate_to: str, scope: str = "all"):
        self.delegator = delegator
        self.delegate_to = delegate_to
        self.scope = scope  # "all" or specific proposal type
        self.delegated_at = datetime.now()
        self.active = True

    def revoke(self):
        """Revoke this delegation."""
        self.active = False


class RankedChoiceCounter:
    """
    Implements instant runoff voting for ranked choice.

    Algorithm:
    1. Count first-choice votes
    2. If candidate has >50%, they win
    3. Else, eliminate candidate with fewest votes
    4. Redistribute eliminated candidate's votes to next choices
    5. Repeat until winner found
    """

    def __init__(self, votes: List[RankedVote], options: List[str]):
        self.votes = votes
        self.options = set(options)
        self.eliminated = set()
        self.rounds = []

    def count_votes(self) -> Dict:
        """
        Run instant runoff algorithm.

        Returns:
            Dictionary with winner, rounds, and final tally
        """
        round_num = 1

        while True:
            # Count current round
            tallies = self._count_current_preferences()
            total_votes = sum(tallies.values())

            # Record round
            self.rounds.append({
                'round': round_num,
                'tallies': tallies.copy(),
                'eliminated': list(self.eliminated)
            })

            # Check for majority winner
            for option, count in tallies.items():
                if total_votes > 0 and count / total_votes > 0.5:
                    return {
                        'winner': option,
                        'rounds': self.rounds,
                        'final_tally': tallies,
                        'method': 'ranked_choice'
                    }

            # No majority - eliminate lowest
            if len(tallies) == 1:
                # Only one option left
                winner = list(tallies.keys())[0]
                return {
                    'winner': winner,
                    'rounds': self.rounds,
                    'final_tally': tallies,
                    'method': 'ranked_choice'
                }

            # Eliminate candidate with fewest votes
            min_votes = min(tallies.values())
            to_eliminate = [opt for opt, count in tallies.items() if count == min_votes]

            # If tie for last place, eliminate oldest
            if len(to_eliminate) > 1:
                to_eliminate = [to_eliminate[0]]  # Simple tiebreak

            self.eliminated.update(to_eliminate)

            # Remove eliminated from all ballots
            for vote in self.votes:
                vote.remove_eliminated(self.eliminated)

            round_num += 1

            # Safety: prevent infinite loops
            if round_num > 100:
                logger.error("Ranked choice voting exceeded 100 rounds")
                return {
                    'winner': None,
                    'error': 'No winner after 100 rounds',
                    'rounds': self.rounds
                }

    def _count_current_preferences(self) -> Dict[str, int]:
        """Count first-choice preferences (after eliminations)."""
        tallies = {opt: 0 for opt in self.options if opt not in self.eliminated}

        for vote in self.votes:
            first_choice = vote.get_choice_at_rank(0)
            if first_choice and first_choice in tallies:
                tallies[first_choice] += 1

        return tallies


class LiquidDemocracyManager:
    """
    Manages vote delegation in liquid democracy.

    Features:
    - Transitive delegation (A → B → C chains)
    - Cycle detection (prevent A → B → A)
    - Scope-based delegation (all votes vs specific types)
    - Vote weight calculation (count delegated votes)
    """

    def __init__(self):
        self.delegations: Dict[str, DelegatedVote] = {}  # delegator → DelegatedVote

    def delegate_vote(self, delegator: str, delegate_to: str, scope: str = "all"):
        """
        Delegate voting power to another user.

        Args:
            delegator: User delegating their vote
            delegate_to: User receiving voting power
            scope: Scope of delegation ("all" or specific proposal type)
        """
        # Check for cycles
        if self._creates_cycle(delegator, delegate_to):
            raise ValueError(f"Delegation would create cycle: {delegator} → {delegate_to}")

        # Create delegation
        delegation = DelegatedVote(delegator, delegate_to, scope)
        self.delegations[delegator] = delegation

        logger.info(f"Vote delegated: {delegator} → {delegate_to} (scope: {scope})")

    def revoke_delegation(self, delegator: str):
        """Revoke vote delegation."""
        if delegator in self.delegations:
            self.delegations[delegator].revoke()
            del self.delegations[delegator]
            logger.info(f"Delegation revoked: {delegator}")

    def get_vote_weight(self, user_id: str, scope: str = "all") -> int:
        """
        Calculate vote weight (1 + delegated votes).

        Args:
            user_id: User ID
            scope: Scope to calculate weight for

        Returns:
            Total vote weight
        """
        # Start with own vote
        weight = 1

        # Add delegated votes
        for delegator, delegation in self.delegations.items():
            if delegation.active and delegation.delegate_to == user_id:
                # Check scope
                if delegation.scope == "all" or delegation.scope == scope:
                    # Recursively count delegator's weight
                    weight += self.get_vote_weight(delegator, scope)

        return weight

    def resolve_delegation_chain(self, user_id: str, scope: str = "all") -> Optional[str]:
        """
        Resolve delegation chain to final voter.

        Args:
            user_id: Starting user
            scope: Scope to resolve for

        Returns:
            Final user who will cast vote (or None if user votes directly)
        """
        visited = set()
        current = user_id

        while current in self.delegations:
            delegation = self.delegations[current]

            if not delegation.active:
                break

            # Check scope
            if delegation.scope != "all" and delegation.scope != scope:
                break

            # Check for cycles (shouldn't happen due to validation)
            if current in visited:
                logger.error(f"Cycle detected in delegation chain: {visited}")
                break

            visited.add(current)
            current = delegation.delegate_to

        return current if current != user_id else None

    def _creates_cycle(self, delegator: str, delegate_to: str) -> bool:
        """
        Check if delegation would create a cycle.

        Args:
            delegator: User delegating
            delegate_to: User receiving delegation

        Returns:
            True if cycle detected
        """
        visited = set([delegator])
        current = delegate_to

        while current in self.delegations:
            if current in visited:
                return True

            visited.add(current)
            current = self.delegations[current].delegate_to

        return False


class QuadraticVotingManager:
    """
    Implements quadratic voting to prevent plutocracy.

    Each user gets N voice credits.
    Cost of k votes = k^2 credits.
    This makes buying votes expensive (diminishing returns).
    """

    def __init__(self, credits_per_user: int = 100):
        self.credits_per_user = credits_per_user
        self.user_credits: Dict[str, int] = {}  # user_id → remaining credits
        self.user_votes: Dict[str, Dict[str, int]] = {}  # user_id → {option_id → vote_count}

    def allocate_votes(self, user_id: str, option_id: str, votes: int):
        """
        Allocate votes to an option using voice credits.

        Args:
            user_id: User ID
            option_id: Option to vote for
            votes: Number of votes to allocate

        Raises:
            ValueError if insufficient credits
        """
        # Initialize user if first vote
        if user_id not in self.user_credits:
            self.user_credits[user_id] = self.credits_per_user
            self.user_votes[user_id] = {}

        # Calculate cost (quadratic)
        cost = votes ** 2

        # Check credits
        if cost > self.user_credits[user_id]:
            raise ValueError(
                f"Insufficient credits: need {cost}, have {self.user_credits[user_id]}"
            )

        # Deduct credits
        self.user_credits[user_id] -= cost

        # Record votes
        if option_id not in self.user_votes[user_id]:
            self.user_votes[user_id][option_id] = 0
        self.user_votes[user_id][option_id] += votes

        logger.info(f"Quadratic vote: {user_id} allocated {votes} votes to {option_id} (cost: {cost})")

    def get_total_votes(self, option_id: str) -> int:
        """Get total votes for an option."""
        total = 0
        for user_votes in self.user_votes.values():
            total += user_votes.get(option_id, 0)
        return total

    def get_user_remaining_credits(self, user_id: str) -> int:
        """Get user's remaining voice credits."""
        return self.user_credits.get(user_id, self.credits_per_user)


class ApprovalVotingCounter:
    """
    Implements approval voting.

    Each user can vote for multiple options (approve/disapprove).
    Option with most approvals wins.
    """

    def __init__(self):
        self.approvals: Dict[str, Set[str]] = {}  # option_id → set of user_ids

    def approve(self, user_id: str, option_ids: List[str]):
        """
        Approve multiple options.

        Args:
            user_id: User ID
            option_ids: List of option IDs to approve
        """
        for option_id in option_ids:
            if option_id not in self.approvals:
                self.approvals[option_id] = set()
            self.approvals[option_id].add(user_id)

    def get_results(self) -> Dict:
        """
        Get approval voting results.

        Returns:
            Dictionary with winner and tallies
        """
        tallies = {opt: len(voters) for opt, voters in self.approvals.items()}

        if not tallies:
            return {'winner': None, 'tallies': {}}

        winner = max(tallies, key=tallies.get)

        return {
            'winner': winner,
            'tallies': tallies,
            'method': 'approval'
        }


class ScoreVotingCounter:
    """
    Implements score voting (range voting).

    Each user rates options on scale (e.g., 0-10).
    Option with highest average score wins.
    """

    def __init__(self, min_score: int = 0, max_score: int = 10):
        self.min_score = min_score
        self.max_score = max_score
        self.scores: Dict[str, List[int]] = {}  # option_id → list of scores

    def submit_scores(self, user_id: str, scores: Dict[str, int]):
        """
        Submit scores for options.

        Args:
            user_id: User ID
            scores: Dictionary of option_id → score
        """
        for option_id, score in scores.items():
            # Validate score
            if not (self.min_score <= score <= self.max_score):
                raise ValueError(
                    f"Score {score} out of range [{self.min_score}, {self.max_score}]"
                )

            if option_id not in self.scores:
                self.scores[option_id] = []
            self.scores[option_id].append(score)

    def get_results(self) -> Dict:
        """
        Get score voting results.

        Returns:
            Dictionary with winner and average scores
        """
        averages = {}
        for option_id, scores_list in self.scores.items():
            if scores_list:
                averages[option_id] = sum(scores_list) / len(scores_list)

        if not averages:
            return {'winner': None, 'averages': {}}

        winner = max(averages, key=averages.get)

        return {
            'winner': winner,
            'averages': averages,
            'method': 'score',
            'score_range': [self.min_score, self.max_score]
        }
