"""
Persistence layer for tuning state.

Saves/loads learned parameters across sessions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TuningStateManager:
    """
    Manages persistent storage of tuning state.

    Stores:
    - Learned parameter values
    - Thompson Sampling bandit states
    - Tuning history and statistics
    """

    def __init__(self, state_dir: str = "./tuning_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "learned_params.json"

    def save_state(self, coordinator_state: Dict[str, Any]):
        """
        Save complete tuning state to disk.

        Args:
            coordinator_state: State dict from MasterTuningCoordinator
        """
        state = {
            'tuning_version': '1.0.0',
            'last_updated': datetime.now().isoformat(),
            'total_queries_seen': coordinator_state.get('total_queries', 0),
            'total_tuning_cycles': coordinator_state.get('total_cycles', 0),
            'learned_parameters': coordinator_state.get('agent_states', {}),
            'meta_bandit': coordinator_state.get('meta_bandit_state', {}),
            'global_metrics': coordinator_state.get('global_metrics', {}),
        }

        # Write atomically (write to temp, then rename)
        temp_file = self.state_file.with_suffix('.tmp')

        try:
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

            logger.info(f"Saved tuning state: {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save tuning state: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load tuning state from disk.

        Returns:
            State dict if found, None otherwise
        """
        if not self.state_file.exists():
            logger.info("No saved tuning state found, starting fresh")
            return None

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            logger.info(f"Loaded tuning state from {self.state_file}")
            logger.info(f"  Total queries: {state.get('total_queries_seen', 0)}")
            logger.info(f"  Total cycles: {state.get('total_tuning_cycles', 0)}")

            return state

        except Exception as e:
            logger.error(f"Failed to load tuning state: {e}")
            return None

    def save_backup(self):
        """Save backup copy of current state."""
        if not self.state_file.exists():
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.state_dir / f"learned_params_backup_{timestamp}.json"

        try:
            import shutil
            shutil.copy(self.state_file, backup_file)
            logger.info(f"Saved backup: {backup_file}")

        except Exception as e:
            logger.error(f"Failed to save backup: {e}")

    def get_tuning_history(self) -> Dict[str, Any]:
        """Get tuning history from saved state."""
        state = self.load_state()
        if state is None:
            return {}

        return {
            'total_queries': state.get('total_queries_seen', 0),
            'total_cycles': state.get('total_tuning_cycles', 0),
            'agent_success_rates': self._compute_success_rates(state),
            'last_updated': state.get('last_updated'),
        }

    def _compute_success_rates(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Compute success rate for each agent."""
        rates = {}

        agent_states = state.get('learned_parameters', {})
        for agent_name, agent_state in agent_states.items():
            total = agent_state.get('total_tuning_cycles', 0)
            successful = agent_state.get('successful_tunings', 0)

            if total > 0:
                rates[agent_name] = successful / total
            else:
                rates[agent_name] = 0.0

        return rates

    def clear_state(self):
        """Clear saved state (use carefully!)."""
        if self.state_file.exists():
            self.save_backup()
            self.state_file.unlink()
            logger.warning("Cleared tuning state")
