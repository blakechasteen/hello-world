"""
PromptlyBridge - Integration between HoloLoom and Promptly
===========================================================

Bridges HoloLoom's agentic chat with Promptly's prompt management system.
Provides import/export of prompts and template synchronization.
"""

from pathlib import Path
from typing import List, Dict, Optional
import sys
import logging

logger = logging.getLogger(__name__)

# Add Promptly to path if installed separately
promptly_path = Path(__file__).parent.parent.parent / "Promptly" / "promptly"
if promptly_path.exists():
    sys.path.insert(0, str(promptly_path.parent))

# Try to import Promptly, but make it truly optional
PROMPTLY_AVAILABLE = False
try:
    from promptly.promptly import Promptly
    from promptly.loop_composition import LoopEngine
    PROMPTLY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Promptly not available: {e}")
    # Create stub classes
    class Promptly:
        pass
    class LoopEngine:
        pass
except Exception as e:
    logger.warning(f"Promptly import failed: {e}")
    # Create stub classes
    class Promptly:
        pass
    class LoopEngine:
        pass


class PromptlyBridge:
    """Bridge between HoloLoom and Promptly"""

    def __init__(self, promptly_dir: str = ".promptly"):
        if not PROMPTLY_AVAILABLE:
            raise ImportError("Promptly framework not installed")

        self.promptly = Promptly(promptly_dir=promptly_dir)
        try:
            self.loop_engine = LoopEngine(self.promptly)
        except Exception:
            self.loop_engine = None
            logger.warning("LoopEngine not available (Promptly version may not support it)")

    def list_prompts(self) -> List[Dict]:
        """List all prompts"""
        prompts = self.promptly.list_prompts()
        return [
            {
                'name': p.name,
                'version': p.version,
                'description': p.description,
                'created_at': p.created_at,
                'tags': p.tags if hasattr(p, 'tags') else []
            }
            for p in prompts
        ]

    def get_prompt(self, name: str, version: Optional[str] = None) -> Dict:
        """Get a specific prompt"""
        prompt = self.promptly.get_prompt(name, version)
        return {
            'name': prompt.name,
            'version': prompt.version,
            'content': prompt.content,
            'description': prompt.description,
            'tags': prompt.tags if hasattr(prompt, 'tags') else [],
            'created_at': prompt.created_at
        }

    def save_prompt(self, name: str, content: str, description: str = "", tags: List[str] = None):
        """Save a new prompt"""
        self.promptly.save_prompt(
            name=name,
            content=content,
            description=description,
            tags=tags or []
        )
        logger.info(f"Saved prompt: {name}")

    def execute_prompt(self, name: str, variables: Dict = None, version: Optional[str] = None) -> str:
        """Execute a prompt"""
        result = self.promptly.execute(
            prompt_name=name,
            variables=variables or {},
            version=version
        )
        return result

    def run_loop(self, prompt_name: str, iterations: int = 3, variables: Dict = None) -> List[str]:
        """Run a prompt in a loop"""
        if not self.loop_engine:
            raise RuntimeError("LoopEngine not available")

        results = self.loop_engine.run_loop(
            prompt_name=prompt_name,
            iterations=iterations,
            initial_variables=variables or {}
        )
        return results

    def import_from_conversation(self, messages: List[Dict]) -> str:
        """Convert conversation messages to a Promptly prompt"""
        # Extract user messages as the prompt content
        user_messages = [msg for msg in messages if msg.get('role') == 'user']

        if not user_messages:
            return ""

        # Combine user messages into a prompt template
        prompt_content = "\n\n".join([msg['content'] for msg in user_messages])
        return prompt_content

    def export_to_conversation(self, prompt_name: str) -> List[Dict]:
        """Convert a Promptly prompt to conversation format"""
        prompt = self.get_prompt(prompt_name)

        # Create a conversation with the prompt as the user message
        messages = [
            {
                'role': 'user',
                'content': prompt['content']
            }
        ]

        return messages
