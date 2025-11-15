#!/usr/bin/env python3
"""
Base protocol definitions for Promptly plugins
"""

from typing import Protocol, Dict, List, Optional, Any, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class EvaluatorPlugin(Protocol):
    """Protocol for custom evaluator plugins"""

    @property
    def name(self) -> str:
        """Unique name for this evaluator"""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this evaluator does"""
        ...

    @abstractmethod
    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate the actual output against expected output

        Args:
            actual: The actual output from the model
            expected: The expected output or reference
            context: Optional context dictionary with additional information

        Returns:
            float: Score between 0.0 and 1.0
        """
        ...

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get detailed metrics for the evaluation (optional)

        Args:
            actual: The actual output from the model
            expected: The expected output or reference
            context: Optional context dictionary

        Returns:
            Dict with detailed metrics
        """
        score = self.evaluate(actual, expected, context)
        return {"score": score}


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for custom storage backends"""

    @property
    def name(self) -> str:
        """Unique name for this storage backend"""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of this storage backend"""
        ...

    @abstractmethod
    def init_storage(self, storage_path: str) -> None:
        """
        Initialize the storage backend

        Args:
            storage_path: Path to storage location
        """
        ...

    @abstractmethod
    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt to storage

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            str: Commit hash or identifier
        """
        ...

    @abstractmethod
    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a prompt from storage

        Args:
            name: Prompt name
            branch: Branch name
            version: Optional specific version
            commit_hash: Optional specific commit hash

        Returns:
            Dict with prompt data or None if not found
        """
        ...

    @abstractmethod
    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        List all prompts on a branch

        Args:
            branch: Branch name

        Returns:
            List of prompt metadata dictionaries
        """
        ...

    @abstractmethod
    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """
        Create a new branch

        Args:
            branch_name: Name of the new branch
            from_branch: Source branch to branch from
        """
        ...

    @abstractmethod
    def get_current_branch(self) -> str:
        """
        Get the current branch name

        Returns:
            str: Current branch name
        """
        ...

    @abstractmethod
    def set_current_branch(self, branch_name: str) -> None:
        """
        Set the current branch

        Args:
            branch_name: Branch to switch to
        """
        ...

    @abstractmethod
    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get commit history

        Args:
            name: Optional prompt name to filter by
            branch: Branch name
            limit: Maximum number of commits to return

        Returns:
            List of commit dictionaries
        """
        ...

    @abstractmethod
    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """
        Save evaluation results

        Args:
            eval_data: Dictionary containing evaluation data
        """
        ...

    @abstractmethod
    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """
        Save a prompt chain

        Args:
            chain_data: Dictionary containing chain data
        """
        ...

    @abstractmethod
    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt chain

        Args:
            name: Chain name

        Returns:
            Dict with chain data or None if not found
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close/cleanup the storage backend"""
        ...


@runtime_checkable
class ChainStepProcessor(Protocol):
    """Protocol for custom chain step processors"""

    @property
    def name(self) -> str:
        """Unique name for this processor"""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of this processor"""
        ...

    @abstractmethod
    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chain step

        Args:
            step_input: Input data for this step
            step_config: Configuration for this step

        Returns:
            Dict: Output data to pass to next step
        """
        ...

    def pre_process(self, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional pre-processing hook

        Args:
            step_input: Input data

        Returns:
            Modified input data
        """
        return step_input

    def post_process(self, step_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional post-processing hook

        Args:
            step_output: Output data

        Returns:
            Modified output data
        """
        return step_output


class BaseEvaluator:
    """Base class for evaluators (alternative to Protocol)"""

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        raise NotImplementedError("Subclasses must implement evaluate()")

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        score = self.evaluate(actual, expected, context)
        return {"score": score}


class BaseStorageBackend:
    """Base class for storage backends (alternative to Protocol)"""

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def init_storage(self, storage_path: str) -> None:
        raise NotImplementedError("Subclasses must implement init_storage()")

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        raise NotImplementedError("Subclasses must implement save_prompt()")

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement get_prompt()")

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement list_prompts()")

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        raise NotImplementedError("Subclasses must implement create_branch()")

    def get_current_branch(self) -> str:
        raise NotImplementedError("Subclasses must implement get_current_branch()")

    def set_current_branch(self, branch_name: str) -> None:
        raise NotImplementedError("Subclasses must implement set_current_branch()")

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement get_commit_history()")

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        raise NotImplementedError("Subclasses must implement save_evaluation()")

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        raise NotImplementedError("Subclasses must implement save_chain()")

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement get_chain()")

    def close(self) -> None:
        pass


class BaseChainStepProcessor:
    """Base class for chain step processors (alternative to Protocol)"""

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement process()")

    def pre_process(self, step_input: Dict[str, Any]) -> Dict[str, Any]:
        return step_input

    def post_process(self, step_output: Dict[str, Any]) -> Dict[str, Any]:
        return step_output
