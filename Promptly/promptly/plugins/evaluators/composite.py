#!/usr/bin/env python3
"""
Composite Evaluator Plugins

Combines multiple evaluators using various strategies:
- Weighted average (combine scores with weights)
- Voting ensemble (majority/unanimous voting)
- Cascading evaluation (short-circuit on failure)
- Min/Max aggregation
"""

from typing import Dict, Any, Optional, List, Callable
from ..base import BaseEvaluator, EvaluatorPlugin


class WeightedAverageEvaluator(BaseEvaluator):
    """
    Weighted average composite evaluator.

    Combines multiple evaluators using weighted averaging.
    Useful for balancing different quality dimensions.
    """

    def __init__(self, evaluators: List[tuple], normalize_weights: bool = True):
        """
        Initialize weighted average evaluator

        Args:
            evaluators: List of (evaluator, weight) tuples
                       e.g., [(bleu_eval, 0.4), (rouge_eval, 0.3), (semantic_eval, 0.3)]
            normalize_weights: Whether to normalize weights to sum to 1.0
        """
        super().__init__(
            name="weighted_average",
            description="Weighted average of multiple evaluators"
        )

        if not evaluators:
            raise ValueError("Must provide at least one evaluator")

        self.evaluators = []
        self.weights = []

        for item in evaluators:
            if isinstance(item, tuple) and len(item) == 2:
                evaluator, weight = item
                self.evaluators.append(evaluator)
                self.weights.append(float(weight))
            else:
                # Assume equal weight if not specified
                self.evaluators.append(item)
                self.weights.append(1.0)

        # Normalize weights if requested
        if normalize_weights:
            total_weight = sum(self.weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in self.weights]

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using weighted average

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context

        Returns:
            float: Weighted average score
        """
        if not actual:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for evaluator, weight in zip(self.evaluators, self.weights):
            try:
                score = evaluator.evaluate(actual, expected, context)
                weighted_sum += score * weight
                total_weight += weight
            except Exception as e:
                print(f"Warning: Evaluator {evaluator.name} failed ({e}), skipping")
                continue

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed metrics from all evaluators"""
        individual_scores = {}
        individual_metrics = {}

        for evaluator, weight in zip(self.evaluators, self.weights):
            try:
                score = evaluator.evaluate(actual, expected, context)
                metrics = evaluator.get_metrics(actual, expected, context)

                individual_scores[evaluator.name] = {
                    'score': score,
                    'weight': weight,
                    'weighted_score': score * weight
                }
                individual_metrics[evaluator.name] = metrics

            except Exception as e:
                individual_scores[evaluator.name] = {
                    'error': str(e),
                    'weight': weight
                }

        return {
            'score': self.evaluate(actual, expected, context),
            'individual_scores': individual_scores,
            'individual_metrics': individual_metrics,
            'strategy': 'weighted_average'
        }


class VotingEnsembleEvaluator(BaseEvaluator):
    """
    Voting ensemble evaluator.

    Makes binary pass/fail decisions based on evaluator votes.
    Supports majority voting and unanimous voting.
    """

    def __init__(self, evaluators: List[EvaluatorPlugin],
                 threshold: float = 0.5,
                 voting_strategy: str = "majority",
                 min_votes: Optional[int] = None):
        """
        Initialize voting ensemble evaluator

        Args:
            evaluators: List of evaluator instances
            threshold: Score threshold for pass/fail decision (per evaluator)
            voting_strategy: 'majority', 'unanimous', or 'any'
            min_votes: Minimum number of votes required (for majority)
        """
        super().__init__(
            name="voting_ensemble",
            description=f"Voting ensemble ({voting_strategy}) evaluator"
        )

        if not evaluators:
            raise ValueError("Must provide at least one evaluator")

        self.evaluators = evaluators
        self.threshold = threshold
        self.voting_strategy = voting_strategy.lower()
        self.min_votes = min_votes

        if self.voting_strategy not in ['majority', 'unanimous', 'any']:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using voting ensemble

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context

        Returns:
            float: 1.0 if vote passes, 0.0 if fails
        """
        if not actual:
            return 0.0

        votes = []

        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(actual, expected, context)
                # Binary vote: pass if score >= threshold
                vote = 1 if score >= self.threshold else 0
                votes.append(vote)
            except Exception as e:
                print(f"Warning: Evaluator {evaluator.name} failed ({e}), counting as fail vote")
                votes.append(0)

        if not votes:
            return 0.0

        pass_votes = sum(votes)
        total_votes = len(votes)

        # Apply voting strategy
        if self.voting_strategy == "unanimous":
            # All must pass
            result = 1.0 if pass_votes == total_votes else 0.0

        elif self.voting_strategy == "any":
            # At least one must pass
            result = 1.0 if pass_votes > 0 else 0.0

        else:  # majority
            # More than half must pass
            required_votes = self.min_votes if self.min_votes is not None else (total_votes / 2)
            result = 1.0 if pass_votes > required_votes else 0.0

        return result

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed voting metrics"""
        votes = {}
        scores = {}

        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(actual, expected, context)
                vote = 1 if score >= self.threshold else 0

                votes[evaluator.name] = vote
                scores[evaluator.name] = score

            except Exception as e:
                votes[evaluator.name] = 0
                scores[evaluator.name] = None

        pass_votes = sum(votes.values())
        total_votes = len(votes)

        return {
            'score': self.evaluate(actual, expected, context),
            'votes': votes,
            'scores': scores,
            'pass_votes': pass_votes,
            'total_votes': total_votes,
            'pass_rate': pass_votes / total_votes if total_votes > 0 else 0.0,
            'threshold': self.threshold,
            'strategy': self.voting_strategy
        }


class CascadingEvaluator(BaseEvaluator):
    """
    Cascading evaluator.

    Evaluates in sequence, short-circuiting on failure.
    Useful for fast rejection of obviously bad outputs.
    """

    def __init__(self, evaluators: List[tuple],
                 stop_on_failure: bool = True,
                 aggregate: str = "min"):
        """
        Initialize cascading evaluator

        Args:
            evaluators: List of (evaluator, min_threshold) tuples
                       e.g., [(length_eval, 0.5), (toxicity_eval, 0.8), (quality_eval, 0.7)]
            stop_on_failure: Stop evaluation if any evaluator fails threshold
            aggregate: How to aggregate scores ('min', 'max', 'average', 'product')
        """
        super().__init__(
            name="cascading",
            description="Cascading evaluator with short-circuit"
        )

        if not evaluators:
            raise ValueError("Must provide at least one evaluator")

        self.stages = []

        for item in evaluators:
            if isinstance(item, tuple) and len(item) == 2:
                evaluator, threshold = item
                self.stages.append({
                    'evaluator': evaluator,
                    'threshold': float(threshold)
                })
            else:
                # No threshold specified, always pass
                self.stages.append({
                    'evaluator': item,
                    'threshold': 0.0
                })

        self.stop_on_failure = stop_on_failure
        self.aggregate = aggregate.lower()

        if self.aggregate not in ['min', 'max', 'average', 'product']:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using cascading strategy

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context

        Returns:
            float: Aggregated score (or 0.0 if failed before completion)
        """
        if not actual:
            return 0.0

        scores = []

        for stage in self.stages:
            evaluator = stage['evaluator']
            threshold = stage['threshold']

            try:
                score = evaluator.evaluate(actual, expected, context)
                scores.append(score)

                # Check if this stage failed
                if self.stop_on_failure and score < threshold:
                    # Short-circuit: return 0.0 immediately
                    return 0.0

            except Exception as e:
                print(f"Warning: Evaluator {evaluator.name} failed ({e})")
                if self.stop_on_failure:
                    return 0.0
                scores.append(0.0)

        # All stages passed, aggregate scores
        if not scores:
            return 0.0

        if self.aggregate == "min":
            return min(scores)
        elif self.aggregate == "max":
            return max(scores)
        elif self.aggregate == "average":
            return sum(scores) / len(scores)
        else:  # product
            result = 1.0
            for score in scores:
                result *= score
            return result

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed cascading metrics"""
        stage_results = []
        all_scores = []
        failed_stage = None

        for i, stage in enumerate(self.stages):
            evaluator = stage['evaluator']
            threshold = stage['threshold']

            try:
                score = evaluator.evaluate(actual, expected, context)
                passed = score >= threshold

                stage_results.append({
                    'stage': i,
                    'evaluator': evaluator.name,
                    'score': score,
                    'threshold': threshold,
                    'passed': passed
                })

                all_scores.append(score)

                if self.stop_on_failure and not passed:
                    failed_stage = i
                    break

            except Exception as e:
                stage_results.append({
                    'stage': i,
                    'evaluator': evaluator.name,
                    'error': str(e),
                    'threshold': threshold,
                    'passed': False
                })

                if self.stop_on_failure:
                    failed_stage = i
                    break

        return {
            'score': self.evaluate(actual, expected, context),
            'stage_results': stage_results,
            'completed_stages': len(stage_results),
            'total_stages': len(self.stages),
            'failed_stage': failed_stage,
            'all_passed': failed_stage is None,
            'aggregate_method': self.aggregate
        }


class MinMaxEvaluator(BaseEvaluator):
    """
    Min/Max aggregation evaluator.

    Returns minimum or maximum score across evaluators.
    Useful for conservative (min) or optimistic (max) evaluation.
    """

    def __init__(self, evaluators: List[EvaluatorPlugin], mode: str = "min"):
        """
        Initialize min/max evaluator

        Args:
            evaluators: List of evaluator instances
            mode: 'min' or 'max'
        """
        super().__init__(
            name=f"{mode}_aggregate",
            description=f"{mode.upper()} aggregation of multiple evaluators"
        )

        if not evaluators:
            raise ValueError("Must provide at least one evaluator")

        self.evaluators = evaluators
        self.mode = mode.lower()

        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using min/max aggregation

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context

        Returns:
            float: Min or max score
        """
        if not actual:
            return 0.0

        scores = []

        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(actual, expected, context)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Evaluator {evaluator.name} failed ({e}), skipping")
                continue

        if not scores:
            return 0.0

        return min(scores) if self.mode == "min" else max(scores)

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed min/max metrics"""
        scores = {}

        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(actual, expected, context)
                scores[evaluator.name] = score
            except Exception as e:
                scores[evaluator.name] = None

        valid_scores = [s for s in scores.values() if s is not None]

        return {
            'score': self.evaluate(actual, expected, context),
            'scores': scores,
            'min_score': min(valid_scores) if valid_scores else None,
            'max_score': max(valid_scores) if valid_scores else None,
            'avg_score': sum(valid_scores) / len(valid_scores) if valid_scores else None,
            'mode': self.mode
        }


class ConditionalEvaluator(BaseEvaluator):
    """
    Conditional evaluator.

    Selects which evaluator to use based on a condition function.
    Useful for adaptive evaluation strategies.
    """

    def __init__(self, evaluator_map: Dict[str, EvaluatorPlugin],
                 condition_func: Callable[[str, str, Optional[Dict[str, Any]]], str],
                 default_evaluator: Optional[str] = None):
        """
        Initialize conditional evaluator

        Args:
            evaluator_map: Dict mapping condition results to evaluators
                          e.g., {'short': length_eval, 'long': quality_eval}
            condition_func: Function that returns which evaluator to use
                           Signature: (actual, expected, context) -> str
            default_evaluator: Default evaluator key if condition fails
        """
        super().__init__(
            name="conditional",
            description="Conditional evaluator with dynamic selection"
        )

        if not evaluator_map:
            raise ValueError("Must provide at least one evaluator")

        self.evaluator_map = evaluator_map
        self.condition_func = condition_func
        self.default_evaluator = default_evaluator or list(evaluator_map.keys())[0]

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using conditional selection

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context

        Returns:
            float: Score from selected evaluator
        """
        if not actual:
            return 0.0

        # Determine which evaluator to use
        try:
            condition_result = self.condition_func(actual, expected, context)
            evaluator_key = condition_result if condition_result in self.evaluator_map else self.default_evaluator
        except Exception as e:
            print(f"Warning: Condition function failed ({e}), using default evaluator")
            evaluator_key = self.default_evaluator

        evaluator = self.evaluator_map[evaluator_key]

        try:
            return evaluator.evaluate(actual, expected, context)
        except Exception as e:
            print(f"Warning: Selected evaluator {evaluator_key} failed ({e})")
            return 0.0

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed conditional metrics"""
        # Determine which evaluator was selected
        try:
            condition_result = self.condition_func(actual, expected, context)
            evaluator_key = condition_result if condition_result in self.evaluator_map else self.default_evaluator
        except Exception as e:
            evaluator_key = self.default_evaluator
            condition_result = None

        evaluator = self.evaluator_map[evaluator_key]

        try:
            score = evaluator.evaluate(actual, expected, context)
            metrics = evaluator.get_metrics(actual, expected, context)
        except Exception as e:
            score = 0.0
            metrics = {'error': str(e)}

        return {
            'score': score,
            'selected_evaluator': evaluator_key,
            'condition_result': condition_result,
            'evaluator_metrics': metrics,
            'available_evaluators': list(self.evaluator_map.keys())
        }


class ThresholdedEvaluator(BaseEvaluator):
    """
    Thresholded evaluator.

    Applies threshold and returns binary pass/fail or scaled score.
    """

    def __init__(self, base_evaluator: EvaluatorPlugin,
                 threshold: float = 0.5,
                 mode: str = "binary"):
        """
        Initialize thresholded evaluator

        Args:
            base_evaluator: Base evaluator to wrap
            threshold: Score threshold
            mode: 'binary' (0 or 1), 'scale_above' (scale scores above threshold),
                  'scale_below' (scale scores below threshold)
        """
        super().__init__(
            name=f"thresholded_{base_evaluator.name}",
            description=f"Thresholded version of {base_evaluator.name}"
        )

        self.base_evaluator = base_evaluator
        self.threshold = threshold
        self.mode = mode.lower()

        if self.mode not in ['binary', 'scale_above', 'scale_below']:
            raise ValueError(f"Unknown mode: {mode}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate with threshold applied

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context

        Returns:
            float: Thresholded score
        """
        base_score = self.base_evaluator.evaluate(actual, expected, context)

        if self.mode == "binary":
            return 1.0 if base_score >= self.threshold else 0.0

        elif self.mode == "scale_above":
            if base_score < self.threshold:
                return 0.0
            # Scale from threshold to 1.0 -> 0.0 to 1.0
            return (base_score - self.threshold) / (1.0 - self.threshold)

        else:  # scale_below
            if base_score >= self.threshold:
                return 1.0
            # Scale from 0.0 to threshold -> 0.0 to 1.0
            return base_score / self.threshold if self.threshold > 0 else 0.0

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed threshold metrics"""
        base_score = self.base_evaluator.evaluate(actual, expected, context)
        thresholded_score = self.evaluate(actual, expected, context)

        return {
            'score': thresholded_score,
            'base_score': base_score,
            'threshold': self.threshold,
            'mode': self.mode,
            'passed': base_score >= self.threshold
        }
