#!/usr/bin/env python3
"""
Aggregation Processor for Promptly Chains

Aggregate results from parallel branches with statistical methods,
voting/consensus mechanisms, outlier detection, and confidence scoring.
"""

import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from collections import Counter, defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from promptly.plugins.base import BaseChainStepProcessor


class AggregationStrategy(Enum):
    """Aggregation strategies"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS = "consensus"
    FIRST = "first"
    BEST = "best"
    CONCAT = "concat"
    MERGE = "merge"


class VotingMethod(Enum):
    """Voting methods"""
    MAJORITY = "majority"
    PLURALITY = "plurality"
    RANKED_CHOICE = "ranked_choice"
    APPROVAL = "approval"
    WEIGHTED = "weighted"


class OutlierDetection(Enum):
    """Outlier detection methods"""
    IQR = "iqr"  # Interquartile range
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"


class AggregationProcessor(BaseChainStepProcessor):
    """
    Aggregate results from parallel execution branches.

    Configuration format:
    {
        "type": "aggregation",
        "strategy": "voting",  # mean, median, mode, voting, consensus, etc.
        "source_field": "parallel_results",
        "target_field": "aggregated_result",
        "voting": {
            "method": "majority",  # majority, plurality, ranked_choice
            "threshold": 0.5,  # for consensus
            "weights": {...}  # for weighted voting
        },
        "statistical": {
            "detect_outliers": true,
            "outlier_method": "iqr",
            "remove_outliers": true
        },
        "confidence": {
            "enabled": true,
            "method": "agreement",  # agreement, variance, entropy
            "threshold": 0.7
        },
        "merge_strategy": {
            "conflicts": "prefer_first",  # prefer_first, prefer_last, merge_all
            "nested": true
        },
        "weights": {
            "model_a": 0.5,
            "model_b": 0.3,
            "model_c": 0.2
        },
        "quality_filter": {
            "enabled": true,
            "min_score": 0.5,
            "field": "quality_score"
        }
    }

    Examples:
    - Aggregate multiple LLM responses
    - Consensus from parallel agents
    - Statistical analysis of results
    - Ensemble methods
    """

    def __init__(self):
        super().__init__(
            name="aggregation",
            description="Aggregate results from parallel branches with voting and consensus"
        )
        self._custom_aggregators: Dict[str, Callable] = {}
        self._quality_scorers: Dict[str, Callable] = {}

    def register_aggregator(self, name: str, aggregator: Callable) -> None:
        """Register custom aggregation function"""
        self._custom_aggregators[name] = aggregator

    def register_quality_scorer(self, name: str, scorer: Callable) -> None:
        """Register custom quality scoring function"""
        self._quality_scorers[name] = scorer

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate parallel results.

        Args:
            step_input: Input data with parallel results
            step_config: Aggregation configuration

        Returns:
            Aggregated result with metadata
        """
        strategy = step_config.get("strategy", "voting")
        source_field = step_config.get("source_field", "parallel_results")
        target_field = step_config.get("target_field", "aggregated_result")

        # Get results to aggregate
        results = self._get_field(step_input, source_field)

        if not results:
            return {
                **step_input,
                "aggregation_error": "No results to aggregate"
            }

        # Apply quality filter if configured
        quality_config = step_config.get("quality_filter", {})
        if quality_config.get("enabled", False):
            results = self._filter_by_quality(results, quality_config)

        # Detect and handle outliers
        statistical_config = step_config.get("statistical", {})
        if statistical_config.get("detect_outliers", False):
            results, outliers = self._detect_outliers(results, statistical_config)
        else:
            outliers = []

        # Aggregate based on strategy
        if strategy in self._custom_aggregators:
            aggregated = self._custom_aggregators[strategy](results, step_config)
        elif strategy == AggregationStrategy.MEAN.value:
            aggregated = self._aggregate_mean(results)
        elif strategy == AggregationStrategy.MEDIAN.value:
            aggregated = self._aggregate_median(results)
        elif strategy == AggregationStrategy.MODE.value:
            aggregated = self._aggregate_mode(results)
        elif strategy == AggregationStrategy.MAX.value:
            aggregated = self._aggregate_max(results)
        elif strategy == AggregationStrategy.MIN.value:
            aggregated = self._aggregate_min(results)
        elif strategy == AggregationStrategy.SUM.value:
            aggregated = self._aggregate_sum(results)
        elif strategy == AggregationStrategy.VOTING.value:
            aggregated = self._aggregate_voting(results, step_config)
        elif strategy == AggregationStrategy.WEIGHTED_VOTING.value:
            aggregated = self._aggregate_weighted_voting(results, step_config)
        elif strategy == AggregationStrategy.CONSENSUS.value:
            aggregated = self._aggregate_consensus(results, step_config)
        elif strategy == AggregationStrategy.FIRST.value:
            aggregated = results[0] if results else None
        elif strategy == AggregationStrategy.BEST.value:
            aggregated = self._aggregate_best(results, step_config)
        elif strategy == AggregationStrategy.CONCAT.value:
            aggregated = self._aggregate_concat(results, step_config)
        elif strategy == AggregationStrategy.MERGE.value:
            aggregated = self._aggregate_merge(results, step_config)
        else:
            aggregated = results[0] if results else None

        # Calculate confidence score
        confidence_config = step_config.get("confidence", {})
        confidence_score = None
        if confidence_config.get("enabled", False):
            confidence_score = self._calculate_confidence(results, aggregated, confidence_config)

        # Build result
        result = {
            **step_input,
            "aggregation_metadata": {
                "strategy": strategy,
                "input_count": len(results) + len(outliers),
                "aggregated_count": len(results),
                "outliers_count": len(outliers),
                "outliers": outliers,
                "confidence_score": confidence_score
            }
        }

        self._set_field(result, target_field, aggregated)

        return result

    def _aggregate_mean(self, results: List[Any]) -> float:
        """Calculate mean of numeric results"""
        numeric_results = [float(r) for r in results if self._is_numeric(r)]

        if not numeric_results:
            return None

        return statistics.mean(numeric_results)

    def _aggregate_median(self, results: List[Any]) -> float:
        """Calculate median of numeric results"""
        numeric_results = [float(r) for r in results if self._is_numeric(r)]

        if not numeric_results:
            return None

        return statistics.median(numeric_results)

    def _aggregate_mode(self, results: List[Any]) -> Any:
        """Calculate mode (most common value)"""
        if not results:
            return None

        # Convert to hashable types
        hashable_results = []
        for r in results:
            if isinstance(r, (list, dict)):
                hashable_results.append(str(r))
            else:
                hashable_results.append(r)

        try:
            return statistics.mode(hashable_results)
        except statistics.StatisticsError:
            # No unique mode, return most common
            counter = Counter(hashable_results)
            return counter.most_common(1)[0][0] if counter else None

    def _aggregate_max(self, results: List[Any]) -> Any:
        """Get maximum value"""
        numeric_results = [float(r) for r in results if self._is_numeric(r)]
        return max(numeric_results) if numeric_results else None

    def _aggregate_min(self, results: List[Any]) -> Any:
        """Get minimum value"""
        numeric_results = [float(r) for r in results if self._is_numeric(r)]
        return min(numeric_results) if numeric_results else None

    def _aggregate_sum(self, results: List[Any]) -> float:
        """Calculate sum of numeric results"""
        numeric_results = [float(r) for r in results if self._is_numeric(r)]
        return sum(numeric_results) if numeric_results else None

    def _aggregate_voting(self, results: List[Any], config: Dict[str, Any]) -> Any:
        """Aggregate using voting"""
        voting_config = config.get("voting", {})
        method = voting_config.get("method", "majority")

        if method == VotingMethod.MAJORITY.value:
            return self._voting_majority(results, voting_config)
        elif method == VotingMethod.PLURALITY.value:
            return self._voting_plurality(results)
        elif method == VotingMethod.RANKED_CHOICE.value:
            return self._voting_ranked_choice(results, voting_config)
        else:
            return self._voting_plurality(results)

    def _voting_majority(self, results: List[Any], config: Dict[str, Any]) -> Any:
        """Majority voting (>50%)"""
        threshold = config.get("threshold", 0.5)

        # Count votes
        votes = Counter(str(r) for r in results)
        total_votes = len(results)

        # Find majority
        for value, count in votes.most_common():
            if count / total_votes > threshold:
                return value

        # No majority found
        return None

    def _voting_plurality(self, results: List[Any]) -> Any:
        """Plurality voting (most votes)"""
        votes = Counter(str(r) for r in results)
        most_common = votes.most_common(1)
        return most_common[0][0] if most_common else None

    def _voting_ranked_choice(self, results: List[Any], config: Dict[str, Any]) -> Any:
        """Ranked choice voting (instant runoff)"""
        # Simplified implementation
        # In real RCV, results would be ranked lists
        return self._voting_plurality(results)

    def _aggregate_weighted_voting(self, results: List[Any], config: Dict[str, Any]) -> Any:
        """Weighted voting based on source weights"""
        weights = config.get("weights", {})

        # If results are dictionaries with source info
        weighted_votes = defaultdict(float)

        for i, result in enumerate(results):
            if isinstance(result, dict):
                source = result.get("source", f"source_{i}")
                value = result.get("value", result)
                weight = weights.get(source, 1.0)
            else:
                source = f"source_{i}"
                value = result
                weight = weights.get(source, 1.0)

            weighted_votes[str(value)] += weight

        # Get highest weighted value
        if weighted_votes:
            return max(weighted_votes.items(), key=lambda x: x[1])[0]

        return None

    def _aggregate_consensus(self, results: List[Any], config: Dict[str, Any]) -> Optional[Any]:
        """Require consensus (agreement) among results"""
        voting_config = config.get("voting", {})
        threshold = voting_config.get("threshold", 0.8)

        # Count agreement
        votes = Counter(str(r) for r in results)
        total_votes = len(results)

        # Check for consensus
        for value, count in votes.most_common():
            if count / total_votes >= threshold:
                return value

        # No consensus
        return None

    def _aggregate_best(self, results: List[Any], config: Dict[str, Any]) -> Any:
        """Select best result based on quality score"""
        quality_field = config.get("quality_field", "score")

        scored_results = []

        for result in results:
            if isinstance(result, dict):
                score = result.get(quality_field, 0.0)
            else:
                # Use custom scorer if available
                scorer_name = config.get("quality_scorer")
                if scorer_name and scorer_name in self._quality_scorers:
                    score = self._quality_scorers[scorer_name](result)
                else:
                    score = 0.0

            scored_results.append((result, score))

        if scored_results:
            return max(scored_results, key=lambda x: x[1])[0]

        return None

    def _aggregate_concat(self, results: List[Any], config: Dict[str, Any]) -> str:
        """Concatenate results"""
        separator = config.get("separator", "\n")

        # Convert to strings and join
        string_results = [str(r) for r in results]
        return separator.join(string_results)

    def _aggregate_merge(self, results: List[Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge dictionary results"""
        merge_strategy = config.get("merge_strategy", {})
        conflict_resolution = merge_strategy.get("conflicts", "prefer_first")
        merge_nested = merge_strategy.get("nested", True)

        merged = {}

        for result in results:
            if not isinstance(result, dict):
                continue

            for key, value in result.items():
                if key not in merged:
                    merged[key] = value
                else:
                    # Handle conflict
                    if conflict_resolution == "prefer_first":
                        pass  # Keep existing
                    elif conflict_resolution == "prefer_last":
                        merged[key] = value
                    elif conflict_resolution == "merge_all":
                        # Merge as list
                        if not isinstance(merged[key], list):
                            merged[key] = [merged[key]]
                        merged[key].append(value)
                    elif conflict_resolution == "merge_nested" and merge_nested:
                        # Recursively merge dicts
                        if isinstance(merged[key], dict) and isinstance(value, dict):
                            merged[key] = self._merge_dicts(merged[key], value)

        return merged

    def _merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries"""
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def _filter_by_quality(self, results: List[Any], quality_config: Dict[str, Any]) -> List[Any]:
        """Filter results by quality score"""
        min_score = quality_config.get("min_score", 0.5)
        score_field = quality_config.get("field", "quality_score")

        filtered = []

        for result in results:
            if isinstance(result, dict):
                score = result.get(score_field, 0.0)
            else:
                score = 1.0  # Default score

            if score >= min_score:
                filtered.append(result)

        return filtered

    def _detect_outliers(
        self,
        results: List[Any],
        config: Dict[str, Any]
    ) -> Tuple[List[Any], List[Any]]:
        """Detect outliers in results"""
        method = config.get("outlier_method", "iqr")
        remove_outliers = config.get("remove_outliers", True)

        # Extract numeric values
        numeric_results = []
        non_numeric_results = []

        for r in results:
            if self._is_numeric(r):
                numeric_results.append((r, float(r)))
            else:
                non_numeric_results.append(r)

        if len(numeric_results) < 4:
            # Not enough data for outlier detection
            return results, []

        # Detect outliers
        if method == OutlierDetection.IQR.value:
            outliers, inliers = self._detect_outliers_iqr(numeric_results)
        elif method == OutlierDetection.Z_SCORE.value:
            outliers, inliers = self._detect_outliers_zscore(numeric_results)
        else:
            outliers, inliers = [], numeric_results

        # Combine with non-numeric results
        if remove_outliers:
            remaining = [r[0] for r in inliers] + non_numeric_results
            outlier_values = [r[0] for r in outliers]
        else:
            remaining = results
            outlier_values = [r[0] for r in outliers]

        return remaining, outlier_values

    def _detect_outliers_iqr(
        self,
        results: List[Tuple[Any, float]]
    ) -> Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]:
        """Detect outliers using IQR method"""
        values = [r[1] for r in results]

        # Calculate Q1, Q3, IQR
        q1 = statistics.quantiles(values, n=4)[0]
        q3 = statistics.quantiles(values, n=4)[2]
        iqr = q3 - q1

        # Calculate bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Separate outliers and inliers
        outliers = [r for r in results if r[1] < lower_bound or r[1] > upper_bound]
        inliers = [r for r in results if lower_bound <= r[1] <= upper_bound]

        return outliers, inliers

    def _detect_outliers_zscore(
        self,
        results: List[Tuple[Any, float]],
        threshold: float = 3.0
    ) -> Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]:
        """Detect outliers using Z-score method"""
        values = [r[1] for r in results]

        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        if stdev == 0:
            return [], results

        # Calculate Z-scores
        outliers = []
        inliers = []

        for r in results:
            z_score = abs((r[1] - mean) / stdev)
            if z_score > threshold:
                outliers.append(r)
            else:
                inliers.append(r)

        return outliers, inliers

    def _calculate_confidence(
        self,
        results: List[Any],
        aggregated: Any,
        config: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for aggregated result"""
        method = config.get("method", "agreement")

        if method == "agreement":
            return self._confidence_agreement(results, aggregated)
        elif method == "variance":
            return self._confidence_variance(results)
        elif method == "entropy":
            return self._confidence_entropy(results)
        else:
            return self._confidence_agreement(results, aggregated)

    def _confidence_agreement(self, results: List[Any], aggregated: Any) -> float:
        """Confidence based on agreement with aggregated result"""
        if not results:
            return 0.0

        # Count how many results match aggregated
        matches = sum(1 for r in results if str(r) == str(aggregated))
        return matches / len(results)

    def _confidence_variance(self, results: List[Any]) -> float:
        """Confidence based on variance (lower variance = higher confidence)"""
        numeric_results = [float(r) for r in results if self._is_numeric(r)]

        if len(numeric_results) < 2:
            return 1.0

        variance = statistics.variance(numeric_results)
        # Normalize to 0-1 (higher variance = lower confidence)
        # This is a simple heuristic
        return 1.0 / (1.0 + variance)

    def _confidence_entropy(self, results: List[Any]) -> float:
        """Confidence based on entropy (lower entropy = higher confidence)"""
        if not results:
            return 0.0

        # Count occurrences
        counts = Counter(str(r) for r in results)
        total = len(results)

        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p ** 0.5)  # Simplified entropy

        # Normalize to 0-1 (lower entropy = higher confidence)
        max_entropy = 1.0
        return 1.0 - min(entropy / max_entropy, 1.0)

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _get_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def _set_field(self, data: Dict[str, Any], field_path: str, value: Any) -> None:
        """Set nested field value"""
        parts = field_path.split(".")
        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value


__all__ = [
    "AggregationProcessor",
    "AggregationStrategy",
    "VotingMethod",
    "OutlierDetection"
]
