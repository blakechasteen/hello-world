"""
Adaptive Moonshot Query Classifier
===================================
Enhanced MoonshotQueryClassifier with full Phase 3 adaptive learning integration.

Integrates 4 adaptive learning components:
1. PatternMiner: Mines patterns from production logs → improves Tier 1 patterns
2. ContinuousValidator: Hourly validation → detects regressions automatically
3. AdaptiveUpdater: Safe pattern deployment → A/B testing + gradual rollout
4. PerformanceReporter: Daily/weekly reports → Prometheus + Slack/email alerts

Architecture:
- Foreground: Fast classification (inherited from MoonshotQueryClassifier)
- Background: Adaptive learning loop (pattern mining, validation, deployment, reporting)

Author: Claude Code
Date: November 13, 2025
Phase: 3 - Adaptive Learning (Day 11-12)
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .learning import (
    AdaptiveUpdater,
    ContinuousValidator,
    DeploymentStrategy,
    Pattern,
    PatternMiner,
    PatternScore,
    PerformanceReporter,
    create_validation_set,
)
from .query_classifier_moonshot import (
    ClassificationResult,
    MoonshotQueryClassifier,
)


class AdaptiveMoonshotClassifier:
    """
    MoonshotQueryClassifier with integrated adaptive learning.

    Combines fast multi-tier classification with continuous learning:
    - Logs all classifications to JSONL
    - Mines patterns hourly from logs
    - Validates accuracy hourly
    - Deploys high-quality patterns safely
    - Generates daily/weekly reports
    """

    def __init__(
        self,
        enable_semantic_tier: bool = True,
        enable_adaptive_learning: bool = True,
        data_dir: Path | None = None,
        validation_set_path: Path | None = None,
        background_learning: bool = True,
        learning_update_interval: float = 3600.0,  # 1 hour
    ):
        """
        Initialize adaptive classifier.

        Args:
            enable_semantic_tier: Enable Tier 3 semantic embeddings
            enable_adaptive_learning: Enable Phase 3 adaptive learning
            data_dir: Directory for logs, patterns, reports (default: ./data)
            validation_set_path: Path to validation set JSON
            background_learning: Run background learning loop
            learning_update_interval: Background learning interval (seconds)
        """
        # Base classifier
        self.classifier = MoonshotQueryClassifier(
            enable_semantic_tier=enable_semantic_tier,
            enable_adaptive_learning=enable_adaptive_learning
        )

        # Paths
        self.data_dir = data_dir or Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.data_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.classification_log_path = self.logs_dir / "classifications.jsonl"
        self.patterns_path = self.data_dir / "learned_patterns.json"

        # Adaptive learning components
        self.enable_adaptive_learning = enable_adaptive_learning
        self.pattern_miner: PatternMiner | None = None
        self.validator: ContinuousValidator | None = None
        self.updater: AdaptiveUpdater | None = None
        self.reporter: PerformanceReporter | None = None

        if enable_adaptive_learning:
            self._initialize_adaptive_components(validation_set_path)

        # Background learning
        self.background_learning = background_learning
        self.learning_update_interval = learning_update_interval
        self._learning_task: asyncio.Task | None = None
        self._should_stop = False

    def _initialize_adaptive_components(self, validation_set_path: Path | None):
        """Initialize Phase 3 adaptive learning components."""

        # 1. PatternMiner - mines patterns from classification logs
        self.pattern_miner = PatternMiner(
            stats_path=str(self.classification_log_path),
            min_support=10,          # Pattern must appear ≥10 times
            min_precision=0.95,      # Pattern must be 95%+ accurate
            min_recall=0.30,         # Minimum coverage
            max_patterns=50          # Top 50 patterns
        )

        # 2. ContinuousValidator - hourly validation checks
        val_set_path = validation_set_path or (self.data_dir / "validation_set.json")

        # Create default validation set if not exists
        if not val_set_path.exists():
            default_queries = self._create_default_validation_set()
            create_validation_set(default_queries, str(val_set_path))

        self.validator = ContinuousValidator(
            classifier=self.classifier,
            validation_set_path=str(val_set_path),
            regression_threshold=0.02,  # 2% accuracy drop triggers alert
            baseline_accuracy=None       # Will be computed on first validation
        )

        # 3. AdaptiveUpdater - safe pattern deployment
        self.updater = AdaptiveUpdater(
            classifier=self.classifier,
            validator=self.validator,
            strategy=DeploymentStrategy.GRADUAL,  # 7-day gradual rollout
            regression_threshold=0.02
        )

        # 4. PerformanceReporter - daily/weekly reports
        self.reporter = PerformanceReporter(
            validator=self.validator,
            updater=self.updater,
            pattern_miner=self.pattern_miner,
            output_dir=str(self.reports_dir)
        )

    def _create_default_validation_set(self) -> list[tuple[str, str]]:
        """Create default validation set with known examples."""
        return [
            # TRIVIAL queries (20 examples)
            ("hi", "trivial"),
            ("hello", "trivial"),
            ("thanks", "trivial"),
            ("thank you", "trivial"),
            ("ok", "trivial"),
            ("bye", "trivial"),
            ("goodbye", "trivial"),
            ("hey", "trivial"),
            ("yo", "trivial"),
            ("sure", "trivial"),
            ("yes", "trivial"),
            ("no", "trivial"),
            ("cool", "trivial"),
            ("nice", "trivial"),
            ("great", "trivial"),
            ("perfect", "trivial"),
            ("help", "trivial"),
            ("stop", "trivial"),
            ("wait", "trivial"),
            ("continue", "trivial"),

            # SIMPLE queries (20 examples)
            ("what is Python?", "simple"),
            ("what are transformers?", "simple"),
            ("define machine learning", "simple"),
            ("who is Einstein?", "simple"),
            ("when is Christmas?", "simple"),
            ("where is Paris?", "simple"),
            ("what is attention?", "simple"),
            ("what are embeddings?", "simple"),
            ("define neural network", "simple"),
            ("what is backpropagation?", "simple"),
            ("who invented transformers?", "simple"),
            ("when was GPT created?", "simple"),
            ("where did AI start?", "simple"),
            ("what is BERT?", "simple"),
            ("define gradient descent", "simple"),
            ("what is CNN?", "simple"),
            ("what are RNNs?", "simple"),
            ("who created PyTorch?", "simple"),
            ("when was TensorFlow released?", "simple"),
            ("what is reinforcement learning?", "simple"),

            # COMPLEX queries (20 examples)
            ("explain how neural networks work", "complex"),
            ("describe attention mechanism with examples", "complex"),
            ("why do transformers use self-attention?", "complex"),
            ("how does backpropagation work in detail?", "complex"),
            ("explain gradient descent with examples", "complex"),
            ("describe CNN architecture in detail", "complex"),
            ("how do RNNs handle sequences?", "complex"),
            ("explain LSTM and its variants", "complex"),
            ("why is attention better than RNNs?", "complex"),
            ("describe the transformer architecture", "complex"),
            ("how does BERT work?", "complex"),
            ("explain GPT architecture", "complex"),
            ("describe batch normalization in detail", "complex"),
            ("how does dropout prevent overfitting?", "complex"),
            ("explain Adam optimizer mechanism", "complex"),
            ("describe Q-learning algorithm", "complex"),
            ("how does PPO work?", "complex"),
            ("explain policy gradients", "complex"),
            ("describe actor-critic methods", "complex"),
            ("how do GANs work?", "complex"),

            # RESEARCH queries (15 examples)
            ("analyze tradeoffs between Thompson Sampling and epsilon-greedy", "research"),
            ("compare and contrast supervised and unsupervised learning", "research"),
            ("evaluate advantages and disadvantages of neural vs symbolic AI", "research"),
            ("analyze all attention mechanisms comprehensively", "research"),
            ("compare RNNs vs Transformers vs State Space Models", "research"),
            ("evaluate tradeoffs of different RL algorithms", "research"),
            ("analyze pros and cons of different optimizers", "research"),
            ("compare CNNs vs Vision Transformers comprehensively", "research"),
            ("evaluate different normalization techniques", "research"),
            ("analyze tradeoffs between model size and performance", "research"),
            ("compare supervised vs semi-supervised vs self-supervised learning", "research"),
            ("evaluate different attention patterns in transformers", "research"),
            ("analyze exploration vs exploitation tradeoffs", "research"),
            ("compare different tokenization strategies", "research"),
            ("evaluate efficiency vs accuracy tradeoffs in neural architecture search", "research"),
        ]

    def classify(self, query_text: str) -> ClassificationResult:
        """
        Classify query and log to JSONL for adaptive learning.

        Args:
            query_text: Query string

        Returns:
            ClassificationResult with complexity and metadata
        """
        # Classify using base classifier
        result = self.classifier.classify(query_text)

        # Log classification to JSONL (for pattern mining)
        if self.enable_adaptive_learning:
            self._log_classification(query_text, result)

        return result

    def _log_classification(self, query_text: str, result: ClassificationResult):
        """Log classification to JSONL for pattern mining."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_text': query_text,
            'complexity': result.complexity.value,
            'confidence': result.confidence,
            'tier_used': result.tier_used,
            'latency_ms': result.latency_ms,
            'detected_patterns': list(result.detected_patterns),
        }

        with open(self.classification_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    async def start_background_learning(self):
        """Start background adaptive learning loop."""
        if not self.background_learning or not self.enable_adaptive_learning:
            return

        if self._learning_task is not None:
            return  # Already running

        self._should_stop = False
        self._learning_task = asyncio.create_task(self._learning_loop())

    async def stop_background_learning(self):
        """Stop background learning loop."""
        if self._learning_task is None:
            return

        self._should_stop = True
        await self._learning_task
        self._learning_task = None

    async def _learning_loop(self):
        """Background learning loop - runs every hour."""
        while not self._should_stop:
            try:
                await asyncio.sleep(self.learning_update_interval)

                if self._should_stop:
                    break

                # Run adaptive learning cycle
                await self._run_learning_cycle()

            except Exception as e:
                print(f"[AdaptiveLearning] Error in learning loop: {e}")
                continue

    async def _run_learning_cycle(self):
        """
        Run one cycle of adaptive learning.

        Steps:
        1. Mine patterns from classification logs
        2. Validate current accuracy
        3. Deploy high-quality patterns (if available)
        4. Generate daily/weekly reports (if due)
        """
        print(f"\n[AdaptiveLearning] Starting learning cycle at {datetime.now()}")

        # Step 1: Mine patterns from logs
        if self.classification_log_path.exists():
            print("[AdaptiveLearning] Mining patterns from logs...")
            patterns = self.pattern_miner.mine_patterns(days_lookback=7)

            if patterns:
                print(f"[AdaptiveLearning] Mined {len(patterns)} high-quality patterns")

                # Save patterns to JSON
                with open(self.patterns_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        [asdict(p) for p in patterns],
                        f,
                        indent=2
                    )

        # Step 2: Validate current accuracy
        print("[AdaptiveLearning] Running hourly validation...")
        validation_result = await self.validator.validate_hourly(sample_size=20)

        print(f"[AdaptiveLearning] Validation accuracy: {validation_result.overall_accuracy:.1%}")

        if validation_result.regression_detected:
            print("[AdaptiveLearning] ⚠️  REGRESSION DETECTED!")

            # Generate alert
            if self.validator.regression_alerts:
                alert = self.validator.regression_alerts[-1]

                # Format for Slack/email
                slack_msg = self.reporter.format_slack_alert(alert)
                email_alert = self.reporter.format_email_alert(alert)

                print(f"[AdaptiveLearning] Slack alert:\n{slack_msg}")
                # In production, would POST to Slack webhook and send email

        # Step 3: Deploy high-quality patterns (if any)
        if self.patterns_path.exists():
            with open(self.patterns_path, encoding='utf-8') as f:
                pattern_data = json.load(f)

            if pattern_data:
                # Convert to Pattern objects
                patterns = [
                    Pattern(
                        regex=p['regex'],
                        complexity=p['complexity'],
                        score=PatternScore(**p['score']),
                        examples=p['examples']
                    )
                    for p in pattern_data
                ]

                # Deploy top 5 patterns
                top_patterns = patterns[:5]

                if top_patterns:
                    print(f"[AdaptiveLearning] Deploying {len(top_patterns)} top patterns...")
                    deployment_result = await self.updater.deploy_patterns(top_patterns)

                    print(f"[AdaptiveLearning] Deployment: {deployment_result.current_phase.value}")
                    print(f"[AdaptiveLearning] Patterns deployed: {deployment_result.patterns_deployed}")

        # Step 4: Generate daily/weekly reports (if due)
        now = datetime.now()

        # Daily report at 9am UTC
        if now.hour == 9 and now.minute < 60:  # Within first hour of 9am
            print("[AdaptiveLearning] Generating daily report...")
            daily_report = self.reporter.generate_daily_report()
            print(f"[AdaptiveLearning] Daily accuracy: {daily_report.overall_accuracy:.1%}")
            print(f"[AdaptiveLearning] Queries classified: {daily_report.queries_classified}")

        # Weekly report on Sundays at 9am UTC
        if now.weekday() == 6 and now.hour == 9 and now.minute < 60:
            print("[AdaptiveLearning] Generating weekly report...")
            weekly_report = self.reporter.generate_weekly_report()
            print(f"[AdaptiveLearning] Weekly accuracy: {weekly_report.overall_accuracy:.1%}")
            print(f"[AdaptiveLearning] Total queries: {weekly_report.total_queries}")
            print("[AdaptiveLearning] Recommendations:")
            for i, rec in enumerate(weekly_report.recommendations[:3], 1):
                print(f"[AdaptiveLearning]   {i}. {rec}")

        print("[AdaptiveLearning] Learning cycle complete\n")

    def get_learning_statistics(self) -> dict:
        """Get comprehensive adaptive learning statistics."""
        stats = {
            'base_classifier': self.classifier.get_stats_summary(),
            'total_queries_logged': 0,
            'patterns_discovered': 0,
            'patterns_deployed': 0,
            'validation_accuracy': 0.0,
            'regression_alerts': 0,
        }

        # Count logged queries
        if self.classification_log_path.exists():
            with open(self.classification_log_path, encoding='utf-8') as f:
                stats['total_queries_logged'] = sum(1 for _ in f)

        # Count patterns
        if self.patterns_path.exists():
            with open(self.patterns_path, encoding='utf-8') as f:
                pattern_data = json.load(f)
                stats['patterns_discovered'] = len(pattern_data)

        # Validation stats
        if self.validator:
            stats['validation_accuracy'] = self.validator.baseline_accuracy or 0.0
            stats['regression_alerts'] = len(self.validator.regression_alerts)

        # Deployment stats
        if self.updater:
            deployment_status = self.updater.get_deployment_status()
            stats['patterns_deployed'] = deployment_status['pattern_versions']

        return stats


# Convenience function
async def create_adaptive_classifier(
    enable_semantic: bool = True,
    enable_learning: bool = True,
    start_background: bool = True,
    data_dir: Path | None = None
) -> AdaptiveMoonshotClassifier:
    """
    Create and initialize adaptive classifier.

    Args:
        enable_semantic: Enable Tier 3 semantic embeddings
        enable_learning: Enable Phase 3 adaptive learning
        start_background: Start background learning loop
        data_dir: Data directory for logs and reports

    Returns:
        Initialized AdaptiveMoonshotClassifier
    """
    classifier = AdaptiveMoonshotClassifier(
        enable_semantic_tier=enable_semantic,
        enable_adaptive_learning=enable_learning,
        background_learning=start_background,
        data_dir=data_dir
    )

    if start_background:
        await classifier.start_background_learning()

    return classifier
