"""
Consciousness Stack Experiments

Automated experiment runner that systematically tests:
1. Fusion impact (ON vs OFF)
2. Complexity scaling (LITE → RESEARCH)
3. Token budget effects (2000 → 8000)
4. Memory limits (5 → 20)

Generates detailed comparison reports with metrics and visualizations.

Run: python experiments/run_experiments.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the simple UI components
from ui.consciousness_ui_simple import (
    DemoMemoryBackend,
    SimpleAwareness,
    SimpleMemoryFusion,
    SimpleContextPacker,
    SimpleDualStream,
    MemoryNode,
    AwarenessContext,
    PackedContext
)


@dataclass
class ExperimentResult:
    """Single experiment result"""
    experiment_name: str
    query: str
    parameters: Dict[str, Any]
    
    # Awareness metrics
    awareness_confidence: float
    awareness_domain: str
    
    # Memory metrics
    memories_retrieved: int
    max_depth: int
    avg_composite_score: float
    fusion_enabled: bool
    
    # Packing metrics
    total_tokens: int
    token_budget: int
    token_usage_pct: float
    elements_included: int
    elements_compressed: int
    elements_excluded: int
    avg_importance: float
    min_importance: float
    
    # Performance metrics
    total_time_ms: float
    fusion_time_ms: float
    packing_time_ms: float
    generation_time_ms: float
    
    # Timestamp
    timestamp: str


@dataclass
class ExperimentComparison:
    """Comparison between two experiments"""
    name: str
    baseline: ExperimentResult
    variant: ExperimentResult
    
    # Deltas
    memories_delta: int
    depth_delta: int
    score_delta: float
    tokens_delta: int
    importance_delta: float
    time_delta: float
    
    # Percentages
    memories_change_pct: float
    tokens_change_pct: float
    importance_change_pct: float
    time_change_pct: float


class ExperimentRunner:
    """Automated experiment runner"""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.backend = DemoMemoryBackend()
        self.awareness = SimpleAwareness()
        
        self.test_query = "What are the applications of quantum computing?"
    
    async def run_single_experiment(
        self,
        experiment_name: str,
        query: str,
        complexity: str,
        use_fusion: bool,
        max_memories: int,
        token_budget: int
    ) -> ExperimentResult:
        """Run a single experiment configuration"""
        
        print(f"\n🧪 Running: {experiment_name}")
        print(f"   Query: {query}")
        print(f"   Params: complexity={complexity}, fusion={use_fusion}, max_mem={max_memories}, budget={token_budget}")
        
        start_time = time.perf_counter()
        
        # 1. Awareness
        awareness_ctx = await self.awareness.analyze(query)
        
        # 2. Memory Fusion
        fusion_start = time.perf_counter()
        memories = []
        max_depth = 0
        avg_score = 0.0
        
        if use_fusion:
            max_passes = {"LITE": 1, "FAST": 2, "FULL": 3, "RESEARCH": 4}.get(complexity, 3)
            fusion = SimpleMemoryFusion(self.backend, max_passes=max_passes)
            memory_nodes = await fusion.retrieve_with_fusion(query, max_results=max_memories)
            memories = memory_nodes
            max_depth = max((n.retrieval_depth for n in memory_nodes), default=0)
            avg_score = sum(n.composite_score for n in memory_nodes) / len(memory_nodes) if memory_nodes else 0
        else:
            # Simple retrieval without fusion
            items = await self.backend.retrieve_with_threshold(query, threshold=0.6, limit=max_memories)
            for item in items:
                memories.append(MemoryNode(
                    content=item['content'],
                    relevance=item['relevance'],
                    composite_score=item['relevance'],
                    retrieval_depth=0,
                    timestamp=item['timestamp']
                ))
            avg_score = sum(m.composite_score for m in memories) / len(memories) if memories else 0
        
        fusion_time = (time.perf_counter() - fusion_start) * 1000
        
        # 3. Context Packing
        packer = SimpleContextPacker(token_budget=token_budget)
        packed = await packer.pack(query, awareness_ctx, memories)
        
        # 4. Generation
        generator = SimpleDualStream()
        gen_start = time.perf_counter()
        internal, external, gen_time = await generator.generate(query, packed.formatted_text)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            query=query,
            parameters={
                "complexity": complexity,
                "use_fusion": use_fusion,
                "max_memories": max_memories,
                "token_budget": token_budget
            },
            awareness_confidence=awareness_ctx.confidence,
            awareness_domain=f"{awareness_ctx.domain}/{awareness_ctx.subdomain}",
            memories_retrieved=len(memories),
            max_depth=max_depth,
            avg_composite_score=avg_score,
            fusion_enabled=use_fusion,
            total_tokens=packed.total_tokens,
            token_budget=token_budget,
            token_usage_pct=(packed.total_tokens / (token_budget * 0.65)) * 100,
            elements_included=packed.elements_included,
            elements_compressed=packed.elements_compressed,
            elements_excluded=packed.elements_excluded,
            avg_importance=packed.avg_importance,
            min_importance=packed.min_importance,
            total_time_ms=total_time,
            fusion_time_ms=fusion_time,
            packing_time_ms=packed.packing_time_ms,
            generation_time_ms=gen_time,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"   ✅ Complete: {len(memories)} memories, {packed.total_tokens} tokens, {total_time:.2f}ms")
        
        return result
    
    def compare_results(
        self,
        name: str,
        baseline: ExperimentResult,
        variant: ExperimentResult
    ) -> ExperimentComparison:
        """Compare two experiment results"""
        
        memories_delta = variant.memories_retrieved - baseline.memories_retrieved
        depth_delta = variant.max_depth - baseline.max_depth
        score_delta = variant.avg_composite_score - baseline.avg_composite_score
        tokens_delta = variant.total_tokens - baseline.total_tokens
        importance_delta = variant.avg_importance - baseline.avg_importance
        time_delta = variant.total_time_ms - baseline.total_time_ms
        
        return ExperimentComparison(
            name=name,
            baseline=baseline,
            variant=variant,
            memories_delta=memories_delta,
            depth_delta=depth_delta,
            score_delta=score_delta,
            tokens_delta=tokens_delta,
            importance_delta=importance_delta,
            time_delta=time_delta,
            memories_change_pct=(memories_delta / baseline.memories_retrieved * 100) if baseline.memories_retrieved else 0,
            tokens_change_pct=(tokens_delta / baseline.total_tokens * 100) if baseline.total_tokens else 0,
            importance_change_pct=(importance_delta / baseline.avg_importance * 100) if baseline.avg_importance else 0,
            time_change_pct=(time_delta / baseline.total_time_ms * 100) if baseline.total_time_ms else 0
        )
    
    async def experiment_1_fusion_impact(self) -> List[ExperimentResult]:
        """Experiment 1: Compare Fusion ON vs OFF"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: FUSION IMPACT")
        print("="*80)
        print("Compare multipass graph crawling vs simple retrieval")
        
        results = []
        
        # Baseline: Fusion OFF
        baseline = await self.run_single_experiment(
            experiment_name="Fusion OFF (Baseline)",
            query=self.test_query,
            complexity="FULL",
            use_fusion=False,
            max_memories=10,
            token_budget=4000
        )
        results.append(baseline)
        
        # Variant: Fusion ON
        variant = await self.run_single_experiment(
            experiment_name="Fusion ON",
            query=self.test_query,
            complexity="FULL",
            use_fusion=True,
            max_memories=10,
            token_budget=4000
        )
        results.append(variant)
        
        # Compare
        comparison = self.compare_results("Fusion Impact", baseline, variant)
        self._print_comparison(comparison)
        
        return results
    
    async def experiment_2_complexity_scaling(self) -> List[ExperimentResult]:
        """Experiment 2: Scale complexity LITE → RESEARCH"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: COMPLEXITY SCALING")
        print("="*80)
        print("Test all complexity levels with fusion enabled")
        
        results = []
        
        for complexity in ["LITE", "FAST", "FULL", "RESEARCH"]:
            result = await self.run_single_experiment(
                experiment_name=f"Complexity: {complexity}",
                query=self.test_query,
                complexity=complexity,
                use_fusion=True,
                max_memories=10,
                token_budget=4000
            )
            results.append(result)
        
        # Compare consecutive levels
        for i in range(len(results) - 1):
            comparison = self.compare_results(
                f"{results[i].parameters['complexity']} → {results[i+1].parameters['complexity']}",
                results[i],
                results[i+1]
            )
            self._print_comparison(comparison)
        
        return results
    
    async def experiment_3_token_budget(self) -> List[ExperimentResult]:
        """Experiment 3: Adjust token budget 2000 → 8000"""
        print("\n" + "="*80)
        print("EXPERIMENT 3: TOKEN BUDGET IMPACT")
        print("="*80)
        print("Test compression behavior with varying budgets")
        
        results = []
        
        for budget in [2000, 3000, 4000, 6000, 8000]:
            result = await self.run_single_experiment(
                experiment_name=f"Budget: {budget}",
                query=self.test_query,
                complexity="FULL",
                use_fusion=True,
                max_memories=10,
                token_budget=budget
            )
            results.append(result)
        
        # Compare first vs last
        comparison = self.compare_results(
            "Budget 2000 → 8000",
            results[0],
            results[-1]
        )
        self._print_comparison(comparison)
        
        return results
    
    async def experiment_4_memory_limits(self) -> List[ExperimentResult]:
        """Experiment 4: Limit memories 5 → 20"""
        print("\n" + "="*80)
        print("EXPERIMENT 4: MEMORY LIMITS")
        print("="*80)
        print("Test quality vs quantity tradeoffs")
        
        results = []
        
        for max_mem in [5, 8, 10, 15, 20]:
            result = await self.run_single_experiment(
                experiment_name=f"Max Memories: {max_mem}",
                query=self.test_query,
                complexity="FULL",
                use_fusion=True,
                max_memories=max_mem,
                token_budget=4000
            )
            results.append(result)
        
        # Compare first vs last
        comparison = self.compare_results(
            "Memories 5 → 20",
            results[0],
            results[-1]
        )
        self._print_comparison(comparison)
        
        return results
    
    def _print_comparison(self, comp: ExperimentComparison):
        """Print comparison in formatted style"""
        print(f"\n📊 COMPARISON: {comp.name}")
        print(f"   Baseline: {comp.baseline.experiment_name}")
        print(f"   Variant:  {comp.variant.experiment_name}")
        print(f"\n   Deltas:")
        print(f"   • Memories:   {comp.memories_delta:+d} ({comp.memories_change_pct:+.1f}%)")
        print(f"   • Max Depth:  {comp.depth_delta:+d}")
        print(f"   • Avg Score:  {comp.score_delta:+.3f}")
        print(f"   • Tokens:     {comp.tokens_delta:+d} ({comp.tokens_change_pct:+.1f}%)")
        print(f"   • Importance: {comp.importance_delta:+.3f} ({comp.importance_change_pct:+.1f}%)")
        print(f"   • Time:       {comp.time_delta:+.2f}ms ({comp.time_change_pct:+.1f}%)")
    
    def save_results(self, all_results: List[ExperimentResult], filename: str):
        """Save results to JSON"""
        output_file = self.output_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(all_results),
            "test_query": self.test_query,
            "results": [asdict(r) for r in all_results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def generate_report(self, all_results: List[ExperimentResult]):
        """Generate markdown report"""
        report_file = self.output_dir / "experiment_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Consciousness Stack Experiment Results\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Query**: {self.test_query}\n\n")
            f.write(f"**Total Experiments**: {len(all_results)}\n\n")
            
            f.write("---\n\n")
            
            # Group by experiment type
            exp1 = [r for r in all_results if "Fusion" in r.experiment_name and "Budget" not in r.experiment_name]
            exp2 = [r for r in all_results if "Complexity" in r.experiment_name]
            exp3 = [r for r in all_results if "Budget" in r.experiment_name]
            exp4 = [r for r in all_results if "Max Memories" in r.experiment_name]
            
            # Experiment 1: Fusion Impact
            if exp1:
                f.write("## Experiment 1: Fusion Impact\n\n")
                f.write("| Metric | Fusion OFF | Fusion ON | Delta | Change % |\n")
                f.write("|--------|-----------|-----------|-------|----------|\n")
                
                baseline = [r for r in exp1 if "OFF" in r.experiment_name][0]
                variant = [r for r in exp1 if "ON" in r.experiment_name and "OFF" not in r.experiment_name][0]
                
                metrics = [
                    ("Memories Retrieved", baseline.memories_retrieved, variant.memories_retrieved),
                    ("Max Depth", baseline.max_depth, variant.max_depth),
                    ("Avg Score", baseline.avg_composite_score, variant.avg_composite_score),
                    ("Total Tokens", baseline.total_tokens, variant.total_tokens),
                    ("Avg Importance", baseline.avg_importance, variant.avg_importance),
                    ("Total Time (ms)", baseline.total_time_ms, variant.total_time_ms),
                ]
                
                for name, base_val, var_val in metrics:
                    delta = var_val - base_val
                    change_pct = (delta / base_val * 100) if base_val else 0
                    f.write(f"| {name} | {base_val:.2f} | {var_val:.2f} | {delta:+.2f} | {change_pct:+.1f}% |\n")
                
                f.write("\n")
            
            # Experiment 2: Complexity Scaling
            if exp2:
                f.write("## Experiment 2: Complexity Scaling\n\n")
                f.write("| Complexity | Memories | Max Depth | Avg Score | Tokens | Importance | Time (ms) |\n")
                f.write("|-----------|----------|-----------|-----------|--------|------------|----------|\n")
                
                for r in exp2:
                    f.write(f"| {r.parameters['complexity']} | {r.memories_retrieved} | {r.max_depth} | "
                           f"{r.avg_composite_score:.3f} | {r.total_tokens} | {r.avg_importance:.2%} | {r.total_time_ms:.2f} |\n")
                
                f.write("\n")
            
            # Experiment 3: Token Budget
            if exp3:
                f.write("## Experiment 3: Token Budget Impact\n\n")
                f.write("| Budget | Tokens Used | Usage % | Compressed | Excluded | Importance | Time (ms) |\n")
                f.write("|--------|-------------|---------|------------|----------|------------|----------|\n")
                
                for r in exp3:
                    f.write(f"| {r.token_budget} | {r.total_tokens} | {r.token_usage_pct:.1f}% | "
                           f"{r.elements_compressed} | {r.elements_excluded} | {r.avg_importance:.2%} | {r.total_time_ms:.2f} |\n")
                
                f.write("\n")
            
            # Experiment 4: Memory Limits
            if exp4:
                f.write("## Experiment 4: Memory Limits\n\n")
                f.write("| Max Memories | Retrieved | Avg Score | Tokens | Importance | Time (ms) |\n")
                f.write("|-------------|-----------|-----------|--------|------------|----------|\n")
                
                for r in exp4:
                    f.write(f"| {r.parameters['max_memories']} | {r.memories_retrieved} | "
                           f"{r.avg_composite_score:.3f} | {r.total_tokens} | {r.avg_importance:.2%} | {r.total_time_ms:.2f} |\n")
                
                f.write("\n")
            
            # Summary
            f.write("---\n\n")
            f.write("## Key Findings\n\n")
            
            if exp1:
                baseline = [r for r in exp1 if "OFF" in r.experiment_name][0]
                variant = [r for r in exp1 if "ON" in r.experiment_name and "OFF" not in r.experiment_name][0]
                depth_gain = variant.max_depth - baseline.max_depth
                f.write(f"### Fusion Impact\n")
                f.write(f"- Graph depth increased by **{depth_gain} hops** with fusion enabled\n")
                f.write(f"- Score changed by **{(variant.avg_composite_score - baseline.avg_composite_score):.3f}**\n")
                f.write(f"- Time overhead: **{(variant.total_time_ms - baseline.total_time_ms):.2f}ms**\n\n")
            
            if exp2:
                lite = exp2[0]
                research = exp2[-1]
                f.write(f"### Complexity Scaling\n")
                f.write(f"- LITE → RESEARCH increased depth from **{lite.max_depth}** to **{research.max_depth}** hops\n")
                f.write(f"- Retrieved **{research.memories_retrieved - lite.memories_retrieved}** more memories\n")
                f.write(f"- Time increased by **{(research.total_time_ms - lite.total_time_ms):.2f}ms**\n\n")
            
            if exp3:
                small = exp3[0]
                large = exp3[-1]
                f.write(f"### Token Budget\n")
                f.write(f"- 2000 → 8000 token budget reduced compression from **{small.elements_compressed}** to **{large.elements_compressed}** items\n")
                f.write(f"- Importance changed by **{(large.avg_importance - small.avg_importance):.3f}**\n")
                f.write(f"- Token usage: {small.token_usage_pct:.1f}% → {large.token_usage_pct:.1f}%\n\n")
            
            if exp4:
                few = exp4[0]
                many = exp4[-1]
                f.write(f"### Memory Limits\n")
                f.write(f"- 5 → 20 memories changed avg score from **{few.avg_composite_score:.3f}** to **{many.avg_composite_score:.3f}**\n")
                f.write(f"- Token usage increased from **{few.total_tokens}** to **{many.total_tokens}**\n")
                f.write(f"- Importance changed by **{(many.avg_importance - few.avg_importance):.3f}**\n\n")
        
        print(f"📄 Report generated: {report_file}")
    
    async def run_all_experiments(self):
        """Run all experiments and generate reports"""
        print("\n" + "🧪" * 40)
        print("CONSCIOUSNESS STACK EXPERIMENTS".center(80))
        print("🧪" * 40 + "\n")
        
        all_results = []
        
        # Run all experiments
        all_results.extend(await self.experiment_1_fusion_impact())
        all_results.extend(await self.experiment_2_complexity_scaling())
        all_results.extend(await self.experiment_3_token_budget())
        all_results.extend(await self.experiment_4_memory_limits())
        
        # Save and report
        self.save_results(all_results, "all_experiments.json")
        self.generate_report(all_results)
        
        print("\n" + "="*80)
        print("✅ ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  • all_experiments.json - Raw data")
        print(f"  • experiment_report.md - Formatted report")


async def main():
    """Main entry point"""
    runner = ExperimentRunner()
    await runner.run_all_experiments()


if __name__ == "__main__":
    asyncio.run(main())
