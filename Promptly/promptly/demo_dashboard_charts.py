#!/usr/bin/env python3
"""
Demo: Web Dashboard with Charts
================================
Populate analytics database with sample data and launch dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tools.prompt_analytics import PromptAnalytics, PromptExecution
from datetime import datetime, timedelta
import random
import time

print("=" * 60)
print("Promptly Dashboard Charts Demo")
print("=" * 60)

# Initialize analytics
analytics = PromptAnalytics()

print("\n1. Populating database with sample data...")

# Sample prompt names
prompts = [
    "sql_optimizer",
    "code_reviewer",
    "ui_designer",
    "system_architect",
    "refactoring_expert",
    "security_auditor",
    "api_designer",
    "test_generator"
]

# Generate executions over last 30 days
base_time = datetime.now() - timedelta(days=30)
total_executions = 0

for day in range(30):
    current_date = base_time + timedelta(days=day)

    # Random number of executions per day (1-10)
    daily_executions = random.randint(1, 10)

    for _ in range(daily_executions):
        # Pick random prompt
        prompt = random.choice(prompts)

        # Generate realistic metrics
        execution_time = random.uniform(5.0, 30.0)
        quality_score = random.uniform(0.7, 0.95)
        success = random.random() > 0.05  # 95% success rate

        # Simulate improvement trend for some prompts
        if prompt in ["sql_optimizer", "code_reviewer"]:
            quality_score += (day / 30) * 0.1  # Improving trend
            quality_score = min(quality_score, 0.98)

        # Simulate degrading trend for others
        elif prompt == "security_auditor":
            quality_score -= (day / 30) * 0.05  # Degrading trend

        # Add some time of day variation
        hour = random.randint(0, 23)
        timestamp = current_date + timedelta(hours=hour)

        # Record execution
        execution = PromptExecution(
            prompt_id=f"{prompt}_{day}_{_}",
            prompt_name=prompt,
            execution_time=execution_time,
            quality_score=quality_score if success else None,
            success=success,
            model="llama3.2:3b",
            backend="ollama",
            tokens_used=random.randint(100, 1000) if success else None,
            cost=0.0,  # Ollama is free
            timestamp=timestamp.isoformat()
        )

        analytics.record_execution(execution)
        total_executions += 1

print(f"   [OK] Created {total_executions} sample executions across {len(prompts)} prompts")

# Get summary
summary = analytics.get_summary()
print(f"\n2. Database populated:")
print(f"   - Total Executions: {summary['total_executions']}")
print(f"   - Unique Prompts: {summary['unique_prompts']}")
print(f"   - Success Rate: {summary['success_rate']:.1f}%")
print(f"   - Avg Execution Time: {summary['avg_execution_time']:.1f}s")
print(f"   - Avg Quality Score: {summary['avg_quality_score']:.2f}")

# Show top prompts
print(f"\n3. Top prompts by quality:")
top_prompts = analytics.get_top_prompts(metric='quality', limit=3)
for i, stats in enumerate(top_prompts, 1):
    print(f"   {i}. {stats.prompt_name}: {stats.avg_quality_score:.2f} quality ({stats.total_executions} runs)")

# Show recommendations
print(f"\n4. Recommendations:")
recommendations = analytics.get_recommendations()
if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
    for rec in recommendations[:3]:
        if isinstance(rec, dict):
            print(f"   - {rec.get('prompt_name', 'Unknown')}: {rec.get('recommendation', '')}")
else:
    print("   - No issues detected, all prompts performing well!")

print("\n" + "=" * 60)
print("Dashboard Ready!")
print("=" * 60)
print("\nNow run:")
print("  python web_dashboard.py")
print("\nThen open: http://localhost:5000")
print("\nYou'll see:")
print("  - 4 beautiful charts (timeline, success, quality, time dist)")
print("  - Real data from 30 days of executions")
print("  - Interactive visualizations")
print("  - All prompts with stats")
print("  - Recommendations")
print("  - Top performers")
print("\n" + "=" * 60)
