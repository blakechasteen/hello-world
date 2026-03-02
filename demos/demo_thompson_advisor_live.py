"""
Thompson Sampling Advisor - Live Demo with Real HoloLoom Policy

Runs 50 queries through the actual HoloLoom policy engine to collect
real bandit statistics and provide advisor recommendations.
"""

import asyncio
import torch
import random
import sys

# Add parent to path
sys.path.insert(0, '.')

from hololoom.policy.unified import create_policy, BanditStrategy


async def run_experiment():
    print('=' * 70)
    print('THOMPSON SAMPLING ADVISOR - REAL HOLOLOOM POLICY')
    print('=' * 70)
    print()

    # Create actual HoloLoom policy with Thompson Sampling
    policy = create_policy(
        mem_dim=384,
        emb=None,  # No embedding needed for this test
        scales=[96, 192, 384],
        bandit_strategy=BanditStrategy.PURE_THOMPSON,
        epsilon=0.1
    )

    # Access tools via core, bandit is direct attribute
    tools = policy.core.tools
    print(f'Policy created with {len(tools)} tools: {tools}')
    print(f'Bandit strategy: PURE_THOMPSON')
    print()

    # Get initial bandit stats
    # Note: TSBandit uses success/fail instead of alpha/beta
    # alpha = success + 1 (prior), beta = fail + 1 (prior)
    print('Initial Bandit State:')
    initial_stats = policy.bandit.get_stats()
    for i, tool in enumerate(tools):
        stats = initial_stats[i]
        alpha = stats["success"] + 1  # Beta distribution prior
        beta = stats["fail"] + 1
        print(f'  {tool}: alpha={alpha:.1f}, beta={beta:.1f}, pulls={stats["pulls"]:.0f}')
    print()

    # Simulate 50 queries with varying complexity
    queries = [
        # Simple factual (should favor 'answer')
        ('What is Python?', 0.95, 'answer'),
        ('Define recursion', 0.92, 'answer'),
        ('What is 2+2?', 0.99, 'answer'),
        ('Hello', 0.98, 'answer'),
        ('What is a variable?', 0.94, 'answer'),

        # Research queries (should favor 'research')
        ('Compare Thompson Sampling vs UCB', 0.82, 'research'),
        ('Analyze ML tradeoffs', 0.78, 'research'),
        ('What are the pros and cons of Python?', 0.80, 'research'),
        ('Deep dive into neural networks', 0.75, 'research'),
        ('Explain transformer architecture in detail', 0.77, 'research'),

        # Exploration queries (should favor 'explore')
        ('Tell me something interesting', 0.70, 'explore'),
        ('Surprise me with a fact', 0.68, 'explore'),
        ('What should I learn next?', 0.72, 'explore'),

        # Clarification queries (should favor 'clarify')
        ('What do you mean by that?', 0.85, 'clarify'),
        ('Can you explain more?', 0.88, 'clarify'),
    ]

    # Repeat to get 50 total
    all_queries = queries * 4  # 60 queries total
    random.shuffle(all_queries)
    all_queries = all_queries[:50]

    print(f'Running {len(all_queries)} queries through Thompson Sampling...')
    print()

    # Track decisions
    decisions = {tool: 0 for tool in tools}
    explorations = 0
    exploitations = 0

    for i, (query, expected_conf, expected_tool) in enumerate(all_queries, 1):
        # Create dummy features (384-dim)
        features = torch.randn(1, 384)

        # Get policy decision - directly use bandit for pure Thompson Sampling
        with torch.no_grad():
            # Sample from bandit (Thompson Sampling)
            probs = torch.ones(len(tools)) / len(tools)  # Uniform prior for demo
            selected_idx, _ = policy.bandit.select_with_strategy(probs.numpy())
            selected_tool = tools[selected_idx]

        # Simulate outcome based on match with expected tool
        if selected_tool == expected_tool:
            success = random.random() < expected_conf
            confidence = expected_conf if success else expected_conf * 0.5
        else:
            success = random.random() < 0.4  # Lower success if wrong tool
            confidence = 0.5 if success else 0.3

        # Update bandit - reward is confidence if successful, 0 otherwise
        reward = confidence if success else 0.0
        policy.bandit.update(selected_idx, reward)

        # Track
        decisions[selected_tool] += 1

        # Determine if exploration or exploitation
        bandit_stats = policy.bandit.get_stats()
        best_tool_idx = max(
            range(len(tools)),
            key=lambda x: (bandit_stats[x]['success'] + 1) /
                         (bandit_stats[x]['success'] + bandit_stats[x]['fail'] + 2)
        )
        if selected_idx == best_tool_idx:
            exploitations += 1
        else:
            explorations += 1

        # Print every 10 queries
        if i % 10 == 0:
            status = "OK" if success else "FAIL"
            print(f'  Query {i:2d}: {selected_tool:10s} | {status:4s} | Conf: {confidence:.2f}')

    print()
    print('=' * 70)
    print('FINAL BANDIT STATISTICS (after 50 pulls)')
    print('=' * 70)
    print()

    final_stats = policy.bandit.get_stats()

    print(f'{"Tool":<12} {"alpha":>8} {"beta":>8} {"Pulls":>7} {"E[X]":>8} {"Uncertainty":>12}')
    print('-' * 60)

    for i, tool in enumerate(tools):
        stats = final_stats[i]
        a = stats['success'] + 1  # Beta prior
        b = stats['fail'] + 1
        expected = a / (a + b)
        uncertainty = (a * b) / ((a + b)**2 * (a + b + 1))
        pulls = decisions[tool]
        print(f'{tool:<12} {a:>8.2f} {b:>8.2f} {pulls:>7d} {expected:>8.3f} {uncertainty:>12.6f}')

    print()
    print('=' * 70)
    print('THOMPSON SAMPLING ADVISOR ANALYSIS')
    print('=' * 70)
    print()

    # Find best tool (by index)
    def get_expected(idx):
        s = final_stats[idx]
        a = s['success'] + 1
        b = s['fail'] + 1
        return a / (a + b)

    def calc_unc(idx):
        s = final_stats[idx]
        a = s['success'] + 1
        b = s['fail'] + 1
        return (a * b) / ((a + b)**2 * (a + b + 1))

    best_idx = max(range(len(tools)), key=get_expected)
    best_tool = tools[best_idx]
    best_exp = get_expected(best_idx)

    # Find most uncertain
    most_uncertain_idx = max(range(len(tools)), key=calc_unc)
    most_uncertain = tools[most_uncertain_idx]

    print(f'[BEST TOOL]      {best_tool}: E[X] = {best_exp:.3f}')
    print(f'[MOST UNCERTAIN] {most_uncertain}: Var = {calc_unc(most_uncertain_idx):.6f}')
    print(f'[EXPLORATION]    {explorations} decisions ({explorations/50*100:.1f}%)')
    print(f'[EXPLOITATION]   {exploitations} decisions ({exploitations/50*100:.1f}%)')
    print()

    print('TOOL USAGE DISTRIBUTION:')
    for tool in sorted(tools, key=lambda t: decisions[t], reverse=True):
        bar = '#' * (decisions[tool] * 2)
        print(f'  {tool:<12} {decisions[tool]:>3} {bar}')

    print()
    print('RECOMMENDATIONS:')
    for i, tool in enumerate(tools):
        stats = final_stats[i]
        a = stats['success'] + 1  # Beta prior
        b = stats['fail'] + 1
        expected = a / (a + b)
        uncertainty = calc_unc(i)
        pulls = decisions[tool]

        if pulls < 5:
            print(f'  [!] {tool}: Under-explored ({pulls} pulls) - increase exploration')
        elif uncertainty > 0.02:
            print(f'  [?] {tool}: High uncertainty ({uncertainty:.4f}) - needs more data')
        elif expected < 0.5:
            print(f'  [-] {tool}: Low success rate ({expected:.1%}) - may need tuning')
        else:
            print(f'  [OK] {tool}: Good performance (E[X]={expected:.3f}, {pulls} pulls)')

    print()
    print('=' * 70)
    print('CONVERGENCE ANALYSIS')
    print('=' * 70)
    print()

    # Check if converged
    uncertainties = [calc_unc(i) for i in range(len(tools))]
    avg_uncertainty = sum(uncertainties) / len(uncertainties)

    if avg_uncertainty < 0.01:
        print('STATUS: CONVERGED - Bandit has stable estimates')
    elif avg_uncertainty < 0.03:
        print('STATUS: CONVERGING - Estimates becoming stable')
    else:
        print('STATUS: LEARNING - More exploration needed')

    print(f'Average uncertainty: {avg_uncertainty:.6f}')
    print(f'Target uncertainty: < 0.01 for stable estimates')
    print(f'Recommended additional pulls: ~{int(50 * (avg_uncertainty / 0.01))} per tool')


if __name__ == '__main__':
    asyncio.run(run_experiment())
