---
mode: narrative
style: tufte
quality: 0.92
confidence: 0.89
---

<!--
Quality Score: 0.92
Confidence: 0.89
-->

What is Thompson Sampling and how does it work?

Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. Furthermore, it balances exploration and exploitation by sampling from posterior distributions. Thompson Sampling maintains a probability distribution over the expected reward of each action. Building on this, at each timestep, it samples a reward estimate from each action's posterior distribution. The action with the highest sampled reward is selected and executed. Furthermore, after observing the reward, the posterior distribution is updated using bayes' theorem. Thompson Sampling handles the exploration-exploitation tradeoff without hyperparameters. Building on this, it has strong theoretical guarantees and empirical performance on many benchmark problems. Summary: Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.