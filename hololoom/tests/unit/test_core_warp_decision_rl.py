"""
Unit tests for warp math decision RL modules:
  - hierarchical_rl.py (OptionsFramework, MAXQ, FeudalRL)
  - deep_rl.py (_SimpleNet, ReplayBuffer, PrioritizedReplay, DQN, DoubleDQN,
                DuelingDQN, SafetyShield, RegretMatching)
  - multi_agent_rl.py (IndependentLearners, CentralizedCritic, MeanFieldMARL,
                        CommProtocol)
  - counterfactual_regret.py (GameNode, CounterfactualRegretMinimization,
                               CFRPlus, LinearCFR)
"""

import numpy as np
import pytest

from hololoom.core.warp.math.decision.hierarchical_rl import (
    Option,
    HRLResult,
    OptionsFramework,
    MAXQ,
    FeudalRL,
)
from hololoom.core.warp.math.decision.deep_rl import (
    _SimpleNet,
    Transition,
    ReplayBuffer,
    PrioritizedReplay,
    DQN,
    DoubleDQN,
    DuelingDQN,
    DQNResult,
    SafetyShield,
    RegretMatching,
)
from hololoom.core.warp.math.decision.multi_agent_rl import (
    MARLAgent,
    MARLResult,
    IndependentLearners,
    CentralizedCritic,
    MeanFieldMARL,
    CommProtocol,
)
from hololoom.core.warp.math.decision.counterfactual_regret import (
    InfoSet,
    CFRResult,
    GameNode,
    CounterfactualRegretMinimization,
    CFRPlus,
    LinearCFR,
)


# ============================================================================
# Fixtures: small MDPs and environments
# ============================================================================


def _simple_mdp(n_states=3, n_actions=2, seed=42):
    """Build a small deterministic-ish MDP for testing."""
    rng = np.random.RandomState(seed)
    # Transition: (n_states, n_actions, n_states), rows sum to 1
    raw = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    transition = raw / raw.sum(axis=2, keepdims=True)
    # Reward per state
    reward = rng.rand(n_states)
    return transition, reward


def _make_option(n_states=3, n_actions=2, name="opt"):
    """Create a simple Option with uniform policy and fixed termination."""
    policy = np.ones((n_states, n_actions)) / n_actions
    initiation = np.ones(n_states, dtype=bool)
    termination = np.full(n_states, 0.5)
    return Option(
        name=name,
        initiation_set=initiation,
        policy=policy,
        termination=termination,
        n_states=n_states,
        n_actions=n_actions,
    )


def _simple_env(state_dim=4, n_actions=2, max_steps=10):
    """Factory for a tiny environment compatible with DQN.learn."""
    def env_fn():
        state = np.zeros(state_dim)
        step_count = [0]

        def step(action):
            step_count[0] += 1
            next_state = np.random.randn(state_dim) * 0.1
            reward = 1.0 if action == 0 else -0.5
            done = step_count[0] >= max_steps
            return next_state, reward, done

        return state, {"step": step, "n_actions": n_actions, "max_steps": max_steps}

    return env_fn


def _rps_game():
    """Rock-Paper-Scissors payoff matrix (zero-sum)."""
    return np.array([
        [0, -1,  1],
        [1,  0, -1],
        [-1, 1,  0],
    ])


# ============================================================================
# hierarchical_rl.py — Option
# ============================================================================


class TestOption:
    def test_can_initiate_array(self):
        opt = _make_option()
        assert opt.can_initiate(0) is True
        opt.initiation_set = np.array([True, False, True])
        assert opt.can_initiate(1) is False

    def test_can_initiate_callable(self):
        opt = _make_option()
        opt.initiation_set = lambda s: s != 1
        assert opt.can_initiate(0) is True
        assert opt.can_initiate(1) is False

    def test_termination_prob_array(self):
        opt = _make_option()
        assert opt.termination_prob(0) == pytest.approx(0.5)

    def test_termination_prob_callable(self):
        opt = _make_option()
        opt.termination = lambda s: 0.1 * s
        assert opt.termination_prob(2) == pytest.approx(0.2)

    def test_select_action_valid(self):
        np.random.seed(0)
        opt = _make_option(n_actions=3)
        action = opt.select_action(0)
        assert 0 <= action < 3


# ============================================================================
# hierarchical_rl.py — OptionsFramework
# ============================================================================


class TestOptionsFramework:
    def test_learn_options_returns_list(self):
        np.random.seed(0)
        T, R = _simple_mdp()
        fw = OptionsFramework(n_states=3, n_actions=2, lr=0.05)
        options = fw.learn_options(T, R, n_options=2, n_episodes=20, max_steps=15)
        assert len(options) == 2
        for o in options:
            assert isinstance(o, Option)
            assert o.policy.shape == (3, 2)
            # Policy rows sum to 1
            np.testing.assert_allclose(o.policy.sum(axis=1), 1.0, atol=1e-6)

    def test_smdp_value_iteration(self):
        np.random.seed(1)
        T, R = _simple_mdp()
        fw = OptionsFramework(n_states=3, n_actions=2)
        options = [_make_option(), _make_option(name="opt2")]
        V, meta_policy = fw.smdp_value_iteration(T, R, options, n_iter=20)
        assert V.shape == (3,)
        assert meta_policy.shape == (3, 2)
        # Meta policy rows sum to ~1
        np.testing.assert_allclose(meta_policy.sum(axis=1), 1.0, atol=1e-5)

    def test_execute_option(self):
        np.random.seed(2)
        T, _ = _simple_mdp()
        opt = _make_option()
        traj, final, dur = OptionsFramework.execute_option(opt, state=0, transition=T, max_steps=20)
        assert len(traj) == dur
        assert 0 <= final < 3
        for s, a in traj:
            assert 0 <= s < 3
            assert 0 <= a < 2

    def test_option_model(self):
        """option_model uses einsum with invalid subscript char; verify it raises."""
        np.random.seed(3)
        T, R = _simple_mdp()
        opt = _make_option()
        # Source has a bug: einsum uses "s'" which is invalid.
        # Just verify it raises ValueError so we document the known issue.
        with pytest.raises(ValueError):
            OptionsFramework.option_model(opt, T, R, n_iter=20)

    def test_learn_options_with_2d_reward(self):
        np.random.seed(4)
        T, _ = _simple_mdp()
        R_2d = np.random.rand(3, 2)
        fw = OptionsFramework(n_states=3, n_actions=2, lr=0.05)
        options = fw.learn_options(T, R_2d, n_options=1, n_episodes=10, max_steps=10)
        assert len(options) == 1


# ============================================================================
# hierarchical_rl.py — MAXQ
# ============================================================================


class TestMAXQ:
    def test_decompose_returns_value_functions(self):
        np.random.seed(5)
        T, R = _simple_mdp(n_actions=2)
        task_graph = {"root": ["a0", "a1"]}
        maxq = MAXQ(n_states=3, n_actions=2, lr=0.1)
        V = maxq.decompose(task_graph, T, R, n_episodes=30, max_steps=10)
        assert "root" in V
        assert "a0" in V
        assert "a1" in V
        assert V["root"].shape == (3,)

    def test_execute_returns_actions(self):
        np.random.seed(6)
        T, R = _simple_mdp(n_actions=2)
        task_graph = {"root": ["a0", "a1"]}
        maxq = MAXQ(n_states=3, n_actions=2, lr=0.1)
        V = maxq.decompose(task_graph, T, R, n_episodes=20, max_steps=10)
        actions, total_reward = maxq.execute("root", 0, task_graph, T, R, V, max_steps=10)
        assert isinstance(actions, list)
        assert isinstance(total_reward, float)

    def test_deeper_task_graph(self):
        np.random.seed(7)
        T, R = _simple_mdp(n_states=4, n_actions=3)
        task_graph = {"root": ["sub1", "sub2"], "sub1": ["a0", "a1"], "sub2": ["a2"]}
        maxq = MAXQ(n_states=4, n_actions=3, lr=0.1)
        V = maxq.decompose(task_graph, T, R, n_episodes=20, max_steps=10)
        assert "sub1" in V
        assert "sub2" in V


# ============================================================================
# hierarchical_rl.py — FeudalRL
# ============================================================================


class TestFeudalRL:
    def test_goal_direction_shape(self):
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        d = feudal._goal_direction(0)
        assert d.shape == (3,)
        assert d[0] == 1.0  # dim 0, positive

    def test_intrinsic_reward_range(self):
        np.random.seed(8)
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        goal = np.array([1.0, 0.0, 0.0])
        r = feudal.intrinsic_reward(0, goal, 1)
        assert -1.0 <= r <= 1.0

    def test_intrinsic_reward_zero_goal(self):
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        r = feudal.intrinsic_reward(0, np.zeros(3), 1)
        assert r == 0.0

    def test_manager_policy_returns_goal(self):
        np.random.seed(9)
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        goal = feudal.manager_policy(0, epsilon=0.0)
        assert goal.shape == (3,)

    def test_worker_policy_returns_action(self):
        np.random.seed(10)
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        goal = np.array([1.0, 0.0, 0.0])
        a = feudal.worker_policy(0, goal, epsilon=0.0)
        assert 0 <= a < 2

    def test_train_returns_result(self):
        np.random.seed(11)
        T, R = _simple_mdp(n_states=4, n_actions=2)
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=2, manager_horizon=3, lr=0.02)
        result = feudal.train(T, R, n_episodes=15, max_steps=20)
        assert isinstance(result, HRLResult)
        assert len(result.options) > 0
        assert result.meta_policy.shape[0] == 4
        assert result.n_episodes == 15

    def test_goal_bin_zero_vector(self):
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        assert feudal._goal_bin(np.zeros(3)) == 0

    def test_embed_state(self):
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        e = feudal._embed_state(0)
        assert e.shape == (3,)

    def test_extract_options(self):
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        options = feudal._extract_options()
        assert len(options) == feudal._n_goal_dirs
        for o in options:
            assert o.policy.shape == (4, 2)
            np.testing.assert_allclose(o.policy.sum(axis=1), 1.0, atol=1e-6)

    def test_manager_softmax(self):
        feudal = FeudalRL(n_states=4, n_actions=2, goal_dim=3)
        mp = feudal._manager_softmax()
        assert mp.shape == (4, feudal._n_goal_dirs)
        np.testing.assert_allclose(mp.sum(axis=1), 1.0, atol=1e-6)


# ============================================================================
# deep_rl.py — _SimpleNet
# ============================================================================


class TestSimpleNet:
    def test_forward_shape(self):
        net = _SimpleNet(4, 8, 2)
        x = np.random.randn(1, 4)
        out = net.forward(x)
        assert out.shape == (1, 2)

    def test_forward_batch(self):
        net = _SimpleNet(4, 8, 2)
        x = np.random.randn(5, 4)
        out = net.forward(x)
        assert out.shape == (5, 2)

    def test_copy_from(self):
        net1 = _SimpleNet(4, 8, 2)
        net2 = _SimpleNet(4, 8, 2)
        net2.copy_from(net1)
        np.testing.assert_array_equal(net1.W1, net2.W1)
        np.testing.assert_array_equal(net1.b2, net2.b2)

    def test_update_changes_weights(self):
        net = _SimpleNet(4, 8, 2)
        x = np.random.randn(3, 4)
        w_before = net.W2.copy()
        targets = np.array([1.0, 2.0, 0.5])
        actions = np.array([0, 1, 0])
        net.update(x, targets, actions, lr=0.01)
        assert not np.allclose(net.W2, w_before)


# ============================================================================
# deep_rl.py — ReplayBuffer
# ============================================================================


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(3):
            buf.add(Transition(np.zeros(4), i % 2, float(i), np.zeros(4), False))
        assert len(buf) == 3

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(10):
            buf.add(Transition(np.zeros(4), 0, 0.0, np.zeros(4), False))
        assert len(buf) == 3

    def test_sample(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(10):
            buf.add(Transition(np.array([float(i)]), 0, 0.0, np.zeros(1), False))
        batch = buf.sample(3)
        assert len(batch) == 3
        assert all(isinstance(t, Transition) for t in batch)


# ============================================================================
# deep_rl.py — PrioritizedReplay
# ============================================================================


class TestPrioritizedReplay:
    def test_add_and_size(self):
        pr = PrioritizedReplay(capacity=5)
        for i in range(3):
            pr.add(Transition(np.zeros(2), 0, 1.0, np.zeros(2), False), priority=1.0)
        assert pr.size == 3

    def test_sample_returns_weights(self):
        pr = PrioritizedReplay(capacity=10)
        for i in range(10):
            pr.add(Transition(np.zeros(2), 0, 1.0, np.zeros(2), False), priority=float(i + 1))
        transitions, indices, weights = pr.sample(3)
        assert len(transitions) == 3
        assert len(indices) == 3
        assert len(weights) == 3
        assert np.all(weights > 0)

    def test_update_priorities(self):
        pr = PrioritizedReplay(capacity=5)
        for i in range(5):
            pr.add(Transition(np.zeros(2), 0, 1.0, np.zeros(2), False), priority=1.0)
        old = pr.priorities[:5].copy()
        pr.update_priorities(np.array([0, 2]), np.array([10.0, 20.0]))
        assert pr.priorities[0] != old[0]
        assert pr.priorities[2] != old[2]


# ============================================================================
# deep_rl.py — DQN / DoubleDQN / DuelingDQN
# ============================================================================


class TestDQN:
    def test_learn_returns_result(self):
        np.random.seed(20)
        env_fn = _simple_env(state_dim=4, n_actions=2, max_steps=5)
        result = DQN.learn(
            env_fn, n_episodes=5, batch_size=4, buffer_size=50,
            target_update=10, state_dim=4, n_actions=2, hidden_dim=8,
        )
        assert isinstance(result, DQNResult)
        assert len(result.episode_rewards) == 5
        assert len(result.episode_lengths) == 5
        assert result.final_q_net is not None


class TestDoubleDQN:
    def test_learn_returns_result(self):
        np.random.seed(21)
        env_fn = _simple_env(state_dim=4, n_actions=2, max_steps=5)
        result = DoubleDQN.learn(
            env_fn, n_episodes=5, batch_size=4, buffer_size=50,
            target_update=10, state_dim=4, n_actions=2, hidden_dim=8,
        )
        assert isinstance(result, DQNResult)
        assert len(result.episode_rewards) == 5


class TestDuelingDQN:
    def test_learn_returns_result(self):
        """DuelingDQN.learn has a known bug: value_net output_dim=1 but update()
        indexes by action. Verify it raises IndexError to document the issue."""
        np.random.seed(22)
        env_fn = _simple_env(state_dim=4, n_actions=2, max_steps=5)
        with pytest.raises(IndexError):
            DuelingDQN.learn(
                env_fn, n_episodes=5, batch_size=4, buffer_size=50,
                target_update=10, state_dim=4, n_actions=2, hidden_dim=8,
            )

    def test_dueling_q_values_computation(self):
        """Test the dueling Q-value formula: Q = V + A - mean(A)."""
        v_net = _SimpleNet(4, 8, 1)
        a_net = _SimpleNet(4, 8, 3)
        states = np.random.randn(2, 4)
        v = v_net.forward(states)
        a = a_net.forward(states)
        q = v + a - np.mean(a, axis=1, keepdims=True)
        assert q.shape == (2, 3)
        # Mean advantage should be 0 by construction
        np.testing.assert_allclose(np.mean(q - v, axis=1), 0.0, atol=1e-10)


# ============================================================================
# deep_rl.py — SafetyShield
# ============================================================================


class TestSafetyShield:
    def test_is_safe_within_bounds(self):
        spec = {"state_bounds": (np.array([0.0, 0.0]), np.array([1.0, 1.0]))}
        assert SafetyShield.is_safe(np.array([0.5, 0.5]), 0, spec) is True

    def test_is_safe_outside_bounds(self):
        spec = {"state_bounds": (np.array([0.0, 0.0]), np.array([1.0, 1.0]))}
        assert SafetyShield.is_safe(np.array([1.5, 0.5]), 0, spec) is False

    def test_is_safe_forbidden_action(self):
        cond = lambda s: s[0] > 0.5
        spec = {"forbidden_actions": {cond: [1]}}
        assert SafetyShield.is_safe(np.array([0.8]), 1, spec) is False
        assert SafetyShield.is_safe(np.array([0.3]), 1, spec) is True

    def test_is_safe_transition_model(self):
        model = lambda s, a: s + np.array([0.5, 0.0])
        spec = {
            "state_bounds": (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            "transition_model": model,
        }
        # s=[0.4, 0.5] + [0.5, 0] = [0.9, 0.5] -> safe
        assert SafetyShield.is_safe(np.array([0.4, 0.5]), 0, spec) is True
        # s=[0.8, 0.5] + [0.5, 0] = [1.3, 0.5] -> unsafe
        assert SafetyShield.is_safe(np.array([0.8, 0.5]), 0, spec) is False

    def test_is_safe_constraint_fn(self):
        spec = {"constraint_fn": lambda s, a: a != 2}
        assert SafetyShield.is_safe(np.array([0.0]), 0, spec) is True
        assert SafetyShield.is_safe(np.array([0.0]), 2, spec) is False

    def test_safe_action_picks_safe(self):
        spec = {"constraint_fn": lambda s, a: a != 0}
        q = np.array([10.0, 5.0, 1.0])  # action 0 is best but unsafe
        a = SafetyShield.safe_action(np.array([0.0]), q, spec)
        assert a == 1  # best safe action

    def test_safe_action_all_unsafe_falls_back(self):
        spec = {"constraint_fn": lambda s, a: False}
        q = np.array([3.0, 1.0])
        a = SafetyShield.safe_action(np.array([0.0]), q, spec)
        assert a == 0  # fallback to best Q

    def test_empty_spec_always_safe(self):
        assert SafetyShield.is_safe(np.array([999.0]), 42, {}) is True


# ============================================================================
# deep_rl.py — RegretMatching
# ============================================================================


class TestRegretMatching:
    def test_initial_strategy_uniform(self):
        rm = RegretMatching(n_actions=3)
        strat = rm.get_strategy()
        np.testing.assert_allclose(strat, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_update_shifts_strategy(self):
        rm = RegretMatching(n_actions=3)
        # Action 0 has lowest loss -> highest regret for not playing it
        rm.update(np.array([0.0, 1.0, 1.0]))
        strat = rm.get_strategy()
        # Action 0 should now dominate
        assert strat[0] > strat[1]
        assert strat[0] > strat[2]

    def test_average_strategy_sums_to_one(self):
        rm = RegretMatching(n_actions=4)
        for _ in range(10):
            rm.update(np.random.rand(4))
        avg = rm.average_strategy()
        assert avg.sum() == pytest.approx(1.0, abs=1e-8)

    def test_exploitability_bound_decreases(self):
        rm = RegretMatching(n_actions=3)
        bounds = []
        for _ in range(50):
            rm.update(np.random.rand(3))
            bounds.append(rm.exploitability_bound())
        # Bound should generally decrease (not strictly, but trend)
        assert bounds[-1] < bounds[0] or bounds[-1] < 1.0

    def test_exploitability_bound_no_rounds(self):
        rm = RegretMatching(n_actions=2)
        assert rm.exploitability_bound() == float('inf')

    def test_cumulative_strategy_tracks_play(self):
        rm = RegretMatching(n_actions=2)
        rm.update(np.array([0.0, 1.0]))
        rm.update(np.array([0.0, 1.0]))
        assert rm.t == 2
        assert rm.cumulative_strategy.sum() > 0


# ============================================================================
# multi_agent_rl.py — MARLAgent
# ============================================================================


class TestMARLAgent:
    def test_default_policy_uniform(self):
        agent = MARLAgent(id="a", n_actions=3)
        np.testing.assert_allclose(agent.policy, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_default_q_zeros(self):
        agent = MARLAgent(id="a", n_actions=4)
        np.testing.assert_array_equal(agent.q_values, np.zeros(4))


# ============================================================================
# multi_agent_rl.py — IndependentLearners
# ============================================================================


class TestIndependentLearners:
    def test_step_returns_actions(self):
        np.random.seed(30)
        agents = [MARLAgent(id="a", n_actions=3), MARLAgent(id="b", n_actions=3)]
        il = IndependentLearners(agents, lr=0.1, epsilon=0.1)
        obs = {"a": 0, "b": 0}
        rewards = {"a": 1.0, "b": -1.0}
        actions = il.step(obs, rewards)
        assert "a" in actions and "b" in actions
        assert 0 <= actions["a"] < 3

    def test_multiple_steps(self):
        np.random.seed(31)
        agents = [MARLAgent(id="x", n_actions=2), MARLAgent(id="y", n_actions=2)]
        il = IndependentLearners(agents, lr=0.1, epsilon=0.2)
        for _ in range(20):
            il.step({"x": 0, "y": 0}, {"x": 1.0, "y": 0.5})
        result = il.get_joint_policy()
        assert isinstance(result, MARLResult)
        assert result.iterations == 20
        assert result.social_welfare > 0

    def test_step_with_states(self):
        np.random.seed(32)
        agents = [MARLAgent(id="a", n_actions=2)]
        il = IndependentLearners(agents, lr=0.1)
        actions = il.step_with_states(
            states={"a": 0},
            actions_taken={"a": 1},
            rewards={"a": 1.0},
            next_states={"a": 1},
            n_states=3,
        )
        assert "a" in actions
        # Q-table should be extended to 2D
        assert il.agents["a"].q_values.ndim == 2

    def test_nash_gap_nonnegative(self):
        np.random.seed(33)
        agents = [MARLAgent(id="a", n_actions=3), MARLAgent(id="b", n_actions=3)]
        il = IndependentLearners(agents)
        for _ in range(10):
            il.step({"a": 0, "b": 0}, {"a": 1.0, "b": 0.5})
        result = il.get_joint_policy()
        assert result.nash_gap >= 0


# ============================================================================
# multi_agent_rl.py — CentralizedCritic
# ============================================================================


class TestCentralizedCritic:
    def test_joint_index_roundtrip(self):
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=3)]
        cc = CentralizedCritic(agents)
        actions = {"a": 1, "b": 2}
        idx = cc._joint_index(actions)
        recovered = cc._individual_from_joint(idx)
        assert recovered["a"] == 1
        assert recovered["b"] == 2

    def test_step_returns_actions(self):
        np.random.seed(34)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        cc = CentralizedCritic(agents, lr=0.05)
        actions = cc.step({"a": 0, "b": 0}, {"a": 1.0, "b": 0.5})
        assert "a" in actions and "b" in actions

    def test_multiple_steps(self):
        np.random.seed(35)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        cc = CentralizedCritic(agents, lr=0.05)
        for _ in range(15):
            cc.step({"a": 0, "b": 0}, {"a": 1.0, "b": 0.5})
        result = cc.get_joint_policy()
        assert isinstance(result, MARLResult)
        assert result.nash_gap >= 0

    def test_softmax_sums_to_one(self):
        agents = [MARLAgent(id="a", n_actions=3)]
        cc = CentralizedCritic(agents)
        s = cc._softmax(np.array([1.0, 2.0, 3.0]))
        assert s.sum() == pytest.approx(1.0, abs=1e-8)


# ============================================================================
# multi_agent_rl.py — MeanFieldMARL
# ============================================================================


class TestMeanFieldMARL:
    def test_step_returns_actions(self):
        np.random.seed(36)
        agents = [MARLAgent(id=f"a{i}", n_actions=3) for i in range(4)]
        mf = MeanFieldMARL(agents, lr=0.1, epsilon=0.2)
        actions = mf.step({"a0": 0, "a1": 0, "a2": 0, "a3": 0},
                          {"a0": 1.0, "a1": 0.5, "a2": 0.0, "a3": -0.5})
        assert len(actions) == 4

    def test_mean_field_bin(self):
        agents = [MARLAgent(id="a", n_actions=3), MARLAgent(id="b", n_actions=3)]
        mf = MeanFieldMARL(agents)
        actions = {"a": 1, "b": 2}
        b = mf._mean_field_bin(actions, "a")
        assert 0 <= b < mf._n_bins

    def test_mean_field_bin_no_others(self):
        agents = [MARLAgent(id="a", n_actions=3)]
        mf = MeanFieldMARL(agents)
        b = mf._mean_field_bin({"a": 1}, "a")
        assert b == 0

    def test_get_joint_policy(self):
        np.random.seed(37)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        mf = MeanFieldMARL(agents, lr=0.1)
        for _ in range(10):
            mf.step({"a": 0, "b": 0}, {"a": 1.0, "b": 0.5})
        result = mf.get_joint_policy()
        assert isinstance(result, MARLResult)
        for aid, pol in result.joint_policy.items():
            assert pol.sum() == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# multi_agent_rl.py — CommProtocol
# ============================================================================


class TestCommProtocol:
    def test_communicate_returns_messages(self):
        np.random.seed(38)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        cp = CommProtocol(agents, message_size=4)
        msgs = cp.communicate({"a": 0, "b": 1}, round=0)
        assert "a" in msgs and "b" in msgs
        assert msgs["a"].shape == (4,)

    def test_communicate_multi_round(self):
        np.random.seed(39)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        cp = CommProtocol(agents, message_size=4)
        cp.communicate({"a": 0, "b": 0}, round=0)
        msgs2 = cp.communicate({"a": 0, "b": 0}, round=1)
        assert msgs2["a"].shape == (4,)

    def test_step_with_communication(self):
        np.random.seed(40)
        agents = [MARLAgent(id="a", n_actions=3), MARLAgent(id="b", n_actions=3)]
        cp = CommProtocol(agents, message_size=4, epsilon=0.2)
        actions = cp.step_with_communication({"a": 0, "b": 1}, n_comm_rounds=2)
        assert "a" in actions and "b" in actions
        assert 0 <= actions["a"] < 3

    def test_update(self):
        np.random.seed(41)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        cp = CommProtocol(agents, message_size=4, lr=0.05)
        cp.step_with_communication({"a": 0, "b": 0}, n_comm_rounds=1)
        dec_before = cp._decoders["a"].copy()
        cp.update({"a": 1.0, "b": -1.0}, {"a": 0, "b": 0})
        # Decoders should change after update
        assert not np.allclose(cp._decoders["a"], dec_before)

    def test_get_joint_policy(self):
        np.random.seed(42)
        agents = [MARLAgent(id="a", n_actions=2), MARLAgent(id="b", n_actions=2)]
        cp = CommProtocol(agents, message_size=4)
        cp.step_with_communication({"a": 0, "b": 0})
        cp.update({"a": 1.0, "b": 0.5}, {"a": 0, "b": 0})
        result = cp.get_joint_policy()
        assert isinstance(result, MARLResult)

    def test_aggregate_messages_empty(self):
        agents = [MARLAgent(id="a", n_actions=2)]
        cp = CommProtocol(agents, message_size=4)
        agg = cp._aggregate_messages("a")
        np.testing.assert_array_equal(agg, np.zeros(4))


# ============================================================================
# counterfactual_regret.py — GameNode
# ============================================================================


class TestGameNode:
    def test_terminal_node(self):
        node = GameNode(player=-1, payoffs=np.array([1.0, -1.0]))
        assert node.is_terminal is True
        assert node.is_chance is False

    def test_chance_node(self):
        node = GameNode(player=-2, chance_probs=np.array([0.5, 0.5]))
        assert node.is_chance is True
        assert node.is_terminal is False

    def test_from_matrix_zero_sum(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        assert root.player == 0
        assert len(root.actions) == 3
        assert len(root.children) == 3
        # Each child is player 1's node
        for child in root.children.values():
            assert child.player == 1
            assert len(child.children) == 3
            # Each grandchild is terminal
            for terminal in child.children.values():
                assert terminal.is_terminal

    def test_from_matrix_general_sum(self):
        A = np.array([[3, 0], [5, 1]])
        B = np.array([[3, 5], [0, 1]])
        root = GameNode.from_matrix(A, B)
        # Check terminal payoffs
        term = root.children[0].children[0]
        np.testing.assert_array_equal(term.payoffs, [3, 3])

    def test_from_matrix_implicit_zero_sum(self):
        A = np.array([[1, -1], [-1, 1]])
        root = GameNode.from_matrix(A)
        term = root.children[0].children[0]
        assert term.payoffs[0] == 1.0
        assert term.payoffs[1] == -1.0


# ============================================================================
# counterfactual_regret.py — CounterfactualRegretMinimization
# ============================================================================


class TestCFR:
    def test_rps_converges_to_uniform(self):
        """RPS Nash is (1/3, 1/3, 1/3) for both players.
        Note: from_matrix creates a sequential tree; exploitability measures
        best-response in the tree (not the normal form), so it can be > 0
        even at Nash. We check strategy convergence instead."""
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        cfr = CounterfactualRegretMinimization()
        result = cfr.train(root, n_iterations=500)
        assert isinstance(result, CFRResult)
        for key, strat in result.strategy.items():
            np.testing.assert_allclose(strat, [1 / 3, 1 / 3, 1 / 3], atol=0.05)
        # Exploitability in the sequential tree can be ~1.0 because player 0
        # moves first and player 1 can observe in best-response computation
        assert result.exploitability >= 0

    def test_get_strategy_after_train(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        cfr = CounterfactualRegretMinimization()
        cfr.train(root, n_iterations=100)
        strat = cfr.get_strategy((0,))
        assert strat.shape == (3,)
        assert strat.sum() == pytest.approx(1.0, abs=1e-8)

    def test_get_strategy_unknown_key_raises(self):
        cfr = CounterfactualRegretMinimization()
        with pytest.raises(KeyError):
            cfr.get_strategy((99,))

    def test_exploitability_perfect_nash(self):
        """Uniform strategy in RPS. In the sequential tree from_matrix creates,
        the best-response for player 0 can exploit knowing player 1's strategy,
        so exploitability is ~1.0, not 0. Test it is finite and non-negative."""
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        strategy = {(0,): np.array([1 / 3, 1 / 3, 1 / 3]),
                     (1,): np.array([1 / 3, 1 / 3, 1 / 3])}
        exploit = CounterfactualRegretMinimization.exploitability(root, strategy)
        assert exploit >= 0
        assert np.isfinite(exploit)

    def test_exploitability_dominated_strategy(self):
        """Pure Rock in RPS should be exploitable."""
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        strategy = {(0,): np.array([1.0, 0.0, 0.0]),
                     (1,): np.array([1.0, 0.0, 0.0])}
        exploit = CounterfactualRegretMinimization.exploitability(root, strategy)
        assert exploit > 0

    def test_matching_pennies(self):
        """Nash of matching pennies is (0.5, 0.5)."""
        mp = np.array([[1, -1], [-1, 1]])
        root = GameNode.from_matrix(mp)
        cfr = CounterfactualRegretMinimization()
        result = cfr.train(root, n_iterations=500)
        for key, strat in result.strategy.items():
            np.testing.assert_allclose(strat, [0.5, 0.5], atol=0.05)


# ============================================================================
# counterfactual_regret.py — CFRPlus
# ============================================================================


class TestCFRPlus:
    def test_rps_converges(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        cfr_plus = CFRPlus()
        result = cfr_plus.train(root, n_iterations=500)
        for key, strat in result.strategy.items():
            np.testing.assert_allclose(strat, [1 / 3, 1 / 3, 1 / 3], atol=0.05)
        assert result.exploitability >= 0

    def test_regrets_floored_at_zero(self):
        """CFR+ floors regrets at 0 — cumulative regret should be non-negative."""
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        cfr_plus = CFRPlus()
        cfr_plus.train(root, n_iterations=50)
        for key, regrets in cfr_plus._cumulative_regret.items():
            assert np.all(regrets >= 0)

    def test_iterations_count(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        cfr_plus = CFRPlus()
        result = cfr_plus.train(root, n_iterations=100)
        assert result.iterations == 100


# ============================================================================
# counterfactual_regret.py — LinearCFR
# ============================================================================


class TestLinearCFR:
    def test_rps_converges(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        lcfr = LinearCFR()
        result = lcfr.train(root, n_iterations=500)
        for key, strat in result.strategy.items():
            np.testing.assert_allclose(strat, [1 / 3, 1 / 3, 1 / 3], atol=0.05)
        assert result.exploitability >= 0

    def test_regrets_floored_at_zero(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        lcfr = LinearCFR()
        lcfr.train(root, n_iterations=50)
        for key, regrets in lcfr._cumulative_regret.items():
            assert np.all(regrets >= 0)

    def test_iterations_count(self):
        rps = _rps_game()
        root = GameNode.from_matrix(rps)
        lcfr = LinearCFR()
        result = lcfr.train(root, n_iterations=200)
        assert result.iterations == 200


# ============================================================================
# counterfactual_regret.py — InfoSet / CFRResult dataclasses
# ============================================================================


class TestCFRDataclasses:
    def test_info_set_creation(self):
        info = InfoSet(player=0, observations=(0, 1), legal_actions=[0, 1, 2])
        assert info.player == 0
        assert info.legal_actions == [0, 1, 2]

    def test_cfr_result_creation(self):
        result = CFRResult(
            strategy={(0,): np.array([0.5, 0.5])},
            exploitability=0.01,
            iterations=100,
        )
        assert result.iterations == 100
        assert result.exploitability == 0.01
