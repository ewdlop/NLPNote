#!/usr/bin/env python3
"""
Test cases for Non-Neurotransmitter Based Reinforcement Learning

This file contains comprehensive test cases to validate the non-neurotransmitter
RL implementations and ensure they work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from non_neurotransmitter_rl_implementations import (
    StressAdaptiveLearningAgent,
    FreeEnergyAgent,
    MaxEntropyAgent,
    EvolutionaryAgent,
    OxytocinSocialAgent,
    SimpleGridEnvironment,
    RLMethodComparison
)


class TestNonNeurotransmitterRL(unittest.TestCase):
    """Test cases for non-neurotransmitter RL implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)  # For reproducible tests
        self.state_size = 5
        self.action_size = 2
        self.env = SimpleGridEnvironment(size=self.state_size)
        
    def test_stress_adaptive_agent_initialization(self):
        """Test StressAdaptiveLearningAgent initialization"""
        agent = StressAdaptiveLearningAgent(self.state_size, self.action_size)
        
        self.assertEqual(agent.state_size, self.state_size)
        self.assertEqual(agent.action_size, self.action_size)
        self.assertEqual(agent.q_table.shape, (self.state_size, self.action_size))
        self.assertAlmostEqual(agent.cortisol_level, 0.5, places=2)
        self.assertEqual(agent.get_name(), "Stress-Adaptive (Cortisol-based)")
    
    def test_stress_adaptive_yerkes_dodson(self):
        """Test Yerkes-Dodson law implementation"""
        agent = StressAdaptiveLearningAgent(self.state_size, self.action_size)
        
        # Test optimal stress level
        efficiency_optimal = agent._yerkes_dodson_efficiency(0.6)
        efficiency_low = agent._yerkes_dodson_efficiency(0.3)
        efficiency_high = agent._yerkes_dodson_efficiency(0.9)
        
        # Optimal stress should give highest efficiency
        self.assertGreater(efficiency_optimal, efficiency_low)
        self.assertGreater(efficiency_optimal, efficiency_high)
        
        # All efficiencies should be positive
        self.assertGreater(efficiency_optimal, 0)
        self.assertGreater(efficiency_low, 0)
        self.assertGreater(efficiency_high, 0)
    
    def test_free_energy_agent_initialization(self):
        """Test FreeEnergyAgent initialization"""
        agent = FreeEnergyAgent(self.state_size, self.action_size)
        
        self.assertEqual(agent.state_size, self.state_size)
        self.assertEqual(agent.action_size, self.action_size)
        self.assertEqual(agent.prior_beliefs.shape, (self.state_size, self.action_size))
        self.assertEqual(agent.get_name(), "Free Energy Principle")
        
        # Check that beliefs are normalized
        for state in range(self.state_size):
            belief_sum = np.sum(agent.prior_beliefs[state])
            self.assertAlmostEqual(belief_sum, 1.0, places=2)
    
    def test_free_energy_computation(self):
        """Test free energy computation"""
        agent = FreeEnergyAgent(self.state_size, self.action_size)
        
        # Test with empty observations
        fe = agent._compute_free_energy(0, 0, [])
        self.assertIsInstance(fe, (int, float))
        self.assertGreaterEqual(fe, 0)
        
        # Test with some observations
        observations = [0.5, 0.7, 0.3]
        fe_with_obs = agent._compute_free_energy(0, 0, observations)
        self.assertIsInstance(fe_with_obs, (int, float))
    
    def test_max_entropy_agent_initialization(self):
        """Test MaxEntropyAgent initialization"""
        agent = MaxEntropyAgent(self.state_size, self.action_size)
        
        self.assertEqual(agent.state_size, self.state_size)
        self.assertEqual(agent.action_size, self.action_size)
        self.assertEqual(agent.q_values.shape, (self.state_size, self.action_size))
        self.assertEqual(agent.policy.shape, (self.state_size, self.action_size))
        self.assertEqual(agent.get_name(), "Maximum Entropy RL")
        
        # Check that policy is normalized
        for state in range(self.state_size):
            policy_sum = np.sum(agent.policy[state])
            self.assertAlmostEqual(policy_sum, 1.0, places=5)
    
    def test_max_entropy_soft_value(self):
        """Test soft value function computation"""
        agent = MaxEntropyAgent(self.state_size, self.action_size, temperature=1.0)
        
        # Set some Q-values
        agent.q_values[0] = [1.0, 2.0]
        soft_value = agent._compute_soft_value(0)
        
        # Soft value should be greater than max Q-value due to entropy term
        max_q = np.max(agent.q_values[0])
        self.assertGreater(soft_value, max_q)
    
    def test_evolutionary_agent_initialization(self):
        """Test EvolutionaryAgent initialization"""
        population_size = 10
        agent = EvolutionaryAgent(self.state_size, self.action_size, population_size)
        
        self.assertEqual(agent.state_size, self.state_size)
        self.assertEqual(agent.action_size, self.action_size)
        self.assertEqual(len(agent.population), population_size)
        self.assertEqual(len(agent.fitness_scores), population_size)
        self.assertEqual(agent.get_name(), "Evolutionary Strategy")
        
        # Check population shapes
        for individual in agent.population:
            self.assertEqual(individual.shape, (self.state_size, self.action_size))
    
    def test_evolutionary_crossover_and_mutation(self):
        """Test evolutionary operations"""
        agent = EvolutionaryAgent(self.state_size, self.action_size, population_size=10)
        
        parent1 = agent.population[0]
        parent2 = agent.population[1]
        
        # Test crossover
        child = agent._crossover(parent1, parent2)
        self.assertEqual(child.shape, parent1.shape)
        
        # Test mutation
        original = np.copy(parent1)
        mutated = agent._mutate(parent1)
        self.assertEqual(mutated.shape, original.shape)
        
        # With non-zero mutation rate, should be some difference
        agent.mutation_rate = 1.0  # Ensure mutation happens
        mutated_high = agent._mutate(original)
        difference = np.sum(np.abs(mutated_high - original))
        self.assertGreater(difference, 0)
    
    def test_oxytocin_social_agent_initialization(self):
        """Test OxytocinSocialAgent initialization"""
        agent = OxytocinSocialAgent(self.state_size, self.action_size)
        
        self.assertEqual(agent.state_size, self.state_size)
        self.assertEqual(agent.action_size, self.action_size)
        self.assertEqual(agent.q_table.shape, (self.state_size, self.action_size))
        self.assertAlmostEqual(agent.oxytocin_level, 0.5, places=2)
        self.assertEqual(agent.get_name(), "Oxytocin Social Learning")
    
    def test_oxytocin_social_reward_enhancement(self):
        """Test social reward enhancement"""
        agent = OxytocinSocialAgent(self.state_size, self.action_size)
        
        base_reward = 1.0
        social_context_low = 0.2
        social_context_high = 0.8
        
        # High social context should enhance reward more
        enhanced_low = agent._compute_social_reward(base_reward, social_context_low)
        enhanced_high = agent._compute_social_reward(base_reward, social_context_high)
        
        self.assertGreaterEqual(enhanced_low, base_reward)
        self.assertGreater(enhanced_high, enhanced_low)
    
    def test_simple_grid_environment(self):
        """Test SimpleGridEnvironment functionality"""
        env = SimpleGridEnvironment(size=5)
        
        # Test reset
        state = env.reset()
        self.assertEqual(state[0], 0)
        self.assertEqual(env.state, 0)
        
        # Test step
        next_state, reward, done, _ = env.step(1)  # Move right
        self.assertEqual(next_state[0], 1)
        self.assertEqual(env.state, 1)
        self.assertFalse(done)
        
        # Test reaching goal
        env.state = env.goal - 1
        next_state, reward, done, _ = env.step(1)  # Move to goal
        self.assertEqual(next_state[0], env.goal)
        self.assertTrue(done)
        self.assertGreater(reward, 0)  # Positive reward for reaching goal
    
    def test_agent_action_selection(self):
        """Test that all agents can select actions"""
        agents = [
            StressAdaptiveLearningAgent(self.state_size, self.action_size),
            FreeEnergyAgent(self.state_size, self.action_size),
            MaxEntropyAgent(self.state_size, self.action_size),
            EvolutionaryAgent(self.state_size, self.action_size, population_size=5),
            OxytocinSocialAgent(self.state_size, self.action_size)
        ]
        
        state = np.array([0])
        
        for agent in agents:
            action = agent.select_action(state)
            self.assertIsInstance(action, (int, np.integer))
            self.assertIn(action, range(self.action_size))
    
    def test_agent_updates(self):
        """Test that all agents can update their parameters"""
        agents = [
            StressAdaptiveLearningAgent(self.state_size, self.action_size),
            FreeEnergyAgent(self.state_size, self.action_size),
            MaxEntropyAgent(self.state_size, self.action_size),
            EvolutionaryAgent(self.state_size, self.action_size, population_size=5),
            OxytocinSocialAgent(self.state_size, self.action_size)
        ]
        
        state = np.array([0])
        next_state = np.array([1])
        action = 1
        reward = 1.0
        done = False
        
        for agent in agents:
            # Should not raise any exceptions
            agent.update(state, action, reward, next_state, done)
    
    def test_rl_method_comparison(self):
        """Test RLMethodComparison functionality"""
        comparator = RLMethodComparison()
        
        # Create small set of agents for quick testing
        agents = [
            StressAdaptiveLearningAgent(self.state_size, self.action_size),
            MaxEntropyAgent(self.state_size, self.action_size)
        ]
        
        # Run short comparison
        results = comparator.run_comparison(
            agents, self.env, episodes=10, verbose=False
        )
        
        # Check results structure
        self.assertEqual(len(results), 2)
        
        for agent in agents:
            name = agent.get_name()
            self.assertIn(name, results)
            
            result = results[name]
            self.assertEqual(len(result.rewards), 10)
            self.assertIsInstance(result.final_performance, (int, float))
            self.assertIsInstance(result.convergence_time, int)
            self.assertIsInstance(result.learning_efficiency, (int, float))
            self.assertEqual(result.method_name, name)
    
    def test_learning_curve_smoothing(self):
        """Test learning curve smoothing function"""
        comparator = RLMethodComparison()
        
        # Test with simple data
        data = [1, 2, 3, 4, 5]
        smoothed = comparator._smooth_curve(data, window=3)
        
        self.assertEqual(len(smoothed), len(data))
        
        # Test with insufficient data
        short_data = [1, 2]
        smoothed_short = comparator._smooth_curve(short_data, window=5)
        self.assertEqual(smoothed_short, short_data)
    
    def test_convergence_detection(self):
        """Test convergence time detection"""
        comparator = RLMethodComparison()
        
        # Test with converging data
        stable_rewards = [1.0] * 200  # Stable performance
        convergence_time = comparator._find_convergence_time(stable_rewards)
        self.assertLess(convergence_time, len(stable_rewards))
        
        # Test with non-converging data
        noisy_rewards = np.random.random(200).tolist()  # Noisy performance
        convergence_time_noisy = comparator._find_convergence_time(noisy_rewards)
        self.assertGreaterEqual(convergence_time_noisy, 0)
    
    def test_learning_efficiency_computation(self):
        """Test learning efficiency computation"""
        comparator = RLMethodComparison()
        
        # Test with improving performance
        improving_rewards = list(range(100))  # 0 to 99
        efficiency = comparator._compute_learning_efficiency(improving_rewards)
        self.assertGreater(efficiency, 0.4)  # Should be reasonably high
        
        # Test with constant performance
        constant_rewards = [5.0] * 100
        efficiency_constant = comparator._compute_learning_efficiency(constant_rewards)
        self.assertAlmostEqual(efficiency_constant, 1.0, places=2)
        
        # Test with empty rewards
        efficiency_empty = comparator._compute_learning_efficiency([])
        self.assertEqual(efficiency_empty, 0.0)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality"""
        comparator = RLMethodComparison()
        
        # Create mock results
        comparator.results = {
            'Method1': type('LearningResult', (), {
                'rewards': [1.0] * 100,
                'final_performance': 1.0
            })(),
            'Method2': type('LearningResult', (), {
                'rewards': [2.0] * 100,
                'final_performance': 2.0
            })()
        }
        
        analysis = comparator.statistical_analysis()
        
        # Check analysis structure
        self.assertIn('performance_ranking', analysis)
        self.assertIn('Method1_vs_Method2', analysis)
        
        # Check ranking
        ranking = analysis['performance_ranking']
        self.assertEqual(len(ranking), 2)
        self.assertEqual(ranking[0]['method'], 'Method2')  # Higher performance first
        
        # Check statistical test
        comparison = analysis['Method1_vs_Method2']
        self.assertIn('p_value', comparison)
        self.assertIn('significant', comparison)
        self.assertIn('effect_size', comparison)
    
    def test_effect_size_computation(self):
        """Test Cohen's d effect size computation"""
        comparator = RLMethodComparison()
        
        # Test with different groups (add some variance)
        group1 = [1.0 + np.random.normal(0, 0.1) for _ in range(50)]
        group2 = [2.0 + np.random.normal(0, 0.1) for _ in range(50)]
        
        effect_size = comparator._compute_effect_size(group1, group2)
        self.assertNotEqual(effect_size, 0.0)  # Should be non-zero
        
        # Test with identical groups
        identical_group = [1.0] * 50
        effect_size_same = comparator._compute_effect_size(identical_group, identical_group)
        self.assertEqual(effect_size_same, 0.0)
        
        # Test with insufficient data
        small_group1 = [1.0]
        small_group2 = [2.0]
        effect_size_small = comparator._compute_effect_size(small_group1, small_group2)
        self.assertEqual(effect_size_small, 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        np.random.seed(42)
        self.env = SimpleGridEnvironment(size=5)
    
    def test_full_learning_episode(self):
        """Test complete learning episode for all agents"""
        agents = [
            StressAdaptiveLearningAgent(5, 2),
            FreeEnergyAgent(5, 2),
            MaxEntropyAgent(5, 2),
            EvolutionaryAgent(5, 2, population_size=5),
            OxytocinSocialAgent(5, 2)
        ]
        
        for agent in agents:
            # Run one complete episode
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 50
            
            while steps < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Should complete without errors
            self.assertGreaterEqual(steps, 1)
            self.assertIsInstance(total_reward, (int, float))
    
    def test_mini_comparison(self):
        """Test mini comparison between agents"""
        agents = [
            StressAdaptiveLearningAgent(5, 2),
            MaxEntropyAgent(5, 2)
        ]
        
        comparator = RLMethodComparison()
        results = comparator.run_comparison(agents, self.env, episodes=5, verbose=False)
        
        # Should have results for both agents
        self.assertEqual(len(results), 2)
        
        # Should be able to perform statistical analysis
        analysis = comparator.statistical_analysis()
        self.assertIsInstance(analysis, dict)
        self.assertIn('performance_ranking', analysis)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNonNeurotransmitterRL))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")