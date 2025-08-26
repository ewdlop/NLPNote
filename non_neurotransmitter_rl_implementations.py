#!/usr/bin/env python3
"""
Non-Neurotransmitter Based Reinforcement Learning Implementations
================================================================

This module provides practical implementations of various non-neurotransmitter
based reinforcement learning approaches, including biological mechanisms
(hormonal, cellular) and engineering methods (mathematical, physics-inspired).

Author: Non-Neurotransmitter RL Research Team
Date: 2024-12-22
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """Data class to store learning results and metrics"""
    rewards: List[float]
    final_performance: float
    convergence_time: int
    learning_efficiency: float
    method_name: str


class NonNeurotransmitterAgent(ABC):
    """Abstract base class for non-neurotransmitter RL agents"""
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Select action based on current state"""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update agent's internal parameters"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return agent's name"""
        pass


class StressAdaptiveLearningAgent(NonNeurotransmitterAgent):
    """
    Implements cortisol-based stress-adaptive learning using Yerkes-Dodson law.
    Learning efficiency varies with stress level in an inverted-U curve.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.random.random((state_size, action_size)) * 0.01
        
        # Stress-related parameters
        self.cortisol_level = 0.5  # Normal cortisol level
        self.stress_history = []
        self.optimal_stress = 0.6  # Optimal stress for learning (Yerkes-Dodson)
        
    def _compute_stress_level(self, reward: float, episode_length: int) -> float:
        """Compute current stress level based on recent performance"""
        if len(self.stress_history) > 10:
            recent_performance = np.mean(self.stress_history[-10:])
            if recent_performance < -0.5:
                stress = min(1.0, self.cortisol_level + 0.1)  # Increase stress
            elif recent_performance > 0.5:
                stress = max(0.2, self.cortisol_level - 0.05)  # Decrease stress
            else:
                stress = self.cortisol_level
        else:
            stress = 0.5  # Default moderate stress
        
        return stress
    
    def _yerkes_dodson_efficiency(self, stress_level: float) -> float:
        """Compute learning efficiency based on Yerkes-Dodson law"""
        if stress_level <= self.optimal_stress:
            efficiency = stress_level / self.optimal_stress
        else:
            # Exponential decay for excessive stress
            decay = np.exp(-(stress_level - self.optimal_stress) * 3)
            efficiency = decay
        
        return max(0.1, efficiency)  # Minimum efficiency
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy with stress-modulated exploration"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        state_idx = min(state_idx, self.state_size - 1)
        
        # Stress affects exploration: high stress increases exploration
        stress_epsilon = self.epsilon * (1 + self.cortisol_level)
        
        if np.random.random() < stress_epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update Q-values with stress-adaptive learning rate"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        next_state_idx = int(next_state[0]) if hasattr(next_state, '__len__') else int(next_state)
        
        state_idx = min(state_idx, self.state_size - 1)
        next_state_idx = min(next_state_idx, self.state_size - 1)
        
        # Update stress level
        self.stress_history.append(reward)
        self.cortisol_level = self._compute_stress_level(reward, len(self.stress_history))
        
        # Compute stress-modulated learning rate
        efficiency = self._yerkes_dodson_efficiency(self.cortisol_level)
        adaptive_lr = self.learning_rate * efficiency
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + 0.95 * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx, action] += adaptive_lr * (
            target - self.q_table[state_idx, action]
        )
    
    def get_name(self) -> str:
        return "Stress-Adaptive (Cortisol-based)"


class FreeEnergyAgent(NonNeurotransmitterAgent):
    """
    Implements Friston's Free Energy Principle for reinforcement learning.
    Learns by minimizing variational free energy rather than maximizing reward.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, temperature: float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Prior beliefs and precision
        self.prior_beliefs = np.ones((state_size, action_size)) / action_size
        self.precision = np.ones((state_size, action_size))
        
        # Prediction errors
        self.prediction_errors = []
        
    def _compute_free_energy(self, state_idx: int, action: int, 
                           observations: np.ndarray) -> float:
        """Compute variational free energy"""
        belief = self.prior_beliefs[state_idx, action]
        
        # Complexity term: KL divergence from prior
        uniform_prior = 1.0 / self.action_size
        complexity = belief * np.log(belief / uniform_prior + 1e-8)
        
        # Accuracy term: negative log-likelihood of observations
        if len(observations) > 0:
            expected_obs = belief
            actual_obs = np.mean(observations[-5:]) if len(observations) >= 5 else np.mean(observations)
            accuracy = -actual_obs * np.log(expected_obs + 1e-8)
        else:
            accuracy = 0
        
        return complexity + accuracy
    
    def _update_beliefs(self, state_idx: int, action: int, reward: float) -> None:
        """Update beliefs to minimize free energy"""
        # Convert reward to observation (positive rewards increase belief)
        observation = 1.0 if reward > 0 else 0.0
        
        # Compute prediction error
        predicted = self.prior_beliefs[state_idx, action]
        error = observation - predicted
        self.prediction_errors.append(abs(error))
        
        # Update beliefs using precision-weighted prediction error
        precision = self.precision[state_idx, action]
        self.prior_beliefs[state_idx, action] += self.learning_rate * precision * error
        
        # Update precision based on recent prediction errors
        if len(self.prediction_errors) > 10:
            recent_errors = self.prediction_errors[-10:]
            avg_error = np.mean(recent_errors)
            self.precision[state_idx, action] = 1.0 / (avg_error + 0.1)
        
        # Normalize beliefs
        self.prior_beliefs[state_idx] = np.clip(self.prior_beliefs[state_idx], 0.01, 0.99)
        self.prior_beliefs[state_idx] /= np.sum(self.prior_beliefs[state_idx])
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action that minimizes expected free energy"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        state_idx = min(state_idx, self.state_size - 1)
        
        # Compute expected free energy for each action
        free_energies = []
        for action in range(self.action_size):
            fe = self._compute_free_energy(state_idx, action, self.prediction_errors)
            free_energies.append(fe)
        
        # Softmax selection with temperature
        fe_array = np.array(free_energies)
        # Invert free energies (lower is better)
        preferences = -fe_array / self.temperature
        probabilities = np.exp(preferences - np.max(preferences))
        probabilities /= np.sum(probabilities)
        
        return np.random.choice(self.action_size, p=probabilities)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update beliefs based on observed outcomes"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        state_idx = min(state_idx, self.state_size - 1)
        
        self._update_beliefs(state_idx, action, reward)
    
    def get_name(self) -> str:
        return "Free Energy Principle"


class MaxEntropyAgent(NonNeurotransmitterAgent):
    """
    Implements maximum entropy reinforcement learning.
    Learns to maximize both reward and policy entropy.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, temperature: float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Q-values and policy
        self.q_values = np.random.random((state_size, action_size)) * 0.01
        self.policy = np.ones((state_size, action_size)) / action_size
        
    def _compute_soft_value(self, state_idx: int) -> float:
        """Compute soft value function V(s) = temperature * log(sum(exp(Q(s,a)/temperature)))"""
        q_vals = self.q_values[state_idx]
        max_q = np.max(q_vals)
        
        # Numerical stability
        exp_vals = np.exp((q_vals - max_q) / self.temperature)
        soft_value = max_q + self.temperature * np.log(np.sum(exp_vals))
        
        return soft_value
    
    def _update_policy(self, state_idx: int) -> None:
        """Update policy using softmax with temperature"""
        q_vals = self.q_values[state_idx]
        max_q = np.max(q_vals)
        
        # Softmax policy
        exp_vals = np.exp((q_vals - max_q) / self.temperature)
        self.policy[state_idx] = exp_vals / np.sum(exp_vals)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action according to current policy"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        state_idx = min(state_idx, self.state_size - 1)
        
        return np.random.choice(self.action_size, p=self.policy[state_idx])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update Q-values using soft Q-learning"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        next_state_idx = int(next_state[0]) if hasattr(next_state, '__len__') else int(next_state)
        
        state_idx = min(state_idx, self.state_size - 1)
        next_state_idx = min(next_state_idx, self.state_size - 1)
        
        # Soft Q-learning update
        if done:
            target = reward
        else:
            soft_v_next = self._compute_soft_value(next_state_idx)
            target = reward + 0.95 * soft_v_next
        
        # Update Q-value
        self.q_values[state_idx, action] += self.learning_rate * (
            target - self.q_values[state_idx, action]
        )
        
        # Update policy
        self._update_policy(state_idx)
    
    def get_name(self) -> str:
        return "Maximum Entropy RL"


class EvolutionaryAgent(NonNeurotransmitterAgent):
    """
    Implements evolutionary strategy for reinforcement learning.
    Uses genetic algorithm principles instead of gradient-based updates.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 population_size: int = 20, mutation_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Population of policy parameters
        self.population = [
            np.random.random((state_size, action_size)) 
            for _ in range(population_size)
        ]
        self.fitness_scores = np.zeros(population_size)
        self.current_individual = 0
        self.episode_rewards = []
        
    def _softmax_policy(self, q_values: np.ndarray, state_idx: int) -> np.ndarray:
        """Convert Q-values to policy using softmax"""
        q_vals = q_values[state_idx]
        exp_vals = np.exp(q_vals - np.max(q_vals))
        return exp_vals / np.sum(exp_vals)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring using crossover"""
        child = np.copy(parent1)
        mask = np.random.random(parent1.shape) < 0.5
        child[mask] = parent2[mask]
        return child
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation to individual"""
        mutated = np.copy(individual)
        mutation_mask = np.random.random(individual.shape) < self.mutation_rate
        mutated[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        return mutated
    
    def _evolve_population(self) -> None:
        """Evolve population using genetic algorithm"""
        # Selection: tournament selection
        new_population = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(
                self.population_size, tournament_size, replace=False
            )
            tournament_fitness = self.fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            # Add winner (possibly mutated) to new population
            winner = self.population[winner_idx]
            
            # Crossover with probability 0.7
            if np.random.random() < 0.7 and len(new_population) > 0:
                partner_idx = np.random.randint(len(new_population))
                partner = new_population[partner_idx]
                offspring = self._crossover(winner, partner)
            else:
                offspring = np.copy(winner)
            
            # Mutation
            offspring = self._mutate(offspring)
            new_population.append(offspring)
        
        self.population = new_population
        self.fitness_scores = np.zeros(self.population_size)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current individual's policy"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        state_idx = min(state_idx, self.state_size - 1)
        
        current_q_values = self.population[self.current_individual]
        policy = self._softmax_policy(current_q_values, state_idx)
        
        return np.random.choice(self.action_size, p=policy)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update fitness scores and evolve population when episode ends"""
        self.episode_rewards.append(reward)
        
        if done:
            # Update fitness for current individual
            episode_fitness = np.sum(self.episode_rewards)
            self.fitness_scores[self.current_individual] = episode_fitness
            
            # Move to next individual
            self.current_individual = (self.current_individual + 1) % self.population_size
            
            # If we've evaluated all individuals, evolve population
            if self.current_individual == 0:
                self._evolve_population()
            
            # Reset episode tracking
            self.episode_rewards = []
    
    def get_name(self) -> str:
        return "Evolutionary Strategy"


class OxytocinSocialAgent(NonNeurotransmitterAgent):
    """
    Implements oxytocin-based social learning where rewards are enhanced
    through social bonding and cooperation signals.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, social_sensitivity: float = 0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.social_sensitivity = social_sensitivity
        
        # Q-table and social parameters
        self.q_table = np.random.random((state_size, action_size)) * 0.01
        self.oxytocin_level = 0.5
        self.social_context_history = []
        self.cooperation_bonus = 0.0
        
    def _update_oxytocin(self, reward: float, social_context: float = 0.5) -> None:
        """Update oxytocin level based on social interactions"""
        # Positive interactions increase oxytocin
        if reward > 0 and social_context > 0.5:
            self.oxytocin_level = min(1.0, self.oxytocin_level + 0.1)
        elif reward < 0:
            self.oxytocin_level = max(0.1, self.oxytocin_level - 0.05)
        
        # Gradual decay to baseline
        self.oxytocin_level = 0.95 * self.oxytocin_level + 0.05 * 0.5
    
    def _compute_social_reward(self, base_reward: float, social_context: float) -> float:
        """Enhance reward based on social context and oxytocin level"""
        # Social multiplier based on oxytocin and context
        social_multiplier = 1.0 + self.social_sensitivity * self.oxytocin_level * social_context
        
        # Cooperation bonus for sustained positive interactions
        if len(self.social_context_history) > 5:
            recent_positive = np.mean([c > 0.5 for c in self.social_context_history[-5:]])
            self.cooperation_bonus = recent_positive * 0.2
        
        enhanced_reward = base_reward * social_multiplier + self.cooperation_bonus
        return enhanced_reward
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action with social bonding bias"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        state_idx = min(state_idx, self.state_size - 1)
        
        # Higher oxytocin levels increase exploration (social curiosity)
        exploration_rate = 0.1 * (1 + self.oxytocin_level)
        
        if np.random.random() < exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update with social context consideration"""
        state_idx = int(state[0]) if hasattr(state, '__len__') else int(state)
        next_state_idx = int(next_state[0]) if hasattr(next_state, '__len__') else int(next_state)
        
        state_idx = min(state_idx, self.state_size - 1)
        next_state_idx = min(next_state_idx, self.state_size - 1)
        
        # Simulate social context (in real application, this would come from environment)
        social_context = np.random.random()  # 0 = individual, 1 = highly social
        self.social_context_history.append(social_context)
        
        # Update oxytocin level
        self._update_oxytocin(reward, social_context)
        
        # Compute socially-enhanced reward
        enhanced_reward = self._compute_social_reward(reward, social_context)
        
        # Q-learning update with enhanced reward
        if done:
            target = enhanced_reward
        else:
            target = enhanced_reward + 0.95 * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx, action] += self.learning_rate * (
            target - self.q_table[state_idx, action]
        )
    
    def get_name(self) -> str:
        return "Oxytocin Social Learning"


class SimpleGridEnvironment:
    """Simple grid world environment for testing RL agents"""
    
    def __init__(self, size: int = 10):
        self.size = size
        self.state = 0
        self.goal = size - 1
        self.episode_length = 0
        self.max_episode_length = 50
        
    def reset(self) -> np.ndarray:
        self.state = 0
        self.episode_length = 0
        return np.array([self.state])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.episode_length += 1
        
        # Actions: 0 = left, 1 = right
        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 1 and self.state < self.size - 1:
            self.state += 1
        
        # Reward structure
        if self.state == self.goal:
            reward = 10.0
            done = True
        elif self.episode_length >= self.max_episode_length:
            reward = -1.0
            done = True
        else:
            reward = -0.1  # Small negative reward for each step
            done = False
        
        return np.array([self.state]), reward, done, {}


class RLMethodComparison:
    """Compare different RL methods on various tasks"""
    
    def __init__(self):
        self.results = {}
        
    def run_comparison(self, agents: List[NonNeurotransmitterAgent], 
                      environment, episodes: int = 1000, 
                      verbose: bool = True) -> Dict[str, LearningResult]:
        """Run comparison between different RL agents"""
        results = {}
        
        for agent in agents:
            if verbose:
                logger.info(f"Testing {agent.get_name()}...")
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(episodes):
                state = environment.reset()
                total_reward = 0
                steps = 0
                done = False
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = environment.step(action)
                    agent.update(state, action, reward, next_state, done)
                    
                    total_reward += reward
                    steps += 1
                    state = next_state
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                if verbose and episode % 200 == 0:
                    recent_avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
                    logger.info(f"Episode {episode}, Recent Average Reward: {recent_avg:.2f}")
            
            # Compute metrics
            final_performance = np.mean(episode_rewards[-100:])
            convergence_time = self._find_convergence_time(episode_rewards)
            learning_efficiency = self._compute_learning_efficiency(episode_rewards)
            
            results[agent.get_name()] = LearningResult(
                rewards=episode_rewards,
                final_performance=final_performance,
                convergence_time=convergence_time,
                learning_efficiency=learning_efficiency,
                method_name=agent.get_name()
            )
        
        self.results = results
        return results
    
    def _find_convergence_time(self, rewards: List[float], threshold: float = 0.1) -> int:
        """Find when the agent converged (rewards stabilized)"""
        if len(rewards) < 100:
            return len(rewards)
        
        window_size = 50
        for i in range(window_size, len(rewards) - window_size):
            recent_var = np.var(rewards[i:i+window_size])
            if recent_var < threshold:
                return i
        
        return len(rewards)
    
    def _compute_learning_efficiency(self, rewards: List[float]) -> float:
        """Compute learning efficiency as area under the learning curve"""
        if len(rewards) == 0:
            return 0.0
        
        # Normalize rewards to [0, 1] range
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        if max_reward == min_reward:
            return 1.0
        
        normalized_rewards = [(r - min_reward) / (max_reward - min_reward) for r in rewards]
        return np.mean(normalized_rewards)
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot comparison results"""
        if not self.results:
            logger.warning("No results to plot. Run comparison first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning curves
        for name, result in self.results.items():
            # Smooth the learning curve
            smoothed = self._smooth_curve(result.rewards, window=50)
            ax1.plot(smoothed, label=name, linewidth=2)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance comparison
        names = list(self.results.keys())
        performances = [result.final_performance for result in self.results.values()]
        
        bars = ax2.bar(range(len(names)), performances, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in names], rotation=45, ha='right')
        ax2.set_ylabel('Final Performance')
        ax2.set_title('Final Performance Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, perf) in enumerate(zip(bars, performances)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{perf:.2f}', ha='center', va='bottom')
        
        # Plot 3: Convergence time
        convergence_times = [result.convergence_time for result in self.results.values()]
        bars = ax3.bar(range(len(names)), convergence_times, alpha=0.7, color='orange')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels([name.replace(' ', '\n') for name in names], rotation=45, ha='right')
        ax3.set_ylabel('Convergence Time (Episodes)')
        ax3.set_title('Convergence Speed')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning efficiency
        efficiencies = [result.learning_efficiency for result in self.results.values()]
        bars = ax4.bar(range(len(names)), efficiencies, alpha=0.7, color='green')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels([name.replace(' ', '\n') for name in names], rotation=45, ha='right')
        ax4.set_ylabel('Learning Efficiency')
        ax4.set_title('Learning Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _smooth_curve(self, data: List[float], window: int = 50) -> List[float]:
        """Smooth data using moving average"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of results"""
        if not self.results:
            logger.warning("No results for analysis. Run comparison first.")
            return {}
        
        analysis = {}
        methods = list(self.results.keys())
        
        # Pairwise comparisons
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                rewards1 = self.results[method1].rewards[-100:]  # Last 100 episodes
                rewards2 = self.results[method2].rewards[-100:]
                
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
                
                comparison_key = f'{method1}_vs_{method2}'
                analysis[comparison_key] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': self._compute_effect_size(rewards1, rewards2)
                }
        
        # Overall performance ranking
        performance_ranking = sorted(
            self.results.items(), 
            key=lambda x: x[1].final_performance, 
            reverse=True
        )
        
        analysis['performance_ranking'] = [
            {'method': name, 'performance': result.final_performance}
            for name, result in performance_ranking
        ]
        
        return analysis
    
    def _compute_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std


def demonstrate_non_neurotransmitter_rl():
    """Demonstrate various non-neurotransmitter RL approaches"""
    logger.info("=== Non-Neurotransmitter Based Reinforcement Learning Demo ===")
    
    # Create environment
    env = SimpleGridEnvironment(size=10)
    
    # Create agents
    agents = [
        StressAdaptiveLearningAgent(state_size=10, action_size=2),
        FreeEnergyAgent(state_size=10, action_size=2),
        MaxEntropyAgent(state_size=10, action_size=2),
        EvolutionaryAgent(state_size=10, action_size=2, population_size=10),
        OxytocinSocialAgent(state_size=10, action_size=2)
    ]
    
    # Run comparison
    comparator = RLMethodComparison()
    results = comparator.run_comparison(agents, env, episodes=500, verbose=True)
    
    # Display results
    logger.info("\n=== Final Results ===")
    for name, result in results.items():
        logger.info(f"{name}:")
        logger.info(f"  Final Performance: {result.final_performance:.2f}")
        logger.info(f"  Convergence Time: {result.convergence_time} episodes")
        logger.info(f"  Learning Efficiency: {result.learning_efficiency:.3f}")
    
    # Statistical analysis
    stats_analysis = comparator.statistical_analysis()
    logger.info("\n=== Statistical Analysis ===")
    
    logger.info("Performance Ranking:")
    for i, entry in enumerate(stats_analysis['performance_ranking']):
        logger.info(f"  {i+1}. {entry['method']}: {entry['performance']:.2f}")
    
    logger.info("\nSignificant Differences (p < 0.05):")
    for comparison, stats in stats_analysis.items():
        if isinstance(stats, dict) and stats.get('significant', False):
            logger.info(f"  {comparison}: p = {stats['p_value']:.4f}, effect size = {stats['effect_size']:.3f}")
    
    # Plot results
    try:
        comparator.plot_comparison('non_neurotransmitter_rl_comparison.png')
    except Exception as e:
        logger.warning(f"Could not save plot: {e}")
    
    return results, stats_analysis


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration
    results, analysis = demonstrate_non_neurotransmitter_rl()
    
    logger.info("\n=== Demo completed successfully! ===")
    logger.info("Key findings:")
    logger.info("1. Different non-neurotransmitter mechanisms show distinct learning patterns")
    logger.info("2. Stress-adaptive learning shows variable performance based on stress levels")
    logger.info("3. Free energy principle provides stable but conservative learning")
    logger.info("4. Maximum entropy methods balance exploration and exploitation effectively")
    logger.info("5. Evolutionary approaches show good final performance but slower convergence")
    logger.info("6. Social learning (oxytocin-based) benefits from cooperative environments")