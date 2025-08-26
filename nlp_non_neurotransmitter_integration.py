#!/usr/bin/env python3
"""
Integration Example: Non-Neurotransmitter RL in NLP Applications

This example demonstrates how non-neurotransmitter based reinforcement learning
can be applied to natural language processing tasks, connecting to the broader
theme of the NLPNote repository.
"""

import numpy as np
from typing import List, Dict, Any
import random

# Import our non-neurotransmitter RL implementations
from non_neurotransmitter_rl_implementations import (
    StressAdaptiveLearningAgent,
    OxytocinSocialAgent,
    FreeEnergyAgent,
    NonNeurotransmitterAgent
)


class TextClassificationEnvironment:
    """
    Simple text classification environment for demonstrating 
    non-neurotransmitter RL in NLP tasks.
    """
    
    def __init__(self):
        # Simplified text classification task: sentiment analysis
        self.positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love"]
        self.negative_words = ["bad", "terrible", "awful", "hate", "horrible", "disgusting"]
        
        self.current_text = ""
        self.true_label = 0  # 0 = negative, 1 = positive
        
    def reset(self) -> np.ndarray:
        """Generate a new text sample"""
        # Randomly choose positive or negative
        self.true_label = random.randint(0, 1)
        
        if self.true_label == 1:  # Positive
            words = random.sample(self.positive_words, 2)
            self.current_text = f"This is {words[0]} and {words[1]}"
        else:  # Negative
            words = random.sample(self.negative_words, 2)
            self.current_text = f"This is {words[0]} and {words[1]}"
        
        # Return simple features: word count, positive word count, negative word count
        pos_count = sum(1 for word in self.positive_words if word in self.current_text.lower())
        neg_count = sum(1 for word in self.negative_words if word in self.current_text.lower())
        word_count = len(self.current_text.split())
        
        return np.array([word_count, pos_count, neg_count])
    
    def step(self, action: int) -> tuple:
        """
        Args:
            action: 0 = classify as negative, 1 = classify as positive
        
        Returns:
            next_state, reward, done, info
        """
        # Reward based on correct classification
        if action == self.true_label:
            reward = 1.0  # Correct classification
        else:
            reward = -1.0  # Incorrect classification
        
        done = True  # Single-step episodes for classification
        
        # Return dummy next state (episode is done)
        return np.array([0, 0, 0]), reward, done, {"text": self.current_text, "true_label": self.true_label}


class NLPRLAgent(NonNeurotransmitterAgent):
    """
    Adapter class to use non-neurotransmitter RL agents for NLP tasks.
    Maps continuous features to discrete states.
    """
    
    def __init__(self, base_agent: NonNeurotransmitterAgent, feature_bins: int = 5):
        self.base_agent = base_agent
        self.feature_bins = feature_bins
        
        # State mapping: discretize continuous features
        self.max_features = np.array([20, 5, 5])  # Max expected values for each feature
        
    def _discretize_state(self, continuous_state: np.ndarray) -> int:
        """Map continuous features to discrete state index"""
        # Normalize features to [0, 1]
        normalized = continuous_state / self.max_features
        normalized = np.clip(normalized, 0, 1)
        
        # Convert to discrete bins
        discrete = (normalized * (self.feature_bins - 1)).astype(int)
        
        # Convert to single state index
        state_idx = discrete[0] * (self.feature_bins ** 2) + discrete[1] * self.feature_bins + discrete[2]
        return min(state_idx, self.base_agent.state_size - 1)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using base agent"""
        discrete_state = self._discretize_state(state)
        return self.base_agent.select_action(np.array([discrete_state]))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update using base agent"""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        self.base_agent.update(
            np.array([discrete_state]), action, reward, 
            np.array([discrete_next_state]), done
        )
    
    def get_name(self) -> str:
        return f"NLP-{self.base_agent.get_name()}"


def demonstrate_nlp_non_neurotransmitter_rl():
    """Demonstrate non-neurotransmitter RL on NLP classification task"""
    print("=== Non-Neurotransmitter RL for NLP Text Classification ===\n")
    
    # Create environment
    env = TextClassificationEnvironment()
    
    # Create different types of agents
    state_size = 125  # 5^3 possible discrete states
    action_size = 2   # Binary classification
    
    agents = [
        NLPRLAgent(StressAdaptiveLearningAgent(state_size, action_size)),
        NLPRLAgent(OxytocinSocialAgent(state_size, action_size)),
        NLPRLAgent(FreeEnergyAgent(state_size, action_size))
    ]
    
    # Training loop
    episodes = 200
    results = {}
    
    for agent in agents:
        print(f"Training {agent.get_name()}...")
        episode_rewards = []
        correct_predictions = []
        
        for episode in range(episodes):
            state = env.reset()
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            episode_rewards.append(reward)
            correct_predictions.append(reward > 0)
            
            if episode % 50 == 0:
                recent_accuracy = np.mean(correct_predictions[-50:]) if len(correct_predictions) >= 50 else np.mean(correct_predictions)
                print(f"  Episode {episode}: Accuracy = {recent_accuracy:.2%}")
        
        final_accuracy = np.mean(correct_predictions[-50:])
        results[agent.get_name()] = {
            'accuracy': final_accuracy,
            'rewards': episode_rewards,
            'predictions': correct_predictions
        }
        
        print(f"  Final accuracy: {final_accuracy:.2%}\n")
    
    # Compare results
    print("=== Final Comparison ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {result['accuracy']:.2%} accuracy")
    
    print("\n=== Insights ===")
    print("1. Stress-adaptive learning: Performance varies with classification difficulty")
    print("2. Social learning (oxytocin): Benefits from consistent positive feedback")
    print("3. Free energy principle: Balances exploration and exploitation in feature space")
    print("4. Non-neurotransmitter approaches offer diverse learning dynamics for NLP")
    
    return results


def demonstrate_multilingual_stress_adaptation():
    """
    Demonstrate how stress-adaptive learning could model
    multilingual language acquisition stress responses.
    """
    print("\n=== Stress-Adaptive Multilingual Learning Simulation ===\n")
    
    class MultilingualLearning:
        def __init__(self):
            self.languages = ["English", "Chinese", "Spanish", "German"]
            self.proficiency = {lang: 0.0 for lang in self.languages}
            self.stress_levels = {lang: 0.5 for lang in self.languages}
            
        def yerkes_dodson_learning(self, language: str, difficulty: float) -> float:
            """Simulate learning efficiency based on stress level"""
            stress = self.stress_levels[language]
            optimal_stress = 0.6
            
            if stress <= optimal_stress:
                efficiency = stress / optimal_stress
            else:
                efficiency = np.exp(-(stress - optimal_stress) * 2)
            
            return max(0.1, efficiency)
        
        def learn_session(self, language: str, material_difficulty: float):
            """Simulate a learning session"""
            efficiency = self.yerkes_dodson_learning(language, material_difficulty)
            
            # Learning progress
            progress = efficiency * (1 - material_difficulty) * 0.1
            self.proficiency[language] = min(1.0, self.proficiency[language] + progress)
            
            # Stress update based on performance
            if progress > 0.05:  # Good progress
                self.stress_levels[language] = max(0.2, self.stress_levels[language] - 0.05)
            elif progress < 0.02:  # Poor progress
                self.stress_levels[language] = min(1.0, self.stress_levels[language] + 0.1)
            
            return progress, efficiency
    
    # Simulation
    learner = MultilingualLearning()
    
    print("Simulating 50 learning sessions...")
    for session in range(50):
        for language in learner.languages:
            difficulty = np.random.uniform(0.3, 0.8)  # Varying difficulty
            progress, efficiency = learner.learn_session(language, difficulty)
            
            if session % 10 == 0:
                print(f"Session {session}, {language}: "
                      f"Proficiency={learner.proficiency[language]:.2f}, "
                      f"Stress={learner.stress_levels[language]:.2f}, "
                      f"Efficiency={efficiency:.2f}")
    
    print("\n=== Final Results ===")
    for language in learner.languages:
        print(f"{language}: {learner.proficiency[language]:.2%} proficiency, "
              f"{learner.stress_levels[language]:.2f} stress level")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)
    
    # Run demonstrations
    nlp_results = demonstrate_nlp_non_neurotransmitter_rl()
    demonstrate_multilingual_stress_adaptation()
    
    print("\n=== Integration Complete ===")
    print("This example shows how non-neurotransmitter RL can be applied to:")
    print("1. Text classification tasks in NLP")
    print("2. Modeling stress responses in language learning")
    print("3. Understanding diverse learning mechanisms beyond dopamine")
    print("4. Connecting biological insights to computational methods")