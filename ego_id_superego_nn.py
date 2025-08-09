"""
Ego-Id-Superego Neural Network Implementation in PyTorch

This module implements a neural network architecture inspired by Freud's structural model
of the psyche, consisting of three interconnected components:
- Id: Instinctual, pleasure-seeking neural network
- Ego: Rational, reality-testing neural network  
- Superego: Moral, conscience-based neural network

The architecture is designed for NLP tasks including text generation, sentiment analysis,
and human expression evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PsycheComponent(Enum):
    """Enumeration of psychological components"""
    ID = "id"
    EGO = "ego" 
    SUPEREGO = "superego"


@dataclass
class PsycheOutput:
    """Output from a psychological component"""
    component: PsycheComponent
    hidden_state: torch.Tensor
    confidence: float
    activation_pattern: torch.Tensor
    decision_weight: float


@dataclass
class IntegratedResponse:
    """Integrated response from all three components"""
    id_output: PsycheOutput
    ego_output: PsycheOutput
    superego_output: PsycheOutput
    final_output: torch.Tensor
    decision_rationale: Dict[str, float]
    conflict_level: float


class IdNetwork(nn.Module):
    """
    Id Network: Represents the instinctual, pleasure-seeking component
    - Focuses on immediate gratification and emotional responses
    - High-dimensional representations for raw desires and impulses
    - Non-linear activations to represent unconscious drives
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(IdNetwork, self).__init__()
        self.component = PsycheComponent.ID
        
        # Multi-layer architecture with high non-linearity
        self.emotion_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Strong non-linearity for emotional responses
            nn.Dropout(0.3)
        )
        
        self.impulse_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.desire_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Bound desires between 0 and 1
        )
        
        # Attention mechanism for emotional focus
        self.emotion_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> PsycheOutput:
        batch_size = x.size(0)
        
        # Emotional processing
        emotions = self.emotion_layer(x)
        
        # Self-attention for emotional focus
        emotions_reshaped = emotions.unsqueeze(1)  # (batch, 1, hidden_dim)
        attended_emotions, _ = self.emotion_attention(
            emotions_reshaped, emotions_reshaped, emotions_reshaped
        )
        attended_emotions = attended_emotions.squeeze(1)
        
        # Impulse processing
        impulses = self.impulse_layer(attended_emotions)
        
        # Desire generation
        desires = self.desire_layer(impulses)
        
        # Calculate confidence
        confidence = self.confidence_estimator(desires).mean().item()
        
        # Decision weight (id is impulsive, high weight for immediate responses)
        decision_weight = torch.norm(desires).item() / desires.numel()
        
        return PsycheOutput(
            component=self.component,
            hidden_state=desires,
            confidence=confidence,
            activation_pattern=impulses,
            decision_weight=decision_weight
        )


class EgoNetwork(nn.Module):
    """
    Ego Network: Represents the rational, reality-testing component
    - Balances between id desires and superego constraints
    - Logical reasoning and planning capabilities
    - Memory and context integration
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(EgoNetwork, self).__init__()
        self.component = PsycheComponent.EGO
        
        # Rational processing layers
        self.reasoning_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # More controlled activation
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Memory integration with LSTM
        self.memory_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        # Planning and decision making
        self.planning_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.reality_test_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Balanced output
        )
        
        # Confidence based on consistency
        self.consistency_estimator = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> PsycheOutput:
        batch_size = x.size(0)
        
        # Rational reasoning
        rational_state = self.reasoning_layer(x)
        
        # Memory integration
        if memory_context is not None:
            combined_input = torch.cat([rational_state.unsqueeze(1), memory_context], dim=1)
        else:
            combined_input = rational_state.unsqueeze(1)
            
        memory_output, _ = self.memory_lstm(combined_input)
        integrated_state = memory_output[:, -1, :]  # Take last output
        
        # Planning
        planned_state = self.planning_layer(integrated_state)
        
        # Reality testing
        ego_output = self.reality_test_layer(planned_state)
        
        # Calculate confidence based on consistency
        confidence = self.consistency_estimator(ego_output).mean().item()
        
        # Decision weight (ego is balanced, moderate weight)
        decision_weight = 0.5 + 0.3 * torch.sigmoid(torch.norm(ego_output)).item()
        
        return PsycheOutput(
            component=self.component,
            hidden_state=ego_output,
            confidence=confidence,
            activation_pattern=planned_state,
            decision_weight=decision_weight
        )


class SuperegoNetwork(nn.Module):
    """
    Superego Network: Represents the moral, conscience-based component
    - Enforces social norms and moral standards
    - Evaluates ethical implications
    - Provides moral constraints and idealistic goals
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(SuperegoNetwork, self).__init__()
        self.component = PsycheComponent.SUPEREGO
        
        # Moral evaluation layers
        self.moral_evaluation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),  # Slightly negative activation for criticism
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Social norm integration
        self.social_norm_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Idealistic goal formation
        self.ideal_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Moral constraint output
        self.constraint_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Balanced moral judgment
        )
        
        # Moral certainty estimator
        self.moral_certainty = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> PsycheOutput:
        batch_size = x.size(0)
        
        # Moral evaluation
        moral_state = self.moral_evaluation(x)
        
        # Social norm consideration
        social_state = self.social_norm_layer(moral_state)
        
        # Idealistic processing
        ideal_state = self.ideal_layer(social_state)
        
        # Generate moral constraints
        superego_output = self.constraint_layer(ideal_state)
        
        # Calculate moral certainty as confidence
        confidence = self.moral_certainty(superego_output).mean().item()
        
        # Decision weight (superego provides strong moral guidance)
        moral_strength = torch.abs(superego_output).mean().item()
        decision_weight = 0.7 + 0.3 * moral_strength
        
        return PsycheOutput(
            component=self.component,
            hidden_state=superego_output,
            confidence=confidence,
            activation_pattern=ideal_state,
            decision_weight=decision_weight
        )


class EgoIdSuperegoIntegrator(nn.Module):
    """
    Integration layer that combines outputs from all three psychological components
    - Resolves conflicts between components
    - Produces final integrated response
    - Provides transparency in decision-making process
    """
    
    def __init__(self, component_output_dim: int = 128, integration_dim: int = 256):
        super(EgoIdSuperegoIntegrator, self).__init__()
        
        # Integration networks
        self.conflict_analyzer = nn.Sequential(
            nn.Linear(component_output_dim * 3, integration_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(integration_dim, integration_dim // 2),
            nn.ReLU()
        )
        
        # Dynamic weight calculation
        self.weight_calculator = nn.Sequential(
            nn.Linear(integration_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Final output generation
        self.output_generator = nn.Sequential(
            nn.Linear(component_output_dim * 3 + 3, integration_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(integration_dim, component_output_dim),
            nn.Tanh()
        )
        
        # Conflict level estimator
        self.conflict_estimator = nn.Sequential(
            nn.Linear(integration_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, id_output: PsycheOutput, ego_output: PsycheOutput, 
                superego_output: PsycheOutput) -> IntegratedResponse:
        
        # Combine all component outputs
        combined_hidden = torch.cat([
            id_output.hidden_state,
            ego_output.hidden_state, 
            superego_output.hidden_state
        ], dim=-1)
        
        # Analyze conflicts
        conflict_features = self.conflict_analyzer(combined_hidden)
        
        # Calculate dynamic weights
        component_weights = self.weight_calculator(conflict_features)
        
        # Adjust weights by component confidence and decision weight
        id_weight = component_weights[:, 0] * id_output.confidence * id_output.decision_weight
        ego_weight = component_weights[:, 1] * ego_output.confidence * ego_output.decision_weight
        superego_weight = component_weights[:, 2] * superego_output.confidence * superego_output.decision_weight
        
        # Normalize weights
        total_weight = id_weight + ego_weight + superego_weight + 1e-8
        id_weight_norm = id_weight / total_weight
        ego_weight_norm = ego_weight / total_weight
        superego_weight_norm = superego_weight / total_weight
        
        # Create final input with weights
        weights_tensor = torch.stack([id_weight_norm, ego_weight_norm, superego_weight_norm], dim=-1)
        final_input = torch.cat([combined_hidden, weights_tensor], dim=-1)
        
        # Generate final output
        final_output = self.output_generator(final_input)
        
        # Calculate conflict level
        conflict_level = self.conflict_estimator(conflict_features).mean().item()
        
        # Create decision rationale
        decision_rationale = {
            'id_weight': id_weight_norm.mean().item(),
            'ego_weight': ego_weight_norm.mean().item(), 
            'superego_weight': superego_weight_norm.mean().item(),
            'id_confidence': id_output.confidence,
            'ego_confidence': ego_output.confidence,
            'superego_confidence': superego_output.confidence
        }
        
        return IntegratedResponse(
            id_output=id_output,
            ego_output=ego_output,
            superego_output=superego_output,
            final_output=final_output,
            decision_rationale=decision_rationale,
            conflict_level=conflict_level
        )


class EgoIdSuperegoNeuralNetwork(nn.Module):
    """
    Complete Ego-Id-Superego Neural Network
    - Integrates all three psychological components
    - Provides comprehensive analysis and decision-making
    - Suitable for various NLP tasks
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(EgoIdSuperegoNeuralNetwork, self).__init__()
        
        # Initialize the three psychological components
        self.id_network = IdNetwork(input_dim, hidden_dim, output_dim)
        self.ego_network = EgoNetwork(input_dim, hidden_dim, output_dim)
        self.superego_network = SuperegoNetwork(input_dim, hidden_dim, output_dim)
        
        # Integration layer
        self.integrator = EgoIdSuperegoIntegrator(output_dim)
        
        # Optional task-specific heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Positive, Negative, Neutral
            nn.Softmax(dim=-1)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 basic emotions
            nn.Softmax(dim=-1)
        )
        
        self.text_generation_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # Project back to input space
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None, 
                task: str = "integration") -> Dict[str, Any]:
        """
        Forward pass through the complete network
        
        Args:
            x: Input tensor
            memory_context: Optional memory context for ego network
            task: Specific task to perform ("integration", "sentiment", "emotion", "generation")
        
        Returns:
            Dictionary containing results from all components and specific task output
        """
        
        # Process through all three components
        id_output = self.id_network(x)
        ego_output = self.ego_network(x, memory_context)
        superego_output = self.superego_network(x)
        
        # Integrate outputs
        integrated_response = self.integrator(id_output, ego_output, superego_output)
        
        # Task-specific processing
        task_output = None
        if task == "sentiment":
            task_output = self.sentiment_head(integrated_response.final_output)
        elif task == "emotion":
            task_output = self.emotion_head(integrated_response.final_output)
        elif task == "generation":
            task_output = self.text_generation_head(integrated_response.final_output)
        
        return {
            'integrated_response': integrated_response,
            'task_output': task_output,
            'psyche_analysis': {
                'id_activation': id_output.hidden_state.mean().item(),
                'ego_activation': ego_output.hidden_state.mean().item(),
                'superego_activation': superego_output.hidden_state.mean().item(),
                'dominant_component': max(
                    integrated_response.decision_rationale.items(),
                    key=lambda x: x[1] if 'weight' in x[0] else 0
                )[0].replace('_weight', ''),
                'psychological_balance': 1.0 - integrated_response.conflict_level
            }
        }
    
    def analyze_text_psychologically(self, text_embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Perform comprehensive psychological analysis of text
        """
        result = self.forward(text_embedding, task="integration")
        
        # Additional psychological insights
        integrated = result['integrated_response']
        
        psychological_profile = {
            'instinctual_response': {
                'strength': integrated.id_output.confidence,
                'impulse_level': integrated.id_output.decision_weight,
                'emotional_intensity': torch.norm(integrated.id_output.hidden_state).item()
            },
            'rational_response': {
                'logical_consistency': integrated.ego_output.confidence,
                'planning_depth': integrated.ego_output.decision_weight,
                'reality_grounding': torch.mean(torch.abs(integrated.ego_output.hidden_state)).item()
            },
            'moral_response': {
                'ethical_certainty': integrated.superego_output.confidence,
                'moral_strength': integrated.superego_output.decision_weight,
                'idealistic_level': torch.norm(integrated.superego_output.hidden_state).item()
            },
            'psychological_dynamics': {
                'internal_conflict': integrated.conflict_level,
                'decision_clarity': 1.0 - integrated.conflict_level,
                'component_harmony': min(integrated.decision_rationale.values()) / max(integrated.decision_rationale.values())
            }
        }
        
        result['psychological_profile'] = psychological_profile
        return result


# Example usage and testing functions
def create_sample_text_embeddings(batch_size: int = 4, embedding_dim: int = 384) -> torch.Tensor:
    """Create sample text embeddings for testing"""
    return torch.randn(batch_size, embedding_dim)


def demonstrate_ego_id_superego_network():
    """Demonstrate the ego-id-superego neural network"""
    print("=== Ego-Id-Superego Neural Network Demonstration ===\n")
    
    # Network parameters
    input_dim = 384  # Typical text embedding dimension
    hidden_dim = 256
    output_dim = 128
    batch_size = 4
    
    # Create the network
    network = EgoIdSuperegoNeuralNetwork(input_dim, hidden_dim, output_dim)
    
    # Create sample inputs
    sample_texts = create_sample_text_embeddings(batch_size, input_dim)
    
    print(f"Network initialized with:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Batch size: {batch_size}\n")
    
    # Test different tasks
    tasks = ["integration", "sentiment", "emotion", "generation"]
    
    for task in tasks:
        print(f"--- Task: {task.upper()} ---")
        
        with torch.no_grad():
            result = network(sample_texts, task=task)
            
            integrated = result['integrated_response']
            print(f"Decision rationale: {integrated.decision_rationale}")
            print(f"Conflict level: {integrated.conflict_level:.3f}")
            print(f"Psyche analysis: {result['psyche_analysis']}")
            
            if result['task_output'] is not None:
                print(f"Task output shape: {result['task_output'].shape}")
                print(f"Task output sample: {result['task_output'][0][:5].tolist()}")
            
        print()
    
    # Demonstrate psychological analysis
    print("--- Psychological Analysis ---")
    with torch.no_grad():
        analysis = network.analyze_text_psychologically(sample_texts[0:1])
        profile = analysis['psychological_profile']
        
        print("Instinctual Response:")
        for key, value in profile['instinctual_response'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nRational Response:")
        for key, value in profile['rational_response'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nMoral Response:")
        for key, value in profile['moral_response'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nPsychological Dynamics:")
        for key, value in profile['psychological_dynamics'].items():
            print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_ego_id_superego_network()