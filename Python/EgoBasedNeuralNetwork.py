"""
Ego-Based Neural Network Implementation
自我導向神經網路實現

This module implements an ego-based neural network that incorporates:
1. Confirmation bias in training
2. Parameter inertia (staying close to initial parameters)
3. Output inertia (consistency with previous outputs)

The ego mechanism helps the model maintain consistency with its initial "personality"
while still learning from new data, mimicking certain aspects of human learning biases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class EgoMLP(nn.Module):
    """
    Multi-Layer Perceptron with ego-based learning capabilities.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output dimension (number of classes)
        hidden_dim (int): Hidden layer dimension (default: 128)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class EgoBasedTrainer:
    """
    Trainer class for ego-based neural network learning.
    
    This trainer implements ego-based learning with:
    - Confirmation bias weighting
    - Parameter inertia
    - Output inertia
    """
    
    def __init__(
        self,
        model: EgoMLP,
        lambda_param: float = 5e-3,  # Parameter inertia strength
        lambda_output: float = 1e-2,  # Output inertia strength
        alpha: float = 0.7,  # Confirmation bias strength
        learning_rate: float = 1e-3
    ):
        """
        Initialize the ego-based trainer.
        
        Args:
            model: The EgoMLP model to train
            lambda_param: Strength of parameter inertia
            lambda_output: Strength of output inertia
            alpha: Confirmation bias weight (higher = more bias toward agreeing predictions)
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.lambda_param = lambda_param
        self.lambda_output = lambda_output
        self.alpha = alpha
        
        # Store initial parameters as the "ego" baseline
        self.initial_params = [p.detach().clone() for p in model.parameters()]
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Track previous outputs for output inertia
        self.prev_logits: Optional[torch.Tensor] = None
    
    def compute_confirmation_bias_weights(
        self, 
        current_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute confirmation bias weights based on agreement with previous predictions.
        
        Args:
            current_logits: Current model predictions
            targets: Ground truth labels
            
        Returns:
            Weights for the loss function
        """
        if self.prev_logits is None:
            # No previous predictions, use uniform weights
            return torch.ones_like(targets, dtype=torch.float)
        
        with torch.no_grad():
            # Check if previous predictions agreed with ground truth
            prev_predictions = self.prev_logits.argmax(dim=1)
            agreement = (prev_predictions == targets)
            
            # Higher weight for examples where we previously agreed with truth
            # Lower weight for examples where we previously disagreed
            weights = torch.where(
                agreement,
                torch.full_like(agreement, self.alpha, dtype=torch.float),
                torch.full_like(agreement, 1 - self.alpha, dtype=torch.float)
            )
            
        return weights
    
    def compute_parameter_inertia_loss(self) -> torch.Tensor:
        """
        Compute parameter inertia loss to keep parameters close to initial values.
        
        Returns:
            Parameter inertia loss
        """
        param_loss = torch.tensor(0.0, requires_grad=True)
        
        for current_param, initial_param in zip(self.model.parameters(), self.initial_params):
            param_loss = param_loss + (current_param - initial_param).pow(2).sum()
        
        return self.lambda_param * param_loss
    
    def compute_output_inertia_loss(self, current_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute output inertia loss to maintain consistency with previous outputs.
        
        Args:
            current_logits: Current model outputs
            
        Returns:
            Output inertia loss
        """
        if self.prev_logits is None:
            return torch.tensor(0.0, requires_grad=True)
        
        return self.lambda_output * F.mse_loss(current_logits, self.prev_logits)
    
    def train_step(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """
        Perform one training step.
        
        Args:
            inputs: Input features
            targets: Target labels
            
        Returns:
            Tuple of (total_loss, task_loss, param_loss, output_loss)
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(inputs)
        
        # Compute confirmation bias weights
        weights = self.compute_confirmation_bias_weights(logits, targets)
        
        # Task loss with confirmation bias weighting
        task_loss = (F.cross_entropy(logits, targets, reduction='none') * weights).mean()
        
        # Parameter inertia loss
        param_loss = self.compute_parameter_inertia_loss()
        
        # Output inertia loss
        output_loss = self.compute_output_inertia_loss(logits)
        
        # Total loss
        total_loss = task_loss + param_loss + output_loss
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        # Update previous logits for next iteration
        self.prev_logits = logits.detach()
        
        return (
            total_loss.item(),
            task_loss.item(),
            param_loss.item(),
            output_loss.item()
        )
    
    def train(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        epochs: int = 200,
        verbose: bool = True
    ) -> List[Tuple[float, float, float, float]]:
        """
        Train the ego-based neural network.
        
        Args:
            inputs: Training input features
            targets: Training target labels
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            List of loss tuples for each epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            losses = self.train_step(inputs, targets)
            loss_history.append(losses)
            
            if verbose and (epoch + 1) % 50 == 0:
                total_loss, task_loss, param_loss, output_loss = losses
                print(f"Epoch {epoch + 1}/{epochs}:")
                print(f"  Total Loss: {total_loss:.6f}")
                print(f"  Task Loss: {task_loss:.6f}")
                print(f"  Parameter Loss: {param_loss:.6f}")
                print(f"  Output Loss: {output_loss:.6f}")
        
        return loss_history


def create_toy_dataset(n_samples: int = 512, input_dim: int = 20, n_classes: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a toy dataset for testing the ego-based neural network.
    
    Args:
        n_samples: Number of samples
        input_dim: Input feature dimension
        n_classes: Number of output classes
        
    Returns:
        Tuple of (features, labels)
    """
    torch.manual_seed(42)  # For reproducibility
    
    features = torch.randn(n_samples, input_dim)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    return features, labels


def demo_ego_based_learning():
    """
    Demonstrate the ego-based neural network with a toy example.
    """
    print("=== Ego-Based Neural Network Demo ===")
    print("自我導向神經網路演示")
    print()
    
    # Create toy dataset
    N, d, k = 512, 20, 3
    x, y = create_toy_dataset(N, d, k)
    
    print(f"Dataset: {N} samples, {d} features, {k} classes")
    
    # Create model
    model = EgoMLP(input_dim=d, output_dim=k)
    
    # Create trainer with ego-based learning
    trainer = EgoBasedTrainer(
        model=model,
        lambda_param=5e-3,   # Parameter inertia strength
        lambda_output=1e-2,  # Output inertia strength
        alpha=0.7,           # Confirmation bias strength
        learning_rate=1e-3
    )
    
    print("\nTraining with ego-based learning...")
    print("使用自我導向學習進行訓練...")
    
    # Train the model
    loss_history = trainer.train(x, y, epochs=200, verbose=True)
    
    print("\nTraining completed!")
    print("訓練完成！")
    
    # Evaluate final performance
    with torch.no_grad():
        final_logits = model(x)
        final_predictions = final_logits.argmax(dim=1)
        accuracy = (final_predictions == y).float().mean()
        print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    return model, trainer, loss_history


if __name__ == "__main__":
    # Run the demo
    demo_ego_based_learning()