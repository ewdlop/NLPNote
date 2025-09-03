"""
Physics Path Formulations: Differential vs Integral Approaches
Demonstrates concepts from physics path formulations in computational contexts.

This module provides implementations that illustrate the differences between
path differentiation and path integral formulations, with applications to
language processing and computational physics.
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod


@dataclass
class PathPoint:
    """Represents a point along a path in configuration space."""
    coordinates: np.ndarray
    time: float
    action_contribution: float = 0.0


class PhysicalSystem(ABC):
    """Abstract base class for physical systems."""
    
    @abstractmethod
    def lagrangian(self, position: np.ndarray, velocity: np.ndarray, time: float) -> float:
        """Compute Lagrangian L = T - V"""
        pass
    
    @abstractmethod
    def equations_of_motion(self, state: np.ndarray, time: float) -> np.ndarray:
        """Return derivatives for differential equation solver"""
        pass


class HarmonicOscillator(PhysicalSystem):
    """Simple harmonic oscillator for demonstrating path formulations."""
    
    def __init__(self, mass: float = 1.0, omega: float = 1.0):
        self.mass = mass
        self.omega = omega
    
    def lagrangian(self, position: np.ndarray, velocity: np.ndarray, time: float) -> float:
        """L = ½mv² - ½mω²x²"""
        kinetic = 0.5 * self.mass * np.sum(velocity**2)
        potential = 0.5 * self.mass * self.omega**2 * np.sum(position**2)
        return kinetic - potential
    
    def equations_of_motion(self, state: np.ndarray, time: float) -> np.ndarray:
        """State = [position, velocity], returns [velocity, acceleration]"""
        pos, vel = state[0], state[1]
        acceleration = -self.omega**2 * pos
        return np.array([vel, acceleration])


class PathDifferentialSolver:
    """Solves physics problems using differential equation approaches."""
    
    def __init__(self, system: PhysicalSystem):
        self.system = system
    
    def runge_kutta_4(self, 
                     initial_state: np.ndarray,
                     time_span: Tuple[float, float],
                     num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve differential equations using 4th-order Runge-Kutta method.
        
        Returns:
            times, states: Arrays of time points and corresponding states
        """
        t_start, t_end = time_span
        dt = (t_end - t_start) / num_points
        
        times = np.linspace(t_start, t_end, num_points + 1)
        states = np.zeros((num_points + 1, len(initial_state)))
        states[0] = initial_state
        
        for i in range(num_points):
            t = times[i]
            y = states[i]
            
            k1 = dt * self.system.equations_of_motion(y, t)
            k2 = dt * self.system.equations_of_motion(y + k1/2, t + dt/2)
            k3 = dt * self.system.equations_of_motion(y + k2/2, t + dt/2)
            k4 = dt * self.system.equations_of_motion(y + k3, t + dt)
            
            states[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return times, states
    
    def compute_action(self, path: List[PathPoint]) -> float:
        """Compute action S = ∫ L dt along a given path."""
        total_action = 0.0
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            dt = p2.time - p1.time
            
            if dt > 0:
                # Approximate velocity
                velocity = (p2.coordinates - p1.coordinates) / dt
                # Compute Lagrangian at midpoint
                mid_pos = (p1.coordinates + p2.coordinates) / 2
                lagrangian = self.system.lagrangian(mid_pos, velocity, p1.time)
                total_action += lagrangian * dt
        
        return total_action


class PathIntegralSolver:
    """Implements path integral formulation approaches."""
    
    def __init__(self, system: PhysicalSystem):
        self.system = system
        self.hbar = 1.0  # Set ℏ = 1 in natural units
    
    def generate_random_path(self, 
                           start_pos: np.ndarray,
                           end_pos: np.ndarray,
                           time_span: Tuple[float, float],
                           num_points: int = 100,
                           noise_scale: float = 0.1) -> List[PathPoint]:
        """Generate a random path between start and end points."""
        t_start, t_end = time_span
        times = np.linspace(t_start, t_end, num_points)
        
        # Linear interpolation as base path
        base_path = np.outer(times, end_pos - start_pos) / (t_end - t_start)
        base_path += start_pos
        
        # Add random fluctuations
        noise = np.random.normal(0, noise_scale, (num_points, len(start_pos)))
        # Ensure endpoints are fixed
        noise[0] = noise[-1] = 0
        
        path_coords = base_path + noise
        
        path = []
        for i, t in enumerate(times):
            point = PathPoint(coordinates=path_coords[i], time=t)
            path.append(point)
        
        return path
    
    def compute_path_amplitude(self, path: List[PathPoint]) -> complex:
        """Compute quantum amplitude exp(iS/ℏ) for a given path."""
        action = self._compute_path_action(path)
        return cmath.exp(1j * action / self.hbar)
    
    def _compute_path_action(self, path: List[PathPoint]) -> float:
        """Compute action along path."""
        total_action = 0.0
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            dt = p2.time - p1.time
            
            if dt > 0:
                velocity = (p2.coordinates - p1.coordinates) / dt
                mid_pos = (p1.coordinates + p2.coordinates) / 2
                lagrangian = self.system.lagrangian(mid_pos, velocity, p1.time)
                total_action += lagrangian * dt
        
        return total_action
    
    def monte_carlo_path_integral(self,
                                start_pos: np.ndarray,
                                end_pos: np.ndarray,
                                time_span: Tuple[float, float],
                                num_paths: int = 1000,
                                num_points_per_path: int = 50) -> complex:
        """
        Approximate path integral using Monte Carlo sampling.
        
        Returns the quantum amplitude ⟨x_f|exp(-iHt/ℏ)|x_i⟩
        """
        total_amplitude = 0.0 + 0.0j
        
        for _ in range(num_paths):
            # Generate random path
            path = self.generate_random_path(
                start_pos, end_pos, time_span, num_points_per_path
            )
            
            # Compute amplitude contribution
            amplitude = self.compute_path_amplitude(path)
            total_amplitude += amplitude
        
        # Normalize by number of paths
        return total_amplitude / num_paths


class LanguagePathProcessor:
    """
    Demonstrates path-like concepts in language processing.
    Shows analogy between physics path formulations and NLP approaches.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Simple word embedding simulation
        self.embeddings = np.random.randn(vocab_size, 64)
    
    def token_to_vector(self, token: str) -> np.ndarray:
        """Convert token to vector representation."""
        # Simple hash-based embedding lookup
        idx = hash(token) % self.vocab_size
        return self.embeddings[idx]
    
    def differential_approach(self, tokens: List[str]) -> List[np.ndarray]:
        """
        Differential-like: Process tokens sequentially (RNN-style).
        Each step depends only on current input and previous hidden state.
        """
        hidden_size = 64
        hidden_state = np.zeros(hidden_size)
        outputs = []
        
        # Simple RNN computation
        W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        W_x = np.random.randn(hidden_size, 64) * 0.1
        
        for token in tokens:
            token_vec = self.token_to_vector(token)
            # h_t = tanh(W_h * h_{t-1} + W_x * x_t)
            hidden_state = np.tanh(W_h @ hidden_state + W_x @ token_vec)
            outputs.append(hidden_state.copy())
        
        return outputs
    
    def path_integral_approach(self, tokens: List[str]) -> List[np.ndarray]:
        """
        Path integral-like: Consider all possible attention patterns (Transformer-style).
        Each output considers all inputs simultaneously.
        """
        token_vecs = [self.token_to_vector(token) for token in tokens]
        n_tokens = len(token_vecs)
        outputs = []
        
        for i in range(n_tokens):
            # Compute attention weights to all positions
            query = token_vecs[i]
            attention_weights = []
            
            for j in range(n_tokens):
                key = token_vecs[j]
                # Simplified attention: dot product similarity
                attention = np.dot(query, key) / np.sqrt(len(query))
                attention_weights.append(attention)
            
            # Softmax normalization
            attention_weights = np.array(attention_weights)
            attention_weights = np.exp(attention_weights)
            attention_weights /= np.sum(attention_weights)
            
            # Weighted sum of all token vectors
            output = np.zeros_like(query)
            for j, weight in enumerate(attention_weights):
                output += weight * token_vecs[j]
            
            outputs.append(output)
        
        return outputs


class TopologicalPathAnalyzer:
    """
    Analyzes topological properties of paths, connecting to homotopy theory
    discussed in the repository's three.md file.
    """
    
    def compute_winding_number(self, path: List[Tuple[float, float]]) -> int:
        """
        Compute winding number of a closed path around origin.
        Related to π₁(S¹) = ℤ discussed in three.md
        """
        if len(path) < 3:
            return 0
        
        total_angle = 0.0
        for i in range(len(path)):
            p1 = path[i]
            p2 = path[(i + 1) % len(path)]
            
            # Convert to complex numbers
            z1 = complex(p1[0], p1[1])
            z2 = complex(p2[0], p2[1])
            
            if abs(z1) > 1e-10 and abs(z2) > 1e-10:
                # Compute angle change
                angle_change = cmath.phase(z2/z1)
                # Ensure we take the shortest angular path
                if angle_change > np.pi:
                    angle_change -= 2*np.pi
                elif angle_change < -np.pi:
                    angle_change += 2*np.pi
                total_angle += angle_change
        
        # Winding number is total angle divided by 2π
        return round(total_angle / (2 * np.pi))
    
    def path_integral_with_topology(self, 
                                  paths: List[List[Tuple[float, float]]]) -> complex:
        """
        Path integral that accounts for topological sectors.
        Different winding numbers contribute with different phases.
        """
        total_amplitude = 0.0 + 0.0j
        
        for path in paths:
            # Compute "action" (simplified as path length squared)
            action = 0.0
            for i in range(len(path) - 1):
                dx = path[i+1][0] - path[i][0]
                dy = path[i+1][1] - path[i][1]
                action += dx**2 + dy**2
            
            # Compute topological contribution
            winding = self.compute_winding_number(path)
            
            # Path integral amplitude with topological phase
            amplitude = cmath.exp(1j * action) * cmath.exp(2j * np.pi * winding)
            total_amplitude += amplitude
        
        return total_amplitude / len(paths) if paths else 0


def demonstrate_physics_path_formulations():
    """
    Demonstrate the differences between differential and path integral approaches
    using a simple harmonic oscillator.
    """
    print("=== Physics Path Formulations Demo ===\n")
    
    # Set up harmonic oscillator
    oscillator = HarmonicOscillator(mass=1.0, omega=2.0)
    
    # Initial conditions
    initial_position = 1.0
    initial_velocity = 0.0
    initial_state = np.array([initial_position, initial_velocity])
    time_span = (0.0, np.pi / 2.0)  # Quarter period for omega=2.0
    
    print("1. Differential Approach (Classical trajectory):")
    # Solve using differential equations
    diff_solver = PathDifferentialSolver(oscillator)
    times, states = diff_solver.runge_kutta_4(initial_state, time_span, 500)
    
    print(f"   Initial position: {initial_position:.3f}")
    print(f"   Final position: {states[-1, 0]:.3f}")
    print(f"   Expected (analytical): {-initial_position:.3f} at t=π/ω = {np.pi/oscillator.omega:.3f}")
    
    # Compute action along classical path
    classical_path = [
        PathPoint(np.array([states[i, 0]]), times[i]) 
        for i in range(len(times))
    ]
    classical_action = diff_solver.compute_action(classical_path)
    print(f"   Classical action: {classical_action:.3f}")
    
    print("\n2. Path Integral Approach (Quantum amplitude):")
    # Path integral calculation
    path_solver = PathIntegralSolver(oscillator)
    
    start_pos = np.array([initial_position])
    end_pos = np.array([states[-1, 0]])
    
    # Monte Carlo path integral
    amplitude = path_solver.monte_carlo_path_integral(
        start_pos, end_pos, time_span, num_paths=500, num_points_per_path=50
    )
    
    print(f"   Quantum amplitude: {amplitude:.3f}")
    print(f"   Probability: |amplitude|² = {abs(amplitude)**2:.3f}")
    print(f"   Phase: {cmath.phase(amplitude):.3f}")
    
    print("\n3. Language Processing Analogy:")
    # Demonstrate language processing analogy
    lang_processor = LanguagePathProcessor()
    tokens = ["quantum", "path", "integral", "formulation"]
    
    # Differential-like processing
    diff_outputs = lang_processor.differential_approach(tokens)
    print(f"   Differential processing (RNN-like): {len(diff_outputs)} hidden states")
    
    # Path integral-like processing
    integral_outputs = lang_processor.path_integral_approach(tokens)
    print(f"   Path integral processing (Transformer-like): {len(integral_outputs)} contextualized representations")
    
    print("\n4. Topological Path Analysis:")
    # Demonstrate topological concepts
    topo_analyzer = TopologicalPathAnalyzer()
    
    # Create paths with different winding numbers
    path1 = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)]  # winding = 1
    path2 = [(1, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]  # winding = -1
    path3 = [(0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, 0)]  # winding = 1
    
    paths = [path1, path2, path3]
    
    for i, path in enumerate(paths):
        winding = topo_analyzer.compute_winding_number(path)
        print(f"   Path {i+1} winding number: {winding}")
    
    result = topo_analyzer.path_integral_with_topology(paths)
    print(f"   Topological path integral: {result:.3f}")


if __name__ == "__main__":
    demonstrate_physics_path_formulations()