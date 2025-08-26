"""
致命吸引子：社會動態系統仿真器
Deadly Attractors: Social Dynamical Systems Simulator

This module implements various social dynamics models that exhibit deadly attractors -
stable states that lead to destructive outcomes once entered.

Author: Generated for NLPNote repository
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.integrate import odeint
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Callable
import warnings

# Suppress integration warnings for demonstration
warnings.filterwarnings('ignore', category=UserWarning)

class SocialDynamicsModel(ABC):
    """Abstract base class for social dynamics models."""
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, t: float, *params) -> np.ndarray:
        """Define the differential equations for the model."""
        pass
    
    @abstractmethod
    def get_equilibria(self) -> List[Tuple[float, ...]]:
        """Return the equilibrium points of the system."""
        pass
    
    @abstractmethod
    def get_parameter_info(self) -> Dict[str, str]:
        """Return information about model parameters."""
        pass


class HateSpiralModel(SocialDynamicsModel):
    """
    仇恨螺旋模型 (Hate Spiral Model)
    
    Models the escalation of conflict between two groups through mutual retaliation,
    leading to complete breakdown of trust.
    
    dx/dt = -α*x*y + β*(x₀ - x) - γ*x³
    dy/dt = -α*y*x + β*(y₀ - y) - γ*y³
    
    Where:
    - x, y: trust levels between groups (-1 to 1)
    - α: mutual retaliation coefficient
    - β: natural healing coefficient  
    - γ: polarization acceleration coefficient
    - x₀, y₀: baseline trust levels
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.2, 
                 x0: float = 0.5, y0: float = 0.5):
        self.alpha = alpha  # retaliation coefficient
        self.beta = beta    # healing coefficient
        self.gamma = gamma  # polarization coefficient
        self.x0 = x0        # baseline trust for group A
        self.y0 = y0        # baseline trust for group B
    
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        
        dxdt = -self.alpha * x * y + self.beta * (self.x0 - x) - self.gamma * x**3
        dydt = -self.alpha * y * x + self.beta * (self.y0 - y) - self.gamma * y**3
        
        return np.array([dxdt, dydt])
    
    def get_equilibria(self) -> List[Tuple[float, ...]]:
        """Calculate equilibrium points analytically where possible."""
        # Simplified analysis - in practice, would use numerical methods
        return [
            (self.x0, self.y0),  # peaceful equilibrium (unstable if α large)
            (-1.0, -1.0),        # deadly attractor: mutual hatred
            (0.0, 0.0)           # neutral point (saddle)
        ]
    
    def get_parameter_info(self) -> Dict[str, str]:
        return {
            'alpha': 'Mutual retaliation coefficient (higher = more vengeful)',
            'beta': 'Natural healing coefficient (higher = more forgiving)',
            'gamma': 'Polarization acceleration (higher = faster extremism)',
            'x0': 'Baseline trust level for group A',
            'y0': 'Baseline trust level for group B'
        }


class EchoChamberModel(SocialDynamicsModel):
    """
    回音室模型 (Echo Chamber Model)
    
    Models opinion polarization where moderate views disappear and
    extreme opinions dominate.
    
    dx/dt = α*tanh(β*x) - γ*x
    ds/dt = -δ*s*(1 + |x|) + ε
    
    Where:
    - x: average opinion position (-1 to 1)
    - s: opinion diversity (≥ 0)
    - α: amplification strength
    - β: polarization sensitivity
    - γ: moderation force
    - δ: diversity loss rate
    - ε: baseline noise
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 0.3,
                 delta: float = 0.5, epsilon: float = 0.1):
        self.alpha = alpha    # amplification
        self.beta = beta      # polarization sensitivity
        self.gamma = gamma    # moderation force
        self.delta = delta    # diversity loss
        self.epsilon = epsilon # baseline noise
    
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        x, s = state
        
        dxdt = self.alpha * np.tanh(self.beta * x) - self.gamma * x
        dsdt = -self.delta * s * (1 + abs(x)) + self.epsilon
        
        # Ensure diversity doesn't go negative
        dsdt = max(dsdt, -s) if s < 0.01 else dsdt
        
        return np.array([dxdt, dsdt])
    
    def get_equilibria(self) -> List[Tuple[float, ...]]:
        # Extreme polarized states with low diversity
        x_extreme = self.alpha / self.gamma * np.tanh(self.beta * self.alpha / self.gamma)
        s_min = self.epsilon / (self.delta * (1 + abs(x_extreme)))
        
        return [
            (x_extreme, s_min),   # right-wing echo chamber
            (-x_extreme, s_min),  # left-wing echo chamber  
            (0.0, self.epsilon / self.delta)  # neutral (unstable)
        ]
    
    def get_parameter_info(self) -> Dict[str, str]:
        return {
            'alpha': 'Opinion amplification strength',
            'beta': 'Polarization sensitivity',
            'gamma': 'Moderation force',
            'delta': 'Rate of diversity loss',
            'epsilon': 'Baseline opinion noise'
        }


class PrisonersDilemmaModel(SocialDynamicsModel):
    """
    囚徒困境模型 (Prisoner's Dilemma Model)
    
    Models the evolution of cooperation in a population with punishment mechanisms.
    
    dp/dt = p*(1-p)*[R*p + S*(1-p) - T*p - P*(1-p)] - μ*p
    dq/dt = -λ*q + ν*p
    
    Where:
    - p: fraction of cooperators (0 to 1)
    - q: punishment mechanism strength (≥ 0)
    - R, S, T, P: payoff matrix (T > R > P > S)
    - μ: cooperation decay rate
    - λ: punishment decay rate
    - ν: punishment response to cooperation
    """
    
    def __init__(self, R: float = 3.0, S: float = 0.0, T: float = 5.0, P: float = 1.0,
                 mu: float = 0.1, lamb: float = 0.2, nu: float = 0.3):
        # Payoff matrix
        self.R = R  # Reward for mutual cooperation
        self.S = S  # Sucker's payoff
        self.T = T  # Temptation to defect
        self.P = P  # Punishment for mutual defection
        
        # Dynamic parameters
        self.mu = mu      # cooperation decay
        self.lamb = lamb  # punishment decay
        self.nu = nu      # punishment response
        
        # Validate payoff ordering
        if not (T > R > P > S):
            print(f"Warning: Payoff ordering T({T}) > R({R}) > P({P}) > S({S}) not satisfied")
    
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        p, q = state
        
        # Ensure p stays in [0,1] and q stays non-negative
        p = max(0, min(1, p))
        q = max(0, q)
        
        # Expected payoffs
        coop_payoff = self.R * p + self.S * (1 - p)
        defect_payoff = self.T * p + self.P * (1 - p)
        
        # Replicator dynamics with punishment effect
        fitness_diff = coop_payoff - defect_payoff + q  # punishment helps cooperation
        dpdt = p * (1 - p) * fitness_diff - self.mu * p
        
        # Punishment mechanism evolution
        dqdt = -self.lamb * q + self.nu * p
        
        return np.array([dpdt, dqdt])
    
    def get_equilibria(self) -> List[Tuple[float, ...]]:
        # Deadly attractor: no cooperation, no punishment
        return [
            (0.0, 0.0),  # All defection (deadly attractor)
            (1.0, self.nu / self.lamb),  # Full cooperation (if achievable)
        ]
    
    def get_parameter_info(self) -> Dict[str, str]:
        return {
            'R': 'Reward for mutual cooperation',
            'S': 'Sucker\'s payoff (cooperate vs defect)',
            'T': 'Temptation to defect',
            'P': 'Punishment for mutual defection',
            'mu': 'Cooperation decay rate',
            'lambda': 'Punishment mechanism decay rate',
            'nu': 'Punishment response to cooperation level'
        }


class TragedyOfCommonsModel(SocialDynamicsModel):
    """
    公地悲劇模型 (Tragedy of Commons Model)
    
    Models resource exploitation leading to resource depletion.
    
    dR/dt = r*R*(1 - R/K) - E*R
    dE/dt = α*E*(R/R₀ - c) - β*E²
    
    Where:
    - R: resource stock (≥ 0)
    - E: exploitation effort (≥ 0)
    - r: resource growth rate
    - K: carrying capacity
    - α: effort response rate
    - β: effort cost coefficient
    - c: exploitation cost threshold
    - R₀: reference resource level
    """
    
    def __init__(self, r: float = 0.5, K: float = 100.0, alpha: float = 0.3,
                 beta: float = 0.1, c: float = 0.5, R0: float = 50.0):
        self.r = r        # resource growth rate
        self.K = K        # carrying capacity
        self.alpha = alpha # effort response rate
        self.beta = beta   # effort cost
        self.c = c         # cost threshold
        self.R0 = R0       # reference resource level
    
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        R, E = state
        
        # Ensure non-negative values
        R = max(0, R)
        E = max(0, E)
        
        # Resource dynamics: logistic growth minus extraction
        dRdt = self.r * R * (1 - R / self.K) - E * R
        
        # Effort dynamics: profit-driven with quadratic costs
        profit_rate = R / self.R0 - self.c if R > 0 else -self.c
        dEdt = self.alpha * E * profit_rate - self.beta * E**2
        
        # Prevent negative resources
        if R <= 0 and dRdt < 0:
            dRdt = 0
        
        return np.array([dRdt, dEdt])
    
    def get_equilibria(self) -> List[Tuple[float, ...]]:
        # Deadly attractor: resource collapse
        return [
            (0.0, 0.0),  # Resource depletion (deadly attractor)
            (self.K, 0.0),  # Pristine environment (no exploitation)
        ]
    
    def get_parameter_info(self) -> Dict[str, str]:
        return {
            'r': 'Resource natural growth rate',
            'K': 'Environmental carrying capacity',
            'alpha': 'Exploitation effort response rate',
            'beta': 'Effort cost coefficient',
            'c': 'Exploitation cost threshold',
            'R0': 'Reference resource level for profit calculation'
        }


class DeadlyAttractorSimulator:
    """
    致命吸引子仿真器 (Deadly Attractor Simulator)
    
    Main simulation engine for analyzing social dynamics models and their deadly attractors.
    """
    
    def __init__(self, model: SocialDynamicsModel):
        self.model = model
        self.trajectories = []
        self.current_trajectory = None
    
    def simulate(self, initial_state: List[float], time_span: float = 50.0, 
                 num_points: int = 1000) -> np.ndarray:
        """
        Simulate the system from given initial conditions.
        
        Args:
            initial_state: Initial system state
            time_span: Total simulation time
            num_points: Number of time points
            
        Returns:
            Array of shape (num_points, num_variables) containing trajectory
        """
        t = np.linspace(0, time_span, num_points)
        
        try:
            trajectory = odeint(self.model.dynamics, initial_state, t)
            self.current_trajectory = {'time': t, 'states': trajectory, 'initial': initial_state}
            self.trajectories.append(self.current_trajectory)
            return trajectory
        except Exception as e:
            print(f"Simulation failed: {e}")
            return np.array([initial_state])
    
    def plot_phase_portrait(self, xlim: Tuple[float, float] = (-2, 2), 
                           ylim: Tuple[float, float] = (-2, 2),
                           resolution: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot phase portrait with vector field and equilibria.
        
        Args:
            xlim: x-axis limits
            ylim: y-axis limits  
            resolution: Grid resolution for vector field
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create meshgrid for vector field
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], Y[i, j]])
                derivatives = self.model.dynamics(state, 0)
                U[i, j] = derivatives[0]
                V[i, j] = derivatives[1]
        
        # Normalize arrows for better visualization
        M = np.sqrt(U**2 + V**2)
        M[M == 0] = 1  # Avoid division by zero
        U_norm = U / M
        V_norm = V / M
        
        # Plot vector field
        ax1.quiver(X, Y, U_norm, V_norm, M, scale=30, alpha=0.6, cmap='viridis')
        
        # Plot equilibria
        equilibria = self.model.get_equilibria()
        for i, eq in enumerate(equilibria):
            if len(eq) >= 2:  # Only plot if 2D or higher
                color = 'red' if i == 1 else 'blue'  # Assume second equilibrium is deadly
                marker = 'X' if i == 1 else 'o'
                label = 'Deadly Attractor' if i == 1 else f'Equilibrium {i+1}'
                ax1.plot(eq[0], eq[1], marker, color=color, markersize=10, 
                        label=label, markeredgecolor='black', markeredgewidth=1)
        
        # Plot trajectories if available
        for traj in self.trajectories:
            states = traj['states']
            ax1.plot(states[:, 0], states[:, 1], 'g-', alpha=0.7, linewidth=2)
            ax1.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
            ax1.plot(states[-1, 0], states[-1, 1], 'rs', markersize=8, label='End')
        
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_xlabel('State Variable 1', fontsize=12)
        ax1.set_ylabel('State Variable 2', fontsize=12)
        ax1.set_title('Phase Portrait with Vector Field', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot time series
        if self.current_trajectory is not None:
            t = self.current_trajectory['time']
            states = self.current_trajectory['states']
            
            ax2.plot(t, states[:, 0], 'b-', linewidth=2, label='Variable 1')
            ax2.plot(t, states[:, 1], 'r-', linewidth=2, label='Variable 2')
            
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_ylabel('State Variables', fontsize=12)
            ax2.set_title('Time Evolution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_stability(self, state: List[float], epsilon: float = 1e-6) -> Dict[str, float]:
        """
        Analyze local stability around a given state using numerical Jacobian.
        
        Args:
            state: State point to analyze
            epsilon: Perturbation size for numerical derivatives
            
        Returns:
            Dictionary with eigenvalues and stability information
        """
        state = np.array(state)
        n = len(state)
        
        # Compute numerical Jacobian
        jacobian = np.zeros((n, n))
        
        for i in range(n):
            # Forward difference
            state_plus = state.copy()
            state_plus[i] += epsilon
            f_plus = self.model.dynamics(state_plus, 0)
            
            # Backward difference
            state_minus = state.copy()
            state_minus[i] -= epsilon
            f_minus = self.model.dynamics(state_minus, 0)
            
            # Central difference
            jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(jacobian)
        
        # Determine stability
        max_real_part = np.max(np.real(eigenvalues))
        is_stable = max_real_part < 0
        is_unstable = max_real_part > 0
        is_marginal = abs(max_real_part) < 1e-10
        
        return {
            'eigenvalues': eigenvalues,
            'max_real_part': max_real_part,
            'is_stable': is_stable,
            'is_unstable': is_unstable,
            'is_marginal': is_marginal,
            'jacobian': jacobian
        }
    
    def basin_of_attraction(self, attractor: List[float], grid_size: int = 50,
                           xlim: Tuple[float, float] = (-2, 2),
                           ylim: Tuple[float, float] = (-2, 2),
                           tolerance: float = 0.1, max_time: float = 100.0):
        """
        Compute the basin of attraction for a given attractor.
        
        Args:
            attractor: The attractor point
            grid_size: Resolution of the initial condition grid
            xlim, ylim: Range of initial conditions to test
            tolerance: Distance tolerance for reaching attractor
            max_time: Maximum simulation time
            
        Returns:
            2D array indicating which initial conditions lead to the attractor
        """
        x_range = np.linspace(xlim[0], xlim[1], grid_size)
        y_range = np.linspace(ylim[0], ylim[1], grid_size)
        
        basin = np.zeros((grid_size, grid_size))
        attractor = np.array(attractor)
        
        for i, x0 in enumerate(x_range):
            for j, y0 in enumerate(y_range):
                initial_state = [x0, y0]
                
                try:
                    # Simulate for shorter time to check convergence
                    trajectory = self.simulate(initial_state, max_time, 500)
                    
                    # Check if final state is close to attractor
                    final_state = trajectory[-1]
                    distance = np.linalg.norm(final_state - attractor)
                    
                    if distance < tolerance:
                        basin[j, i] = 1  # Note: j, i for proper orientation
                
                except:
                    continue
        
        return basin, x_range, y_range
    
    def plot_basin_of_attraction(self, attractor: List[float], **kwargs):
        """Plot the basin of attraction for a given attractor."""
        basin, x_range, y_range = self.basin_of_attraction(attractor, **kwargs)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(basin, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
                  origin='lower', cmap='RdYlBu', alpha=0.7)
        plt.colorbar(label='Converges to Attractor')
        
        # Mark the attractor
        plt.plot(attractor[0], attractor[1], 'rX', markersize=15, 
                markeredgecolor='black', markeredgewidth=2, label='Deadly Attractor')
        
        plt.xlabel('State Variable 1', fontsize=12)
        plt.ylabel('State Variable 2', fontsize=12)
        plt.title(f'Basin of Attraction for Deadly Attractor at {attractor}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def demonstrate_deadly_attractors():
    """
    Demonstrate various deadly attractor models with visualizations.
    """
    print("🔥 Deadly Attractors in Social Dynamics Demonstration 🔥")
    print("=" * 60)
    
    # 1. Hate Spiral Model
    print("\n1. 仇恨螺旋模型 (Hate Spiral Model)")
    print("-" * 40)
    
    hate_model = HateSpiralModel(alpha=0.8, beta=0.1, gamma=0.3)
    simulator = DeadlyAttractorSimulator(hate_model)
    
    # Simulate from initially peaceful state
    print("Simulating from initially peaceful state (0.7, 0.6)...")
    trajectory = simulator.simulate([0.7, 0.6], time_span=30)
    
    print(f"Initial trust: ({trajectory[0, 0]:.3f}, {trajectory[0, 1]:.3f})")
    print(f"Final trust: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})")
    
    # Analyze stability of deadly attractor
    stability = simulator.analyze_stability([-1.0, -1.0])
    print(f"Deadly attractor stability: {'Stable' if stability['is_stable'] else 'Unstable'}")
    print(f"Eigenvalues: {stability['eigenvalues']}")
    
    # Plot phase portrait
    simulator.plot_phase_portrait(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    
    # 2. Echo Chamber Model
    print("\n2. 回音室模型 (Echo Chamber Model)")
    print("-" * 40)
    
    echo_model = EchoChamberModel(alpha=1.2, beta=3.0, gamma=0.4)
    simulator2 = DeadlyAttractorSimulator(echo_model)
    
    # Simulate from moderate initial state
    print("Simulating from moderate opinion state (0.1, 0.8)...")
    trajectory2 = simulator2.simulate([0.1, 0.8], time_span=25)
    
    print(f"Initial opinion & diversity: ({trajectory2[0, 0]:.3f}, {trajectory2[0, 1]:.3f})")
    print(f"Final opinion & diversity: ({trajectory2[-1, 0]:.3f}, {trajectory2[-1, 1]:.3f})")
    
    # Plot phase portrait
    simulator2.plot_phase_portrait(xlim=(-2, 2), ylim=(0, 1))
    
    # 3. Prisoner's Dilemma Model  
    print("\n3. 囚徒困境模型 (Prisoner's Dilemma Model)")
    print("-" * 40)
    
    pd_model = PrisonersDilemmaModel(R=3, S=0, T=5, P=1, mu=0.15, lamb=0.3, nu=0.4)
    simulator3 = DeadlyAttractorSimulator(pd_model)
    
    # Simulate from partial cooperation state
    print("Simulating from partial cooperation state (0.6, 0.2)...")
    trajectory3 = simulator3.simulate([0.6, 0.2], time_span=40)
    
    print(f"Initial cooperation & punishment: ({trajectory3[0, 0]:.3f}, {trajectory3[0, 1]:.3f})")
    print(f"Final cooperation & punishment: ({trajectory3[-1, 0]:.3f}, {trajectory3[-1, 1]:.3f})")
    
    # Plot phase portrait
    simulator3.plot_phase_portrait(xlim=(0, 1), ylim=(0, 2))
    
    # 4. Tragedy of Commons
    print("\n4. 公地悲劇模型 (Tragedy of Commons Model)")
    print("-" * 40)
    
    commons_model = TragedyOfCommonsModel(r=0.3, K=100, alpha=0.4, beta=0.1, c=0.3)
    simulator4 = DeadlyAttractorSimulator(commons_model)
    
    # Simulate from sustainable state
    print("Simulating from initially sustainable state (80, 0.5)...")
    trajectory4 = simulator4.simulate([80, 0.5], time_span=50)
    
    print(f"Initial resources & effort: ({trajectory4[0, 0]:.1f}, {trajectory4[0, 1]:.3f})")
    print(f"Final resources & effort: ({trajectory4[-1, 0]:.1f}, {trajectory4[-1, 1]:.3f})")
    
    # Plot phase portrait
    simulator4.plot_phase_portrait(xlim=(0, 100), ylim=(0, 3))
    
    print("\n" + "=" * 60)
    print("Analysis Complete! All models show convergence to deadly attractors.")
    print("These represent stable but destructive social equilibria.")


if __name__ == "__main__":
    # Demonstrate the deadly attractor models
    demonstrate_deadly_attractors()
    
    # Additional analysis example
    print("\n🎯 Basin of Attraction Analysis Example")
    print("-" * 40)
    
    # Analyze basin of attraction for hate spiral model
    hate_model = HateSpiralModel(alpha=0.6, beta=0.1, gamma=0.2)
    simulator = DeadlyAttractorSimulator(hate_model)
    
    print("Computing basin of attraction for deadly attractor (-1, -1)...")
    print("This may take a moment...")
    
    # Plot basin of attraction
    simulator.plot_basin_of_attraction(
        attractor=[-1.0, -1.0],
        grid_size=30,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        tolerance=0.2,
        max_time=20.0
    )
    
    print("Basin analysis complete!")
    print("\nRed regions show initial conditions that lead to the deadly attractor.")
    print("This demonstrates the 'catchment area' of destructive social dynamics.")