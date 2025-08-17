"""
Numerical Euler-Lagrangian Solver using SciPy

This module provides numerical methods for solving Euler-Lagrange equations,
which are fundamental in variational calculus and physics. The Euler-Lagrange
equation has the form:

d/dt(∂L/∂q̇) - ∂L/∂q = 0

where L(q, q̇, t) is the Lagrangian function, q is the generalized coordinate,
and q̇ is its time derivative.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Any
import warnings


def numerical_derivative(func, x, dx=1e-8):
    """
    Compute numerical derivative using central difference.
    
    Parameters:
    -----------
    func : callable
        Function to differentiate
    x : float
        Point at which to evaluate derivative
    dx : float
        Step size for finite difference
        
    Returns:
    --------
    float
        Approximate derivative
    """
    return (func(x + dx) - func(x - dx)) / (2 * dx)


class EulerLagrangianSolver:
    """
    A comprehensive solver for Euler-Lagrange equations using numerical methods.
    
    This class provides various numerical approaches to solve variational problems
    and differential equations arising from Lagrangian mechanics.
    """
    
    def __init__(self, lagrangian: Callable[[np.ndarray, np.ndarray, float], float],
                 n_coordinates: int = 1):
        """
        Initialize the Euler-Lagrangian solver.
        
        Parameters:
        -----------
        lagrangian : callable
            The Lagrangian function L(q, q_dot, t) where:
            - q: array of generalized coordinates
            - q_dot: array of generalized velocities  
            - t: time
        n_coordinates : int
            Number of generalized coordinates
        """
        self.lagrangian = lagrangian
        self.n_coordinates = n_coordinates
        self._cache = {}
        
    def compute_euler_lagrange_equation(self, q: np.ndarray, q_dot: np.ndarray, 
                                      q_ddot: np.ndarray, t: float, 
                                      coordinate_idx: int = 0) -> float:
        """
        Compute the Euler-Lagrange equation for a specific coordinate.
        
        The equation is: d/dt(∂L/∂q̇) - ∂L/∂q = 0
        
        Parameters:
        -----------
        q : np.ndarray
            Generalized coordinates
        q_dot : np.ndarray
            Generalized velocities
        q_ddot : np.ndarray
            Generalized accelerations
        t : float
            Time
        coordinate_idx : int
            Index of the coordinate to compute the equation for
            
        Returns:
        --------
        float
            Value of the Euler-Lagrange equation (should be 0 for solutions)
        """
        # Partial derivative with respect to q
        def L_partial_q(q_var):
            q_temp = q.copy()
            q_temp[coordinate_idx] = q_var
            return self.lagrangian(q_temp, q_dot, t)
        
        dL_dq = numerical_derivative(L_partial_q, q[coordinate_idx], dx=1e-8)
        
        # Partial derivative with respect to q_dot
        def L_partial_q_dot(q_dot_var):
            q_dot_temp = q_dot.copy()
            q_dot_temp[coordinate_idx] = q_dot_var
            return self.lagrangian(q, q_dot_temp, t)
        
        dL_dq_dot = numerical_derivative(L_partial_q_dot, q_dot[coordinate_idx], dx=1e-8)
        
        # Time derivative of ∂L/∂q̇
        # This requires computing the total derivative
        def dL_dq_dot_func(t_var):
            return derivative(L_partial_q_dot, q_dot[coordinate_idx], dx=1e-8)
        
        # Approximate the time derivative using finite differences
        dt = 1e-6
        dL_dq_dot_dt = numerical_derivative(lambda t_var: numerical_derivative(L_partial_q_dot, 
                                 q_dot[coordinate_idx] + q_ddot[coordinate_idx] * (t_var - t), 
                                 dx=1e-8), t, dx=dt)
        
        # For computational efficiency, we can use the chain rule:
        # d/dt(∂L/∂q̇) = ∂²L/∂q∂q̇ * q̇ + ∂²L/∂q̇² * q̈ + ∂²L/∂q̇∂t
        
        # Second partial derivatives
        def L_qq_dot(q_var):
            q_temp = q.copy()
            q_temp[coordinate_idx] = q_var
            def inner(q_dot_var):
                q_dot_temp = q_dot.copy()
                q_dot_temp[coordinate_idx] = q_dot_var
                return self.lagrangian(q_temp, q_dot_temp, t)
            return numerical_derivative(inner, q_dot[coordinate_idx], dx=1e-8)
        
        d2L_dq_dq_dot = numerical_derivative(L_qq_dot, q[coordinate_idx], dx=1e-8)
        
        def L_q_dot_q_dot(q_dot_var):
            q_dot_temp = q_dot.copy()
            q_dot_temp[coordinate_idx] = q_dot_var
            def inner(q_dot_var2):
                q_dot_temp2 = q_dot.copy()
                q_dot_temp2[coordinate_idx] = q_dot_var2
                return self.lagrangian(q, q_dot_temp2, t)
            return numerical_derivative(inner, q_dot[coordinate_idx], dx=1e-8)
        
        d2L_dq_dot2 = numerical_derivative(L_q_dot_q_dot, q_dot[coordinate_idx], dx=1e-8)
        
        # Time derivative approximation
        d_dt_dL_dq_dot = (d2L_dq_dq_dot * q_dot[coordinate_idx] + 
                         d2L_dq_dot2 * q_ddot[coordinate_idx])
        
        # Euler-Lagrange equation
        euler_lagrange = d_dt_dL_dq_dot - dL_dq
        
        return euler_lagrange
    
    def solve_trajectory(self, initial_conditions: Dict[str, np.ndarray], 
                        time_span: Tuple[float, float], 
                        n_points: int = 1000,
                        method: str = 'RK45') -> Dict[str, np.ndarray]:
        """
        Solve for the trajectory using the Euler-Lagrange equations.
        
        Parameters:
        -----------
        initial_conditions : dict
            Dictionary with keys 'q0' (initial positions) and 'q_dot0' (initial velocities)
        time_span : tuple
            (t_start, t_end) for the integration
        n_points : int
            Number of time points to evaluate
        method : str
            Integration method for scipy.integrate.solve_ivp
            
        Returns:
        --------
        dict
            Dictionary containing 't', 'q', and 'q_dot' arrays
        """
        q0 = initial_conditions['q0']
        q_dot0 = initial_conditions['q_dot0']
        
        if len(q0) != self.n_coordinates or len(q_dot0) != self.n_coordinates:
            raise ValueError(f"Initial conditions must have {self.n_coordinates} coordinates")
        
        # Create state vector [q1, q2, ..., qn, q_dot1, q_dot2, ..., q_dotn]
        y0 = np.concatenate([q0, q_dot0])
        
        def system_of_odes(t, y):
            """
            Convert Euler-Lagrange equations to a system of first-order ODEs.
            """
            n = self.n_coordinates
            q = y[:n]
            q_dot = y[n:]
            
            # We need to solve for q_ddot from the Euler-Lagrange equations
            # This is generally a system of nonlinear equations
            
            def residual_function(q_ddot):
                residuals = np.zeros(n)
                for i in range(n):
                    residuals[i] = self.compute_euler_lagrange_equation(
                        q, q_dot, q_ddot, t, i)
                return residuals
            
            # Solve for accelerations
            try:
                q_ddot_solution = scipy.optimize.fsolve(residual_function, 
                                                      np.zeros(n), xtol=1e-12)
            except:
                # Fallback to a simpler approach
                q_ddot_solution = np.zeros(n)
                warnings.warn("Could not solve for accelerations, using zero acceleration")
            
            # Return derivatives [q_dot, q_ddot]
            dydt = np.concatenate([q_dot, q_ddot_solution])
            return dydt
        
        # Solve the ODE system
        t_eval = np.linspace(time_span[0], time_span[1], n_points)
        
        try:
            solution = scipy.integrate.solve_ivp(
                system_of_odes, time_span, y0, 
                t_eval=t_eval, method=method,
                rtol=1e-8, atol=1e-10
            )
            
            if not solution.success:
                raise RuntimeError(f"Integration failed: {solution.message}")
            
            n = self.n_coordinates
            result = {
                't': solution.t,
                'q': solution.y[:n],
                'q_dot': solution.y[n:]
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to solve trajectory: {str(e)}")
    
    def find_stationary_action_path(self, boundary_conditions: Dict[str, Tuple], 
                                  time_span: Tuple[float, float],
                                  n_segments: int = 100) -> Dict[str, np.ndarray]:
        """
        Find the path that makes the action stationary using variational methods.
        
        This implements the principle of least action by discretizing the path
        and optimizing the action integral.
        
        Parameters:
        -----------
        boundary_conditions : dict
            Dictionary with 'q_start' and 'q_end' as tuples of (time, position)
        time_span : tuple
            (t_start, t_end) for the path
        n_segments : int
            Number of segments to discretize the path
            
        Returns:
        --------
        dict
            Dictionary containing the optimal path 't', 'q', 'q_dot'
        """
        t_start, t_end = time_span
        q_start = np.array(boundary_conditions['q_start'][1])
        q_end = np.array(boundary_conditions['q_end'][1])
        
        # Create time grid
        t_grid = np.linspace(t_start, t_end, n_segments + 1)
        dt = t_grid[1] - t_grid[0]
        
        # Initial guess: linear interpolation between boundary conditions
        n_coords = self.n_coordinates
        n_params = (n_segments - 1) * n_coords  # Interior points only
        
        def action_functional(params):
            """
            Compute the action integral for a given path.
            """
            # Reconstruct the full path including boundary conditions
            q_path = np.zeros((n_coords, n_segments + 1))
            q_path[:, 0] = q_start
            q_path[:, -1] = q_end
            
            # Fill in interior points
            param_idx = 0
            for i in range(1, n_segments):
                for j in range(n_coords):
                    q_path[j, i] = params[param_idx]
                    param_idx += 1
            
            # Compute velocities using finite differences
            q_dot_path = np.zeros_like(q_path)
            for i in range(n_segments + 1):
                if i == 0:
                    q_dot_path[:, i] = (q_path[:, i+1] - q_path[:, i]) / dt
                elif i == n_segments:
                    q_dot_path[:, i] = (q_path[:, i] - q_path[:, i-1]) / dt
                else:
                    q_dot_path[:, i] = (q_path[:, i+1] - q_path[:, i-1]) / (2 * dt)
            
            # Compute action integral using trapezoidal rule
            action = 0.0
            for i in range(n_segments):
                t_i = t_grid[i]
                L_i = self.lagrangian(q_path[:, i], q_dot_path[:, i], t_i)
                
                t_ip1 = t_grid[i + 1]
                L_ip1 = self.lagrangian(q_path[:, i + 1], q_dot_path[:, i + 1], t_ip1)
                
                action += 0.5 * (L_i + L_ip1) * dt
            
            return action
        
        # Initial guess: linear interpolation
        initial_params = []
        for i in range(1, n_segments):
            alpha = i / n_segments
            q_interp = (1 - alpha) * q_start + alpha * q_end
            for j in range(n_coords):
                initial_params.append(q_interp[j])
        
        initial_params = np.array(initial_params)
        
        # Minimize the action
        try:
            result = scipy.optimize.minimize(
                action_functional, initial_params,
                method='BFGS', 
                options={'gtol': 1e-8, 'ftol': 1e-8}
            )
            
            if not result.success:
                warnings.warn(f"Optimization may not have converged: {result.message}")
            
            # Reconstruct the optimal path
            optimal_params = result.x
            q_optimal = np.zeros((n_coords, n_segments + 1))
            q_optimal[:, 0] = q_start
            q_optimal[:, -1] = q_end
            
            param_idx = 0
            for i in range(1, n_segments):
                for j in range(n_coords):
                    q_optimal[j, i] = optimal_params[param_idx]
                    param_idx += 1
            
            # Compute velocities
            q_dot_optimal = np.zeros_like(q_optimal)
            for i in range(n_segments + 1):
                if i == 0:
                    q_dot_optimal[:, i] = (q_optimal[:, i+1] - q_optimal[:, i]) / dt
                elif i == n_segments:
                    q_dot_optimal[:, i] = (q_optimal[:, i] - q_optimal[:, i-1]) / dt
                else:
                    q_dot_optimal[:, i] = (q_optimal[:, i+1] - q_optimal[:, i-1]) / (2 * dt)
            
            return {
                't': t_grid,
                'q': q_optimal,
                'q_dot': q_dot_optimal,
                'action': result.fun,
                'optimization_result': result
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to find stationary action path: {str(e)}")
    
    def plot_trajectory(self, solution: Dict[str, np.ndarray], 
                       coordinate_labels: Optional[List[str]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Plot the trajectory solution.
        
        Parameters:
        -----------
        solution : dict
            Solution dictionary from solve_trajectory or find_stationary_action_path
        coordinate_labels : list, optional
            Labels for the coordinates
        save_path : str, optional
            Path to save the plot
        """
        t = solution['t']
        q = solution['q']
        q_dot = solution['q_dot']
        
        if coordinate_labels is None:
            coordinate_labels = [f'q_{i+1}' for i in range(self.n_coordinates)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot positions
        for i in range(self.n_coordinates):
            ax1.plot(t, q[i], label=coordinate_labels[i], linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position')
        ax1.set_title('Generalized Coordinates vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot velocities
        for i in range(self.n_coordinates):
            ax2.plot(t, q_dot[i], label=f'd{coordinate_labels[i]}/dt', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Generalized Velocities vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def analyze_conservation_laws(self, solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Analyze conservation laws for the given solution.
        
        Parameters:
        -----------
        solution : dict
            Solution dictionary from solve_trajectory
            
        Returns:
        --------
        dict
            Dictionary containing conserved quantities
        """
        t = solution['t']
        q = solution['q']
        q_dot = solution['q_dot']
        
        # Compute energy (if the Lagrangian is autonomous)
        energy = []
        for i in range(len(t)):
            q_i = q[:, i]
            q_dot_i = q_dot[:, i]
            t_i = t[i]
            
            # Energy = Σ(q̇ᵢ * ∂L/∂q̇ᵢ) - L
            total_energy = 0.0
            for j in range(self.n_coordinates):
                def L_partial_q_dot_j(q_dot_var):
                    q_dot_temp = q_dot_i.copy()
                    q_dot_temp[j] = q_dot_var
                    return self.lagrangian(q_i, q_dot_temp, t_i)
                
                dL_dq_dot_j = numerical_derivative(L_partial_q_dot_j, q_dot_i[j], dx=1e-8)
                total_energy += q_dot_i[j] * dL_dq_dot_j
            
            total_energy -= self.lagrangian(q_i, q_dot_i, t_i)
            energy.append(total_energy)
        
        return {
            'energy': np.array(energy),
            'time': t
        }


def create_simple_harmonic_oscillator_lagrangian(mass: float = 1.0, 
                                                spring_constant: float = 1.0) -> Callable:
    """
    Create a Lagrangian for a simple harmonic oscillator.
    
    L = (1/2) * m * q̇² - (1/2) * k * q²
    
    Parameters:
    -----------
    mass : float
        Mass of the oscillator
    spring_constant : float
        Spring constant
        
    Returns:
    --------
    callable
        Lagrangian function
    """
    def lagrangian(q, q_dot, t):
        kinetic_energy = 0.5 * mass * q_dot[0]**2
        potential_energy = 0.5 * spring_constant * q[0]**2
        return kinetic_energy - potential_energy
    
    return lagrangian


def create_pendulum_lagrangian(length: float = 1.0, mass: float = 1.0, 
                              gravity: float = 9.81) -> Callable:
    """
    Create a Lagrangian for a simple pendulum.
    
    L = (1/2) * m * l² * θ̇² + m * g * l * cos(θ)
    
    Parameters:
    -----------
    length : float
        Length of the pendulum
    mass : float
        Mass of the pendulum bob
    gravity : float
        Gravitational acceleration
        
    Returns:
    --------
    callable
        Lagrangian function
    """
    def lagrangian(q, q_dot, t):
        theta = q[0]
        theta_dot = q_dot[0]
        
        kinetic_energy = 0.5 * mass * length**2 * theta_dot**2
        potential_energy = -mass * gravity * length * np.cos(theta)
        
        return kinetic_energy - potential_energy
    
    return lagrangian


def create_double_pendulum_lagrangian(l1: float = 1.0, l2: float = 1.0,
                                    m1: float = 1.0, m2: float = 1.0,
                                    gravity: float = 9.81) -> Callable:
    """
    Create a Lagrangian for a double pendulum.
    
    Parameters:
    -----------
    l1, l2 : float
        Lengths of the pendulum segments
    m1, m2 : float
        Masses of the pendulum bobs
    gravity : float
        Gravitational acceleration
        
    Returns:
    --------
    callable
        Lagrangian function
    """
    def lagrangian(q, q_dot, t):
        theta1, theta2 = q[0], q[1]
        theta1_dot, theta2_dot = q_dot[0], q_dot[1]
        
        # Kinetic energy
        T1 = 0.5 * m1 * l1**2 * theta1_dot**2
        T2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                         2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
        
        # Potential energy
        V1 = -m1 * gravity * l1 * np.cos(theta1)
        V2 = -m2 * gravity * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
        
        return T1 + T2 - V1 - V2
    
    return lagrangian