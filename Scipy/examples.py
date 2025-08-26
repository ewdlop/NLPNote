"""
Examples demonstrating the Euler-Lagrangian solver.

This script provides practical examples of how to use the numerical
Euler-Lagrangian solver for various physical systems.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from euler_lagrangian import (
    EulerLagrangianSolver,
    create_simple_harmonic_oscillator_lagrangian,
    create_pendulum_lagrangian,
    create_double_pendulum_lagrangian
)


def example_simple_harmonic_oscillator():
    """
    Example: Simple Harmonic Oscillator
    
    This demonstrates solving the equation of motion for a mass-spring system.
    The analytical solution is q(t) = A*cos(ωt + φ) where ω = √(k/m).
    """
    print("=" * 60)
    print("Example 1: Simple Harmonic Oscillator")
    print("=" * 60)
    
    # Physical parameters
    mass = 1.0
    spring_constant = 4.0  # ω = 2 rad/s
    
    # Create the Lagrangian
    lagrangian = create_simple_harmonic_oscillator_lagrangian(mass, spring_constant)
    
    # Initialize the solver
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
    
    # Initial conditions: q(0) = 1, q̇(0) = 0
    initial_conditions = {
        'q0': np.array([1.0]),
        'q_dot0': np.array([0.0])
    }
    
    # Solve for one complete period
    omega = np.sqrt(spring_constant / mass)
    period = 2 * np.pi / omega
    time_span = (0, 2 * period)
    
    print(f"Physical parameters:")
    print(f"  Mass: {mass} kg")
    print(f"  Spring constant: {spring_constant} N/m")
    print(f"  Natural frequency: {omega:.3f} rad/s")
    print(f"  Period: {period:.3f} s")
    print()
    
    try:
        solution = solver.solve_trajectory(
            initial_conditions, time_span, n_points=500
        )
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        
        # Position vs time
        plt.subplot(2, 2, 1)
        plt.plot(solution['t'], solution['q'][0], 'b-', linewidth=2, label='Numerical')
        # Analytical solution for comparison
        t_analytical = solution['t']
        q_analytical = 1.0 * np.cos(omega * t_analytical)
        plt.plot(t_analytical, q_analytical, 'r--', linewidth=2, label='Analytical')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Simple Harmonic Oscillator - Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Velocity vs time
        plt.subplot(2, 2, 2)
        plt.plot(solution['t'], solution['q_dot'][0], 'b-', linewidth=2, label='Numerical')
        # Analytical velocity
        v_analytical = -omega * 1.0 * np.sin(omega * t_analytical)
        plt.plot(t_analytical, v_analytical, 'r--', linewidth=2, label='Analytical')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Simple Harmonic Oscillator - Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Phase space plot
        plt.subplot(2, 2, 3)
        plt.plot(solution['q'][0], solution['q_dot'][0], 'b-', linewidth=2)
        plt.xlabel('Position (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Phase Space')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Energy conservation
        conservation = solver.analyze_conservation_laws(solution)
        plt.subplot(2, 2, 4)
        plt.plot(conservation['time'], conservation['energy'], 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (J)')
        plt.title('Energy Conservation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/NLPNote/NLPNote/Scipy/simple_harmonic_oscillator.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print some statistics
        energy_variation = np.std(conservation['energy'])
        print(f"Results:")
        print(f"  Energy variation (std): {energy_variation:.2e} J")
        print(f"  Average energy: {np.mean(conservation['energy']):.6f} J")
        print(f"  Theoretical energy: {0.5 * spring_constant * 1.0**2:.6f} J")
        
    except Exception as e:
        print(f"Error solving simple harmonic oscillator: {e}")
    
    print()


def example_pendulum():
    """
    Example: Simple Pendulum
    
    This demonstrates solving the nonlinear pendulum equation.
    For small angles, this reduces to simple harmonic motion.
    """
    print("=" * 60)
    print("Example 2: Simple Pendulum")
    print("=" * 60)
    
    # Physical parameters
    length = 1.0
    mass = 1.0
    gravity = 9.81
    
    # Create the Lagrangian
    lagrangian = create_pendulum_lagrangian(length, mass, gravity)
    
    # Initialize the solver
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
    
    # Initial conditions: θ(0) = π/3 (60 degrees), θ̇(0) = 0
    initial_angle = np.pi / 3
    initial_conditions = {
        'q0': np.array([initial_angle]),
        'q_dot0': np.array([0.0])
    }
    
    # Solve for multiple periods
    omega_small = np.sqrt(gravity / length)  # Small angle approximation
    period_small = 2 * np.pi / omega_small
    time_span = (0, 3 * period_small)
    
    print(f"Physical parameters:")
    print(f"  Length: {length} m")
    print(f"  Mass: {mass} kg")
    print(f"  Gravity: {gravity} m/s²")
    print(f"  Initial angle: {initial_angle * 180 / np.pi:.1f}°")
    print(f"  Small angle frequency: {omega_small:.3f} rad/s")
    print(f"  Small angle period: {period_small:.3f} s")
    print()
    
    try:
        solution = solver.solve_trajectory(
            initial_conditions, time_span, n_points=1000
        )
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        
        # Angular position vs time
        plt.subplot(2, 2, 1)
        plt.plot(solution['t'], solution['q'][0] * 180 / np.pi, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Pendulum - Angular Position')
        plt.grid(True, alpha=0.3)
        
        # Angular velocity vs time
        plt.subplot(2, 2, 2)
        plt.plot(solution['t'], solution['q_dot'][0], 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Pendulum - Angular Velocity')
        plt.grid(True, alpha=0.3)
        
        # Phase space plot
        plt.subplot(2, 2, 3)
        plt.plot(solution['q'][0] * 180 / np.pi, solution['q_dot'][0], 'b-', linewidth=2)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Phase Space')
        plt.grid(True, alpha=0.3)
        
        # Energy conservation
        conservation = solver.analyze_conservation_laws(solution)
        plt.subplot(2, 2, 4)
        plt.plot(conservation['time'], conservation['energy'], 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (J)')
        plt.title('Energy Conservation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/NLPNote/NLPNote/Scipy/pendulum.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print some statistics
        energy_variation = np.std(conservation['energy'])
        print(f"Results:")
        print(f"  Energy variation (std): {energy_variation:.2e} J")
        print(f"  Average energy: {np.mean(conservation['energy']):.6f} J")
        
        # Estimate the actual period from zero crossings
        zero_crossings = []
        for i in range(1, len(solution['q'][0])):
            if (solution['q'][0][i-1] > 0 and solution['q'][0][i] <= 0):
                # Linear interpolation to find exact crossing
                t_cross = solution['t'][i-1] + (solution['t'][i] - solution['t'][i-1]) * \
                         (-solution['q'][0][i-1]) / (solution['q'][0][i] - solution['q'][0][i-1])
                zero_crossings.append(t_cross)
        
        if len(zero_crossings) >= 2:
            actual_period = 2 * (zero_crossings[1] - zero_crossings[0])
            print(f"  Actual period: {actual_period:.3f} s")
            print(f"  Small angle period: {period_small:.3f} s")
            print(f"  Period ratio: {actual_period / period_small:.3f}")
        
    except Exception as e:
        print(f"Error solving pendulum: {e}")
    
    print()


def example_variational_path():
    """
    Example: Variational Path Finding
    
    This demonstrates finding the path that minimizes action using
    the principle of least action (stationary action).
    """
    print("=" * 60)
    print("Example 3: Variational Path Finding")
    print("=" * 60)
    
    # Simple example: particle in a potential field
    # L = (1/2) * m * v² - V(x) where V(x) = (1/2) * k * x²
    mass = 1.0
    spring_constant = 1.0
    
    lagrangian = create_simple_harmonic_oscillator_lagrangian(mass, spring_constant)
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
    
    # Boundary conditions: start at x=0 at t=0, end at x=1 at t=π
    boundary_conditions = {
        'q_start': (0.0, np.array([0.0])),
        'q_end': (np.pi, np.array([1.0]))
    }
    
    time_span = (0.0, np.pi)
    
    print(f"Finding path from x=0 at t=0 to x=1 at t=π")
    print(f"Using simple harmonic oscillator Lagrangian")
    print()
    
    try:
        result = solver.find_stationary_action_path(
            boundary_conditions, time_span, n_segments=50
        )
        
        # Compare with analytical solution
        # For harmonic oscillator, the solution is q(t) = A*sin(ωt) + B*cos(ωt)
        # With boundary conditions q(0)=0, q(π)=1, and ω=1:
        # q(t) = sin(t)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(result['t'], result['q'][0], 'b-', linewidth=2, label='Variational')
        t_analytical = result['t']
        q_analytical = np.sin(t_analytical)
        plt.plot(t_analytical, q_analytical, 'r--', linewidth=2, label='Analytical')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Stationary Action Path')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(result['t'], result['q_dot'][0], 'b-', linewidth=2, label='Variational')
        v_analytical = np.cos(t_analytical)
        plt.plot(t_analytical, v_analytical, 'r--', linewidth=2, label='Analytical')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity from Stationary Action')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/NLPNote/NLPNote/Scipy/variational_path.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results:")
        print(f"  Action value: {result['action']:.6f}")
        print(f"  Optimization successful: {result['optimization_result'].success}")
        
        # Compute error relative to analytical solution
        error = np.sqrt(np.mean((result['q'][0] - q_analytical)**2))
        print(f"  RMS error vs analytical: {error:.6e}")
        
    except Exception as e:
        print(f"Error in variational path finding: {e}")
    
    print()


def example_double_pendulum():
    """
    Example: Double Pendulum (if time permits)
    
    This demonstrates a more complex system with two degrees of freedom.
    """
    print("=" * 60)
    print("Example 4: Double Pendulum")
    print("=" * 60)
    
    # Physical parameters
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    gravity = 9.81
    
    # Create the Lagrangian
    lagrangian = create_double_pendulum_lagrangian(l1, l2, m1, m2, gravity)
    
    # Initialize the solver
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=2)
    
    # Initial conditions: both pendulums at small angles
    initial_conditions = {
        'q0': np.array([np.pi/6, np.pi/4]),  # 30° and 45°
        'q_dot0': np.array([0.0, 0.0])
    }
    
    time_span = (0, 10)  # 10 seconds
    
    print(f"Physical parameters:")
    print(f"  Lengths: l1={l1} m, l2={l2} m")
    print(f"  Masses: m1={m1} kg, m2={m2} kg")
    print(f"  Initial angles: {30}°, {45}°")
    print()
    
    try:
        solution = solver.solve_trajectory(
            initial_conditions, time_span, n_points=1000
        )
        
        # Plot the results
        plt.figure(figsize=(15, 10))
        
        # Angular positions
        plt.subplot(2, 3, 1)
        plt.plot(solution['t'], solution['q'][0] * 180 / np.pi, 'b-', linewidth=2, label='θ₁')
        plt.plot(solution['t'], solution['q'][1] * 180 / np.pi, 'r-', linewidth=2, label='θ₂')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Angular Positions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Angular velocities
        plt.subplot(2, 3, 2)
        plt.plot(solution['t'], solution['q_dot'][0], 'b-', linewidth=2, label='ω₁')
        plt.plot(solution['t'], solution['q_dot'][1], 'r-', linewidth=2, label='ω₂')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Phase space for first pendulum
        plt.subplot(2, 3, 3)
        plt.plot(solution['q'][0] * 180 / np.pi, solution['q_dot'][0], 'b-', linewidth=1)
        plt.xlabel('θ₁ (degrees)')
        plt.ylabel('ω₁ (rad/s)')
        plt.title('Phase Space - Pendulum 1')
        plt.grid(True, alpha=0.3)
        
        # Phase space for second pendulum
        plt.subplot(2, 3, 4)
        plt.plot(solution['q'][1] * 180 / np.pi, solution['q_dot'][1], 'r-', linewidth=1)
        plt.xlabel('θ₂ (degrees)')
        plt.ylabel('ω₂ (rad/s)')
        plt.title('Phase Space - Pendulum 2')
        plt.grid(True, alpha=0.3)
        
        # Configuration space
        plt.subplot(2, 3, 5)
        plt.plot(solution['q'][0] * 180 / np.pi, solution['q'][1] * 180 / np.pi, 'g-', linewidth=1)
        plt.xlabel('θ₁ (degrees)')
        plt.ylabel('θ₂ (degrees)')
        plt.title('Configuration Space')
        plt.grid(True, alpha=0.3)
        
        # Energy conservation
        conservation = solver.analyze_conservation_laws(solution)
        plt.subplot(2, 3, 6)
        plt.plot(conservation['time'], conservation['energy'], 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (J)')
        plt.title('Energy Conservation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/NLPNote/NLPNote/Scipy/double_pendulum.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print some statistics
        energy_variation = np.std(conservation['energy'])
        print(f"Results:")
        print(f"  Energy variation (std): {energy_variation:.2e} J")
        print(f"  Average energy: {np.mean(conservation['energy']):.6f} J")
        
    except Exception as e:
        print(f"Error solving double pendulum: {e}")
        print("Note: Double pendulum is a challenging system that may require fine-tuned parameters")
    
    print()


def main():
    """
    Run all examples.
    """
    print("Numerical Euler-Lagrangian Solver Examples")
    print("==========================================")
    print()
    
    # Run examples
    example_simple_harmonic_oscillator()
    example_pendulum()
    example_variational_path()
    
    # Note: Double pendulum is computationally intensive and may fail
    # Uncomment the next line if you want to try it
    # example_double_pendulum()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()