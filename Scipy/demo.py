#!/usr/bin/env python3
"""
Simple demonstration of the Euler-Lagrangian solver.

This script demonstrates the core functionality without requiring
matplotlib for plotting, making it suitable for environments
where visualization is not available.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import sys
import os

# Add the current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from euler_lagrangian import (
    EulerLagrangianSolver,
    create_simple_harmonic_oscillator_lagrangian,
    create_pendulum_lagrangian
)


def demo_simple_harmonic_oscillator():
    """Demonstrate simple harmonic oscillator solution."""
    print("Simple Harmonic Oscillator Demo")
    print("=" * 40)
    
    # Physical parameters
    mass = 1.0
    spring_constant = 4.0  # ω = 2 rad/s
    omega = np.sqrt(spring_constant / mass)
    
    print(f"Mass: {mass} kg")
    print(f"Spring constant: {spring_constant} N/m")
    print(f"Natural frequency: {omega:.3f} rad/s")
    
    # Create Lagrangian and solver
    lagrangian = create_simple_harmonic_oscillator_lagrangian(mass, spring_constant)
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
    
    # Initial conditions: q(0) = 1, q̇(0) = 0
    initial_conditions = {
        'q0': np.array([1.0]),
        'q_dot0': np.array([0.0])
    }
    
    # Solve for one period
    period = 2 * np.pi / omega
    time_span = (0, period)
    
    try:
        solution = solver.solve_trajectory(
            initial_conditions, time_span, n_points=100
        )
        
        # Analyze results
        print(f"Period: {period:.3f} s")
        print(f"Initial position: {solution['q'][0, 0]:.6f} m")
        print(f"Final position: {solution['q'][0, -1]:.6f} m")
        print(f"Position at t=T/4: {solution['q'][0, 25]:.6f} m")
        
        # Analytical comparison
        t_quarter = period / 4
        q_analytical_quarter = 1.0 * np.cos(omega * t_quarter)
        print(f"Analytical at t=T/4: {q_analytical_quarter:.6f} m")
        
        # Energy analysis
        conservation = solver.analyze_conservation_laws(solution)
        energy_std = np.std(conservation['energy'])
        energy_mean = np.mean(conservation['energy'])
        
        print(f"Energy conservation:")
        print(f"  Mean energy: {energy_mean:.6f} J")
        print(f"  Energy std: {energy_std:.2e} J")
        print(f"  Theoretical energy: {0.5 * spring_constant * 1.0**2:.6f} J")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def demo_pendulum():
    """Demonstrate pendulum solution."""
    print("\nSimple Pendulum Demo")
    print("=" * 40)
    
    # Physical parameters
    length = 1.0
    mass = 1.0
    gravity = 9.81
    
    print(f"Length: {length} m")
    print(f"Mass: {mass} kg")
    print(f"Gravity: {gravity} m/s²")
    
    # Create Lagrangian and solver
    lagrangian = create_pendulum_lagrangian(length, mass, gravity)
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
    
    # Initial conditions: θ(0) = π/6 (30 degrees), θ̇(0) = 0
    initial_angle = np.pi / 6
    initial_conditions = {
        'q0': np.array([initial_angle]),
        'q_dot0': np.array([0.0])
    }
    
    print(f"Initial angle: {initial_angle * 180 / np.pi:.1f}°")
    
    # Solve for multiple oscillations
    omega_small = np.sqrt(gravity / length)  # Small angle approximation
    period_small = 2 * np.pi / omega_small
    time_span = (0, 2 * period_small)
    
    try:
        solution = solver.solve_trajectory(
            initial_conditions, time_span, n_points=200
        )
        
        # Analyze results
        print(f"Small angle period: {period_small:.3f} s")
        print(f"Initial angle: {solution['q'][0, 0] * 180 / np.pi:.3f}°")
        print(f"Final angle: {solution['q'][0, -1] * 180 / np.pi:.3f}°")
        
        # Find actual period from zero crossings
        angles = solution['q'][0]
        times = solution['t']
        
        zero_crossings = []
        for i in range(1, len(angles)):
            if (angles[i-1] > 0 and angles[i] <= 0):
                # Linear interpolation for exact crossing
                t_cross = times[i-1] + (times[i] - times[i-1]) * \
                         (-angles[i-1]) / (angles[i] - angles[i-1])
                zero_crossings.append(t_cross)
        
        if len(zero_crossings) >= 2:
            actual_period = 2 * (zero_crossings[1] - zero_crossings[0])
            print(f"Actual period: {actual_period:.3f} s")
            print(f"Period ratio (actual/small-angle): {actual_period/period_small:.3f}")
        
        # Energy analysis
        conservation = solver.analyze_conservation_laws(solution)
        energy_std = np.std(conservation['energy'])
        energy_mean = np.mean(conservation['energy'])
        
        print(f"Energy conservation:")
        print(f"  Mean energy: {energy_mean:.6f} J")
        print(f"  Energy std: {energy_std:.2e} J")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def demo_variational_path():
    """Demonstrate variational path finding."""
    print("\nVariational Path Finding Demo")
    print("=" * 40)
    
    # Simple boundary value problem
    lagrangian = create_simple_harmonic_oscillator_lagrangian(1.0, 1.0)
    solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)
    
    # Boundary conditions: q(0) = 0, q(π/2) = 1
    boundary_conditions = {
        'q_start': (0.0, np.array([0.0])),
        'q_end': (np.pi/2, np.array([1.0]))
    }
    
    time_span = (0.0, np.pi/2)
    
    print(f"Boundary conditions:")
    print(f"  q(0) = {boundary_conditions['q_start'][1][0]}")
    print(f"  q(π/2) = {boundary_conditions['q_end'][1][0]}")
    
    try:
        result = solver.find_stationary_action_path(
            boundary_conditions, time_span, n_segments=30
        )
        
        print(f"Optimization successful: {result['optimization_result'].success}")
        print(f"Action value: {result['action']:.6f}")
        
        # Compare with analytical solution: q(t) = sin(t) for this problem
        t_mid = np.pi/4
        q_mid_numerical = result['q'][0, len(result['t'])//2]
        q_mid_analytical = np.sin(t_mid)
        
        print(f"At t = π/4:")
        print(f"  Numerical: {q_mid_numerical:.6f}")
        print(f"  Analytical: {q_mid_analytical:.6f}")
        print(f"  Error: {abs(q_mid_numerical - q_mid_analytical):.6e}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all demonstrations."""
    print("Numerical Euler-Lagrangian Solver Demonstration")
    print("=" * 50)
    print()
    
    success_count = 0
    total_demos = 3
    
    # Run demonstrations
    if demo_simple_harmonic_oscillator():
        success_count += 1
    
    if demo_pendulum():
        success_count += 1
    
    if demo_variational_path():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Demonstration Summary: {success_count}/{total_demos} successful")
    
    if success_count == total_demos:
        print("✓ All demonstrations completed successfully!")
        print("The Euler-Lagrangian solver is working correctly.")
    else:
        print("⚠ Some demonstrations failed.")
        print("Check the error messages above for details.")
    
    print("=" * 50)
    
    return success_count == total_demos


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)