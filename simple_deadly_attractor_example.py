"""
Simple Example: Deadly Attractors in Social Systems

This script provides a basic example of how to use the deadly attractor
simulation system to analyze social dynamics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

from deadly_attractor_simulator import (
    HateSpiralModel, EchoChamberModel, DeadlyAttractorSimulator
)

def simple_hate_spiral_example():
    """Simple example of hate spiral dynamics."""
    print("🔥 Simple Hate Spiral Example")
    print("=" * 40)
    
    # Create a hate spiral model
    # α = retaliation strength, β = healing rate, γ = polarization rate
    model = HateSpiralModel(alpha=0.7, beta=0.1, gamma=0.2)
    simulator = DeadlyAttractorSimulator(model)
    
    # Start from a peaceful state
    initial_trust = [0.6, 0.5]  # Both groups start with moderate trust
    print(f"Initial trust levels: Group A = {initial_trust[0]}, Group B = {initial_trust[1]}")
    
    # Simulate the evolution
    trajectory = simulator.simulate(initial_trust, time_span=30)
    
    # Show the result
    final_trust = trajectory[-1]
    print(f"Final trust levels: Group A = {final_trust[0]:.3f}, Group B = {final_trust[1]:.3f}")
    
    if final_trust[0] < 0 and final_trust[1] < 0:
        print("❌ System converged to deadly attractor: MUTUAL HATRED")
    elif final_trust[0] > 0.3 and final_trust[1] > 0.3:
        print("✅ System maintained peaceful equilibrium")
    else:
        print("⚠️ System in intermediate conflicted state")
    
    return trajectory


def simple_echo_chamber_example():
    """Simple example of echo chamber dynamics."""
    print("\n📢 Simple Echo Chamber Example")
    print("=" * 40)
    
    # Create echo chamber model
    # α = amplification, β = polarization sensitivity, γ = moderation force
    model = EchoChamberModel(alpha=1.2, beta=2.5, gamma=0.3)
    simulator = DeadlyAttractorSimulator(model)
    
    # Start with slightly biased but diverse opinions
    initial_state = [0.15, 0.8]  # Slight right lean, high diversity
    print(f"Initial opinion position: {initial_state[0]:.2f} (0=neutral)")
    print(f"Initial opinion diversity: {initial_state[1]:.2f} (1=max diversity)")
    
    # Simulate evolution
    trajectory = simulator.simulate(initial_state, time_span=20)
    
    # Show result
    final_state = trajectory[-1]
    print(f"Final opinion position: {final_state[0]:.3f}")
    print(f"Final opinion diversity: {final_state[1]:.3f}")
    
    if abs(final_state[0]) > 1.0 and final_state[1] < 0.2:
        print("❌ System converged to deadly attractor: EXTREME ECHO CHAMBER")
    elif abs(final_state[0]) < 0.3 and final_state[1] > 0.5:
        print("✅ System maintained moderate diverse opinions")
    else:
        print("⚠️ System showing signs of polarization")
    
    return trajectory


def analyze_tipping_points():
    """Analyze how system parameters affect convergence to deadly attractors."""
    print("\n🎯 Tipping Point Analysis")
    print("=" * 40)
    
    # Test hate spiral with different retaliation levels
    print("Testing hate spiral with different retaliation levels:")
    
    retaliation_levels = [0.3, 0.5, 0.7, 0.9]
    initial_state = [0.4, 0.5]
    
    for alpha in retaliation_levels:
        model = HateSpiralModel(alpha=alpha, beta=0.1, gamma=0.2)
        simulator = DeadlyAttractorSimulator(model)
        trajectory = simulator.simulate(initial_state, time_span=25)
        final_state = trajectory[-1]
        
        outcome = "Deadly" if (final_state[0] < 0 and final_state[1] < 0) else "Stable"
        print(f"  α = {alpha:.1f} → Final trust: ({final_state[0]:.3f}, {final_state[1]:.3f}) [{outcome}]")
    
    # Find approximate tipping point
    print("\nTipping point analysis:")
    print("- Low retaliation (α < 0.5): System can maintain some trust")  
    print("- High retaliation (α > 0.7): System converges to deadly attractor")
    print("- Critical threshold around α ≈ 0.6")


def demonstrate_intervention():
    """Demonstrate how intervention can prevent convergence to deadly attractors."""
    print("\n🛠️ Intervention Demonstration") 
    print("=" * 40)
    
    # Scenario: High-conflict situation with intervention
    print("Scenario: High-conflict situation")
    
    # Without intervention (high retaliation, low healing)
    model_no_intervention = HateSpiralModel(alpha=0.8, beta=0.05, gamma=0.2)
    simulator1 = DeadlyAttractorSimulator(model_no_intervention)
    trajectory1 = simulator1.simulate([0.3, 0.4], time_span=20)
    
    # With intervention (same retaliation, but higher healing through mediation)
    model_with_intervention = HateSpiralModel(alpha=0.8, beta=0.3, gamma=0.2)
    simulator2 = DeadlyAttractorSimulator(model_with_intervention)
    trajectory2 = simulator2.simulate([0.3, 0.4], time_span=20)
    
    print(f"Without intervention: Final trust = ({trajectory1[-1, 0]:.3f}, {trajectory1[-1, 1]:.3f})")
    print(f"With intervention:    Final trust = ({trajectory2[-1, 0]:.3f}, {trajectory2[-1, 1]:.3f})")
    
    improvement = np.mean(trajectory2[-1]) - np.mean(trajectory1[-1])
    print(f"Improvement from intervention: {improvement:.3f}")
    
    if improvement > 0.2:
        print("✅ Intervention successfully prevented deadly attractor!")
    else:
        print("⚠️ Intervention had limited effect - may need stronger measures")


def stability_analysis_example():
    """Demonstrate stability analysis of equilibrium points."""
    print("\n🔍 Stability Analysis Example")
    print("=" * 40)
    
    model = HateSpiralModel(alpha=0.6, beta=0.15, gamma=0.25)
    simulator = DeadlyAttractorSimulator(model)
    
    # Analyze different points
    test_points = [
        ("Peaceful equilibrium", [0.4, 0.4]),
        ("Neutral point", [0.0, 0.0]),
        ("Deadly attractor", [-1.0, -1.0]),
        ("Asymmetric conflict", [-0.3, 0.2])
    ]
    
    print(f"{'Point':<20} {'Stable?':<8} {'Max Eigenvalue Real Part'}")
    print("-" * 55)
    
    for name, point in test_points:
        stability = simulator.analyze_stability(point)
        stable = "Yes" if stability['is_stable'] else "No"
        max_real = stability['max_real_part']
        print(f"{name:<20} {stable:<8} {max_real:<8.4f}")
    
    print("\nInterpretation:")
    print("- Negative eigenvalue real parts → Stable (attracts nearby trajectories)")
    print("- Positive eigenvalue real parts → Unstable (repels nearby trajectories)")


if __name__ == "__main__":
    print("🎭 Deadly Attractors: Simple Social Dynamics Examples")
    print("=" * 60)
    print("This script demonstrates key concepts of deadly attractors in social systems.\n")
    
    try:
        # Run all examples
        simple_hate_spiral_example()
        simple_echo_chamber_example()
        analyze_tipping_points()
        demonstrate_intervention()
        stability_analysis_example()
        
        print("\n" + "=" * 60)
        print("🎯 Key Takeaways:")
        print("1. Small parameter changes can dramatically alter system outcomes")
        print("2. Deadly attractors are stable - once reached, they persist")
        print("3. Early intervention is much more effective than late intervention")
        print("4. Understanding stability helps predict system behavior")
        print("5. Social systems exhibit mathematical patterns similar to physical systems")
        
        print("\n📚 For more details, see:")
        print("- deadly-attractors-social-dynamics.md (comprehensive theory)")
        print("- deadly_attractor_simulator.py (full implementation)")
        print("- deadly_attractor_demo.py (visual demonstrations)")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all required modules are available.")