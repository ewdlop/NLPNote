"""
Deadly Attractors Demo: Interactive Social Dynamics Visualization

This script demonstrates various deadly attractor phenomena in social systems
with visualizations and analysis.

Run this script to see phase portraits and trajectories for different
social dynamics models.
"""

import numpy as np
import matplotlib.pyplot as plt
from deadly_attractor_simulator import (
    HateSpiralModel, EchoChamberModel, PrisonersDilemmaModel, 
    TragedyOfCommonsModel, DeadlyAttractorSimulator
)

def run_hate_spiral_demo():
    """Demonstrate the hate spiral model with multiple scenarios."""
    print("🔥 Hate Spiral Model Demonstration")
    print("=" * 50)
    
    # Create model with different parameter sets
    scenarios = [
        {"name": "High Retaliation", "params": {"alpha": 1.0, "beta": 0.05, "gamma": 0.1}},
        {"name": "Moderate Conflict", "params": {"alpha": 0.5, "beta": 0.15, "gamma": 0.2}},
        {"name": "Forgiving Society", "params": {"alpha": 0.3, "beta": 0.4, "gamma": 0.1}}
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, scenario in enumerate(scenarios):
        model = HateSpiralModel(**scenario["params"])
        simulator = DeadlyAttractorSimulator(model)
        
        # Test multiple initial conditions
        initial_conditions = [
            [0.8, 0.7],   # High mutual trust
            [0.3, 0.4],   # Low trust
            [-0.2, 0.5],  # Asymmetric trust
        ]
        
        colors = ['blue', 'green', 'orange']
        
        for j, init_state in enumerate(initial_conditions):
            trajectory = simulator.simulate(init_state, time_span=25)
            axes[i].plot(trajectory[:, 0], trajectory[:, 1], 
                        color=colors[j], linewidth=2, alpha=0.8,
                        label=f'Start: ({init_state[0]}, {init_state[1]})')
            axes[i].plot(init_state[0], init_state[1], 'o', color=colors[j], markersize=8)
            axes[i].plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=colors[j], markersize=8)
        
        # Mark deadly attractor
        axes[i].plot(-1, -1, 'rX', markersize=15, markeredgecolor='black', 
                    markeredgewidth=2, label='Deadly Attractor')
        
        axes[i].set_xlim(-1.5, 1)
        axes[i].set_ylim(-1.5, 1)
        axes[i].set_xlabel('Group A Trust Level')
        axes[i].set_ylabel('Group B Trust Level')
        axes[i].set_title(f'{scenario["name"]}\nα={scenario["params"]["alpha"]}, β={scenario["params"]["beta"]}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Hate Spiral Model: Different Scenarios', fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
    print("Analysis: Higher retaliation (α) leads faster to deadly attractor.")
    print("Higher forgiveness (β) can sometimes prevent total breakdown.")


def run_echo_chamber_demo():
    """Demonstrate the echo chamber model."""
    print("\n📢 Echo Chamber Model Demonstration")
    print("=" * 50)
    
    model = EchoChamberModel(alpha=1.5, beta=2.5, gamma=0.3, delta=0.6, epsilon=0.1)
    simulator = DeadlyAttractorSimulator(model)
    
    # Show how moderate opinions get polarized
    initial_conditions = [
        [0.1, 0.8],   # Slight right lean, high diversity
        [-0.1, 0.8],  # Slight left lean, high diversity
        [0.0, 0.9],   # Perfectly neutral, high diversity
        [0.5, 0.3],   # Moderate right, low diversity
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, init_state in enumerate(initial_conditions):
        trajectory = simulator.simulate(init_state, time_span=20)
        
        # Phase portrait
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Start: ({init_state[0]:.1f}, {init_state[1]:.1f})')
        ax1.plot(init_state[0], init_state[1], 'o', color=colors[i], markersize=8)
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=colors[i], markersize=8)
        
        # Time series
        t = np.linspace(0, 20, len(trajectory))
        ax2.plot(t, trajectory[:, 0], color=colors[i], linewidth=2, alpha=0.8,
                label=f'Opinion: Start {init_state[0]:.1f}')
    
    # Mark extreme attractors
    equilibria = model.get_equilibria()
    for eq in equilibria:
        if abs(eq[0]) > 0.5:  # Extreme opinions
            ax1.plot(eq[0], eq[1], 'rX', markersize=12, markeredgecolor='black',
                    markeredgewidth=1, alpha=0.7)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Opinion Position (-1=Left, +1=Right)')
    ax1.set_ylabel('Opinion Diversity')
    ax1.set_title('Echo Chamber Phase Portrait')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Opinion Position')
    ax2.set_title('Opinion Polarization Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis: Even small initial biases can lead to extreme polarization.")
    print("Diversity decreases as opinions become more extreme.")


def run_comparative_analysis():
    """Compare multiple deadly attractor models side by side."""
    print("\n⚖️ Comparative Analysis of Deadly Attractors")
    print("=" * 50)
    
    models = [
        ("Hate Spiral", HateSpiralModel(alpha=0.6, beta=0.1, gamma=0.2), [0.5, 0.5]),
        ("Echo Chamber", EchoChamberModel(alpha=1.2, beta=2.0, gamma=0.3), [0.2, 0.7]),
        ("Prisoner's Dilemma", PrisonersDilemmaModel(mu=0.2, lamb=0.3, nu=0.3), [0.7, 0.3]),
        ("Tragedy of Commons", TragedyOfCommonsModel(r=0.4, alpha=0.5, beta=0.1), [60, 0.8])
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (name, model, init_state) in enumerate(models):
        simulator = DeadlyAttractorSimulator(model)
        
        # Simulate trajectory
        trajectory = simulator.simulate(init_state, time_span=30)
        
        # Plot time evolution
        t = np.linspace(0, 30, len(trajectory))
        
        axes[i].plot(t, trajectory[:, 0], 'b-', linewidth=2, label='Variable 1', alpha=0.8)
        axes[i].plot(t, trajectory[:, 1], 'r-', linewidth=2, label='Variable 2', alpha=0.8)
        
        # Mark initial and final points
        axes[i].plot(0, trajectory[0, 0], 'bo', markersize=8, label='Start 1')
        axes[i].plot(0, trajectory[0, 1], 'ro', markersize=8, label='Start 2')
        axes[i].plot(30, trajectory[-1, 0], 'bs', markersize=8, label='End 1')
        axes[i].plot(30, trajectory[-1, 1], 'rs', markersize=8, label='End 2')
        
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('State Variables')
        axes[i].set_title(f'{name} Model\nConvergence to Deadly Attractor')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add text box with final values
        final_text = f'Final: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})'
        axes[i].text(0.7, 0.95, final_text, transform=axes[i].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top')
    
    plt.tight_layout()
    plt.suptitle('Convergence to Deadly Attractors in Social Systems', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
    print("Analysis Complete!")
    print("- All models show convergence to destructive equilibria")
    print("- Different timescales and dynamics, but similar outcomes")
    print("- Early intervention critical before crossing tipping points")


def run_stability_analysis():
    """Analyze the stability properties of deadly attractors."""
    print("\n🔍 Stability Analysis of Deadly Attractors")
    print("=" * 50)
    
    # Analyze hate spiral model in detail
    model = HateSpiralModel(alpha=0.8, beta=0.1, gamma=0.3)
    simulator = DeadlyAttractorSimulator(model)
    
    # Points to analyze
    points = [
        ("Peaceful State", [0.5, 0.5]),
        ("Neutral State", [0.0, 0.0]),
        ("Deadly Attractor", [-1.0, -1.0]),
        ("Partial Conflict", [-0.3, 0.2])
    ]
    
    print(f"{'Point':<20} {'Stability':<12} {'Max Real λ':<12} {'Classification'}")
    print("-" * 65)
    
    for name, point in points:
        stability = simulator.analyze_stability(point)
        
        classification = "Stable" if stability['is_stable'] else \
                        "Unstable" if stability['is_unstable'] else "Marginal"
        
        print(f"{name:<20} {classification:<12} {stability['max_real_part']:<12.4f} "
              f"{str(stability['eigenvalues'])}")
    
    print("\nInterpretation:")
    print("- Negative real parts → Stable (attracts nearby trajectories)")
    print("- Positive real parts → Unstable (repels nearby trajectories)")
    print("- The deadly attractor at (-1,-1) is locally stable")
    print("- Peaceful states are typically unstable when retaliation is strong")


if __name__ == "__main__":
    print("🎭 Deadly Attractors in Social Dynamics: Interactive Demo")
    print("=" * 60)
    print("This demonstration shows how social systems can evolve toward")
    print("destructive stable states - deadly attractors that are difficult to escape.")
    print()
    
    try:
        # Run all demonstrations
        run_hate_spiral_demo()
        run_echo_chamber_demo() 
        run_comparative_analysis()
        run_stability_analysis()
        
        print("\n" + "=" * 60)
        print("🎯 Key Insights from the Analysis:")
        print("1. Small initial differences can lead to drastically different outcomes")
        print("2. Deadly attractors are stable - once reached, systems stay there")
        print("3. Early intervention is crucial before crossing tipping points")
        print("4. Different social phenomena share similar mathematical structures")
        print("5. Prevention is much easier than escape from deadly attractors")
        print("\n📚 See deadly-attractors-social-dynamics.md for detailed theory!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure matplotlib is available for visualizations.")