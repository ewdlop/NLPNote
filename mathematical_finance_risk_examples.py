#!/usr/bin/env python3
"""
Mathematical Finance Risk Management Examples
===========================================

This module provides practical implementations of mathematical finance
concepts for non-monetary risk management applications.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class RiskAnalyzer:
    """
    A class for performing various risk analysis calculations using
    mathematical finance techniques.
    """
    
    def __init__(self):
        self.scenarios = []
        self.risk_metrics = {}
    
    def calculate_var(self, data: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for any type of loss data.
        
        Args:
            data: Array of loss values (can be time delays, quality scores, etc.)
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            VaR value at specified confidence level
        """
        return np.percentile(data, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, data: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            data: Array of loss values
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            CVaR value (expected loss beyond VaR threshold)
        """
        var = self.calculate_var(data, confidence_level)
        return np.mean(data[data <= var])
    
    def portfolio_optimization(self, expected_returns: np.ndarray, 
                             cov_matrix: np.ndarray, 
                             target_return: float = None) -> Dict[str, Any]:
        """
        Optimize resource allocation to minimize risk for given expected benefit.
        
        Args:
            expected_returns: Expected benefits/returns for each resource
            cov_matrix: Covariance matrix of returns
            target_return: Minimum required expected return
            
        Returns:
            Dictionary with optimal weights, risk, and return
        """
        n_assets = len(expected_returns)
        
        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        def portfolio_return(weights):
            return np.dot(weights, expected_returns)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w: portfolio_return(w) - target_return
            })
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_risk, x0, constraints=constraints, bounds=bounds)
        
        if result.success:
            optimal_weights = result.x
            optimal_risk = result.fun
            optimal_return = portfolio_return(optimal_weights)
            
            return {
                'weights': optimal_weights,
                'risk': optimal_risk,
                'return': optimal_return,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def monte_carlo_simulation(self, n_simulations: int = 10000, 
                             distributions: List[stats.rv_continuous] = None,
                             combination_func=None) -> np.ndarray:
        """
        Perform Monte Carlo simulation for risk scenario analysis.
        
        Args:
            n_simulations: Number of simulation runs
            distributions: List of probability distributions for risk factors
            combination_func: Function to combine random variables into final outcome
            
        Returns:
            Array of simulation results
        """
        if distributions is None:
            # Default example: project completion time with three components
            distributions = [
                stats.norm(10, 2),  # Component 1: mean=10, std=2
                stats.norm(15, 3),  # Component 2: mean=15, std=3
                stats.norm(8, 1)    # Component 3: mean=8, std=1
            ]
        
        if combination_func is None:
            # Default: sum all components
            combination_func = lambda x: np.sum(x, axis=1)
        
        # Generate random samples
        samples = np.array([dist.rvs(n_simulations) for dist in distributions]).T
        
        # Combine samples
        results = combination_func(samples)
        
        self.scenarios = results
        return results
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, 
                              risk_free_rate: float = 0) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted performance measurement.
        
        Args:
            returns: Array of returns/benefits
            risk_free_rate: Baseline/risk-free rate
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def maximum_drawdown(self, values: np.ndarray) -> float:
        """
        Calculate maximum drawdown from peak to trough.
        
        Args:
            values: Time series of values (performance, quality scores, etc.)
            
        Returns:
            Maximum drawdown as percentage
        """
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (cumulative_max - values) / cumulative_max
        return np.max(drawdowns)


class ProjectRiskAssessment:
    """
    Specialized class for project risk assessment using mathematical finance techniques.
    """
    
    def __init__(self):
        self.risk_analyzer = RiskAnalyzer()
    
    def assess_timeline_risk(self, tasks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Assess timeline risk for a project with multiple tasks.
        
        Args:
            tasks: Dictionary with task names as keys and (mean_duration, std_duration) tuples
            
        Returns:
            Risk assessment results
        """
        task_names = list(tasks.keys())
        durations = [tasks[task][0] for task in task_names]
        uncertainties = [tasks[task][1] for task in task_names]
        
        # Calculate total project statistics
        total_expected = sum(durations)
        total_variance = sum([std**2 for std in uncertainties])
        total_std = np.sqrt(total_variance)
        
        # Generate Monte Carlo simulations
        distributions = [stats.norm(dur, std) for dur, std in zip(durations, uncertainties)]
        results = self.risk_analyzer.monte_carlo_simulation(
            distributions=distributions,
            combination_func=lambda x: np.sum(x, axis=1)
        )
        
        # Calculate risk metrics
        var_95 = self.risk_analyzer.calculate_var(results, 0.95)
        cvar_95 = self.risk_analyzer.calculate_cvar(results, 0.95)
        
        return {
            'tasks': task_names,
            'expected_duration': total_expected,
            'duration_std': total_std,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'simulation_results': results,
            'probability_exceed_expected': np.mean(results > total_expected)
        }
    
    def resource_allocation_optimization(self, resources: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Optimize resource allocation across project activities.
        
        Args:
            resources: Dictionary with resource info including expected_benefit and risk
            
        Returns:
            Optimal allocation strategy
        """
        resource_names = list(resources.keys())
        expected_returns = np.array([resources[r]['expected_benefit'] for r in resource_names])
        risks = np.array([resources[r]['risk'] for r in resource_names])
        
        # Create covariance matrix (simplified - assumes independence)
        cov_matrix = np.diag(risks**2)
        
        # Add some correlation for realism
        n = len(resource_names)
        for i in range(n):
            for j in range(i+1, n):
                correlation = 0.1  # Small positive correlation
                cov_matrix[i, j] = correlation * risks[i] * risks[j]
                cov_matrix[j, i] = cov_matrix[i, j]
        
        # Optimize allocation
        result = self.risk_analyzer.portfolio_optimization(
            expected_returns, cov_matrix, target_return=np.mean(expected_returns)
        )
        
        if result['success']:
            allocation = dict(zip(resource_names, result['weights']))
            return {
                'allocation': allocation,
                'expected_return': result['return'],
                'portfolio_risk': result['risk'],
                'resource_names': resource_names
            }
        else:
            return {'success': False, 'message': result.get('message', 'Optimization failed')}


def demonstrate_risk_analysis():
    """
    Demonstrate various risk analysis techniques with practical examples.
    """
    print("=== Mathematical Finance Risk Management Demonstration ===\n")
    
    # Initialize risk analyzer
    analyzer = RiskAnalyzer()
    
    # Example 1: Simple VaR and CVaR calculation
    print("1. Value at Risk (VaR) and Conditional VaR Analysis")
    print("-" * 50)
    
    # Simulate project delay data (in days)
    np.random.seed(42)
    project_delays = np.random.gamma(2, 3, 1000)  # Gamma distribution for delays
    
    var_95 = analyzer.calculate_var(project_delays, 0.95)
    cvar_95 = analyzer.calculate_cvar(project_delays, 0.95)
    
    print(f"Average project delay: {np.mean(project_delays):.2f} days")
    print(f"95% VaR (worst 5% threshold): {var_95:.2f} days")
    print(f"95% CVaR (expected loss beyond VaR): {cvar_95:.2f} days")
    print()
    
    # Example 2: Portfolio Optimization for Resource Allocation
    print("2. Resource Allocation Optimization")
    print("-" * 40)
    
    # Define resources with expected benefits and risks
    expected_returns = np.array([0.08, 0.12, 0.15, 0.20])  # Expected efficiency gains
    cov_matrix = np.array([
        [0.04, 0.01, 0.02, -0.01],
        [0.01, 0.09, 0.01, 0.02],
        [0.02, 0.01, 0.16, 0.03],
        [-0.01, 0.02, 0.03, 0.25]
    ])
    
    result = analyzer.portfolio_optimization(expected_returns, cov_matrix, target_return=0.12)
    
    if result['success']:
        resource_names = ['Development', 'Testing', 'Documentation', 'Training']
        print("Optimal resource allocation:")
        for i, name in enumerate(resource_names):
            print(f"  {name}: {result['weights'][i]:.1%}")
        print(f"Expected return: {result['return']:.1%}")
        print(f"Portfolio risk: {result['risk']:.1%}")
    print()
    
    # Example 3: Monte Carlo Simulation
    print("3. Monte Carlo Simulation for Project Timeline")
    print("-" * 45)
    
    # Define task distributions
    distributions = [
        stats.norm(10, 2),  # Task 1: 10±2 days
        stats.norm(15, 3),  # Task 2: 15±3 days
        stats.norm(8, 1),   # Task 3: 8±1 days
        stats.norm(12, 2.5) # Task 4: 12±2.5 days
    ]
    
    simulation_results = analyzer.monte_carlo_simulation(
        n_simulations=10000,
        distributions=distributions
    )
    
    print(f"Monte Carlo Results (10,000 simulations):")
    print(f"  Mean project duration: {np.mean(simulation_results):.1f} days")
    print(f"  Standard deviation: {np.std(simulation_results):.1f} days")
    print(f"  95% confidence interval: [{np.percentile(simulation_results, 2.5):.1f}, {np.percentile(simulation_results, 97.5):.1f}] days")
    print(f"  Probability of exceeding 50 days: {np.mean(simulation_results > 50):.1%}")
    print()
    
    # Example 4: Project Risk Assessment
    print("4. Comprehensive Project Risk Assessment")
    print("-" * 40)
    
    project_assessor = ProjectRiskAssessment()
    
    # Define project tasks
    tasks = {
        'Planning': (5, 1),
        'Development': (20, 4),
        'Testing': (10, 3),
        'Documentation': (8, 2),
        'Deployment': (3, 1)
    }
    
    assessment = project_assessor.assess_timeline_risk(tasks)
    
    print("Project Timeline Risk Assessment:")
    print(f"  Expected duration: {assessment['expected_duration']:.1f} days")
    print(f"  Duration uncertainty (std): {assessment['duration_std']:.1f} days")
    print(f"  95% VaR: {assessment['var_95']:.1f} days")
    print(f"  95% CVaR: {assessment['cvar_95']:.1f} days")
    print(f"  Probability of delay beyond expected: {assessment['probability_exceed_expected']:.1%}")
    print()
    
    # Example 5: Risk-Adjusted Performance Metrics
    print("5. Risk-Adjusted Performance Analysis")
    print("-" * 38)
    
    # Simulate performance data for two strategies
    np.random.seed(42)
    strategy_a = np.random.normal(0.08, 0.12, 252)  # 8% mean, 12% volatility
    strategy_b = np.random.normal(0.06, 0.08, 252)  # 6% mean, 8% volatility
    
    sharpe_a = analyzer.calculate_sharpe_ratio(strategy_a)
    sharpe_b = analyzer.calculate_sharpe_ratio(strategy_b)
    
    # Simulate cumulative performance
    cumulative_a = np.cumprod(1 + strategy_a)
    cumulative_b = np.cumprod(1 + strategy_b)
    
    drawdown_a = analyzer.maximum_drawdown(cumulative_a)
    drawdown_b = analyzer.maximum_drawdown(cumulative_b)
    
    print("Strategy Comparison:")
    print(f"  Strategy A - Sharpe Ratio: {sharpe_a:.3f}, Max Drawdown: {drawdown_a:.1%}")
    print(f"  Strategy B - Sharpe Ratio: {sharpe_b:.3f}, Max Drawdown: {drawdown_b:.1%}")
    print(f"  Better risk-adjusted performance: {'Strategy A' if sharpe_a > sharpe_b else 'Strategy B'}")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_risk_analysis()
    
    print("\n=== Additional Analysis Functions Available ===")
    print("- RiskAnalyzer.calculate_var()")
    print("- RiskAnalyzer.calculate_cvar()")
    print("- RiskAnalyzer.portfolio_optimization()")
    print("- RiskAnalyzer.monte_carlo_simulation()")
    print("- ProjectRiskAssessment.assess_timeline_risk()")
    print("- ProjectRiskAssessment.resource_allocation_optimization()")