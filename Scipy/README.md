# Numerical Euler-Lagrangian Solver using SciPy

This directory contains a comprehensive implementation of numerical methods for solving Euler-Lagrange equations using SciPy. The Euler-Lagrange equations are fundamental in variational calculus and physics, particularly in Lagrangian mechanics.

## Mathematical Background

The Euler-Lagrange equation has the form:

```
d/dt(∂L/∂q̇) - ∂L/∂q = 0
```

where:
- `L(q, q̇, t)` is the Lagrangian function
- `q` represents generalized coordinates
- `q̇` represents generalized velocities
- `t` is time

## Files

### Core Implementation
- **`euler_lagrangian.py`** - Main implementation of the EulerLagrangianSolver class
- **`examples.py`** - Comprehensive examples demonstrating the solver
- **`test_euler_lagrangian.py`** - Test suite for validation

### Generated Output
- **`simple_harmonic_oscillator.png`** - Visualization of harmonic oscillator solution
- **`pendulum.png`** - Visualization of pendulum dynamics
- **`variational_path.png`** - Demonstration of variational path finding

## Features

### 1. EulerLagrangianSolver Class
- Numerical solution of Euler-Lagrange equations
- Support for multiple generalized coordinates
- Trajectory solving using scipy.integrate.solve_ivp
- Variational path finding using the principle of least action
- Energy conservation analysis
- Visualization capabilities

### 2. Pre-built Lagrangian Functions
- Simple harmonic oscillator
- Simple pendulum
- Double pendulum
- Extensible framework for custom Lagrangians

### 3. Numerical Methods
- Automatic differentiation for computing partial derivatives
- Multiple integration methods (RK45, DOP853, etc.)
- Nonlinear equation solving for implicit systems
- Action functional optimization

## Usage Examples

### Simple Harmonic Oscillator
```python
from euler_lagrangian import EulerLagrangianSolver, create_simple_harmonic_oscillator_lagrangian

# Create Lagrangian for mass-spring system
lagrangian = create_simple_harmonic_oscillator_lagrangian(mass=1.0, spring_constant=4.0)

# Initialize solver
solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)

# Set initial conditions
initial_conditions = {
    'q0': np.array([1.0]),      # initial position
    'q_dot0': np.array([0.0])   # initial velocity
}

# Solve trajectory
solution = solver.solve_trajectory(initial_conditions, time_span=(0, 2*np.pi))

# Plot results
solver.plot_trajectory(solution)
```

### Pendulum Dynamics
```python
from euler_lagrangian import create_pendulum_lagrangian

# Create pendulum Lagrangian
lagrangian = create_pendulum_lagrangian(length=1.0, mass=1.0, gravity=9.81)
solver = EulerLagrangianSolver(lagrangian, n_coordinates=1)

# Initial condition: 60 degrees from vertical
initial_conditions = {
    'q0': np.array([np.pi/3]),
    'q_dot0': np.array([0.0])
}

# Solve and analyze
solution = solver.solve_trajectory(initial_conditions, time_span=(0, 10))
conservation = solver.analyze_conservation_laws(solution)
```

### Variational Path Finding
```python
# Find path that minimizes action
boundary_conditions = {
    'q_start': (0.0, np.array([0.0])),
    'q_end': (np.pi, np.array([1.0]))
}

result = solver.find_stationary_action_path(
    boundary_conditions, 
    time_span=(0, np.pi),
    n_segments=50
)
```

## Running the Code

### Install Dependencies
```bash
pip install scipy numpy matplotlib
```

### Run Examples
```bash
cd Scipy
python examples.py
```

### Run Tests
```bash
cd Scipy
python test_euler_lagrangian.py
```

## Applications

This implementation can be used for:

### 1. Physics Education
- Demonstrating classical mechanics principles
- Visualizing energy conservation
- Understanding variational calculus

### 2. Research Applications
- Solving complex mechanical systems
- Optimization problems in physics
- Computational physics simulations

### 3. Engineering Analysis
- Structural dynamics
- Robotics (multi-body systems)
- Control system design

## Mathematical Methods Used

### 1. Numerical Differentiation
- Finite difference approximations for partial derivatives
- Automatic differentiation using scipy.misc.derivative

### 2. Ordinary Differential Equation Solving
- Converting Euler-Lagrange equations to first-order ODE systems
- Using scipy.integrate.solve_ivp with various methods

### 3. Nonlinear Equation Solving
- Solving for accelerations using scipy.optimize.fsolve
- Handling implicit systems of equations

### 4. Optimization
- Action functional minimization using scipy.optimize.minimize
- Variational path finding with boundary conditions

## Limitations and Future Work

### Current Limitations
- Numerical stability for highly nonlinear systems
- Computational complexity for many-body systems
- Limited to conservative systems (no dissipation)

### Potential Extensions
- Support for dissipative systems
- Symplectic integrators for better energy conservation
- Parallel computing for large systems
- Higher-order methods for improved accuracy

## References

- Goldstein, H. "Classical Mechanics" (3rd Edition)
- Arnold, V.I. "Mathematical Methods of Classical Mechanics"
- Hairer, E. "Geometric Numerical Integration"

## [物理,數學] = Lagrangian Mechanics ✓

This implementation bridges physics and mathematics through:
- Variational calculus
- Differential equations
- Numerical analysis
- Optimization theory

## [生物,數學] = Biomechanics Applications

Potential applications in biomechanics:
- Muscle dynamics modeling
- Joint movement analysis
- Gait analysis
- Neural control systems

## [化學,數學] = Molecular Dynamics

Applications in chemistry:
- Molecular vibrations
- Reaction coordinate analysis
- Chemical kinetics
- Statistical mechanics

## [物理, 生物] = Biophysics

Interdisciplinary applications:
- Protein folding dynamics
- Cell membrane mechanics
- DNA dynamics
- Enzyme kinetics

## [化學, 生物] = Biochemical Systems

Applications in biochemistry:
- Metabolic pathway dynamics
- Enzyme reaction networks
- Molecular transport
- Signal transduction

## [物理, 化學] = Physical Chemistry

Applications in physical chemistry:
- Thermodynamic systems
- Phase transitions
- Spectroscopy
- Quantum mechanics

## [Numerical Methods, Physics] = Computational Physics ✓

This implementation represents the intersection of numerical methods and physics, providing tools for solving fundamental equations of motion in classical mechanics.
