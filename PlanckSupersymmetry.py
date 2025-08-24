#!/usr/bin/env python3
"""
Planck Scale Supersymmetry Calculator

This module provides computational tools for analyzing supersymmetric theories 
at the Planck scale, including fundamental constants, particle spectra, and 
theoretical predictions.

Author: NLPNote Project
Date: 2024-12-22
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Physical constants in SI units
HBAR = 1.054571817e-34  # J⋅s (reduced Planck constant)
C = 299792458          # m/s (speed of light)
G = 6.67430e-11        # m³⋅kg⁻¹⋅s⁻² (gravitational constant)
ELECTRON_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31   # kg

@dataclass
class PlanckUnits:
    """Planck scale fundamental units"""
    length: float      # meters
    time: float        # seconds  
    mass: float        # kilograms
    energy: float      # joules
    energy_gev: float  # GeV
    force: float       # newtons
    power: float       # watts
    temperature: float # kelvin

@dataclass
class SupersymmetricSpectrum:
    """Supersymmetric particle spectrum"""
    graviton_mass: float
    gravitino_mass: float
    scalar_masses: np.ndarray
    fermion_masses: np.ndarray
    vector_masses: np.ndarray

class PlanckSupersymmetryCalculator:
    """Calculator for Planck scale supersymmetric physics"""
    
    def __init__(self):
        self.planck_units = self._calculate_planck_units()
        
    def _calculate_planck_units(self) -> PlanckUnits:
        """Calculate fundamental Planck scale units"""
        
        # Basic Planck units
        l_planck = math.sqrt(HBAR * G / C**3)
        t_planck = math.sqrt(HBAR * G / C**5)
        m_planck = math.sqrt(HBAR * C / G)
        E_planck = m_planck * C**2
        
        # Derived units
        F_planck = C**4 / G  # Planck force
        P_planck = C**5 / G  # Planck power
        T_planck = E_planck / (1.380649e-23)  # Planck temperature (using Boltzmann constant)
        
        return PlanckUnits(
            length=l_planck,
            time=t_planck,
            mass=m_planck,
            energy=E_planck,
            energy_gev=E_planck / ELECTRON_CHARGE / 1e9,
            force=F_planck,
            power=P_planck,
            temperature=T_planck
        )
    
    def supersymmetric_spectrum(self, n_multiplets: int = 10) -> SupersymmetricSpectrum:
        """
        Generate supersymmetric particle spectrum at Planck scale
        
        Args:
            n_multiplets: Number of supersymmetric multiplets to generate
            
        Returns:
            SupersymmetricSpectrum object with particle masses
        """
        m_planck = self.planck_units.mass
        
        # Massless graviton
        graviton_mass = 0.0
        
        # Gravitino at Planck scale
        gravitino_mass = m_planck
        
        # Generate SUSY multiplets with random variations
        np.random.seed(42)  # For reproducibility
        
        # Scalar superpartners (selectrons, squarks, etc.)
        scalar_masses = np.array([
            m_planck * (0.8 + 0.4 * np.random.random()) 
            for _ in range(n_multiplets)
        ])
        
        # Fermionic superpartners (gauginos, higgsinos, etc.)
        fermion_masses = np.array([
            m_planck * (0.9 + 0.3 * np.random.random()) 
            for _ in range(n_multiplets)
        ])
        
        # Vector bosons (gauge bosons)
        vector_masses = np.array([
            m_planck * (0.7 + 0.6 * np.random.random()) 
            for _ in range(n_multiplets//2)
        ])
        
        return SupersymmetricSpectrum(
            graviton_mass=graviton_mass,
            gravitino_mass=gravitino_mass,
            scalar_masses=scalar_masses,
            fermion_masses=fermion_masses,
            vector_masses=vector_masses
        )
    
    def superalgebra_relations(self, momentum: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate supersymmetric algebra anticommutation relations
        
        Args:
            momentum: Four-momentum vector [E, px, py, pz]
            
        Returns:
            Dictionary containing anticommutation relation results
        """
        # Pauli matrices (2x2)
        sigma_matrices = {
            0: np.array([[1, 0], [0, 1]]),      # σ⁰ = I
            1: np.array([[0, 1], [1, 0]]),      # σ¹ = σₓ  
            2: np.array([[0, -1j], [1j, 0]]),   # σ² = σᵧ
            3: np.array([[1, 0], [0, -1]])      # σ³ = σᵧ
        }
        
        # Calculate {Qα, Q̄β̇} = 2σᵘαβ̇ Pμ
        anticommutator = np.zeros((2, 2), dtype=complex)
        
        for mu in range(4):
            anticommutator += 2 * sigma_matrices[mu] * momentum[mu]
            
        return {
            'anticommutator': anticommutator,
            'momentum': momentum,
            'pauli_matrices': sigma_matrices
        }
    
    def bps_bound(self, central_charge: float) -> float:
        """
        Calculate BPS (Bogomol'nyi-Prasad-Sommerfield) bound
        
        Args:
            central_charge: Value of the central charge Z
            
        Returns:
            BPS mass bound M ≥ |Z|
        """
        return abs(central_charge)
    
    def holomorphic_superpotential(self, phi: complex, params: Dict[str, float]) -> complex:
        """
        Calculate holomorphic superpotential W(Φ)
        
        Args:
            phi: Complex scalar field value
            params: Dictionary with 'm', 'lambda', 'y' parameters
            
        Returns:
            Superpotential value
        """
        m = params.get('m', self.planck_units.mass)
        lam = params.get('lambda', 0.1)
        y = params.get('y', 0.01)
        
        M_planck = self.planck_units.mass
        
        W = (m * phi**2 / 2 + 
             lam * phi**3 / 3 + 
             y * phi**4 / (4 * M_planck))
        
        return W
    
    def f_term_potential(self, phi: complex, params: Dict[str, float]) -> float:
        """
        Calculate F-term contribution to scalar potential
        
        Args:
            phi: Complex scalar field value
            params: Parameters for superpotential
            
        Returns:
            F-term potential |∂W/∂Φ|²
        """
        # Numerical derivative of superpotential
        delta = 1e-10
        dW_dphi = (self.holomorphic_superpotential(phi + delta, params) - 
                   self.holomorphic_superpotential(phi - delta, params)) / (2 * delta)
        
        return abs(dW_dphi)**2
    
    def gravitino_mass(self, vev_superpotential: float) -> float:
        """
        Calculate gravitino mass in supergravity
        
        Args:
            vev_superpotential: Vacuum expectation value of superpotential
            
        Returns:
            Gravitino mass m₃/₂ = <W>/Mₚ²
        """
        M_planck = self.planck_units.mass
        return abs(vev_superpotential) / M_planck**2
    
    def supersymmetry_breaking_scale(self, soft_masses: List[float]) -> float:
        """
        Estimate supersymmetry breaking scale from soft masses
        
        Args:
            soft_masses: List of soft SUSY breaking masses
            
        Returns:
            Typical SUSY breaking scale
        """
        return np.sqrt(np.mean(np.array(soft_masses)**2))
    
    def black_hole_entropy(self, mass: float) -> float:
        """
        Calculate black hole entropy using Bekenstein-Hawking formula
        
        Args:
            mass: Black hole mass in kg
            
        Returns:
            Entropy in units of kB
        """
        # Schwarzschild radius
        r_s = 2 * G * mass / C**2
        
        # Area in Planck units
        area_planck = np.pi * r_s**2 / self.planck_units.length**2
        
        # Bekenstein-Hawking entropy S = A/(4lₚ²)
        entropy = area_planck / 4
        
        return entropy
    
    def hawking_temperature(self, mass: float) -> float:
        """
        Calculate Hawking temperature of black hole
        
        Args:
            mass: Black hole mass in kg
            
        Returns:
            Hawking temperature in Kelvin
        """
        # T_H = ℏc³/(8πGMkB)
        k_b = 1.380649e-23  # Boltzmann constant
        
        T_hawking = (HBAR * C**3) / (8 * np.pi * G * mass * k_b)
        
        return T_hawking
    
    def cosmic_string_tension(self, symmetry_breaking_scale: float) -> float:
        """
        Calculate cosmic string tension from symmetry breaking
        
        Args:
            symmetry_breaking_scale: Energy scale of symmetry breaking
            
        Returns:
            String tension μ in kg/m
        """
        # μ ~ η² where η is the symmetry breaking scale
        mu = (symmetry_breaking_scale / C**2)**2
        
        return mu
    
    def print_planck_units(self):
        """Print formatted Planck scale units"""
        print("=== Planck Scale Units ===")
        print(f"Planck length:      {self.planck_units.length:.3e} m")
        print(f"Planck time:        {self.planck_units.time:.3e} s") 
        print(f"Planck mass:        {self.planck_units.mass:.3e} kg")
        print(f"Planck energy:      {self.planck_units.energy:.3e} J")
        print(f"Planck energy:      {self.planck_units.energy_gev:.3e} GeV")
        print(f"Planck force:       {self.planck_units.force:.3e} N")
        print(f"Planck power:       {self.planck_units.power:.3e} W")
        print(f"Planck temperature: {self.planck_units.temperature:.3e} K")
    
    def print_susy_spectrum(self, spectrum: SupersymmetricSpectrum):
        """Print formatted supersymmetric spectrum"""
        print("\n=== Supersymmetric Particle Spectrum ===")
        print(f"Graviton mass:      {spectrum.graviton_mass:.3e} kg")
        print(f"Gravitino mass:     {spectrum.gravitino_mass:.3e} kg")
        print(f"Scalar masses:      {len(spectrum.scalar_masses)} particles")
        print(f"  Range:            {spectrum.scalar_masses.min():.3e} - {spectrum.scalar_masses.max():.3e} kg")
        print(f"Fermion masses:     {len(spectrum.fermion_masses)} particles")  
        print(f"  Range:            {spectrum.fermion_masses.min():.3e} - {spectrum.fermion_masses.max():.3e} kg")
        print(f"Vector masses:      {len(spectrum.vector_masses)} particles")
        print(f"  Range:            {spectrum.vector_masses.min():.3e} - {spectrum.vector_masses.max():.3e} kg")

def main():
    """Example usage of PlanckSupersymmetryCalculator"""
    
    print("Planck Scale Supersymmetry Calculator")
    print("=====================================\n")
    
    # Initialize calculator
    calc = PlanckSupersymmetryCalculator()
    
    # Display Planck units
    calc.print_planck_units()
    
    # Generate supersymmetric spectrum
    spectrum = calc.supersymmetric_spectrum(n_multiplets=8)
    calc.print_susy_spectrum(spectrum)
    
    # Test superalgebra relations
    print("\n=== Supersymmetric Algebra ===")
    momentum = np.array([calc.planck_units.energy/C, 0, 0, 0])  # Rest frame
    algebra = calc.superalgebra_relations(momentum)
    print(f"Anticommutator matrix:")
    print(algebra['anticommutator'])
    
    # Calculate superpotential example
    print("\n=== Superpotential Example ===")
    phi = 0.1 * calc.planck_units.mass  # Field value
    params = {'m': calc.planck_units.mass, 'lambda': 0.1, 'y': 0.01}
    W = calc.holomorphic_superpotential(phi, params)
    print(f"Superpotential W(Φ): {W:.3e}")
    
    # F-term potential
    V_F = calc.f_term_potential(phi, params)
    print(f"F-term potential:    {V_F:.3e}")
    
    # Gravitino mass
    m_gravitino = calc.gravitino_mass(abs(W))
    print(f"Gravitino mass:      {m_gravitino:.3e} kg")
    
    # Black hole physics example
    print("\n=== Black Hole Physics ===")
    bh_mass = 10 * calc.planck_units.mass  # 10 Planck masses
    entropy = calc.black_hole_entropy(bh_mass)
    temperature = calc.hawking_temperature(bh_mass)
    print(f"Black hole mass:     {bh_mass:.3e} kg")
    print(f"Bekenstein entropy:  {entropy:.3e} kB")
    print(f"Hawking temperature: {temperature:.3e} K")
    
    print("\nCalculation completed!")

if __name__ == "__main__":
    main()