#!/usr/bin/env python3
"""
Comprehensive Examples for Planck Scale Supersymmetry

This file demonstrates various aspects of supersymmetric physics 
at the Planck scale with detailed calculations and visualizations.

Usage: python3 supersymmetry_examples.py
"""

import numpy as np
# import matplotlib.pyplot as plt  # Optional for advanced plotting
from PlanckSupersymmetry import PlanckSupersymmetryCalculator

def demonstrate_planck_scale_hierarchy():
    """Demonstrate the hierarchy of scales at Planck level"""
    
    calc = PlanckSupersymmetryCalculator()
    
    print("=== Planck Scale Hierarchy ===")
    print("The Planck scale represents the fundamental limit where quantum")
    print("effects of gravity become important. Below this scale, classical")
    print("spacetime geometry breaks down.")
    print()
    
    # Compare with other scales
    scales = {
        "Atomic": 1e-10,           # meters (hydrogen atom)
        "Nuclear": 1e-15,          # meters (atomic nucleus)  
        "Quantum Chromodynamics": 1e-18,  # meters (proton size)
        "Electroweak": 1e-19,      # meters (W/Z boson scale)
        "Planck": calc.planck_units.length  # meters
    }
    
    print("Length Scale Comparison:")
    for name, length in scales.items():
        ratio = length / calc.planck_units.length
        print(f"  {name:25}: {length:.2e} m  (×{ratio:.2e} Planck lengths)")
    
    print()
    
    # Energy scales
    print("Energy Scale Comparison:")
    energies = {
        "Chemical bonds": 1e-19,   # ~1 eV in Joules
        "Nuclear binding": 1e-12,  # ~10 MeV 
        "Particle physics": 1e-10, # ~1 GeV
        "Electroweak": 1e-8,       # ~100 GeV
        "Planck": calc.planck_units.energy
    }
    
    for name, energy in energies.items():
        ratio = energy / calc.planck_units.energy
        print(f"  {name:25}: {energy:.2e} J   (×{ratio:.2e} Planck energies)")

def analyze_susy_breaking_scenarios():
    """Analyze different supersymmetry breaking scenarios"""
    
    calc = PlanckSupersymmetryCalculator()
    
    print("\n=== Supersymmetry Breaking Scenarios ===")
    print("At the Planck scale, supersymmetry is expected to be unbroken.")
    print("However, at lower energies, SUSY must be broken to explain")
    print("why we don't observe supersymmetric partners.")
    print()
    
    # Different breaking scales
    breaking_scales = {
        "High scale (GUT)": 1e16,      # GeV
        "Intermediate": 1e10,          # GeV  
        "TeV scale": 1e3,              # GeV
        "Low scale": 1e2               # GeV
    }
    
    print("SUSY Breaking Scale Analysis:")
    for scenario, scale_gev in breaking_scales.items():
        scale_joules = scale_gev * 1.602e-19 * 1e9  # Convert to Joules
        
        # Calculate associated length scale
        # Using E ~ ℏc/L → L ~ ℏc/E
        length_scale = (calc.planck_units.length * calc.planck_units.energy) / scale_joules
        
        # Gravitino mass estimate
        # m₃/₂ ~ (breaking scale)² / M_Planck
        gravitino_mass_estimate = (scale_joules**2) / (calc.planck_units.energy * calc.planck_units.mass)
        
        print(f"  {scenario:20}: {scale_gev:.0e} GeV")
        print(f"    Length scale:    {length_scale:.2e} m")
        print(f"    Gravitino mass:  {gravitino_mass_estimate:.2e} kg")
        print()

def black_hole_susy_thermodynamics():
    """Explore black hole thermodynamics with supersymmetry"""
    
    calc = PlanckSupersymmetryCalculator()
    
    print("=== Black Hole Supersymmetric Thermodynamics ===")
    print("Supersymmetric black holes have special properties due to")
    print("BPS bounds and preserved supersymmetries.")
    print()
    
    # Range of black hole masses
    masses_planck = np.logspace(0, 5, 20)  # 1 to 10^5 Planck masses
    masses_kg = masses_planck * calc.planck_units.mass
    
    entropies = []
    temperatures = []
    
    for mass in masses_kg:
        entropy = calc.black_hole_entropy(mass)
        temperature = calc.hawking_temperature(mass)
        
        entropies.append(entropy)
        temperatures.append(temperature)
    
    entropies = np.array(entropies)
    temperatures = np.array(temperatures)
    
    print("Black Hole Properties vs Mass:")
    print("Mass (M_P)    Entropy (k_B)    Temperature (K)")
    print("-" * 50)
    
    for i in range(0, len(masses_planck), 4):  # Sample every 4th point
        mass_ratio = masses_planck[i]
        entropy = entropies[i]
        temp = temperatures[i]
        print(f"{mass_ratio:8.1e}      {entropy:8.2e}        {temp:8.2e}")
    
    # BPS black hole analysis
    print("\nBPS Black Hole Analysis:")
    print("For supersymmetric black holes, the mass satisfies M = |Z|")
    print("where Z is the central charge. This provides a lower bound.")
    
    central_charges = [1e-10, 1e-5, 1.0, 1e5, 1e10]  # Various central charge values
    for Z in central_charges:
        bps_mass = calc.bps_bound(Z)
        print(f"Central charge Z = {Z:.0e} → BPS mass ≥ {bps_mass:.2e}")

def cosmic_string_network():
    """Analyze cosmic string networks from SUSY breaking"""
    
    calc = PlanckSupersymmetryCalculator()
    
    print("\n=== Cosmic String Networks from SUSY Breaking ===")
    print("When supersymmetry is broken, topological defects like")
    print("cosmic strings can form. These have observable consequences.")
    print()
    
    # Different symmetry breaking scales
    breaking_scales = np.logspace(10, 19, 10)  # 10^10 to 10^19 GeV
    
    print("Cosmic String Properties:")
    print("Breaking Scale (GeV)    String Tension (kg/m)    Gμ/c²")
    print("-" * 60)
    
    for scale_gev in breaking_scales:
        scale_joules = scale_gev * 1.602e-19 * 1e9
        
        # String tension μ ~ η² where η is breaking scale
        tension = calc.cosmic_string_tension(scale_joules)
        
        # Dimensionless string tension Gμ/c²
        dimensionless = (6.67430e-11 * tension) / (299792458**2)
        
        print(f"{scale_gev:12.0e}         {tension:12.2e}        {dimensionless:.2e}")
    
    print("\nObservational signatures:")
    print("- Gravitational waves from cosmic string networks")
    print("- Cosmic microwave background anisotropies")
    print("- Pulsar timing array signals")
    print("- Direct gravitational wave detection")

def demonstrate_holographic_susy():
    """Demonstrate holographic supersymmetry (AdS/CFT)"""
    
    calc = PlanckSupersymmetryCalculator()
    
    print("\n=== Holographic Supersymmetry (AdS/CFT) ===")
    print("The AdS/CFT correspondence relates supersymmetric gravity")
    print("theories in Anti-de Sitter space to conformal field theories.")
    print()
    
    # Example: AdS₅ × S⁵ with N=4 Super Yang-Mills
    print("AdS₅ × S⁵ / N=4 SYM Correspondence:")
    
    # Parameters
    N = 4  # Number of supersymmetries
    g_ym = 1.0  # Yang-Mills coupling
    lambda_t_hooft = g_ym**2 * N  # 't Hooft coupling
    
    # AdS radius in Planck units
    L_ads_planck = 10.0  # Example value
    L_ads = L_ads_planck * calc.planck_units.length
    
    print(f"Number of SUSY generators:  {N}")
    print(f"'t Hooft coupling λ:        {lambda_t_hooft}")
    print(f"AdS radius:                 {L_ads:.2e} m ({L_ads_planck} l_P)")
    
    # Central charge (proportional to N²)
    central_charge = N**2 / 4  # For SU(N) gauge theory
    print(f"Central charge c:           {central_charge}")
    
    # Hawking-Page transition temperature
    T_hp = 1 / (np.pi * L_ads_planck) * calc.planck_units.temperature
    print(f"Hawking-Page temperature:   {T_hp:.2e} K")
    
    print("\nDuality relations:")
    print("- Bulk graviton ↔ Boundary stress tensor")
    print("- Bulk gravitino ↔ Boundary supercurrent")
    print("- Bulk scalars ↔ Boundary operators")
    print("- AdS black holes ↔ Thermal CFT states")

def quantum_gravity_susy_effects():
    """Explore quantum gravity effects in supersymmetric theories"""
    
    calc = PlanckSupersymmetryCalculator()
    
    print("\n=== Quantum Gravity Effects in Supersymmetry ===")
    print("At the Planck scale, quantum gravitational effects become")
    print("important and modify supersymmetric relationships.")
    print()
    
    # Planck scale quantum corrections
    print("Planck Scale Quantum Corrections:")
    
    # Loop expansion parameter
    g_newton = 6.67430e-11 / (calc.planck_units.length**3 / (calc.planck_units.mass * calc.planck_units.time**2))
    print(f"Dimensionless Newton constant: {g_newton:.2e}")
    
    # Supersymmetric non-renormalization theorems
    print("\nNon-Renormalization in SUSY + Gravity:")
    print("- Superpotential: protected by holomorphy")
    print("- Kähler potential: receives quantum corrections")
    print("- Gauge kinetic function: logarithmically corrected")
    
    # Example: corrections to Kähler potential
    phi = 1.0  # Field value in Planck units
    tree_level = phi**2
    one_loop = g_newton * phi**4 * np.log(phi**2)
    
    print(f"\nExample Kähler potential corrections:")
    print(f"Tree level:     K = |Φ|² = {tree_level:.3f}")
    print(f"One-loop:       ΔK ~ g |Φ|⁴ ln|Φ|² = {one_loop:.2e}")
    
    # Planck scale supersymmetry breaking
    print(f"\nPlanck scale SUSY breaking effects:")
    print(f"- Gravitino mass ~ M_Planck = {calc.planck_units.mass:.2e} kg")
    print(f"- Scalar masses ~ M_Planck")
    print(f"- Gaugino masses ~ M_Planck")

def main():
    """Run comprehensive supersymmetry examples"""
    
    print("Comprehensive Planck Scale Supersymmetry Examples")
    print("=" * 55)
    
    # Run all demonstrations
    demonstrate_planck_scale_hierarchy()
    analyze_susy_breaking_scenarios()
    black_hole_susy_thermodynamics()
    cosmic_string_network()
    demonstrate_holographic_susy()
    quantum_gravity_susy_effects()
    
    print("\n" + "=" * 55)
    print("Analysis completed! This demonstrates various aspects")
    print("of supersymmetry at the Planck scale, from fundamental")
    print("scale hierarchies to advanced topics like holography")
    print("and quantum gravity effects.")
    print("\nFor more detailed calculations, see:")
    print("- Supersymmetry at Planck Scale.md")
    print("- 數學/超對稱代數與普朗克尺度.md")
    print("- PlanckSupersymmetry.py module")

if __name__ == "__main__":
    main()