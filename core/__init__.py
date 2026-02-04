"""
Core modules for GRIP grain boundary optimization.

This package contains:
- Bicrystal: Class for constructing grain boundary structures
- Simulation: Orchestration class for running optimizations
- Calculator: Abstract interface for energy calculators
- calculators/: Implementations for different backends (LAMMPS, ASE, generic)
"""

from core.bicrystal import Bicrystal
from core.simulation import Simulation
from core.calculator import Calculator, CalculationResult
from core.interstitial import Interstitial

__all__ = [
    'Bicrystal',
    'Simulation', 
    'Calculator',
    'CalculationResult',
    'Interstitial',
]
