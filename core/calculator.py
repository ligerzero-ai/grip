"""
Abstract base class for energy calculators in GRIP.

This module defines the interface that all calculator implementations must follow.
The design allows relax_structure to return both the relaxed atoms AND energy,
avoiding redundant calculations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Union
import numpy as np
from ase import Atoms


@dataclass
class CalculationResult:
    """
    Container for calculation results.
    
    Attributes:
        atoms: The (possibly relaxed) atomic structure
        energy: Total energy in eV
        energy_per_atom: Energy per atom in eV (optional, computed if not provided)
        converged: Whether the calculation converged
        forces: Forces on atoms in eV/Angstrom (optional)
        stress: Stress tensor (optional)
        metadata: Any additional calculation-specific data
    """
    atoms: Atoms
    energy: float
    converged: bool = True
    energy_per_atom: Optional[float] = None
    forces: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.energy_per_atom is None and self.atoms is not None:
            self.energy_per_atom = self.energy / len(self.atoms)


class Calculator(ABC):
    """
    Abstract base class for energy calculators.
    
    All calculator implementations must inherit from this class and implement
    the required methods. The interface is designed so that:
    
    1. relax_structure() returns BOTH the relaxed structure AND energy
    2. calculate_energy() is only needed for single-point calculations
    3. The main workflow can call relax_structure() once and get both results
    
    Example usage:
        calc = MyCalculator(...)
        result = calc.relax_structure(atoms, **params)
        relaxed_atoms = result.atoms
        energy = result.energy
        gb_energy = calc.get_gb_energy(result, n_atoms, area, e_coh)
    """
    
    @abstractmethod
    def calculate_energy(self, atoms: Atoms, **kwargs) -> CalculationResult:
        """
        Perform a single-point energy calculation (no relaxation).
        
        Args:
            atoms: ASE Atoms object with the structure
            **kwargs: Calculator-specific parameters
            
        Returns:
            CalculationResult with energy and the same atoms (unmodified)
        """
        pass
    
    @abstractmethod
    def relax_structure(self, atoms: Atoms, **kwargs) -> CalculationResult:
        """
        Relax the structure and return both relaxed atoms AND energy.
        
        This is the primary method for GRIP workflows. It performs structural
        relaxation and returns the final energy, avoiding the need to call
        calculate_energy() separately.
        
        Args:
            atoms: ASE Atoms object with the structure
            **kwargs: Calculator-specific parameters (e.g., fmax, steps)
            
        Returns:
            CalculationResult with:
                - atoms: the relaxed structure
                - energy: final energy after relaxation
                - converged: whether relaxation converged
        """
        pass
    
    def get_gb_energy(self, result: CalculationResult, n_atoms: int, 
                      area: float, e_cohesive: float) -> float:
        """
        Calculate grain boundary energy from total energy.
        
        Default implementation uses the standard formula:
            E_gb = (E_total - N * E_coh) / Area * conversion_factor
        
        Can be overridden by subclasses for different conventions.
        
        Args:
            result: CalculationResult from calculate_energy or relax_structure
            n_atoms: Number of atoms in the GB region used for energy calculation
            area: GB area in Angstrom^2
            e_cohesive: Cohesive energy per atom in eV (typically negative)
            
        Returns:
            GB energy in J/m^2
        """
        # E_bulk = N * E_coh (both typically negative)
        E_bulk = n_atoms * e_cohesive
        # E_excess = E_total - E_bulk (positive for GB)
        E_excess = result.energy - E_bulk
        # Convert eV/Angstrom^2 to J/m^2: multiply by 16.021766
        E_gb = E_excess / area * 16.021766
        
        # Handle unphysical negative values
        if E_gb < 0:
            print(f"Warning: Negative GB energy ({E_gb:.4f} J/m²), setting to 100")
            E_gb = 100.0
            
        return E_gb
    
    def run_md(self, atoms: Atoms, temperature: float, steps: int, 
               **kwargs) -> CalculationResult:
        """
        Run molecular dynamics at finite temperature.
        
        Optional method - not all calculators support MD.
        Default implementation raises NotImplementedError.
        
        Args:
            atoms: ASE Atoms object
            temperature: Temperature in Kelvin
            steps: Number of MD steps
            **kwargs: Calculator-specific MD parameters
            
        Returns:
            CalculationResult with final structure and energy
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support molecular dynamics. "
            "Use relax_structure() for static relaxation only."
        )
    
    def supports_md(self) -> bool:
        """Check if this calculator supports molecular dynamics."""
        return False
