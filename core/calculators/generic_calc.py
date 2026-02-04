"""
Generic calculator that wraps user-provided functions.

This is the most flexible calculator - you can provide any function
that takes an ASE Atoms object and returns energy (and optionally
a relaxed structure).

The key design feature is that relax_function returns BOTH the
relaxed structure AND the energy, so you don't need to evaluate twice.
"""

from typing import Callable, Optional, Union, Tuple
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS

from core.calculator import Calculator, CalculationResult


class GenericCalculator(Calculator):
    """
    Generic calculator that wraps user-provided functions.
    
    This calculator is designed to be maximally flexible. You only need to provide
    ONE function - typically relax_function which returns both structure AND energy.
    
    Example usage:
    
        # RECOMMENDED: Just provide relax_function (returns both structure and energy)
        def my_relax(atoms):
            relaxed = optimizer.run(atoms)
            energy = model.predict(relaxed)
            return relaxed, energy  # Returns BOTH!
        
        calc = GenericCalculator(relax_function=my_relax)
        
        # Alternative: Only energy function (no relaxation)
        def my_energy(atoms):
            return some_model.predict(atoms)
        
        calc = GenericCalculator(energy_function=my_energy)
        
        # With MD support
        def my_md(atoms, temperature, steps):
            final_atoms = run_md(atoms, T=temperature, n=steps)
            energy = model.predict(final_atoms)
            return final_atoms, energy
        
        calc = GenericCalculator(relax_function=my_relax, md_function=my_md)
    """
    
    def __init__(
        self,
        relax_function: Optional[Callable[[Atoms], Tuple[Atoms, float]]] = None,
        energy_function: Optional[Callable[[Atoms], float]] = None,
        md_function: Optional[Callable[[Atoms, float, int], Tuple[Atoms, float]]] = None,
        name: str = "GenericCalculator"
    ):
        """
        Initialize the generic calculator.
        
        You typically only need to provide relax_function, which returns BOTH
        the relaxed structure AND the energy in a single call.
        
        Args:
            relax_function: Function that relaxes structure AND returns energy.
                Signature: (atoms: Atoms) -> Tuple[Atoms, float]
                Returns: (relaxed_atoms, final_energy_in_eV)
                This is the primary function for GRIP workflows.
                
            energy_function: Optional function for single-point energy only.
                Signature: (atoms: Atoms) -> float
                Only needed if you want single-point calculations without relaxation,
                or if relax_function returns only atoms (not energy).
                
            md_function: Optional function for molecular dynamics.
                Signature: (atoms: Atoms, temperature: float, steps: int) -> Tuple[Atoms, float]
                Returns: (final_atoms, final_energy_in_eV)
                
            name: Name for this calculator (for logging/debugging)
        """
        if relax_function is None and energy_function is None:
            raise ValueError(
                "Must provide at least one of: relax_function or energy_function"
            )
        
        self.relax_function = relax_function
        self.energy_function = energy_function
        self.md_function = md_function
        self.name = name
    
    def calculate_energy(self, atoms: Atoms, **kwargs) -> CalculationResult:
        """
        Perform single-point energy calculation.
        
        Args:
            atoms: ASE Atoms object
            **kwargs: Passed to energy_function if it accepts them
            
        Returns:
            CalculationResult with energy (atoms unchanged)
        """
        if self.energy_function is not None:
            try:
                energy = self.energy_function(atoms, **kwargs)
            except TypeError:
                energy = self.energy_function(atoms)
            
            return CalculationResult(
                atoms=atoms.copy(),
                energy=energy,
                converged=True,
                metadata={'calculator': self.name, 'type': 'single_point'}
            )
        elif self.relax_function is not None:
            # Fall back to relax_function if no energy_function
            # (less efficient but works)
            return self.relax_structure(atoms, **kwargs)
        else:
            raise RuntimeError("No energy_function or relax_function available")
    
    def relax_structure(self, atoms: Atoms, **kwargs) -> CalculationResult:
        """
        Relax structure and return both relaxed atoms AND energy.
        
        This is the PRIMARY method - it returns both results from a single
        calculation, avoiding the need to evaluate energy twice.
        
        Args:
            atoms: ASE Atoms object
            **kwargs: Passed to relax_function if provided
            
        Returns:
            CalculationResult with:
                - atoms: relaxed structure (or copy of input if no relax_function)
                - energy: final energy after relaxation
                - converged: True (assumed unless relax_function indicates otherwise)
        """
        if self.relax_function is None:
            if self.energy_function is not None:
                # No relaxation - just return single-point energy
                return self.calculate_energy(atoms, **kwargs)
            else:
                raise RuntimeError("No relax_function or energy_function available")
        
        # Call relax function which returns (atoms, energy)
        try:
            result = self.relax_function(atoms, **kwargs)
        except TypeError:
            # Function doesn't accept kwargs
            result = self.relax_function(atoms)
        
        # Handle different return formats
        if isinstance(result, tuple):
            if len(result) == 2:
                relaxed_atoms, energy = result
                converged = True
            elif len(result) == 3:
                relaxed_atoms, energy, converged = result
            else:
                raise ValueError(
                    f"relax_function returned tuple of length {len(result)}, "
                    "expected 2 (atoms, energy) or 3 (atoms, energy, converged)"
                )
        elif isinstance(result, CalculationResult):
            # Already a CalculationResult
            return result
        elif isinstance(result, Atoms):
            # Only returned atoms - need to calculate energy separately
            relaxed_atoms = result
            if self.energy_function is not None:
                energy = self.energy_function(relaxed_atoms)
            else:
                raise ValueError(
                    "relax_function returned only Atoms without energy, "
                    "but no energy_function was provided. "
                    "Either return (atoms, energy) tuple or provide energy_function."
                )
            converged = True
        else:
            raise ValueError(
                f"relax_function returned unexpected type: {type(result)}. "
                "Expected Tuple[Atoms, float] or CalculationResult."
            )
        
        return CalculationResult(
            atoms=relaxed_atoms,
            energy=energy,
            converged=converged,
            metadata={'calculator': self.name, 'type': 'relaxation'}
        )
    
    def run_md(self, atoms: Atoms, temperature: float, steps: int, 
               **kwargs) -> CalculationResult:
        """
        Run molecular dynamics if md_function was provided.
        
        Args:
            atoms: ASE Atoms object
            temperature: Temperature in Kelvin
            steps: Number of MD steps
            **kwargs: Additional parameters passed to md_function
            
        Returns:
            CalculationResult with final structure and energy
        """
        if self.md_function is None:
            raise NotImplementedError(
                f"{self.name} does not have an MD function configured. "
                "Provide md_function to GenericCalculator to enable MD."
            )
        
        try:
            result = self.md_function(atoms, temperature, steps, **kwargs)
        except TypeError:
            result = self.md_function(atoms, temperature, steps)
        
        # Handle return formats
        if isinstance(result, tuple):
            if len(result) == 2:
                final_atoms, energy = result
                converged = True
            elif len(result) == 3:
                final_atoms, energy, converged = result
            else:
                raise ValueError(f"md_function returned unexpected tuple length: {len(result)}")
        elif isinstance(result, CalculationResult):
            return result
        else:
            raise ValueError(f"md_function returned unexpected type: {type(result)}")
        
        return CalculationResult(
            atoms=final_atoms,
            energy=energy,
            converged=converged,
            metadata={
                'calculator': self.name,
                'type': 'md',
                'temperature': temperature,
                'steps': steps
            }
        )
    
    def supports_md(self) -> bool:
        """Check if MD is supported (md_function was provided)."""
        return self.md_function is not None
    
    def __repr__(self):
        features = []
        if self.relax_function:
            features.append("relax")
        if self.energy_function:
            features.append("energy")
        if self.md_function:
            features.append("MD")
        return f"GenericCalculator({self.name}, {'+'.join(features) or 'no functions'})"


# Convenience function to create calculator from ASE calculator
def from_ase_calculator(ase_calc, optimizer_class=BFGS, fmax=0.05, steps=500):
    """
    Create a GenericCalculator from an ASE calculator.
    
    This is a convenience function that wraps an ASE calculator and provides
    both energy calculation and relaxation using ASE's optimizers.
    
    Args:
        ase_calc: An ASE calculator instance (e.g., EMT(), LennardJones())
        optimizer_class: ASE optimizer class to use (default: BFGS)
        fmax: Force convergence criterion in eV/Angstrom
        steps: Maximum optimization steps
        
    Returns:
        GenericCalculator instance
        
    Example:
        from ase.calculators.emt import EMT
        calc = from_ase_calculator(EMT(), fmax=0.01)
    """
    def energy_function(atoms):
        atoms_copy = atoms.copy()
        atoms_copy.calc = ase_calc
        return atoms_copy.get_potential_energy()
    
    def relax_function(atoms):
        atoms_copy = atoms.copy()
        atoms_copy.calc = ase_calc
        opt = optimizer_class(atoms_copy, logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
        energy = atoms_copy.get_potential_energy()
        return atoms_copy, energy, converged
    
    return GenericCalculator(
        energy_function=energy_function,
        relax_function=relax_function,
        name=f"ASE-{ase_calc.__class__.__name__}"
    )
