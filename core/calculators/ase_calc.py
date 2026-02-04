"""
ASE calculator wrapper for GRIP.

This module wraps any ASE-compatible calculator (EMT, LennardJones, 
machine learning potentials, etc.) to work with GRIP.
"""

from typing import Optional, Dict, Any, Type
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalc
from ase.optimize import BFGS, FIRE, LBFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from core.calculator import Calculator, CalculationResult


class ASECalculator(Calculator):
    """
    Wrapper for ASE calculators to use with GRIP.
    
    This allows using any ASE-compatible calculator (EMT, MACE, CHGNet, etc.)
    with the GRIP grain boundary optimization workflow.
    
    The key feature is that relax_structure returns BOTH the relaxed
    structure AND energy from a single optimization run.
    
    Example:
        from ase.calculators.emt import EMT
        calc = ASECalculator(calculator=EMT())
        
        # Or with ML potential
        from mace.calculators import MACECalculator
        mace = MACECalculator(model_path="model.pt")
        calc = ASECalculator(calculator=mace)
    """
    
    def __init__(
        self,
        calculator: Optional[ASECalc] = None,
        calculator_name: Optional[str] = None,
        calculator_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type = BFGS,
        fmax: float = 0.05,
        max_steps: int = 500,
    ):
        """
        Initialize ASE calculator wrapper.
        
        Args:
            calculator: An ASE calculator instance. If provided, used directly.
            calculator_name: Name of ASE calculator to instantiate (e.g., 'EMT')
            calculator_kwargs: Kwargs passed to calculator constructor
            optimizer_class: ASE optimizer class for relaxation (default: BFGS)
            fmax: Force convergence criterion in eV/Angstrom
            max_steps: Maximum optimization steps
        """
        if calculator is not None:
            self.calculator = calculator
        elif calculator_name is not None:
            self.calculator = self._get_calculator(calculator_name, calculator_kwargs or {})
        else:
            raise ValueError("Must provide either calculator instance or calculator_name")
        
        self.optimizer_class = optimizer_class
        self.fmax = fmax
        self.max_steps = max_steps
    
    def _get_calculator(self, name: str, kwargs: Dict[str, Any]) -> ASECalc:
        """Get ASE calculator by name."""
        name_lower = name.lower()
        
        if name_lower == 'emt':
            from ase.calculators.emt import EMT
            return EMT(**kwargs)
        
        elif name_lower == 'lj' or name_lower == 'lennardjones':
            from ase.calculators.lj import LennardJones
            return LennardJones(**kwargs)
        
        elif name_lower == 'mace':
            try:
                from mace.calculators import MACECalculator
                return MACECalculator(**kwargs)
            except ImportError:
                raise ImportError("MACE not installed. Install with: pip install mace-torch")
        
        elif name_lower == 'chgnet':
            try:
                from chgnet.model import CHGNetCalculator
                return CHGNetCalculator(**kwargs)
            except ImportError:
                raise ImportError("CHGNet not installed. Install with: pip install chgnet")
        
        elif name_lower == 'm3gnet':
            try:
                from m3gnet.models import M3GNetCalculator
                return M3GNetCalculator(**kwargs)
            except ImportError:
                raise ImportError("M3GNet not installed. Install with: pip install m3gnet")
        
        else:
            raise ValueError(
                f"Unknown calculator: {name}. "
                f"Supported: EMT, LJ, MACE, CHGNet, M3GNet. "
                f"Or provide a calculator instance directly."
            )
    
    def calculate_energy(self, atoms: Atoms, **kwargs) -> CalculationResult:
        """
        Single-point energy calculation.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            CalculationResult with energy (atoms unchanged)
        """
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator
        
        try:
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()
            converged = True
        except Exception as e:
            return CalculationResult(
                atoms=atoms_copy,
                energy=float('inf'),
                converged=False,
                metadata={'error': str(e)}
            )
        
        return CalculationResult(
            atoms=atoms_copy,
            energy=energy,
            converged=converged,
            forces=forces,
            metadata={'calculator': self.calculator.__class__.__name__, 'type': 'single_point'}
        )
    
    def relax_structure(self, atoms: Atoms, fmax: Optional[float] = None,
                       max_steps: Optional[int] = None, **kwargs) -> CalculationResult:
        """
        Relax structure and return both relaxed atoms AND energy.
        
        Args:
            atoms: ASE Atoms object
            fmax: Force convergence (overrides default)
            max_steps: Max steps (overrides default)
            
        Returns:
            CalculationResult with relaxed atoms and final energy
        """
        fmax = fmax or self.fmax
        max_steps = max_steps or self.max_steps
        
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator
        
        try:
            opt = self.optimizer_class(atoms_copy, logfile=None)
            converged = opt.run(fmax=fmax, steps=max_steps)
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()
        except Exception as e:
            return CalculationResult(
                atoms=atoms_copy,
                energy=float('inf'),
                converged=False,
                metadata={'error': str(e)}
            )
        
        return CalculationResult(
            atoms=atoms_copy,
            energy=energy,
            converged=converged,
            forces=forces,
            metadata={
                'calculator': self.calculator.__class__.__name__,
                'type': 'relaxation',
                'fmax': fmax,
                'steps': opt.nsteps
            }
        )
    
    def run_md(self, atoms: Atoms, temperature: float, steps: int,
               timestep: float = 1.0, friction: float = 0.02,
               seed: int = 12345, **kwargs) -> CalculationResult:
        """
        Run Langevin MD followed by relaxation.
        
        Args:
            atoms: ASE Atoms object
            temperature: Temperature in Kelvin
            steps: Number of MD steps
            timestep: MD timestep in fs (default: 1.0)
            friction: Langevin friction coefficient (default: 0.02)
            seed: Random seed
            
        Returns:
            CalculationResult with final structure and energy
        """
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator
        
        try:
            # Initialize velocities
            np.random.seed(seed)
            MaxwellBoltzmannDistribution(atoms_copy, temperature_K=temperature)
            
            # Run Langevin MD
            dyn = Langevin(
                atoms_copy,
                timestep=timestep * units.fs,
                temperature_K=temperature,
                friction=friction,
                logfile=None
            )
            dyn.run(steps)
            
            # Final relaxation
            opt = self.optimizer_class(atoms_copy, logfile=None)
            converged = opt.run(fmax=self.fmax, steps=self.max_steps)
            
            energy = atoms_copy.get_potential_energy()
            
        except Exception as e:
            return CalculationResult(
                atoms=atoms_copy,
                energy=float('inf'),
                converged=False,
                metadata={'error': str(e)}
            )
        
        return CalculationResult(
            atoms=atoms_copy,
            energy=energy,
            converged=converged,
            metadata={
                'calculator': self.calculator.__class__.__name__,
                'type': 'md',
                'temperature': temperature,
                'steps': steps
            }
        )
    
    def supports_md(self) -> bool:
        """ASE calculators support MD through ASE dynamics."""
        return True
    
    def __repr__(self):
        return f"ASECalculator({self.calculator.__class__.__name__})"
