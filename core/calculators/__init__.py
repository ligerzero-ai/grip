"""
Calculator implementations for GRIP.

This module provides various calculator backends for energy evaluation
and structure relaxation in grain boundary optimization.

Available calculators:
    - GenericCalculator: Wraps any function(Atoms) -> energy
    - LAMMPSCalculator: Interface to LAMMPS
    - ASECalculator: Wrapper for ASE built-in calculators
"""

from core.calculators.generic_calc import GenericCalculator
from core.calculators.lammps_calc import LAMMPSCalculator
from core.calculators.ase_calc import ASECalculator


def get_calculator(config: dict):
    """
    Factory function to create a calculator from configuration.
    
    Args:
        config: Dictionary with calculator configuration.
            Must contain 'engine' key specifying the calculator type.
            
    Returns:
        Calculator instance
        
    Example config:
        {
            'engine': 'lammps',
            'lammps': {
                'binary': '/path/to/lammps',
                'pair_style': 'eam/alloy',
                'pair_coeff': '* * pot.eam.alloy Cu',
            }
        }
    """
    engine = config.get('engine', 'generic').lower()
    
    if engine == 'lammps':
        lmp_config = config.get('lammps', {})
        return LAMMPSCalculator(
            binary=lmp_config.get('binary'),
            pair_style=lmp_config.get('pair_style'),
            pair_coeff=lmp_config.get('pair_coeff'),
            mass=lmp_config.get('mass'),
        )
    
    elif engine == 'ase':
        ase_config = config.get('ase', {})
        return ASECalculator(
            calculator_name=ase_config.get('calculator', 'EMT'),
            calculator_kwargs=ase_config.get('kwargs', {}),
        )
    
    elif engine == 'generic':
        # For generic, user must provide functions separately
        raise ValueError(
            "Generic calculator requires energy_function to be provided. "
            "Use GenericCalculator directly instead of get_calculator()."
        )
    
    else:
        raise ValueError(f"Unknown calculator engine: {engine}")


__all__ = [
    'Calculator',
    'CalculationResult', 
    'GenericCalculator',
    'LAMMPSCalculator',
    'ASECalculator',
    'get_calculator',
]
