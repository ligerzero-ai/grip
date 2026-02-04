"""
Simulation orchestration for GRIP grain boundary optimization.

This module has been refactored to use a generic Calculator interface,
allowing different backends (LAMMPS, ASE calculators, custom functions).
"""

import os
from subprocess import Popen, PIPE, getstatusoutput
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.io import write as ase_write, read as ase_read

from core.bicrystal import Bicrystal
from core.calculator import Calculator, CalculationResult
from utils.unique import clear_best


class Simulation():
    """
    A class for orchestrating grain boundary optimization simulations.
    
    This class manages the simulation workflow, including:
    - Structure manipulation
    - Running calculations via a Calculator interface
    - Storing and managing results
    
    Attributes:
        root (str): The base path for the simulation.
        debug (bool): Flag for running in DEBUG mode.
        pid (int): The current process ID.
        cfold (str): The subfolder for the current process.
        calculator (Calculator): The calculation backend.
        Ecoh (float): Bulk cohesive energy for the specific potential.
    """

    fname_final = "relaxed_structure"  # Base filename for relaxed structures

    def __init__(self,
                 struct: dict,
                 algo: dict,
                 calculator: Optional[Calculator] = None,
                 debug: bool = False):
        """
        Constructs a Simulation object.

        Args:
            struct (dict): Dictionary of structure parameters.
            algo (dict): Dictionary of algorithm parameters.
            calculator (Calculator, optional): Calculator for energy evaluation.
                If None, will try to create from algo['calculator'] config.
            debug (bool, optional): Flag for running in DEBUG mode.

        Returns:
            Simulation object
        """
        self.root = os.getcwd()
        self.debug = debug
        
        # Process ID handling for parallel execution
        if not self.debug:
            if "SLURM_PROCID" in os.environ:
                self.pid = int(os.environ["SLURM_PROCID"])
            elif "PBS_TASKNUM" in os.environ:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                self.pid = comm.Get_rank()
            else:
                self.pid = 0
        else:
            self.pid = 0
            
        self.cfold = os.path.join(algo["dir_calcs"], f"{algo['dir_calcs']}_{self.pid + 1}")
        self.Emult = algo["Emult"]
        self.counter = 0
        self.best_Egb = 1000
        self.nruns = algo["nruns"]
        self.clear_freq = algo["clear_freq"]
        
        # MD parameters (used when calculator supports MD)
        self.md_run = algo.get("MD_run", 0.0)
        if self.debug:
            self.md_steps0 = 1000
            self.md_steps1 = 1000
        else:
            self.md_steps0 = algo.get("MD_min", 0)
            self.md_steps1 = algo.get("MD_max", 0)
        self.md_steps = None
        self.md_var = algo.get("var_steps", 0)
        self.md_T = None
        self.Tmin = algo.get("Tmin", 300)
        self.Tmax = algo.get("Tmax", 1200)
        
        # Material properties
        self.symbol = struct["symbol"]
        self.Ecoh = struct["Ecoh"]
        self.mass = struct.get("mass", 1.0)
        
        # Calculator setup
        if calculator is not None:
            self.calculator = calculator
        elif "calculator" in algo:
            from core.calculators import get_calculator
            self.calculator = get_calculator(algo["calculator"])
        else:
            # Legacy mode: try to create LAMMPS calculator from old-style config
            self._setup_legacy_lammps(struct, algo)
        
        # Store last calculation result
        self.last_result: Optional[CalculationResult] = None

    def _setup_legacy_lammps(self, struct: dict, algo: dict) -> None:
        """Set up LAMMPS calculator from legacy config format."""
        if "lammps_bin" in algo:
            from core.calculators.lammps_calc import LAMMPSCalculator
            self.calculator = LAMMPSCalculator(
                binary=algo["lammps_bin"],
                pair_style=struct["pair_style"],
                pair_coeff=struct["pair_coeff"],
                mass=struct["mass"],
            )
        else:
            self.calculator = None
            if not self.debug:
                print("Warning: No calculator configured!")

    def sample_params(self, rng: np.random.Generator) -> None:
        """
        Selects a random MD temperature and numsteps.

        Args:
            rng (np.random.Generator): Random number generator from NumPy.

        Returns:
            None
        """
        # Randomly choose a temperature in multiples of 100
        self.md_T = rng.choice(np.arange(self.Tmin, self.Tmax + 1, 100))
        
        if rng.random() > self.md_run:
            self.md_steps = 0
        elif self.md_var == 1 and not self.debug:
            # Linearly scale the number of MD steps
            self.md_steps = int(np.round(rng.integers(self.md_steps0, self.md_steps1, 
                                                      endpoint=True), -3))
        elif self.md_var == 2 and not self.debug:
            # Exponentially scale the number of MD steps
            C = np.log(self.md_steps1 / self.md_steps0) if self.md_steps0 > 0 else 0
            self.md_steps = int(np.round(self.md_steps0 *
                                         np.exp(C * rng.random()), -3))
        else:
            self.md_steps = self.md_steps0

    def run_calculation(self, system: Bicrystal, update_gb: bool = True) -> CalculationResult:
        """
        Run calculation using the configured calculator.
        
        This method:
        1. Decides whether to run MD or just relaxation based on md_steps
        2. Calls the appropriate calculator method
        3. Returns BOTH the relaxed structure AND energy
        
        Args:
            system (Bicrystal): Bicrystal object with a GB.
            update_gb (bool): Store the relaxed GB structure into Bicrystal object.

        Returns:
            CalculationResult with relaxed structure and energy
        """
        assert self.md_T is not None and self.md_steps is not None, \
            "Must set temperature and steps first! Call sample_params()."
        assert system.bounds is not None, \
            "Compute GB boundaries using bicrystal.get_bounds() first!"
        
        if self.calculator is None:
            # Debug mode without calculator - return dummy result
            return self._dummy_calculation(system)
        
        # Enter calculation directory
        run_dir = os.path.join(self.root, self.cfold)
        os.chdir(run_dir)
        
        # Set working directory for calculator
        if hasattr(self.calculator, 'working_dir'):
            self.calculator.working_dir = run_dir
        
        try:
            # Choose calculation method based on MD steps
            if self.md_steps > 0 and self.calculator.supports_md():
                # Run MD followed by relaxation
                result = self.calculator.run_md(
                    atoms=system.gb,
                    temperature=self.md_T,
                    steps=self.md_steps,
                    bounds=tuple(system.bounds),
                    seed=self.pid + 1
                )
            else:
                # Just relaxation
                result = self.calculator.relax_structure(
                    atoms=system.gb,
                    bounds=tuple(system.bounds) if system.bounds is not None else None
                )
            
            self.last_result = result
            
            if update_gb:
                system.gb = result.atoms
                system.relaxed = True
                
        finally:
            os.chdir(self.root)
        
        self.counter += 1
        return result

    def _dummy_calculation(self, system: Bicrystal) -> CalculationResult:
        """Create dummy result for debug mode without calculator."""
        dummy_energy = 12.3456789  # Dummy GB energy
        return CalculationResult(
            atoms=system.gb.copy() if system.gb else None,
            energy=dummy_energy * len(system.gb) if system.gb else 0,
            converged=True,
            metadata={'type': 'dummy'}
        )

    def get_gb_energy(self, system: Bicrystal) -> float:
        """
        Calculate the GB energy from the last calculation result.

        Args:
            system (Bicrystal): Bicrystal object with a GB.

        Returns:
            Egb (float): GB energy in Joules per meter squared.
        """
        if self.last_result is None:
            raise RuntimeError("No calculation has been run yet!")
        
        # Calculate GB area (assuming xy plane is GB plane)
        area = system.gb.cell[0, 0] * system.gb.cell[1, 1]  # Angstrom^2
        
        # Get number of atoms in GB region for energy calculation
        # Using the bounds to determine which atoms to count
        if system.bounds is not None:
            lowerb, upperb, pad = system.bounds
            z_positions = system.gb.positions[:, 2]
            z_min = z_positions.min()
            z_max = z_positions.max()
            
            # GB region with padding
            lower_cutoff = z_min + lowerb - pad
            upper_cutoff = z_max - upperb + pad
            
            mask = (z_positions >= lower_cutoff) & (z_positions <= upper_cutoff)
            n_atoms = np.sum(mask)
        else:
            n_atoms = len(system.gb)
        
        # Calculate GB energy using calculator method
        Egb = self.calculator.get_gb_energy(
            result=self.last_result,
            n_atoms=n_atoms,
            area=area,
            e_cohesive=self.Ecoh
        )
        
        system.Egb = Egb
        return Egb

    def store_best_structs(self, system: Bicrystal, best_dir: str = "best",
                          format: str = "lammps-dump") -> None:
        """
        Store the best structures from the simulations.

        Args:
            system (Bicrystal): Bicrystal object with GB.
            best_dir (str, optional): Directory to save structures to.
            format (str, optional): Output format for structures.

        Returns:
            None
        """
        if system.Egb < self.best_Egb * self.Emult:
            if system.Egb < self.best_Egb:
                self.best_Egb = system.Egb

            # Custom file name to differentiate results
            fname_base = f"gb_{system.Egb:.3f}_{system.n:.3f}_" + \
                        f"{system.dxyz[0]:.2f}_{system.dxyz[1]:.2f}_" + \
                        f"{system.rxyz[0]:d}_{system.rxyz[1]:d}_" + \
                        f"{self.md_T:d}_{self.md_steps:d}"

            # Save structure using ASE
            output_path = os.path.join(best_dir, fname_base)
            
            if format == "lammps-dump":
                # Write in LAMMPS dump format for compatibility
                self._write_lammps_dump(output_path, system)
            else:
                # Use ASE for other formats
                ase_write(output_path, system.gb, format=format)

            # Periodically remove duplicate structures
            if self.clear_freq:
                if len(os.listdir(best_dir)) > 4000:
                    print(f"Clearing highest energy from {best_dir} now\n")
                    clear_best(best_dir, extra=True, alpha=0.5)
                elif self.counter % self.clear_freq == 0:
                    print(f"Clearing {best_dir} now\n")
                    clear_best(best_dir)

    def _write_lammps_dump(self, filename: str, system: Bicrystal) -> None:
        """Write structure in LAMMPS dump format for compatibility."""
        assert system.gb is not None, "GB hasn't been created yet!"
        
        with open(filename, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{system.natoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp ss\n")
            for i in range(3):
                f.write(f"0.0 {system.gb.cell[i, i]}\n")
            f.write("ITEM: ATOMS id type x y z c_eng\n")
            
            s = system.gb.get_chemical_symbols()
            s_unique_sorted = sorted(list(dict.fromkeys(s)))
            p = system.gb.get_positions()
            
            # Get per-atom energies if available
            if self.last_result and self.last_result.metadata:
                energies = self.last_result.metadata.get('per_atom_energies', 
                                                         np.zeros(len(p)))
            else:
                energies = np.zeros(len(p))
            
            for i in range(system.natoms):
                idx = s_unique_sorted.index(s[i])
                f.write(f"{i+1} {idx+1} {p[i,0]} {p[i,1]} {p[i,2]} {energies[i]:.6f}\n")
        
        # Append GB energy
        if system.Egb is not None:
            with open(filename, "a") as f:
                f.write(f"Egb = {system.Egb}\n")

    # =========================================================================
    # Legacy methods for backward compatibility
    # =========================================================================
    
    def run_md(self, system: Bicrystal, update_gb: bool = True) -> None:
        """
        Legacy method - calls run_calculation internally.
        
        Maintained for backward compatibility with existing scripts.
        """
        self.run_calculation(system, update_gb)
    
    def write_dummy_lammps_dump(self, filename: str, system: Bicrystal) -> None:
        """Legacy method for writing dummy LAMMPS output."""
        self._write_lammps_dump(filename, system)
