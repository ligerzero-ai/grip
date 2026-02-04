#!/usr/bin/env python3
"""
GRand canonical Interface Predictor (GRIP)

Main entry point for grain boundary optimization using grand canonical sampling.

This version supports multiple calculation backends through the Calculator interface:
- LAMMPS (original backend)
- ASE calculators (EMT, LJ, ML potentials)
- Generic user-provided functions

Usage:
    python main.py                     # Run with default params.yaml
    python main.py -i my_params.yaml   # Run with custom parameters
    python main.py -d                  # Debug mode (terminates early)
    python main.py --help              # Show all options
"""

import os
from argparse import ArgumentParser
from typing import Optional
import numpy as np

from core.bicrystal import Bicrystal
from core.simulation import Simulation
from core.calculator import Calculator
from utils.utils import make_dirs, get_inputs, make_crystals, compute_weights, \
                  get_xy_translation, get_xy_replications


def create_calculator(algo: dict, struct: dict) -> Optional[Calculator]:
    """
    Create a calculator from configuration.
    
    Args:
        algo: Algorithm parameters dictionary
        struct: Structure parameters dictionary
        
    Returns:
        Calculator instance or None for debug mode
    """
    # New-style calculator config
    if "calculator" in algo:
        from core.calculators import get_calculator
        return get_calculator(algo["calculator"])
    
    # Legacy LAMMPS config
    if "lammps_bin" in algo:
        from core.calculators.lammps_calc import LAMMPSCalculator
        return LAMMPSCalculator(
            binary=algo["lammps_bin"],
            pair_style=struct["pair_style"],
            pair_coeff=struct["pair_coeff"],
            mass=struct["mass"],
        )
    
    return None


def main(infile: str, debug: bool, calculator: Optional[Calculator] = None) -> None:
    """
    Performs grand canonical optimization of GB structures.

    Args:
        infile (str): YAML file of simulation parameters.
        debug (bool): Flag for running in DEBUG mode.
        calculator (Calculator, optional): Pre-configured calculator.
            If None, will be created from config file.

    Returns:
        None.
    """
    # Read in parameters from YAML file
    struct, algo = get_inputs(infile, debug)

    # Create calculator if not provided
    if calculator is None:
        calculator = create_calculator(algo, struct)
    
    # Create a Simulation object to orchestrate the simulation
    sim = Simulation(struct, algo, calculator, debug)
    if debug: print(f"Starting GRIP calculations from {sim.root}")
    if debug: print(f"This process is running in {sim.cfold}")
    if debug and calculator: print(f"Using calculator: {calculator}")

    # Create relevant directories
    make_dirs(sim.pid, algo["dir_struct"], algo["dir_calcs"])

    # Use structure parameters to create upper and lower bulk slabs
    lower_0, upper_0, dlat = make_crystals(struct, debug)

    # Compute the weights for replications
    weights = compute_weights(struct)
    if debug: print(f"The weights are: {weights}")

    # Create a Bicrystal object from the two bulk slabs
    bicrystal = Bicrystal(lower_0, upper_0, struct, algo, dlat,
                          make_copy=False, debug=debug)

    ##########################################################################
    # Main optimization loop - samples different GB structures
    ##########################################################################
    
    while sim.counter < sim.nruns or not sim.nruns:

        if debug: print(f"\n~~~~~ Starting simulation iteration {sim.counter+1} ~~~~~\n")

        # Make a copy of the parent slabs for each run
        bicrystal.copy_ul()

        # Sample a random translation and shift the upper slab
        dx, dy = get_xy_translation(upper_0, rng, algo["ngrid"], sim.pid, debug)
        bicrystal.shift_upper(dx, dy)
        if debug: print(f"Translation in (x, y) = [{dx:.4f}, {dy:.4f}]")

        # Get the bounds of the GB region for MD
        bicrystal.get_bounds(algo)
        if debug: print(f"Bounds for simulation (lower, upper, pad): \n{bicrystal.bounds}\n")

        # Sample a replication amount and replicate bicrystal in xy directions
        rx, ry = get_xy_replications(rng, weights)
        bicrystal.replicate(rx, ry)
        if debug: print(f"Replication in (x, y) = [{rx}, {ry}]")

        # Get the number of grain boundary atoms in the upper slab
        bicrystal.get_gbplane_atoms_u()
        if debug: print(f"Num atoms per plane in upper: {bicrystal.npp_u}\n")
        if debug: print(f"Num atoms in bicrystal: {bicrystal.natoms}")

        # Create vacancies in those grain boundary atoms in the upper slab
        bicrystal.defect_upper(algo, rng)

        if debug: print(f"{bicrystal} \n")
        if debug: print(f"Num atoms in bicrystal: {bicrystal.natoms}")
        if debug: print(f"n frac = {bicrystal.n}\n")

        # Perturb the GB atoms randomly
        bicrystal.perturb_atoms(rng)
        if debug: print(f"Perturbing GB atoms by {algo['perturb_u']} and {algo['perturb_l']}\n")

        # Combine the two slabs (upper w/ defects) into a single GB structure
        bicrystal.join_gb(algo)

        # Find interstitial sites and swap atoms in GB region
        swapped_n = bicrystal.find_and_swap_inters(rng)
        if debug: print(f"Swapping {swapped_n} GB atoms with interstitial sites.\n")

        # Write the GB structure to a file (input for calculator)
        input_struct_file = os.path.join(algo["dir_calcs"], f"{algo['dir_calcs']}_{sim.pid+1}", "STRUC")
        bicrystal.write_gb(input_struct_file)

        if debug: print(bicrystal)
        if debug: print(f"GB structure is {bicrystal.gb}\n")

        # Sample parameters like Temperature and Numsteps for this iteration
        sim.sample_params(rng)
        if debug: print(f"The simulation parameters are T={sim.md_T}, N={sim.md_steps}")

        # Run the calculation to produce a final, relaxed GB structure
        if debug: print(f"Running calculation...")
        result = sim.run_calculation(bicrystal, update_gb=True)
        if debug: print(f"Calculation converged: {result.converged}")

        # Get the GB energy
        sim.get_gb_energy(bicrystal)

        if debug: print(bicrystal)
        if debug: print(f"GB structure is {bicrystal.gb}\n")
        if debug: print(f"The GB energy is {bicrystal.Egb} J/m^2\n")

        # Store the energy and save the file to the "best" folder
        sim.store_best_structs(bicrystal)

        if debug:
            assert False, "Terminated early in DEBUG mode."

##############################################################################

if __name__ == "__main__":
    parser = ArgumentParser(description="Perform grand canonical optimization of GBs.")
    parser.add_argument("-i", "--input", type=str, default="params.yaml",
                        help="File containing structure & algorithm parameters.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run in DEBUG mode, which prints variables and terminates early.")
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="Random seed for reproducibility.")
    parser.add_argument("-e", "--engine", type=str, default=None,
                        help="Override calculator engine (lammps, ase, etc.)")
    args = parser.parse_args()
    
    infile = args.input
    debug = args.debug
    seed = args.seed

    # Set up random number generator
    if debug:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    main(infile, debug)
