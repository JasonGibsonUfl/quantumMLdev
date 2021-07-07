import sys
import os
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from uf3.util import user_config
from uf3.data import io
from uf3.representation import distances
from uf3.data.io import DataCoordinator
from uf3.data.composition import ChemicalSystem

if __name__ == "__main__":
    if len(sys.argv) == 2:
        settings_filename = "settings.yaml"
    else:
        settings_filename = sys.argv[-1]
    settings = user_config.read_config(settings_filename)

    element_list = settings["element_list"]
    degree = settings['degree']

    experiment_path = settings['experiment_path']
    output_directory = settings['output_directory']
    filename_pattern = settings['filename_pattern']
    cache_data = settings['cache_data']  # output
    data_filename = settings['data_filename']

    energy_key = settings['energy_key']
    force_key = settings['force_key']
    max_samples = settings['max_samples_per_file']
    min_diff = settings['min_diff']
    vasp_pressure = settings['vasp_pressure']

    analyze_fraction = settings['analyze_fraction']

    verbose = settings['verbose']
    random_seed = settings['random_seed']
    progress_bars = settings['progress_bars']

    np.random.seed(random_seed)
    if verbose >= 1:
        print(settings)

    # Parse data
    data_coordinator = DataCoordinator(energy_key=energy_key,
                                       force_key=force_key)
    data_paths = io.identify_paths(experiment_path=experiment_path,
                                   filename_pattern=filename_pattern)

    io.parse_with_subsampling(data_paths,
                              data_coordinator,
                              max_samples=max_samples,
                              min_diff=min_diff,
                              energy_key=energy_key,
                              vasp_pressure=vasp_pressure,
                              verbose=verbose)
    if cache_data:
        if os.path.isfile(data_filename):
            if verbose >= 1:
                print("Overwriting...")
            os.remove(data_filename)
        io.cache_data(data_coordinator, data_filename, energy_key=energy_key)
        if verbose >= 1:
            print("Cached data:", data_filename)
    df_data = data_coordinator.consolidate()
    if verbose >= 1:
        n_energies = len(df_data)
        n_forces = int(np.sum(df_data["size"]) * 3)
        print("Number of energies:", n_energies)
        print("Number of forces:", n_forces)
