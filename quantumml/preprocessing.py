"""
This module will be used to process the GASP run data for machine learning
"""
import os
import numpy as np
from pymatgen.io.vasp import Xdatcar, Oszicar
from sklearn.cluster import KMeans


def prep_ml_formation_energy(fileroot="."):
    """
    writes .poscar and .energy files with index information for use in model training

    Parameters
    ----------
    """
    n = 100  # number of steps to sample
    i = 0
    for a in os.walk("."):
        directory = a[0]
        s_extension = "poscar"
        e_extension = "energy"
        prefix = ""  # prefix for files, e.g. name of structure
        # e.g. "[root]/[prefix][i].[poscar]" where i=1,2,...,n
        try:
            s_list = Xdatcar(directory + "/XDATCAR").structures
            e_list = [
                step["E0"] for step in Oszicar(directory + "/OSZICAR").ionic_steps
            ]
            if n < len(s_list) - 1:
                # the idea here is to obtain a subset of n energies
                # such that the energies are as evenly-spaced as possible
                # we do this in energy-space not in relaxation-space
                # because energies drop fast and then level off
                idx_to_keep = []
                fitting_data = np.array(e_list)[:, np.newaxis]  # kmeans expects 2D
                kmeans_model = KMeans(n_clusters=n)
                kmeans_model.fit(fitting_data)
                cluster_centers = sorted(kmeans_model.cluster_centers_.flatten())
                for centroid in cluster_centers:
                    closest_idx = np.argmin(np.subtract(e_list, centroid) ** 2)
                    idx_to_keep.append(closest_idx)
                idx_to_keep[-1] = len(e_list) - 1  # replace the last
                idx_list = np.arange(len(s_list))
                idx_batched = np.array_split(idx_list[:-1], n)
                idx_kept = [batch[0] for batch in idx_batched]
                idx_kept.append(idx_list[-1])
            else:
                idx_kept = np.arange(len(e_list))

            for j, idx in enumerate(idx_kept):
                filestem = str(j)
                i2 = str(i)
                s_filename = "{}/{}{}_{}.{}".format(
                    fileroot, prefix, i2, filestem, s_extension
                )
                e_filename = "{}/{}{}_{}.{}".format(
                    fileroot, prefix, i2, filestem, e_extension
                )
                s_list[idx].to(fmt="poscar", filename=s_filename)
                with open(e_filename, "w") as f:
                    f.write(str(e_list[idx]))
            i = i + 1
        except:
            print("noFile")
