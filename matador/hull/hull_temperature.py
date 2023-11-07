# coding: utf-8
# Distributed under the terms of the MIT License.

""" This file implements the `TemperatureDependentHull` class
for assessing phase stability from finite temperature free energies.

"""

import copy
import numpy as np
import tqdm

from matador.hull.hull_ensemble import EnsembleHull
from matador.orm.spectral import VibrationalDOS


class TemperatureDependentHull(EnsembleHull):
    """Leverages `EnsembleHull` to construct temperature dependent
    hulls from phonon calculations.

    """

    data_key = "temperature"
    energy_key = "free_energy_per_atom"

    def __init__(
        self, cursor, energy_key="enthalpy_per_atom", temperatures=None, use_castep_thermo=True, **kwargs
    ):

        self.temperatures = temperatures
        self.use_castep_thermo = use_castep_thermo
        if temperatures is None:
            self.temperatures = np.linspace(0, 800, 21)

        _cursor = copy.deepcopy(cursor)

        # prepare the cursor by computing free energies
        # and store it in the format expected by EnsembleHull
        #print("computing free energies from vibrations...")
        for ind, doc in enumerate(cursor):
            #print(ind, doc["source"])
            if not isinstance(doc, VibrationalDOS):

                #jpd47 computing the virational free energy CAN be VERY SLOW
                #it can also already be present in the doc if a thermodynamics calculation was performed.
                if not self.use_castep_thermo:
                    _doc = VibrationalDOS(doc)
                    temps, vib_free_energies = _doc.vibrational_free_energy(
                        temperatures=self.temperatures
                    )
                else:
                    #print("trying to get vibrational free enrgies direct from the castep run")
                    castep_temps = np.array(doc["thermo_temps"])
                    castep_vib_free_energies = np.array([x for x in doc["thermo_free_energy"].values()])
                    temps = self.temperatures
                    # print("castep_temps", castep_temps.shape)
                    # print("castep_vib_free_energies", castep_vib_free_energies.shape)

                    #these should be per atom...
                    vib_free_energies = [np.interp(T, castep_temps, castep_vib_free_energies)/doc["num_atoms"] for T in temps]


                _cursor[ind][self.data_key] = {}
                _cursor[ind][self.data_key][self.energy_key] = (
                    np.ones_like(self.temperatures) * _cursor[ind][energy_key]
                )
                _cursor[ind][self.data_key][self.energy_key] += vib_free_energies
                _cursor[ind][self.data_key]["temperatures"] = self.temperatures

        super().__init__(
            cursor=_cursor,
            data_key=self.data_key,
            energy_key=self.energy_key,
            chempot_energy_key=energy_key,
            parameter_key="temperatures",
            update_chem_pots=True,
            **kwargs
        )

    def plot_hull(self, **kwargs):
        """Hull plot helper function."""
        from matador.plotting.hull_plotting import plot_temperature_hull

        return plot_temperature_hull(self, **kwargs)
