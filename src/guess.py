#!/usr/bin/python
# coding: utf-8
""" This file implements the attempted fitting of
structures to an experimental diffraction input
with the diffpy package.
"""


def doc2diffpy(doc):
    """ Convert doc into diffpy Structure object. """
    from diffpy.Structure.atom import Atom
    from diffpy.Structure.lattice import Lattice
    from diffpy.Structure.structure import Structure
    from numpy import asarray

    lattice = Lattice(a=doc['lattice_abc'][0][0],
                      b=doc['lattice_abc'][0][1],
                      c=doc['lattice_abc'][0][2],
                      alpha=doc['lattice_abc'][1][0],
                      beta=doc['lattice_abc'][1][1],
                      gamma=doc['lattice_abc'][1][2]
                      )
    atoms = []
    for ind, atom in enumerate(doc['atom_types']):
        atoms.append(Atom(atype=atom,
                          xyz=asarray(doc['positions_frac'][ind])))
    title = None
    for sources in doc['source']:
        if sources.endswith('.res') or sources.endswith('.castep'):
            title = sources.split('/')[-1].split('.')[0].encode('utf-8')

    return Structure(atoms, lattice, title=title)
