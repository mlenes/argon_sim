import numpy as np

# Constants
kB = 1.38e-23 # (kg*m^2)/(K*s^2)
epsilon = 1.65e-21 # kg*m^2/s^2

def len_jones_force(r):
    return 24*(-2*r**(-13) + r**(-7))

def len_jones_pot(r):
    return 4*(r**(-12) - r**(-6))

def rescale_factor(lattice, temperature):
    sum_vi = np.sum([np.linalg.norm(atom.vel) for atom in lattice.field])
    return np.sqrt(((lattice.n_atoms-1)*3*kB*temperature)/(epsilon*sum_vi))

def get_all_atom_positions(lattice):
    return np.stack([atom.pos for atom in lattice.field])
