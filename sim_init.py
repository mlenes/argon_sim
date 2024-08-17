import numpy as np

class Argon:
    def __init__(self, pos, vel) -> None:
        self.pos = pos
        self.vel = vel
        self.force_t = np.array([0, 0, 0])
        self.force_t_h = np.array([0, 0, 0])

class FCC_lattice:
    def __init__(self, n_lattices, lattice_constant, std) -> None:
        self.n_lattices = n_lattices
        self.lattice_constant = lattice_constant
        self.std = std
        self.total_K = 0
        self.total_U = 0
        self.field = self.get_field()
        self.field_size = self.get_field_size()
        self.n_atoms = self.get_n_atoms()

    def get_field_size(self):
        return self.lattice_constant*(self.n_lattices + 1)
    
    def get_field(self):
        n_layers = (self.n_lattices*2) + 1
        field = []
        for layer in range(n_layers):
            if layer % 2 == 0:
                for row in range(self.n_lattices + 1):
                    for column in range(self.n_lattices + 1):
                        field.append(Argon(np.array([(column + 0.5)*self.lattice_constant, 
                                            (row + 0.5)*self.lattice_constant, 
                                            (layer*0.5 + 0.5)*self.lattice_constant]), 
                                            np.array(np.random.normal(0, self.std, 3)))) # corner atoms
                for row in range(self.n_lattices):
                    for column in range(self.n_lattices):
                        field.append(Argon(np.array([(column + 1)*self.lattice_constant, 
                                            (row + 1)*self.lattice_constant, 
                                            (layer*0.5 + 0.5)*self.lattice_constant]), 
                                            np.array(np.random.normal(0, self.std, 3)))) # face atoms
            else:
                for row in range(self.n_lattices + 1):
                    for column in range(self.n_lattices):
                        field.append(Argon(np.array([(column + 1)*self.lattice_constant, 
                                            (row + 0.5)*self.lattice_constant, 
                                            (layer*0.5 + 0.5)*self.lattice_constant]), 
                                            np.array(np.random.normal(0, self.std, 3)))) # face atoms
                for row in range(self.n_lattices):
                    for column in range(self.n_lattices + 1):
                        field.append(Argon(np.array([(column + 0.5)*self.lattice_constant, 
                                            (row + 1)*self.lattice_constant, 
                                            (layer*0.5 + 0.5)*self.lattice_constant]), 
                                            np.array(np.random.normal(0, self.std, 3)))) # face atoms
        return field
    
    def get_n_atoms(self):
        return len(self.field)