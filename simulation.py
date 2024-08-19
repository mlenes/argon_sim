import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sim_tools
import sim_init

# Constants
kB = 1.38e-23 # (kg*m^2)/(K*s^2)
epsilon = 1.65e-21 # kg*m^2/s^2

# Input parameters
timestep = 1e-3 # sqrt(m*sigma**2/epsilon)
temperature = 300 # K
lattice_constant = 2 # sigma
n_lattices = 1 # lattices per row, column, layer
repetitions = 10 # how many times to run the updating algorithm before showing new locations in the animation
std = np.sqrt(kB*temperature/epsilon)

# Update values
def update_positions(lattice, timestep):
    '''
    Update the positions of the input lattice
    '''
    for atom in lattice.field:
        atom.pos = (atom.pos + atom.vel*timestep + (atom.force_t/2)*timestep**2) % lattice.field_size

def update_forces(lattice):
    '''
    Update the forces of the input lattice
    '''
    lattice.total_U = 0
    all_forces = [[] for _ in range(len(lattice.field))]

    for i in range(len(lattice.field) - 1):
        for j in range(i + 1, len(lattice.field)):
            r_vec = (lattice.field[i].pos - lattice.field[j].pos + lattice.field_size/2) % lattice.field_size - lattice.field_size/2
            r = np.linalg.norm(r_vec)
            F = sim_tools.len_jones_force(r)*r_vec/r
            all_forces[i].append(F)
            all_forces[j].append(-1*F)

            lattice.total_U += sim_tools.len_jones_pot(r) # Update the total potential energy

    for i in range(len(lattice.field)):
        lattice.field[i].force_t_h = -1*np.sum(all_forces[i], axis=0) 

def update_velocities(lattice, timestep):
    '''
    Update the velocities of the input lattice
    '''
    lattice.total_K = 0
    for atom in lattice.field:
        atom.vel = atom.vel + (timestep/2)*(atom.force_t_h + atom.force_t)
        lattice.total_K += 0.5*(np.linalg.norm(atom.vel)**2) # Update the total kinetic energy
        atom.force_t = atom.force_t_h # Move up a timestep


# Get initial lattice
lattice = sim_init.FCC_lattice(n_lattices, lattice_constant, std)

# Graph
fig = plt.figure(figsize=(10, 5))
rows = 1
columns = 2

# Initial 3d plot for atoms
ax1 = fig.add_subplot(rows, columns, 1, projection='3d')
ax1.set_xlabel('x [sigma]')
ax1.set_ylabel('y [sigma]')
ax1.set_zlabel('z [sigma]')
ax1.view_init(elev=20., azim=40, roll=0)
ax1.axes.set_xlim3d(left = 0, right = lattice.field_size)
ax1.axes.set_ylim3d(bottom = 0, top = lattice.field_size)
ax1.axes.set_zlim3d(bottom = 0, top = lattice.field_size)

# Initial plot for atom positions
positions = sim_tools.get_all_atom_positions(lattice)
atom_plot = ax1.plot(positions[:,0], positions[:,1], positions[:,2], marker='o', markersize=8, ls='', alpha=0.5)[0]

# Initial 3d plot for energy
ax2 = fig.add_subplot(rows, columns, 2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Energy per atom [epsilon]')
total_pot_energy = ax2.plot([], [], label='Potential energy')[0]
ax2.axes.set_xlim(left = 0, right = 2)
ax2.axes.set_ylim(bottom = -5, top = (10)**2)

running_pot_energy = []

total_kin_energy = ax2.plot([], [], label='Kinetic energy')[0]
running_kin_energy = []

total_energy = ax2.plot([], [], label='Total energy')[0]
running_total_energy = []
ax2.legend()

# Button for initializing temperature
def clicked(event):
    factor = sim_tools.rescale_factor(lattice, temperature)
    for atom in lattice.field:
        atom.vel = atom.vel*factor

axset_t = fig.add_axes([0.4, 0.05, 0.1, 0.075])
set_t_button = Button(axset_t, label='Init Temp', hovercolor='white')
set_t_button.on_clicked(clicked)

time_list = [] # Keeps track of global time

def update(frame):
    for _ in range(repetitions):
        update_positions(lattice, timestep)
        update_forces(lattice)
        update_velocities(lattice, timestep)

        if len(running_kin_energy) < 300:
            running_pot_energy.append(lattice.total_U/lattice.n_atoms)
            running_kin_energy.append(lattice.total_K/lattice.n_atoms)
            running_total_energy.append((lattice.total_K+lattice.total_U)/lattice.n_atoms)

            if time_list == []:
                time_list.append(0)
            else:
                time_list.append(time_list[-1] + timestep)
        else:
            running_pot_energy.append(lattice.total_U/lattice.n_atoms)
            running_kin_energy.append(lattice.total_K/lattice.n_atoms)
            running_total_energy.append((lattice.total_K+lattice.total_U)/lattice.n_atoms)
            running_pot_energy.pop(0)
            running_kin_energy.pop(0)
            running_total_energy.pop(0)

            time_list.append(time_list[-1] + timestep)
            time_list.pop(0)

    positions = sim_tools.get_all_atom_positions(lattice)
    atom_plot.set_data(positions[:,0], positions[:,1])
    atom_plot.set_3d_properties(positions[:,2])   

    total_pot_energy.set_data(time_list, running_pot_energy)
    total_kin_energy.set_data(time_list, running_kin_energy)
    total_energy.set_data(time_list, running_total_energy)
    ax2.axes.set_xlim(left=time_list[0], right=time_list[-1])

# calling the animation function     
anim = animation.FuncAnimation(fig, update, frames = 1000, interval = 1) 

plt.show()

