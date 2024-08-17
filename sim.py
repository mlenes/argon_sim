import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sigma = 3.405e-10 # Meter
kB = 1.380649e-23
epsilon = 119.8 * kB # Kelvin
x_size = 5e-10 # Meter
y_size= 5e-10 # Meter
h = 0.01e-10 # Meter
m = 6.6335e-26 # kg
step = 10e-17 # s

arg_pos_list = [[1e-10, 1e-10], [5e-10, 1e-10], [0e-10, 4e-10]] # Positions of argon atoms
arg_vel_list = [[10000, 0], [0, 20000], [-5000, -3000]] # Velocities of argon atoms

def cbps(pos, x_size, y_size):
    '''
    Cross boundary positions
    All of the positions that are relevent to calculating shortest distance
    '''
    return [[pos[0], pos[1]], [pos[0], pos[1]+y_size], [pos[0], pos[1]-y_size], 
            [pos[0]-x_size, pos[1]], [pos[0]-x_size, pos[1]+y_size], [pos[0]-x_size, pos[1]-y_size], 
            [pos[0]+x_size, pos[1]], [pos[0]+x_size, pos[1]+y_size], [pos[0]+x_size, pos[1]-y_size]]

def r_list_list(arg_pos_list, x_size, y_size):
    r_list_list = []
    for pos1 in arg_pos_list:
        r_list = []
        for pos2 in arg_pos_list:
            if pos1 != pos2:
                r_list.append(np.min([np.linalg.norm([pos1[0] - cbp[0], pos1[1] - cbp[1]]) for cbp in cbps(pos2, x_size, y_size)]))
        r_list_list.append(r_list)
    return r_list_list

def len_jones(r):
    return 4*epsilon*((sigma/r)**(12)-(sigma/r)**(6))

def F_list(arg_pos_list, h):
    F_values = []
    r_values = r_list_list(arg_pos_list, x_size, y_size)
    for r_list in r_values:
        U = np.sum([len_jones(r) for r in r_list])

    for idx, arg in enumerate(arg_pos_list):
        arg_pos_list_copy = arg_pos_list.copy()
        arg_pos_list_copy[idx] = [arg[0]+h, arg[1]]
        r_values_h = r_list_list(arg_pos_list_copy, x_size, y_size)
        for r_list in r_values_h:
            U_h = np.sum([len_jones(r) for r in r_list])
        F_x = -1*(U_h-U)/h

        arg_pos_list_copy = arg_pos_list.copy()
        arg_pos_list_copy[idx] = [arg[0], arg[1]+h]
        r_values_h = r_list_list(arg_pos_list_copy, x_size, y_size)
        for r_list in r_values_h:
            U_h = np.sum([len_jones(r) for r in r_list])
        F_y = -1*(U_h-U)/h
        F_values.append([F_x, F_y])
    return F_values

pos_over_time = [arg_pos_list]
vel_over_time = [arg_vel_list]

current_pos = arg_pos_list
current_vel = arg_vel_list

for i in range(1000):
    Forces = F_list(current_pos, h)
    new_pos = []
    new_vel = []
    for idx, pos in enumerate(current_pos):
        new_x_pos = (pos[0] + current_vel[idx][0]*step) % x_size
        new_y_pos = (pos[1] + current_vel[idx][1]*step) % y_size
        new_pos.append([new_x_pos, new_y_pos])

        new_x_vel = current_vel[idx][0] + Forces[idx][0]*(step/m)
        new_y_vel = current_vel[idx][1] + Forces[idx][1]*(step/m)
        new_vel.append([new_x_vel, new_y_vel])
    current_pos = new_pos
    current_vel = new_vel
    pos_over_time.append(new_pos)
    vel_over_time.append(new_vel)

pos_over_time = np.array(pos_over_time)
vel_over_time =np.array(vel_over_time)

fig, ax = plt.subplots()

scat1 = ax.scatter(pos_over_time[:,0,0], pos_over_time[:,0,1])
scat2 = ax.scatter(pos_over_time[:,1,0], pos_over_time[:,1,1])
scat3 = ax.scatter(pos_over_time[:,2,0], pos_over_time[:,2,1])

def update(frame):
    # for each frame, update the data stored on each artist.
    x1 = pos_over_time[frame,0,0]
    y1 = pos_over_time[frame,0,1]

    x2 = pos_over_time[frame,1,0]
    y2 = pos_over_time[frame,1,1]

    x3 = pos_over_time[frame,2,0]
    y3 = pos_over_time[frame,2,1]
    # update the scatter plot:
    data1 = np.stack([x1, y1]).T
    data2 = np.stack([x2, y2]).T
    data3 = np.stack([x3, y3]).T
    scat1.set_offsets(data1)
    scat2.set_offsets(data2)
    scat3.set_offsets(data3)
    return (scat1, scat2, scat3)

ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=1)
plt.show()