import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

## IMPORT MODULARIZED FUNCTIONS
from fem_functions import lumped_mass_matrix, bar_element_2d, geometry_2d
from force_functions import internal_force_2d, external_force_2d
from plot_animation_functions import animate_displacement_2d, plot_displacement_2d, plot_velocity_2d, plot_mesh
from time_integration import time_step

###### MAIN SCRIPT ######
nx = 20 # number of elements in x
ny = 2 # number of elements in y
total_length_x = 20.0 # length of structure in x (mm)
total_length_y = 2.0 # length of structure in y (mm)
nnode, ndof, coord, connec, dof_id = geometry_2d(nx, ny, total_length_x, total_length_y)
nele = len(connec)
#print(nele)
#print('number of nodes =', nnode)
#print('connec = ', connec)
#print('dof_id =',dof_id)

# material properties
E1 = 30000.0        # Young's Modulus (MPa)
A1 = 0.0789       # Area (mm^2)
rho = 2.4e-6       # Density (kg/mm^3)
Es = np.full(len(connec), E1)   # assign E1 to all elements (Pa)
As = np.full(len(connec), A1)   # assign A1 to all elements (m^2)

# boundary conditions
B_2d = np.where(coord[:, 0] == 0)[0]
B_dof = np.hstack([2 * B_2d, 2 * B_2d + 1]) # x and y DOFS for all nodes at x = 0
#print("Boundary Conditions (B_dof):", B_dof)

# externally applied force function
right_side_nodes = np.where(np.isclose(coord[:, 0], total_length_x))[0]
force_value = -200.0 # applied force (N)

# compute lumped global mass matrix
M = lumped_mass_matrix(rho, As, coord, nele, ndof, connec, dof_id) # mass matrix (Kg)
#print('M = ', M)

# initialize displacement, velocity, acceleration vectors
U = np.zeros(ndof)      # displacement (m)
V_half = np.zeros(ndof) # velocity at n - 1/2 (m/s)
a = np.zeros(ndof)      # acceleration (m/s^2)

# compute initila acceleration at time = 0
F_ext = external_force_2d(ndof, right_side_nodes, force_value) # external force (N)
F_int = internal_force_2d(U, Es, As, coord, ndof, nele, connec, dof_id) # internal force (N)
a = (F_ext - F_int) / M # acceleration (m/s^2)

# boundary conditions at time = 0
U[B_2d] = 0.0                  # displacement (mm)
a[B_2d] = 0.0                  # acceleration (mm/s^2)
V_half[B_2d] = 0.0             # velocity at n - 1/2 (mm/s)

# timestep parameters
dt_crit = time_step(Es, As, coord, nele, connec, rho) # critical time step (s)
dt = dt_crit * 0.8 # scale critical timestep for stabilization (s)
T_total = 0.001 # total simulation time (s)
num_steps = int(T_total / dt) # number of time steps

# display time integration parameters for Lattice FEM simulation
print(f"Lattice FEM simulation running...")
print(f"Time step (dt): {dt}")
print(f"Number of time steps (num_steps): {num_steps}")
print(f"Total simulation time (T_total): {T_total}")

time = np.linspace(0, T_total, num_steps)    # time array (s)
U_history = np.zeros((ndof, num_steps))      # displacement history (m)
V_half_history = np.zeros((ndof, num_steps)) # velocity history (m/s)

# central difference method algorithm
for n in range(num_steps):
    t = n * dt
    
    # velocity at n + 1/2 (m/s)
    V_half += dt * a
    V_half[B_dof] = 0.0
    
    # displacement at n + 1 (m)
    U_new = U + dt * V_half
    U_new[B_dof] = 0.0
    
    # internal force at n + 1 (N)
    F_ext = external_force_2d(ndof, right_side_nodes, force_value)
    F_int = internal_force_2d(U_new, Es, As, coord, ndof, nele, connec, dof_id)
    
    # debug force balance
    #force_balance = F_ext - F_int    
    # if n % 1 == 0:
    #     print(f"Step {n}: Displacement U:\n {U}")
    #     print(f"Step {n}: Velocity V_half:\n {V_half}")
    #     print(f"Internal Force at step {n}:\n", F_int)
    #     print(f"Force Balance at step {n}:\n {force_balance}")

    # acceleration at n + 1 (m/s^2)
    a = (F_ext - F_int) / M
    a[B_dof] = 0.0
    
    # save displacement and velocity history
    U_history[:, n] = U_new
    V_half_history[:, n] = V_half
    
    # update displacement for next timestep
    U = U_new
    #print('t =', t)

# extract DOF indices and print displacements at applied forces
U_disp = U_history[:, -1]   
right_dof_x = 2 * right_side_nodes
right_dof_y = 2 * right_side_nodes + 1
disp_x = U_disp[right_dof_x]
disp_y = U_disp[right_dof_y]

# displacement solution
print("Lattice FEM simulation complete\n") 
print(f"results for Lattice FEM simulation:")
print(f"node #\t x-displacement\t y-displacement")
for i, node in enumerate(right_side_nodes):
    print(f"{node}\t{disp_x[i]:.6e}\t{disp_y[i]:.6e}")
    
scale_factor = 0.4
plot_displacement_2d(time, U_history, right_side_nodes)
plot_velocity_2d(time, V_half_history, right_side_nodes)    
animate_displacement_2d(coord, connec, U_history, scale_factor, time, interval=1)
plot_mesh(coord, connec)