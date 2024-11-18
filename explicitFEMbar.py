import numpy as np
np.set_printoptions(precision =2)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
## IMPORT MODULARIZED FUNCTIONS

from fem_functions import bar_element_1d, lumped_mass_matrix, geometry_1d
from force_functions import internal_force_1d, external_force_1d
from plot_animation_functions import plot_displacement_1d, plot_velocity_1d, animate_displacement_1d
from time_integration import time_step

###### MAIN SCRIPT ######
nx = 100          # number of elements along x-axis
total_length_x = 100.0  # total length
nnode, ndof, coord, connec, dof_id = geometry_1d(nx, total_length_x) # call geometry function to discretize

# material properties
E1 = 2.0        # Youngs modulus (Pa)
A1 = 3.0        # Area (m^2)
rho = 0.8       # Density (kg/m^3)
Es = np.full(len(connec), E1)   # assign E1 to all elements (Pa)
As = np.full(len(connec), A1)   # assign A1 to all elements (m^2)
force_value = -1.0e-1 # applied force (N)

# boundary conditions
B_1d = np.array([0])         # geometrically constrained nodes
R = np.array([0])            # applied value (m)

# compute lumped global mass matrix
M = lumped_mass_matrix(rho, As, coord, nx, ndof, connec, dof_id) # mass matrix (Kg)

# initialize displacement, velocity, acceleration
U = np.zeros(ndof)       # displacement (m)
V_half = np.zeros(ndof)  # velocity at n - 1/2 (m/s)
a = np.zeros(ndof)       # acceleration (m/s^2)

# compute initial acceleration (time = 0)
F_ext = external_force_1d(ndof, force_value) # external force (N)
F_int = internal_force_1d(U, Es, As, coord, ndof, nx, connec, dof_id) # internal force (N)
a = (F_ext - F_int) / M # accelleration (m/s^2)

# boundary conditions (time = 0)
U[B_1d] = R                  # displacement (m)
a[B_1d] = 0.0                # velocity (m/s)
V_half[B_1d] = 0.0           # acceleration (m/s^2)

# time step parameters
dt_crit = time_step(Es, As, coord, nx, connec, rho) # critical time step (s)
dt = dt_crit * 0.8 # scale critical time step to stabilize (s)
T_total = 1000.0 # total simulation time (s)
num_steps = int(T_total / dt) # number of time steps

# display time integration parameters for bar FEM simulation
print(f"Bar FEM simulation running...")
print(f"Time step (dt): {dt}")
print(f"Number of time steps (num_steps): {num_steps}")
print(f"Total simulation time (T_total): {T_total}")

time = np.linspace(0, T_total, num_steps) # time array (s)
U_history = np.zeros((ndof, num_steps))   # displacement history (m)
V_half_history = np.zeros((ndof, num_steps)) # velocity history (m/s)

# central difference method loop
for n in range(num_steps):
    t = n * dt

    # velocity at n + 1/2 (m/s)
    V_half += dt * a
    V_half[B_1d] = 0.0

    # displacement at n + 1 (m)
    U_new = U + dt * V_half
    U_new[B_1d] = R

    # internal force at n + 1 (m/s^2)
    F_ext = external_force_1d(ndof, force_value)
    F_int = internal_force_1d(U_new, Es, As, coord, ndof, nx, connec, dof_id)

    # acceleration at n + 1
    a = (F_ext - F_int) / M
    a[B_1d] = 0.0

    # save displacement history
    U_history[:, n] = U_new
    V_half_history[:, n] = V_half

    # update displacment for next time step
    U = U_new
    #print('t =',t)

# displacement solution
#print("Final Nodal Displacements:")
#print(f"{U}")
#print(dt)

scale_factor = 0.4
plot_displacement_1d(time, U_history, nnode, num_steps)   
plot_velocity_1d(time, V_half_history, nnode, num_steps)
animate_displacement_1d(coord[:, 0], connec, U_history, scale_factor, time, interval = 0.01)