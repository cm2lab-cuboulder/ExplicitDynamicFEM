import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import cProfile
import pstats
from line_profiler import LineProfiler

from fem_functions import (
    lumped_mass_matrix,
    bar_element_1d, geometry_1d,
    bar_element_2d, geometry_2d)

from force_functions import (
    internal_force_1d, external_force_1d,
    internal_force_2d, external_force_2d)

from plot_animation_functions import (
    plot_displacement_1d, plot_velocity_1d, animate_displacement_1d,
    plot_displacement_2d, plot_velocity_2d, animate_displacement_2d,
    plot_mesh)

from time_integration import time_step

def run_1d_simulation():
    nnode, ndof, coord, connec, dof_id = geometry_1d(nx, total_length_x) # call geometry function to discretize
    end_node_index_1d = nnode - 1

    Es = np.full(len(connec), E1)   # assign E1 to all bar elements (Pa)
    As = np.full(len(connec), A1)   # assign A1 to all bar elements (m^2)

    # boundary conditions
    B_1d = np.array([0])            # geometrically constrained nodes
    R_1d = np.array([0])            # applied displacement (m)

    # compute lumped global mass matrix
    M_1d = lumped_mass_matrix(rho, As, coord, nx, ndof, connec, dof_id) # mass matrix (Kg)
    print('M = ', M_1d)
    
    # 1D bar total mass
    L_total_1D = total_length_x
    M_total_1D = rho * A1 * L_total_1D
    print(f"Total Mass in 1D Bar: {M_total_1D}")

    # initialize displacement, velocity, acceleration vectors
    U_1d = np.zeros(ndof)       # displacement (m)
    V_half_1d = np.zeros(ndof)  # velocity at n - 1/2 (m/s)
    a_1d = np.zeros(ndof)       # acceleration (m/s^2)

    # compute initial acceleration (time = 0)
    F_ext = external_force_1d(ndof, force_value) # external force (N)
    F_int = internal_force_1d(U_1d, Es, As, coord, ndof, nx, connec, dof_id) # internal force (N)
    a_1d = (F_ext - F_int) / M_1d # acceleration (m/s^2)

    # boundary conditions (time = 0)
    U_1d[B_1d] = R_1d           # displacement (m)
    V_half_1d[B_1d] = 0.0       # velocity (m/s)
    a_1d[B_1d] = 0.0            # acceleration (m/s^2)

    # time step parameters
    dt_crit_1d = time_step(Es, As, coord, nx, connec, rho) # critical time step (s)
    dt = dt_crit_1d * 0.9 # scale critical time step to stabilize (s)
    num_steps = int(T_total / dt) # number of time steps
    
    # display time integration parameters for bar FEM simulation
    print(f"Bar FEM simulation running...")
    print(f"Time step (dt): {dt}")
    print(f"Number of time steps (num_steps): {num_steps}")
    print(f"Total simulation time (T_total): {T_total}")

    time = np.linspace(0, T_total, num_steps) # time array (s)
    U_history_1d = np.zeros((ndof, num_steps)) # displacement history (m)
    V_half_history_1d = np.zeros((ndof, num_steps)) # velocity history (m/s)

    # central difference method algorithm
    for n in range(num_steps):
        t = n * dt

        # velocity at n + 1/2 (m/s)
        V_half_1d += dt * a_1d
        V_half_1d[B_1d] = 0.0

        # displacement at n + 1 (m)
        U_new_1d = U_1d + dt * V_half_1d
        U_new_1d[B_1d] = R_1d

        # internal force at n + 1 (N)
        F_ext = external_force_1d(ndof, force_value)
        F_int = internal_force_1d(U_new_1d, Es, As, coord, ndof, nx, connec, dof_id)

        # acceleration at n + 1 (m/s^2)
        a_1d = (F_ext - F_int) / M_1d
        a_1d[B_1d] = 0.0

        # save displacement history
        U_history_1d[:, n] = U_new_1d
        V_half_history_1d[:, n] = V_half_1d

        # update displacment for next time step
        U_1d = U_new_1d
        #print('t =',t)
    
        # extract DOF indices and print displacements at applied forces
    # displacement solution 
    #print("Final Nodal Displacement:\n",f"{U_1d}")
        
    plot_displacement_1d(time, U_history_1d, nnode, num_steps)   
    plot_velocity_1d(time, V_half_history_1d, nnode, num_steps)
    animate_displacement_1d(coord[:, 0], connec, U_history_1d, scale_factor, time, interval = 0.01)
    print("Bar FEM simulation complete\n")
    return U_history_1d, V_half_history_1d, time, end_node_index_1d

#@Profile
def run_2d_simulation():
    nnode, ndof, coord, connec, dof_id = geometry_2d(nx, ny, total_length_x, total_length_y)
    nele = len(connec)
    
    ########print debugging
    #print(nele)
    #print('number of nodes =', nnode)
    #print('connec = ', connec)
    #print('dof_id =',dof_id)

    Es = np.full(len(connec), E1)   # assign E1 to all elements (Pa)
    A1_2d = A1 / ny # correct mass distribution for 2D Lattice
    As = np.full(len(connec), A1_2d)   # assign A1 to all elements (m^2)

    num_node = len(coord)
    all_nodes = np.arange(num_node)
    y_dofs_all = 2 * all_nodes + 1

    # boundary conditions
    B_2d = np.where(coord[:, 0] == 0)[0] # geometrically constrain left side of lattice
    #B_dof = np.hstack([2 * B_2d, 2 * B_2d + 1]) # x and y DOFS for all nodes at x = 0
    B_dof = np.hstack([2 * B_2d, y_dofs_all])
    #print("Boundary Conditions (B_dof):", B_dof)

    # externally applied force function
    right_side_nodes = np.where(np.isclose(coord[:, 0], total_length_x))[0]
    
    # extract middle right node for plotting
    y_coords = coord[right_side_nodes, 1]
    mid_y = total_length_y / 2.0
    abs_diff = np.abs(y_coords - mid_y)
    min_diff = np.min(abs_diff)
    indices_of_middle_nodes = np.where(abs_diff == min_diff)[0]
    index_of_middle_node = indices_of_middle_nodes[0]
    middle_node_2d = right_side_nodes[index_of_middle_node]

    # compute lumped global mass matrix
    M_2d = lumped_mass_matrix(rho, As, coord, nele, ndof, connec, dof_id) # mass matrix (Kg)
    print('M = ', M_2d)

    # 2D lattice total mass
    L_total_2D = total_length_x * ny
    M_total_2D = rho * A1_2d * L_total_2D
    print(f"Total Mass in 2D Lattice: {M_total_2D}")

    # initialize displacement, velocity, acceleration vectors
    U_2d = np.zeros(ndof)      # displacement (m)
    V_half_2d = np.zeros(ndof) # velocity at n - 1/2 (m/s)
    a_2d = np.zeros(ndof)      # acceleration (m/s^2)

    # compute initial acceleration (time = 0)
    F_ext = external_force_2d(ndof, right_side_nodes, force_value) # external force (N)
    F_int = internal_force_2d(U_2d, Es, As, coord, ndof, nele, connec, dof_id) # internal force (N)
    a_2d = (F_ext - F_int) / M_2d # acceleration (m/s^2)
    
    # boundary conditions (time = 0)
    U_2d[B_2d] = 0.0             # displacement (m)
    a_2d[B_2d] = 0.0             # velocity (m/s)
    V_half_2d[B_2d] = 0.0        # acceleration (m/s^2)

    # timestep parameters
    dt_crit_2d = time_step(Es, As, coord, nele, connec, rho) # critial time step (s)
    dt = dt_crit_2d * 0.9 # scale critical timestep for stabilization (s)
    num_steps = int(T_total / dt) # number of time steps
    
    # display time integration parameters for Lattice FEM simulation
    print(f"Lattice FEM simulation running...")
    print(f"Time step (dt): {dt}")
    print(f"Number of time steps (num_steps): {num_steps}")
    print(f"Total simulation time (T_total): {T_total}\n")

    time = np.linspace(0, T_total, num_steps)       # time array (s)
    U_history_2d = np.zeros((ndof, num_steps))      # displacement history (m)
    V_half_history_2d = np.zeros((ndof, num_steps)) # velocity history (m/s)

    # central difference method algorithm
    for n in range(num_steps):
        t = n * dt
        
        # velocity at n + 1/2 (m/s)
        V_half_2d += dt * a_2d
        V_half_2d[B_dof] = 0.0
        
        # displacement at n + 1 (m)
        U_new_2d = U_2d + dt * V_half_2d
        U_new_2d[B_dof] = 0.0
        
        # internal force at n + 1 (N)
        F_ext = external_force_2d(ndof, right_side_nodes, force_value)
        F_int = internal_force_2d(U_new_2d, Es, As, coord, ndof, nele, connec, dof_id)
        
        # debug force balance
        #force_balance = F_ext - F_int    
        # if n % 1 == 0:
        #     print(f"Step {n}: Displacement U:\n {U_2d}")
        #     print(f"Step {n}: Velocity V_half:\n {V_half_2d}")
        #     print(f"Internal Force at step {n}:\n", F_int)
        #     print(f"Force Balance at step {n}:\n {force_balance}")

        # acceleration at n + 1 (m/s^2)
        a_2d = (F_ext - F_int) / M_2d
        a_2d[B_dof] = 0.0
        
        # save displacement and velocity history
        U_history_2d[:, n] = U_new_2d
        V_half_history_2d[:, n] = V_half_2d
        
        # update displacement for next timestep
        U_2d = U_new_2d
        #print('t =', t)

    # extract DOF indices and print displacements at applied forces
    U_disp_2d = U_history_2d[:, -1]   
    right_dof_x = 2 * right_side_nodes
    right_dof_y = 2 * right_side_nodes + 1
    disp_x = U_disp_2d[right_dof_x]
    disp_y = U_disp_2d[right_dof_y]
    
    # displacement solution 
    print("Lattice FEM simulation complete\n")
    print(f"results for Lattice FEM simulation:")
    print(f"node #\t x-displacement\t y-displacement")
    for i, node in enumerate(right_side_nodes):
        print(f"{node}\t{disp_x[i]:.6e}\t{disp_y[i]:.6e}")
        
    plot_displacement_2d(time, U_history_2d, right_side_nodes)
    plot_velocity_2d(time, V_half_history_2d, right_side_nodes)    
    animate_displacement_2d(coord, connec, U_history_2d, scale_factor, time, interval =0.01)
    plot_mesh(coord, connec)
    end_node_index_2d = right_side_nodes
    return U_history_2d, V_half_history_2d, time, middle_node_2d

def compare_plots(time_1d, U_history_1d, V_history_1d, end_node_index_1d,
                  time_2d, U_history_2d, V_history_2d, middle_node_2d):
    # Plot displacement comparison
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # displacement plot comparison
    end_node_displacement_1d = U_history_1d[end_node_index_1d,:]
    axs[0].plot(time_1d, end_node_displacement_1d, label='bar displacement')
    middle_node_displacement_2d = U_history_2d[2 * middle_node_2d, :]
    axs[0].plot(time_2d, middle_node_displacement_2d, label='lattice displacement')
    axs[0].set_title('displacement comparison')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('displacement (m)')
    axs[0].legend(bbox_to_anchor=(1.1, 1.05))
    axs[0].grid(True)

    # velocity plot comparison
    end_node_velocity_1d = (V_history_1d[end_node_index_1d,:])
    axs[1].plot(time_1d, end_node_velocity_1d, label='bar velocity')
    middle_node_velocity_2d = V_history_2d[2 * middle_node_2d, :]
    axs[1].plot(time_2d, middle_node_velocity_2d, label='lattice velocity')
    axs[1].set_title('velocity comparison')
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('velocity (m/s)')
    axs[1].legend(bbox_to_anchor=(1.1, 1.05))
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    U_history_1d, V_half_history_1d, time_1d, end_node_index_1d = run_1d_simulation()
    U_history_2d, V_half_history_2d, time_2d, middle_node_2d = run_2d_simulation()

    compare_plots(
        time_1d, U_history_1d, V_half_history_1d, end_node_index_1d,
        time_2d, U_history_2d, V_half_history_2d, middle_node_2d)

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # applied force parameter for all simulation
    force_value = -1.0e-1 # applied force (N)
    
    # discretization for all simulations
    nx = 30 # number of elements in x
    ny = 2 # number of elements in y
    total_length_x = 20.0 # length of structure in x (m)
    total_length_y = 2.0 # length of structure in y (m)
    scale_factor = 0.4
    
    # material properties for all simulations
    E1 = 3.0        # Young's Modulus (Pa)
    A1 = 2.0      # Area (m^2)
    rho = 1.0     # density (kg/m^3)
    
    #steel
    # E1 = 200.0       # Young's Modulus (Pa)
    # A1 = 0.2         # Area (m^2)
    # rho = 7850.0     # density (kg/m^3)
    
    # timestep parameters for all simulations
    T_total = 600 # total simulation time (s)
    
    main()
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # stats.dump_stats('profiling_results.prof')