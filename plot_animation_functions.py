import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# functions for plotting bar displacement and velocity
def plot_displacement_1d(time, U_history, nnode, num_steps):
    end_node_index = nnode - 1
    end_node_displacement = U_history[end_node_index, :]
    plt.figure()
    plt.plot(time[:num_steps], end_node_displacement, label='end node')
    plt.xlabel('time (s)')
    plt.ylabel('displacement (m)')
    plt.title('bar displacement')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.grid(True)
    plt.show()

def plot_velocity_1d(time, V_half_history, nnode, num_steps):
    end_node_index = nnode - 1 
    end_node_velocity = V_half_history[end_node_index, :]
    plt.figure()
    plt.plot(time[:num_steps], end_node_velocity, label='end node')
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')
    plt.title('bar velocity')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.grid(True)
    plt.show()

## animated plot with scaled node
# def animate_displacement_1d(coord, connec, U_history, scale_factor, time_array, interval):

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.set_aspect('equal')
#     ax.set_xlim(np.min(coord) - 1, np.max(coord) + 1)
#     ax.set_ylim(- 1, 1)
#     max_disp = np.max(U_history) * scale_factor
#     min_disp = np.min(U_history) * scale_factor
#     ax.set_xlabel('x-axis')
#     ax.set_title('bar displacement')

#     # prep lines for LineCollections
#     lines = []
#     for i in range(len(connec)):
#         element_nodes = connec[i]
#         x = coord[element_nodes]
#         y = np.zeros_like(x)
#         line_coords = np.column_stack([x, y])
#         lines.append(line_coords)

#     # add elements to LineCollection
#     line_collection = LineCollection(lines, cmap='jet', linewidth=2)
#     ax.add_collection(line_collection)
#     line_collection.set_array(np.zeros(len(lines)))

#     # add time counter
#     time_text = ax.text(0.05, 1.5, '', transform=ax.transAxes)

#     # animation update
#     def update(frame):
#         U_disp = U_history[:, frame]
#         line_coords_updated = []
#         colors = []
          # average element displacement
#         for i, element_nodes in enumerate(connec):
#             element_nodes = np.array(element_nodes)
#             u_x_disp = U_disp[element_nodes] * scale_factor
#             x = coord[element_nodes] + U_disp[element_nodes]
#             y = np.zeros_like(x)
#             line_coords = np.column_stack([x, y])
#             line_coords_updated.append(line_coords)
#             avg_disp = np.mean(u_x_disp)
#             colors.append(avg_disp)

#         # update element colors and position
#         line_collection.set_segments(line_coords_updated)
#         line_collection.set_array(np.array(colors))
#         line_collection.set_clim(vmin=min_disp, vmax=max_disp)
        
#         # set time to CDM time integration
#         time_text.set_text(f'Time = {time_array[frame]:.2f} s')
        
#         return [line_collection, time_text]

#     # colorbar
#     cbar = fig.colorbar(line_collection, ax=ax, orientation='horizontal')
#     cbar.set_label('x-displacement (m)')
#     tick_min = np.floor(min_disp / 5) * 5
#     tick_max = np.ceil(max_disp / 5) * 5
#     total_range = tick_max - tick_min
#     interval = total_range / 10
#     ticks = np.arange(tick_min, tick_max + interval, interval)
#     cbar.set_ticks(ticks)

#     fig.subplots_adjust(bottom=0.2)

#     ani = FuncAnimation(fig, update, frames=range(U_history.shape[1]), blit=False, interval=interval)
#     ani.save('bar_displacement_animation.mp4', writer='ffmpeg', fps=1000/interval)
#     plt.show()

def animate_displacement_1d(coord, connec, U_history, scale_factor, time_array, interval=0.1):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_xlim(np.min(coord) - 1, np.max(coord) + 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Bar Displacement')

    # match color scaling to displacement
    max_disp = np.max(U_history) * scale_factor
    min_disp = np.min(U_history) * scale_factor

    # prep lines for LineCollection
    lines = []
    for element_nodes in connec:
        x = coord[element_nodes]
        y = np.zeros_like(x)
        line_coords = np.column_stack([x, y])
        lines.append(line_coords)

    # add elements to Linecollection
    line_collection = LineCollection(lines, cmap='jet', linewidth=2)
    ax.add_collection(line_collection)

    # initialize co lors
    initial_colors = np.zeros(len(lines))
    line_collection.set_array(initial_colors)
    line_collection.set_clim(vmin=min_disp, vmax=max_disp)

    # color bar config
    cbar = fig.colorbar(line_collection, ax=ax, orientation='horizontal')
    cbar.set_label('X-Displacement (m)')
    tick_min = np.floor(min_disp / 5) * 5
    tick_max = np.ceil(max_disp / 5) * 5
    total_range = tick_max - tick_min
    tick_interval = total_range / 10 if total_range != 0 else 1
    ticks = np.arange(tick_min, tick_max + tick_interval, tick_interval)
    cbar.set_ticks(ticks)
    
    # add time counter
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # animation update
    def update(frame):
        U_disp = U_history[:, frame] * scale_factor
        colors = []
        # average element displacment
        for i, element_nodes in enumerate(connec):
            u_x_disp = U_disp[element_nodes]
            avg_disp = np.mean(u_x_disp)
            colors.append(avg_disp)

        # update element color
        line_collection.set_array(np.array(colors))
        # set time to CDM time integration
        time_text.set_text(f'Time = {time_array[frame]:.2f} s')

        return [line_collection, time_text]

    fig.subplots_adjust(bottom=0.2)

    ani = FuncAnimation(fig, update, frames=range(U_history.shape[1]), blit=False, interval=interval)
    ani.save('bar_displacement_animation.mp4', writer='ffmpeg', fps=10/interval)
    plt.show()

# plot mesh geometry
def plot_mesh(coord, connec):
    plt.figure(figsize=(8, 8))
    # plot elements to verify connectivity
    for element in connec:
        node_indices = element
        x_coords = coord[node_indices, 0]
        y_coords = coord[node_indices, 1]
        plt.plot(x_coords, y_coords, 'b-')
    
    plt.plot(coord[:, 0], coord[:, 1], 'ro')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# functions for plotting lattice dispalcement and velocity
def plot_displacement_2d(time, U_history, right_side_nodes):
    plt.figure()
    for node in right_side_nodes:
        plt.plot(time, U_history[2 * node, :], label=f'Node {node}')
    plt.xlabel('time (s)')
    plt.ylabel('displacement (m)')
    plt.title('lattice displacement')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.grid(True)
    plt.show()
    
def plot_velocity_2d(time, V_half_history, right_side_nodes):
    plt.figure()
    for node in right_side_nodes:
        plt.plot(time, V_half_history[2 * node, :], label=f'Node {node}')
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')
    plt.title('lattice velocity')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.grid(True)
    plt.show()
    
# animate displacement for lattice
# def animate_displacement_2d(coord, connec, U_history, scale_factor, time_array, interval=1):

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.set_aspect('equal')
#     ax.set_xlim(np.min(coord[:, 0]) - 1, np.max(coord[:, 0]) + 1)
#     ax.set_ylim(np.min(coord[:, 1]) - 1, np.max(coord[:, 1]) + 1)
#     max_disp = np.max(U_history) * scale_factor
#     min_disp = np.min(U_history) * scale_factor
#     ax.set_xlabel('x-axis')
#     ax.set_title('lattice displacement')

#     # prep lines for LineCollections
#     lines = []
#     for i in range(len(connec)):
#         element_nodes = connec[i]
#         element_nodes = np.array(element_nodes)
#         x = coord[element_nodes, 0]
#         y = coord[element_nodes, 1]
#         line_coords = np.column_stack([x, y])
#         lines.append(line_coords)

#     # add elements to LineCollection
#     line_collection = LineCollection(lines, cmap='jet', linewidth=2)
#     ax.add_collection(line_collection)
#     # initialize color array
#     line_collection.set_array(np.zeros(len(lines)))

#     # add time counter
#     time_text = ax.text(0.05, 1.5, '', transform=ax.transAxes)

#     # animation update
#     def update(frame):
#         U_disp = U_history[:, frame]
#         line_coords_updated = []
#         colors = []
#         for i, element_nodes in enumerate(connec):
#             element_nodes = np.array(element_nodes)
#             u_x_disp = U_disp[2 * element_nodes] * scale_factor
#             u_y_disp = U_disp[2 * element_nodes + 1] * scale_factor
#             x = coord[element_nodes, 0] + u_x_disp
#             y = coord[element_nodes, 1] + u_y_disp
#             line_coords = np.column_stack([x, y])
#             line_coords_updated.append(line_coords)

#             avg_disp = np.mean(u_x_disp)
#             colors.append(avg_disp)

          # update element colors and position
#         line_collection.set_segments(line_coords_updated)
#         line_collection.set_array(np.array(colors))
#         line_collection.set_clim(vmin=min_disp, vmax=max_disp)
        
#         # set time to CDM time integration
#         time_text.set_text(f'Time = {time_array[frame]:.2f} s')

#         return [line_collection, time_text]

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="15%", pad=0.5)

#     # colorbar
#     cbar = fig.colorbar(line_collection, cax=cax, orientation='horizontal')
#     cbar.set_label('displacement (m)')
#     tick_min = np.floor(min_disp / 5) * 5
#     tick_max = np.ceil(max_disp / 5) * 5
#     total_range = tick_max - tick_min
#     interval = total_range / 10
#     ticks = np.arange(tick_min, tick_max + interval, interval)
#     cbar.set_ticks(ticks)
    
#     ani = FuncAnimation(fig, update, frames=range(U_history.shape[1]), blit=False, interval=interval)
#     ani.save('lattice_displacement_animation.mp4', writer='ffmpeg', fps=1000/interval)
#     plt.show()

def animate_displacement_2d(coord, connec, U_history, scale_factor, time_array, interval=0.1):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_xlim(np.min(coord[:, 0]) - 1, np.max(coord[:, 0]) + 1)
    ax.set_ylim(np.min(coord[:, 1]) - 1, np.max(coord[:, 1]) + 1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Lattice Displacement')

    # match color scaling to displacement
    U_magnitude = np.sqrt(U_history[::2, :]**2 + U_history[1::2, :]**2)
    max_disp = np.max(U_magnitude) * scale_factor
    min_disp = 0  # displacement magnitude is always non-negative

    # prep lines for LineCollections
    lines = []
    for element_nodes in connec:
        element_nodes = np.array(element_nodes)
        x = coord[element_nodes, 0]
        y = coord[element_nodes, 1]
        line_coords = np.column_stack([x, y])
        lines.append(line_coords)

    # add elements to LineCollection
    line_collection = LineCollection(lines, cmap='jet', linewidth=2)
    ax.add_collection(line_collection)

    # initialize co lors
    initial_colors = np.zeros(len(lines))
    line_collection.set_array(initial_colors)
    line_collection.set_clim(vmin=min_disp, vmax=max_disp)
    
    # color bar config
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = fig.colorbar(line_collection, cax=cax, orientation='horizontal')
    cbar.set_label('Displacement Magnitude (scaled)')
    
    # add time counter
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # animation update
    def update(frame):
        U_disp = U_history[:, frame] * scale_factor
        colors = []
        # average element displacement
        for i, element_nodes in enumerate(connec):
            element_nodes = np.array(element_nodes)
            u_x_disp = U_disp[2 * element_nodes]
            u_y_disp = U_disp[2 * element_nodes + 1]
            disp_magnitude = np.sqrt(u_x_disp**2 + u_y_disp**2)
            avg_disp = np.mean(disp_magnitude)
            colors.append(avg_disp)

        # update element colors
        line_collection.set_array(np.array(colors))
        # set time to CDM time integration
        time_text.set_text(f'Time = {time_array[frame]:.2f} s')

        return line_collection, time_text

    ani = FuncAnimation(fig, update, frames=range(U_history.shape[1]), blit=False, interval=interval)
    ani.save('lattice_displacement_animation.mp4', writer='ffmpeg', fps=10/interval)
    plt.show()