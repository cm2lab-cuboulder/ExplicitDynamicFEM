import numpy as np

# local stiffness function for 1d
def bar_element_1d(E, A, ends):
    x1, y1 = ends[0, 0], ends[0, 1]
    x2, y2 = ends[1, 0], ends[1, 1]
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx ** 2 + dy ** 2) # length of element

    k_e = (E * A / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    #print("1d elemental stiffness:", k_e)
    return k_e

# local stiffness function for 2d
def bar_element_2d(E1, A1, ends):
    
    k_e = np.zeros((4, 4))
    
    x1, y1 = ends[0, 0], ends[0, 1]
    x2, y2 = ends[1, 0], ends[1, 1]
    dx = x2 - x1
    dy = y2 - y1
    
    L = np.sqrt(dx ** 2 + dy ** 2)  # length of the element
    
    c = dx / L  # cos theta
    s = dy / L  # sin theta
    
    # compute k_e components
    cc = c**2
    ss = s**2
    cs = c*s

    k_e = (E1*A1/L)*np.array([
        [cc, cs, -cc, -cs],
        [cs, ss, -cs, -ss],
        [-cc, -cs, cc, cs],
        [-cs, -ss, cs, ss]])
    
    #print(f"Element length (L): {L}, Material properties (E1, A1): {E1}, {A1}")
    #print(f"2d Element stiffness matrix (k_e):\n{k_e}")
    return k_e

# lumped mass matrix function
def lumped_mass_matrix(rho, As, coord, nele, ndof, connec, dof_id):
    """
Parameters:
- rho: Density of the material (kg/m^3)
- As: Array of cross-sectional areas for each element (m^2)
- coord: Coordinates of the nodes.
- nele: Number of elements.
- ndof: Total number of degrees of freedom in the structure.
- connec: Connectivity matrix.
- dof_id: Degrees of freedom associated with each element.

    """
    M = np.zeros(ndof)
    for i in range(nele):
        ix = connec[i] # element node indices
        
        ends = coord[ix, :] # element node coordinates
        
        L = np.linalg.norm(ends[1] - ends[0]) # element length
        
        m_e = rho * As[i] * L # element mass
        
        dofs = dof_id[i]  # element DOFs
        
        num_dofs = len(dofs)
        
        m_local = (m_e / num_dofs) * np.ones(num_dofs) # equal mass distribution
        
        M[dofs] += m_local # global lumped mass matrix assembly
        
         #print('m_e =', m_e)
         #M[M==0] = 1e-6
        
    return M

# 1d discretization of finite elements
def geometry_1d(nele, total_length):
    nnode = nele + 1  # number of nodes
    ndof = 1 * nnode  # number of degrees of freedom

    # discretize total length by number of elements
    x_coords = np.linspace(0, total_length, nnode)
    y_coords = np.zeros(nnode) # 1D for now, rework eventually to extend to 2D
    coord = np.column_stack((x_coords, y_coords))

    # connectivity based on number of elements
    connec = np.array([[i, i + 1] for i in range(nele)])
    dof_id = connec.copy()  # DOF IDs match node indices for axial elements

    return nnode, ndof, coord, connec, dof_id

# # 2d discretization of finite elements
# def geometry_2d(nx, ny, total_length_x, total_length_y):
#     nnode = (nx + 1) * (ny + 1)  # total number of nodes
#     ndof = 2 * nnode  # each node has 2 DOFs (x and y)
    
#     # axial coordinates
#     x_coords = np.linspace(0, total_length_x, nx + 1)
#     y_coords = np.linspace(0, total_length_y, ny + 1)
#     X, Y = np.meshgrid(x_coords, y_coords)
    
#     coord = np.column_stack([X.ravel(), Y.ravel()])
    
#     # connectivity matrix for quads
#     connec = []
#     for j in range(ny + 1):
#         for i in range(nx + 1):
#             n = i + j * (nx + 1)
#             # connect to right neighbor
#             if i < nx:
#                 n_right = n + 1
#                 connec.append([n, n_right])
#             # connect to top neighbor
#             if j < ny:
#                 n_top = n + (nx + 1)
#                 connec.append([n, n_top])
#             # connect to top-right neighbor (diagonal)
#             if i < nx and j < ny:
#                 n_top_right = n + (nx + 1) + 1
#                 connec.append([n, n_top_right])
#             # connect to top-left neighbor (diagonal)
#             if i > 0 and j < ny:
#                 n_top_left = n + (nx + 1) - 1
#                 connec.append([n, n_top_left])

#     # DOF ID: for each node, its x and y DOF ids
#     dof_id = []
#     for nodes in connec:
#         element_dof_ids = []
#         for node in nodes:
#             element_dof_ids.append(2 * node)      # x DOF
#             element_dof_ids.append(2 * node + 1)  # y DOF
#         dof_id.append(element_dof_ids)
    
#     return nnode, ndof, coord, connec, dof_id

def geometry_2d(nx, ny, total_length_x, total_length_y):
    x_coords = np.linspace(0, total_length_x, nx + 1)
    y_coords = np.linspace(0, total_length_y, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords)
    coord = np.column_stack([X.ravel(), Y.ravel()])
    nnode_original = (nx + 1) * (ny + 1)
    
    # compute center node
    center_coords = []
    center_node_indices = np.arange(nnode_original, nnode_original + nx * ny).reshape((ny, nx))
    for j in range(ny):
        for i in range(nx):
            x_center = (x_coords[i] + x_coords[i+1]) / 2
            y_center = (y_coords[j] + y_coords[j+1]) / 2
            center_coords.append([x_center, y_center])
    coord = np.vstack([coord, center_coords])  # add to coord
    
    nnode = nnode_original + nx * ny  # total nodes
    ndof = 2 * nnode
    
    # connectivity algorithm
    connec = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = i + j * (nx + 1)
            # connect to right neighbor
            if i < nx:
                n_right = n + 1
                connec.append([n, n_right])
            # connect to top neighbor
            if j < ny:
                n_top = n + (nx + 1)
                connec.append([n, n_top])
            # connect to diagonals via center node
            if i < nx and j < ny:
                center_node = center_node_indices[j, i]
                n_top_right = n + (nx + 1) + 1
                n_top_left = n + (nx + 1) - 1
                # diagonal from n to n_top_right via center node
                connec.append([n, center_node])
                connec.append([center_node, n_top_right])
                # diagonal from n_right to n_top via center node
                n_right = n + 1
                n_top = n + (nx + 1)
                connec.append([n_right, center_node])
                connec.append([center_node, n_top])
    
    # assign DOFs
    dof_id = []
    for nodes in connec:
        element_dof_ids = []
        for node in nodes:
            element_dof_ids.append(2 * node)      # x DOF
            element_dof_ids.append(2 * node + 1)  # y DOF
        dof_id.append(element_dof_ids)
    
    return nnode, ndof, coord, connec, dof_id

