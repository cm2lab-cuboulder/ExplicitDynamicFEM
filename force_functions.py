import numpy as np
from multiprocessing import Pool, Array
import ctypes
from fem_functions import bar_element_1d, bar_element_2d

def internal_force_1d(U, Es, As, coord, ndof, nx, connec, dof_id):
    f_int = np.zeros(ndof)
    for i in range(nx):
        ix = connec[i, :]
        dofs = dof_id[i, :]
        u_e = U[dofs]
        k_e = bar_element_1d(Es[i], As[i], coord[ix, :])
        f_int_e = np.dot(k_e, u_e)
        f_int[dofs] += f_int_e
        
        # print statements for debugging
        #print(f"element displacement: {u_e =}\n")
        #print(f"element stiffness: {k_e =}\n")
        #print(f"element internal forces: {f_int_e =}\n")
    return f_int

# externally applied force function for 1d discretization
def external_force_1d(ndof, force_value):
    F_ext = np.zeros(ndof)
    F_ext[-1] = force_value
    return F_ext

def internal_force_2d(U, Es, As, coord, ndof, nele, connec, dof_id):
    f_int = np.zeros(ndof)
    for i in range(nele):
        ix = connec[i][:2] # get the two nodes forming the bar element
        dofs = dof_id[i][:4] # get the DOFS for the two nodes (x and y for both)
        u_e = U[dofs] # get the displacements at the element's DOFs
        k_e = bar_element_2d(Es[i], As[i], coord[ix, :]) # compute local stiffness matrix
        f_int_e = np.dot(k_e, u_e)
        f_int[dofs] += f_int_e
        
        #print(f"internal force contribution from element {i}: {np.dot(k_e, u_e)}")        
    return f_int

def external_force_2d(ndof, right_side_nodes, force_value):
    F_ext = np.zeros(ndof)
    
    # apply the same force tal all nodes on the right side
    for node in right_side_nodes:
        F_ext[2 * node] = force_value # apply force at x dof indices
    return F_ext