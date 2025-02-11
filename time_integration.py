import numpy as np
#timestep function
def time_step(Es, As, coord, nele, connec, rho):
    dt_crit = np.inf
    for i in range(nele):
        ix = connec[i]
        ends = coord[ix, :]
        L = np.linalg.norm(ends[1] - ends[0])
        E = Es[i]
        c = np.sqrt(E / rho)  # wave speed
        dt_e = L / c
        if dt_e < dt_crit:
            dt_crit = dt_e
    print(f"element length:",L)
    return dt_crit