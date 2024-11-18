This repository contains files that model an explicit dynamic FEM in both a 1D and 2D implementation.
Both the 1D and 2D implementations are initialized with a fixed boundary condition on the left side, and an applied
force on the right side, introducing axial compression into the system. 

The 1D code is just a series of bar elements. The 2D code is discretized as a quad lattice with
cross elements and a central node. Neither of these models are setup to handle shear deformation 
or the analysis of shear force.

FEM functions, Force functions, time integration functions and plotting functions are all
modularized.

explicitFEMbar.py runs the 1D implementation.
explicitFEMlattice.py runs the 2D implementation.
explicitFEM_combined.py runs a comparison of the two models, including plots.
