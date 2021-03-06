# Heat equation FEM and PINN solver

## What this is

This repository solves the heat equation with given initial and boundary conditions with the finite element method (FEM) and physics-informed neural networks (PINN).
Two demos / test problems are implemented. 
For further documentation, see [this](https://github.com/daniellarin22/heat_equation_solver/blob/4d464b1e88a76f3ff35354d4d3bff32cb0a44b18/20220424_BA_Larin_final.pdf) bachelor thesis.

## How it works

This program requires the libraries [FEniCS](https://fenicsproject.org/pub/tutorial/pdf/fenics-tutorial-vol1.pdf) for the FEM solver and [DeepXDE](https://deepxde.readthedocs.io/en/latest/index.html) for the PINN solver as well as NumPy and Matplotlib for scientific computing and plotting. 

After installing the required libraries follow these steps:

1. Clone this repository
2. Adjust configurations for 1D test problem ([here](https://github.com/daniellarin22/heat_equation_solver/blob/989813fa11114325915170ba6771fc94fb5e76f8/main/test_problem_1d.py)) or 2D test problem ([here](https://github.com/daniellarin22/heat_equation_solver/blob/main/main/test_problem_2d.py)), such as mesh resolution, number of time steps, network architecture or number of epochs and usage of hard constraints (1D only)
3. Run either [main/test_problem_1d.py](https://github.com/daniellarin22/heat_equation_solver/blob/989813fa11114325915170ba6771fc94fb5e76f8/main/test_problem_1d.py) or [main/test_problem_2d.py](https://github.com/daniellarin22/heat_equation_solver/blob/main/main/test_problem_2d.py) 
4. Check main/assets/ folder for plots of cost function, error, exact and numerical solutions
