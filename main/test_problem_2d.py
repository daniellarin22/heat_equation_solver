# Import libraries
import matplotlib
matplotlib.use("agg")   # disables gui output (required for WSL)

import os
import time
import deepxde as dde
import numpy as np

from fem.solver import FEMSolver
from pinn.solver import PINNSolver
from services.plot import plot_2d_heatmap, plot_error_measures, plot_losses
from services.tuning import build_hyper_parameter_space
from services.constants import *


# Set default float type for PINN solver
DTYPE = "float64"
dde.config.set_default_float(DTYPE)

# Final time
T = 0.1

####################################################### FEM solver #####################################################

# Start timer
start_fem = time.time()

# Initialize FEM solver class instance
solver_fem = FEMSolver(
    test_problem_str=HEAT_EQUATION_2D,
    T=T
)
solver_fem.setup_test_problem()
solver_fem.set_boundary_types()

# Create mesh and define function space
mesh = solver_fem.set_mesh(
    {NX: 25, NY: 25}
)   # nx = ny = 25 equals h = 0.05656854249492385 with mesh.hmax() -> see Table 3
function_space = solver_fem.set_function_space(
    mesh, {BASIS_FUNCTIONS: "Lagrange", DEGREE: 2}
)

# Define boundary conditions
bcs, measures = solver_fem.set_boundary_conditions_and_measures(mesh)

# Compute solution
solution_fem, time_values_data = solver_fem.solve(bcs, function_space, measures, {NUM_TIME_STEPS: 100, ODE_SOLVER: 1})

# Determine computational time
end_fem = time.time()
timer_fem = end_fem - start_fem

################################################# Hyper parameter tuning ###############################################

# Define run id
run_id = "run_2d_test"

# Number of iterations to train the model
epochs = 1000

# Create hyper parameter product space for tuning
hyper_parameter_product_space = build_hyper_parameter_space(
    maximal_trials=25,
    hyper_parameter_lists=[
        [18],       # 0: NUM_NEURONS
        [6],        # 1: NUM_HIDDEN_LAYER
        [1e-2],     # 2: LEARNING_RATE
        [1000],     # 3: NUM_DOMAIN
        [1000],     # 4: NUM_INITIAL
        [1000],     # 5: NUM_BOUNDARY
    ]
)

##################################################### PINN solver ######################################################

# Initialize timer array
timer_pinn = []

# Iterate over the hyper parameter product space
for hyper_parameter in hyper_parameter_product_space:

    # Define name of folder for saving the figures to for this trial
    name = f"{run_id}_nn{hyper_parameter[0]}_nhl{hyper_parameter[1]}_lr{hyper_parameter[2]}_ndom{hyper_parameter[3]}_" \
           f"nic{hyper_parameter[4]}_nbc{hyper_parameter[5]}_epochs{epochs}"

    folder_path = f"assets/{name}"
    while os.path.isdir(folder_path):
        folder_path += "_new"
    os.mkdir(folder_path)

    print(f"Start trial {hyper_parameter_product_space.index(hyper_parameter) + 1} with hyper_parameter {name}\n")

    # Start timer
    start_pinn = time.time()

    # Initialize PINN solver class instance
    solver_pinn = PINNSolver(
        test_problem_str=HEAT_EQUATION_2D,
        T=T
    )
    solver_pinn.setup_test_problem()
    solver_pinn.set_boundary_types() 
    solver_pinn.set_domain()
    solver_pinn.set_boundary_conditions()

    # Define the PDE problem and configure the network
    pde_solver = solver_pinn.init_time_dependent_pde_solver(
        {NUM_DOMAIN: hyper_parameter[3], NUM_INITIAL: hyper_parameter[4], NUM_BOUNDARY: hyper_parameter[5]}
    )
    neural_net = solver_pinn.init_neural_network(
        {
            NUM_NEURONS: hyper_parameter[0], NUM_HIDDEN_LAYER: hyper_parameter[1],
            ACTIVATION: "tanh", INITIALIZER: "Glorot uniform"
        }
    )

    # Build and train the model
    model, loss, train = solver_pinn.train_model(
        pde_solver,
        neural_net,
        {OPTIMIZER: "adam", LEARNING_RATE: hyper_parameter[2], EPOCHS: epochs, LOSS_WEIGHTS: [1, 1, 1, 1]}
    )

    # Determine computational time
    end_pinn = time.time()
    timer_pinn.append(end_pinn - start_pinn)

    ###################################################### Plotting ####################################################

    # Initialize solution arrays
    exact_solution = np.zeros_like(solution_fem)
    solution_pinn = np.zeros_like(solution_fem)

    # Compute exact and PINN solution on mesh for time steps from FEM solution
    for time_step in range(solution_fem.shape[0]):
        coordinates_data = np.concatenate(
            [
                function_space.tabulate_dof_coordinates(),
                (time_values_data[time_step] * np.ones(function_space.dim())).reshape(-1, 1)
            ],
            axis=1
        )

        exact_solution[time_step, :] = np.array(solver_pinn.solution(coordinates_data)).reshape(-1,)
        solution_pinn[time_step, :] = model.predict(coordinates_data).reshape(-1,)

    # Plot the individual losses from the PINN solution
    plot_losses(
        loss,
        [
            r"$\mathcal{L}_{\mathcal{H}, \mathcal{T}}$",
            r"$\mathcal{L}_{\mathcal{I}, \mathcal{T}}$",
            r"$\mathcal{L}_{\mathcal{B}, D, \mathcal{T}}$",
            r"$\mathcal{L}_{\mathcal{B}, N, \mathcal{T}}$"
        ],
        folder_path
    )

    # Plot exact and numerical solutions as a 2D heatmap
    plot_2d_heatmap(
        {
            EXACT: exact_solution,
            FEM: solution_fem,
            PINN: solution_pinn
        },
        function_space,
        time_values_data,
        [0, 10, 20, 30, 50, 100],
        folder_path
    )

    # Plot error in space between the exact and numerical solutions for p ∈ {2, inf}.
    for error_measure in [2, np.inf]:
        plot_error_measures(
            exact_solution,
            {
                FEM: solution_fem,
                PINN: solution_pinn
            },
            time_values_data,
            error_measure,
            folder_path
        )
