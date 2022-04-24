from typing import Dict, List, Union
import deepxde as dde
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from services.constants import *


def plot_1d_graph(
    solution_dict: Dict[str, np.array],
    function_space: df.FunctionSpace,
    time_mesh: np.array,
    levels: List[int],
    folder_path: str
) -> None:
    """
    Plots exact and numerical solutions of 1D heat equation versus x-coordinate as lines for different times t.

    :param solution_dict:       Dict of exact and numerical solutions as numpy arrays.
    :param function_space:      Finite element function space from FEM solution.
    :param time_mesh:           Different time steps from FEM solution as numpy array.
    :param levels:              List of time indexes to plot the solutions.
    :param folder_path:         Location for saving the figures to.
    :return:                    None
    """
    # Define labels and linestyles for the solutions
    name_dict = {EXACT: "Exakt", FEM: "FEM", PINN: "PINN"}
    linestyle_dict = {EXACT: "solid", FEM: "dashed", PINN: "dashdot"}

    # Compute min and max values of the solutions for setting the y-limits
    solution_min = np.min(list(solution_dict.values()))
    solution_max = np.max(list(solution_dict.values()))

    # Time-stepping
    for step in levels:

        # Iterate over the different solutions
        plt.clf()
        for name, solution in solution_dict.items():
            solution_function = df.Function(function_space)
            solution_function.vector().set_local(solution[step, :])

            # Plot solution
            df.plot(
                solution_function,
                title=f"$t = {time_mesh[step]:.2f}$",
                label=name_dict[name],
                linestyle=linestyle_dict[name],
            )

        # Finalize and save the plot
        plt.ylim((solution_min - 0.2, solution_max + 0.2))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$u,\, u_{h,\tau},\, u_{\theta}$")
        plt.legend()
        plt.savefig(f"{folder_path}/solution_{time_mesh[step]:.2f}.png", dpi=300)


def plot_2d_heatmap(
    solution_dict: Dict[str, np.array],
    function_space: df.FunctionSpace,
    time_mesh: np.array,
    levels: List[int],
    folder_path: str
) -> None:
    """
    Plots exact and numerical solutions of 2D heat equation as a 2D heatmap for different times t.

    :param solution_dict:       Dict of exact and numerical solutions as numpy arrays.
    :param function_space:      Finite element function space from FEM solution.
    :param time_mesh:           Different time steps from FEM solution as numpy array.
    :param levels:              List of time indexes to plot the solutions.
    :param folder_path:         Location for saving the figures to.
    :return:                    None
    """
    # Define labels for the solutions
    name_dict = {EXACT: "$u$", FEM: r"$u_{h,\tau}$", PINN: r"$u_{\theta}$"}

    # Compute min and max values of the solutions for setting the color bar limits
    solution_min = np.min(list(solution_dict.values()))
    solution_max = np.max(list(solution_dict.values()))

    # Time-stepping and iterating over the different solutions
    for name, solution in solution_dict.items():
        for step in levels:
            solution_function = df.Function(function_space)
            solution_function.vector().set_local(solution[step, :])

            # Plot solution
            plt.clf()
            heatmap = df.plot(
                solution_function,
                title=f"$t = {time_mesh[step]:.2f}$",
                cmap="turbo",
                vmin=solution_min,
                vmax=solution_max
            )

            # Finalize and save the plot
            plt.colorbar(heatmap, label=name_dict[name])
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.savefig(f"{folder_path}/heatmap_{name}_{time_mesh[step]:.2f}.png", dpi=300)


def plot_error_measures(
    exact_solution: Union[np.array, None],
    solution_dict: Dict[str, np.array],
    time_mesh: np.array,
    error_measure: Union[int, str, None],
    folder_path: str
) -> None:
    """
    Plots error in space between the exact and numerical solutions versus t-coordinate as lines
    for different p ∈ [1, inf].

    :param exact_solution:      Exact solutions as numpy array if available.
    :param solution_dict:       Dict of numerical solutions as numpy arrays.
    :param time_mesh:           Different time steps from FEM solution as numpy array.
    :param error_measure:       List of p to use (usually p = 2 or p = inf).
    :param folder_path:         Location for saving the figures to.
    :return:                    None
    """
    # Map error measures to Latex code for labels of the y-axis
    error_measure_in_latex = {np.inf: "{\infty}", 2: 2}

    plt.clf()
    if exact_solution is None:
        # Plot error between FEM and PINN solution for approximation of the PINN error
        # Condition: FEM error is significantly smaller than PINN error
        plt.plot(
            time_mesh[1:],
            np.linalg.norm(solution_dict[FEM] - solution_dict[PINN], error_measure, axis=1)[1:]
        )
    else:
        # Plot error of the FEM and PINN solution
        plt.plot(
            time_mesh[1:],
            np.linalg.norm(exact_solution - solution_dict[FEM], error_measure, axis=1)[1:],
            label="$E_{\mathrm{FEM}}$"
        )
        plt.plot(
            time_mesh[1:],
            np.linalg.norm(exact_solution - solution_dict[PINN], error_measure, axis=1)[1:],
            label="$E_{\mathrm{PINN}}$"
        )
        plt.legend()

    # Finalize and save the plot
    plt.xlabel(r"$t$")
    plt.ylabel(f"Fehler für $p = {error_measure_in_latex[error_measure]}$")
    plt.yscale('log', base=10)
    plt.savefig(f"{folder_path}/l_{error_measure}_error.png", dpi=300)


def plot_losses(loss: dde.model.LossHistory, loss_names: List[str], folder_path: str) -> None:
    """
    Plots the individual losses from the PINN solution versus the epochs as lines.

    :param loss:            Loss history object containing the individual losses.
    :param loss_names:      List of labels for the individual losses to be shown in the legend.
    :param folder_path:     Location for saving the figures to.
    :return:                None
    """
    loss_array = np.array(loss.loss_train)

    # Plot loss components
    plt.clf()
    for loss_type in range(loss_array.shape[1]):
        plt.plot(loss.steps, loss_array[:, loss_type], label=loss_names[loss_type])

    # Finalize and save the plot
    plt.xlabel("Anzahl der Iterationen")
    plt.ylabel("Kostenfunktion")
    plt.yscale('log', base=10)
    plt.legend()
    plt.savefig(f"{folder_path}/loss.png", dpi=300)
