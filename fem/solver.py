from typing import Dict, List, Union, Tuple
import dolfin as df
import numpy as np

from services.constants import *
from fem.expressions import EXPRESSIONS_TYPE_DICT
from fem.mesh import DOMAIN_TYPE_DICT
from fem.boundary import Boundary


class FEMSolver:
    """
    Class for solving the heat equation with the finite element method (FEM).
    """

    def __init__(self, test_problem_str: str, T: float) -> None:
        """
        Constructor of this class.
        Will be called when an object is created from this class and
        allows the class to initialize the attributes of the class.

        :param test_problem_str:    Name of the test problem.
        :param T:                   Final time.
        """
        self.test_problem_str = test_problem_str
        self.T = T
        self.domain_type = TEST_PROBLEM_TO_DOMAIN_DICT[self.test_problem_str]

        self.f = None
        self.initial = None
        self.dirichlet = None
        self.neumann = None
        self.robin_r = None
        self.robin_s = None
        self.solution = None

        self.boundaries_assignment_dict = None
        self.num_dirichlet_sectors = None
        self.num_neumann_sectors = None
        self.num_robin_sectors = None

        self.boundary_types = None

    def setup_test_problem(self) -> None:
        """
        Initializing the inhomogeneity of the heat equation, the initial condition,
        the inhomogeneities of the boundary conditions and the solution.

        :return:    None
        """
        self.f, self.initial, self.dirichlet, self.neumann, self.robin_r, self.robin_s, self.solution = \
            EXPRESSIONS_TYPE_DICT[self.domain_type][self.test_problem_str]() 

        # Assign the different parts of the boundary to the different boundary types
        self.boundaries_assignment_dict = {
            DIRICHLET: list(self.dirichlet.keys()),
            NEUMANN: list(self.neumann.keys()),
            ROBIN: list(self.robin_r.keys())
        }
        self.num_dirichlet_sectors = 0 if self.dirichlet is None else len(self.boundaries_assignment_dict[DIRICHLET])
        self.num_neumann_sectors = 0 if self.neumann is None else len(self.boundaries_assignment_dict[NEUMANN])
        self.num_robin_sectors = 0 if self.robin_r is None else len(self.boundaries_assignment_dict[ROBIN])

    def set_boundary_types(self) -> None:
        """
        Defines python functions for the different parts of the boundary.
        The functions should return booleans depending whether a point is on the boundary or not.

        :return:    None
        """
        self.boundary_types = {
            (key, sector): Boundary(self.domain_type, sector).boundary
            for (key, sector) in (
                self.boundaries_assignment_dict[DIRICHLET] +
                self.boundaries_assignment_dict[NEUMANN] +
                self.boundaries_assignment_dict[ROBIN]
            )
        }

    def set_mesh(self, mesh_resolution_args: Dict[str, int]) -> df.mesh:
        """
        Defines the computational mesh.

        :param mesh_resolution_args:    Dict with keys NX (and NY in 2D) with number of cells as value(s).
        :return:                        The mesh.
        """
        return DOMAIN_TYPE_DICT[self.domain_type](mesh_resolution_args)

    @staticmethod
    def set_function_space(mesh: df.Mesh, function_space_args: Dict[str, Union[int, str]]) -> df.FunctionSpace:
        """
        Defines the finite element function space.

        :param mesh:                    The mesh.
        :param function_space_args:     Dict with keys BASIS_FUNCTIONS and DEGREE with
                                        specification of the element family (e. g. Lagrange) and
                                        the degree of the element as values.
        :return:                        Finite element function space.
        """
        return df.FunctionSpace(
            mesh,
            function_space_args[BASIS_FUNCTIONS],
            function_space_args[DEGREE]
        )

    def set_boundary_conditions_and_measures(
        self,
        mesh: df.Mesh
    ) -> Tuple[Dict[Tuple[int, str], Boundary], List[df.Measure]]:
        """
        Defining markers for the different parts of the boundary and
        splitting the boundary integral into parts using the markers.
        
        :param mesh:        The mesh.
        :return:            Dict with keys (marker, part of boundary) and
                            Boundary class instance as values and
                            list of measures.
        """
        boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)

        boundary_conditions = self.boundary_types.copy()
        for key_tuple in boundary_conditions:
            boundary_conditions[key_tuple].mark(boundary_markers, key_tuple[0])

        # Define the measures in terms of the boundary markers
        dx = df.Measure(DX, mesh, subdomain_data=boundary_markers)

        if self.num_neumann_sectors + self.num_robin_sectors > 0:
            ds = df.Measure(DS, mesh, subdomain_data=boundary_markers)
        else:
            ds = None

        return boundary_conditions, [dx, ds]

    def _define_variational_problem(
        self,
        measures: List[df.Measure],
        function_space: df.FunctionSpace,
        time_discretization_args: Dict[str, int]
    ) -> Tuple[List[df.Function], List[df.Function]]:
        """
        Internal function for defining the variational problem.
        For time discretization we use the θ-method, where as
            θ = 0:      explicit Euler method,
            θ = 0.5:    Crank-Nicholson method,
            θ = 1:      implicit Euler method.
        See [Dzi10, Abschnitt 5.7].

        :param measures:                    List of measures.
        :param function_space:              Finite element function space.
        :param time_discretization_args:    Dict with keys NUM_TIME_STEPS and ODE_SOLVER with
                                            number of time steps and ODE solver (θ) as values.
        :return:                            Left and right-hand side of the variational problem and
                                            numerical solutions at the current and previous time steps.
        """
        dt = self.T / time_discretization_args[NUM_TIME_STEPS]      # Time step size
        theta = time_discretization_args[ODE_SOLVER]
        [dx, ds] = measures

        # Define initial function
        u_n = df.interpolate(self.initial.expression, function_space)

        # Define variational problem
        u = df.TrialFunction(function_space)
        v = df.TestFunction(function_space)

        left_hand_side = u * v * dx + dt * theta * df.dot(df.grad(u), df.grad(v)) * dx
        right_hand_side = u_n * v * dx - dt * (1 - theta) * df.dot(df.grad(u_n), df.grad(v)) * dx

        right_hand_side += dt * theta * self.f.expression * v * dx + \
                                dt * (1 - theta) * self.f.expression2 * v * dx

        if self.num_neumann_sectors > 0:
            for key_tuple in self.boundaries_assignment_dict[NEUMANN]:
                right_hand_side += dt * theta * self.neumann[key_tuple].expression * v * ds(key_tuple[0]) + \
                     dt * (1 - theta) * self.neumann[key_tuple].expression2 * v * ds(key_tuple[0])

        if self.num_robin_sectors > 0:
            for key_tuple in self.boundaries_assignment_dict[ROBIN]:
                left_hand_side += dt * theta * self.robin_r[key_tuple].expression * u * v * ds(key_tuple[0])
                right_hand_side -= dt * (1 - theta) * \
                                   self.robin_r[key_tuple].expression2 * u_n * v * ds(key_tuple[0])
                right_hand_side += dt * theta * self.robin_s[key_tuple].expression * v * ds(key_tuple[0]) + \
                     dt * (1 - theta) * self.robin_r[key_tuple].expression2 * v * ds(key_tuple[0])

        return [left_hand_side, right_hand_side], [df.Function(function_space), u_n]

    def _solve_variational_problem(
        self,
        variational_problem: List[df.Function],
        approx_functions: List[df.Function],
        boundary_conditions: Dict[Tuple[int, str], Boundary],
        function_space: df.FunctionSpace,
        time_discretization_args: Dict[str, int]
    ) -> Tuple[np.array, np.array]:
        """
        Internal function for solving the variational problem in the time-stepping loop.

        :param variational_problem:         Left and right-hand side of the variational problem.
        :param approx_functions:            Numerical solutions at the current and previous time steps.
        :param boundary_conditions:         Dict with keys (marker, part of boundary) and
                                            Boundary class instance as values.
        :param function_space:              Finite element function space.
        :param time_discretization_args:    Dict with keys NUM_TIME_STEPS and ODE_SOLVER with
                                            number of time steps and ODE solver (θ) as values.
        :return:                            Numerical solutions and different time steps from the numerical solution
                                            as numpy arrays.
        """
        [u, u_n] = approx_functions

        # Initialize solution and time-values array
        solution_data = np.zeros((time_discretization_args[NUM_TIME_STEPS] + 1, function_space.dim()))
        solution_data[0, :] = np.array(
            df.interpolate(self.initial.expression, function_space).vector()
        )
        time_values_data = np.zeros(time_discretization_args[NUM_TIME_STEPS] + 1)

        # Time-stepping
        dt = self.T / time_discretization_args[NUM_TIME_STEPS]
        t = t_n = 0
        for time_step in range(time_discretization_args[NUM_TIME_STEPS]):

            # Update current time
            t += dt

            # Update time in time-dependent expressions
            if self.solution is not None:
                self.solution.expression.t = t
                self.solution.expression2.t = t_n

            if self.f.expression_type == TIME_DEPENDENT:
                self.f.expression.t = t
                self.f.expression2.t = t_n

            if self.num_dirichlet_sectors > 0:
                for key_tuple in self.boundaries_assignment_dict[DIRICHLET]:
                    if self.dirichlet[key_tuple].expression_type == TIME_DEPENDENT:
                        self.dirichlet[key_tuple].expression.t = t
                        self.dirichlet[key_tuple].expression2.t = t_n

            if self.num_neumann_sectors > 0:
                for key_tuple in self.boundaries_assignment_dict[NEUMANN]:
                    if self.neumann[key_tuple].expression_type == TIME_DEPENDENT:
                        self.neumann[key_tuple].expression.t = t
                        self.neumann[key_tuple].expression2.t = t_n

            if self.num_robin_sectors > 0:
                for key_tuple in self.boundaries_assignment_dict[ROBIN]:
                    if self.robin_r[key_tuple].expression_type == TIME_DEPENDENT:
                        self.robin_r[key_tuple].expression.t = t
                        self.robin_r[key_tuple].expression2.t = t_n
                    if self.robin_s[key_tuple].expression_type == TIME_DEPENDENT:
                        self.robin_s[key_tuple].expression.t = t
                        self.robin_s[key_tuple].expression2.t = t_n

            # Set Dirichlet boundary conditions and compute solution
            if self.num_dirichlet_sectors > 0:
                dirichlet_bcs = []
                for key_tuple in self.boundaries_assignment_dict[DIRICHLET]:
                    dirichlet_bcs.append(
                        df.DirichletBC
                            (
                                function_space,
                                self.dirichlet[key_tuple].expression,
                                boundary_conditions[key_tuple]
                            )
                    )
                df.solve(variational_problem[0] == variational_problem[1], u, dirichlet_bcs)
            else:
                df.solve(variational_problem[0] == variational_problem[1], u)

            # Save solution and time-values for current time
            solution_data[time_step + 1, :] = np.array(u.vector())
            time_values_data[time_step + 1] = t

            # Compute inf-error at vertices
            if self.solution is not None:
                exact_current = df.interpolate(self.solution.expression, function_space)
                error_current = np.abs(np.array(exact_current.vector()) - np.array(u.vector())).max()
                print(f"t = {t}, error = {error_current}")
            else:
                print(f"t = {t}")

            # Update previous solution
            u_n.assign(u)
            t_n = t

        return solution_data, time_values_data

    def solve(
        self,
        boundary_conditions: Dict[Tuple[int, str], Boundary],
        function_space: df.FunctionSpace,
        measures: List[df.Measure],
        time_discretization_args: Dict[str, int]
    ) -> Tuple[np.array, np.array]:
        """
        Defines and solves the variational problem in the time-stepping loop.
        
        :param boundary_conditions:         Dict with keys (marker, part of boundary) and
                                            Boundary class instance as values.
        :param function_space:              Finite element function space.
        :param measures:                    List of measures.
        :param time_discretization_args:    Dict with keys NUM_TIME_STEPS and ODE_SOLVER with
                                            number of time steps and ODE solver (θ) as values.
        :return:                            Numerical solutions and different time steps from the numerical solution
                                            as numpy arrays.
        """
        variational_problem, approx_functions = self._define_variational_problem(
            measures, function_space, time_discretization_args
        )
        return self._solve_variational_problem(
            variational_problem, approx_functions, boundary_conditions, function_space, time_discretization_args
        )
