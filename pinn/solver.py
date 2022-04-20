from typing import Dict, List, Union, Tuple, Callable
import deepxde as dde

from services.constants import *
from pinn.expressions import EXPRESSIONS_TYPE_DICT
from pinn.domain import DOMAIN_TYPE_DICT
from pinn.boundary import Boundary


class PINNSolver:
    """
    Class for solving the heat equation with physics-informed neural networks (PINN).
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
        self.robin = None
        self.solution = None

        self.boundaries_assignment_dict = None
        self.num_dirichlet_sectors = None
        self.num_neumann_sectors = None
        self.num_robin_sectors = None

        self.domain = None
        self.on_boundary = None
        self.initial_boundary_conditions = None

    def setup_test_problem(self) -> None:
        """
        Initializing the inhomogeneity of the heat equation, the initial condition,
        the inhomogeneities of the boundary conditions and the solution.

        :return:    None
        """
        self.f, self.initial, self.dirichlet, self.neumann, self.robin, self.solution = \
            EXPRESSIONS_TYPE_DICT[self.domain_type][self.test_problem_str]()

        # Assign the different parts of the boundary to the different boundary types
        self.boundaries_assignment_dict = {
            DIRICHLET: list(self.dirichlet.keys()),
            NEUMANN: list(self.neumann.keys()),
            ROBIN: list(self.robin.keys())
        }
        self.num_dirichlet_sectors = 0 if self.dirichlet is None else len(self.boundaries_assignment_dict[DIRICHLET])
        self.num_neumann_sectors = 0 if self.neumann is None else len(self.boundaries_assignment_dict[NEUMANN])
        self.num_robin_sectors = 0 if self.robin is None else len(self.boundaries_assignment_dict[ROBIN])

    def set_boundary_types(self) -> None:
        """
        Defines python functions for the different parts of the boundary.
        The functions should return booleans depending whether a point is on the boundary or not.

        :return:    None
        """
        self.on_boundary = {
            key: Boundary(self.domain_type, key).boundary
            for key in (
                self.boundaries_assignment_dict[DIRICHLET] +
                self.boundaries_assignment_dict[NEUMANN] +
                self.boundaries_assignment_dict[ROBIN]
            )
        }

    def set_domain(self) -> None:
        """
        Defines the computational geometry.

        :return:    None
        """
        space_domain = DOMAIN_TYPE_DICT[self.domain_type]()
        time_domain = dde.geometry.TimeDomain(0, self.T)

        # Combine both space and time domain
        self.domain = dde.geometry.GeometryXTime(space_domain, time_domain)

    def set_boundary_conditions(self) -> None:
        """
        Assign the initial condition and the different boundary conditions to the different parts of the boundary.

        :return:    None
        """
        # Initial condition
        self.initial_boundary_conditions = [
            dde.IC(self.domain, self.initial, lambda _, on_initial: on_initial)
        ]

        # Boundary conditions
        if self.num_dirichlet_sectors > 0:
            for key in self.boundaries_assignment_dict[DIRICHLET]:
                self.initial_boundary_conditions.append(
                    dde.DirichletBC(self.domain, self.dirichlet[key], self.on_boundary[key])
                )
        if self.num_neumann_sectors > 0:
            for key in self.boundaries_assignment_dict[NEUMANN]:
                self.initial_boundary_conditions.append(
                    dde.NeumannBC(self.domain, self.neumann[key], self.on_boundary[key])
                )
        if self.num_robin_sectors > 0:
            for key in self.boundaries_assignment_dict[ROBIN]:
                self.initial_boundary_conditions.append(
                    dde.RobinBC(self.domain, self.robin[key], self.on_boundary[key])
                )

    def init_time_dependent_pde_solver(self, pde_solver_args: Dict[str, int]) -> dde.data.TimePDE:
        """
        Defines the time-dependent PDE problem.

        :param pde_solver_args:     Dict with keys NUM_DOMAIN, NUM_BOUNDARY and NUM_INITIAL with
                                    number of training points sampled inside the domain,
                                    on the initial location and on the boundary as values.
        :return:                    Time-dependent PDE solver.
        """
        return dde.data.TimePDE(
            geometryxtime=self.domain,
            pde=self.heat_equation,
            ic_bcs=self.initial_boundary_conditions,
            num_domain=pde_solver_args[NUM_DOMAIN],
            num_boundary=pde_solver_args[NUM_BOUNDARY],
            num_initial=pde_solver_args[NUM_INITIAL],
            train_distribution="pseudo",
            solution=self.solution,
            num_test=None
        )

    def init_neural_network(
        self,
        neural_network_args: Dict[str, int],
        hard_constraint: Callable = None
    ) -> dde.maps.FNN:
        """
        Configures the neural network.

        :param neural_network_args:     Dict with keys NUM_NEURONS, NUM_HIDDEN_LAYER, ACTIVATION and INITIALIZER with
                                        number of neurons per layer, number of hidden layers,
                                        the activation function and kernel initializer as values.
        :param hard_constraint:         Function that transforms the network output
                                        for exact initial and boundary conditions if not None.
        :return:                        Fully-connected neural network.
        """
        net = dde.maps.FNN(
            [self.domain.dim] + \
            [neural_network_args[NUM_NEURONS]] * neural_network_args[NUM_HIDDEN_LAYER] + \
            [1],
            neural_network_args[ACTIVATION],
            neural_network_args[INITIALIZER]
        )
        if hard_constraint is not None:
            net.apply_output_transform(hard_constraint)
        return net

    @staticmethod
    def train_model(
        time_pde_solver: dde.data.TimePDE,
        neural_network: dde.maps.FNN,
        optimizer_args: Dict[str, Union[str, int, List[float]]]
    ) -> Tuple[dde.Model, dde.model.LossHistory, dde.model.TrainState]:
        """
        Builds and trains the model.

        :param time_pde_solver:     Time-dependent PDE solver.
        :param neural_network:      Fully-connected neural network.
        :param optimizer_args:      Dict with keys OPTIMIZER, LEARNING_RATE, LOSS_WEIGHTS and EPOCHS with
                                    name of the optimizer, the learning rate, list of floats to weight
                                    the loss contributions and number of iterations to train the model as values.
        :return:                    Model, loss history and train state objects.
        """
        model = dde.Model(time_pde_solver, neural_network)

        # Resample the training points for PDE losses every given period.
        resampler = dde.callbacks.PDEResidualResampler(period=100)

        # Configure the model for training
        model.compile(
            optimizer_args[OPTIMIZER],
            optimizer_args[LEARNING_RATE],
            loss_weights=optimizer_args[LOSS_WEIGHTS]
        )

        # Trains the model
        loss_history, train_state = model.train(epochs=optimizer_args[EPOCHS], callbacks=[resampler])

        return model, loss_history, train_state

    def heat_equation(self, x, u):
        """
        Defines the PDE residual of the N-dimensional heat equation.

        :param x:   Space-time coordinates as (N + 1)-vectors.
        :param u:   Temperature in point x.
        :return:    PDE residual.
        """
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_t - u_xx - self.f(x)
