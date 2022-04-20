from deepxde.backend import tf
import numpy as np

from services.constants import *


def test_problem_1d():
    """
    Contains the inhomogeneity of the heat equation, the initial condition,
    the inhomogeneities of the boundary conditions and the solution
    of the 1D test problem, see Section 5.1.

    :return:    Data of the test problem as python functions or
                dict with key part of boundary and python functions as values.
    """
    # x = x[:, 0:1], t = x[:, 1:2]

    def f(x): return 0

    def initial(x): return tf.sin(2 * np.pi * x[:, 0:1])

    def dirichlet(x): return 0

    def solution(x): return tf.exp(-4 * np.pi * np.pi * x[:, 1:2]) * tf.sin(2 * np.pi * x[:, 0:1])

    return (
        f,
        initial,
        {ALL: dirichlet},
        {},
        {},
        solution
    )


def test_problem_2d():
    """
    Contains the inhomogeneity of the heat equation, the initial condition,
    the inhomogeneities of the boundary conditions and the solution
    of the 2D test problem, see Section 5.2.

    :return:    Data of the test problem as python functions or
                dict with key part of boundary and python functions as values.
    """
    # x = x[:, 0:1], y = x[:, 1:2], t = x[:, 2:3]

    pi_squared = np.pi * np.pi

    def f(x): return 4 * pi_squared * tf.exp(-4 * pi_squared * x[:, 2:3]) * \
                     tf.cos(2 * np.pi * x[:, 0:1]) * tf.cos(2 * np.pi * x[:, 1:2])

    def initial(x): return tf.cos(2 * np.pi * x[:, 0:1]) * tf.cos(2 * np.pi * x[:, 1:2])

    def dirichlet(x): return tf.exp(-4 * pi_squared * x[:, 2:3])  * tf.cos(2 * np.pi * x[:, 1:2])

    def neumann(x): return 0

    def solution(x): return tf.exp(-4 * pi_squared * x[:, 2:3]) * \
                            tf.cos(2 * np.pi * x[:, 0:1]) * tf.cos(2 * np.pi * x[:, 1:2])

    return (
        f,
        initial,
        {LEFT_RIGHT: dirichlet},
        {TOP_BOTTOM: neumann},
        {},
        solution
    )


# Dicts holding the data of the different test problems for different domain types
EXPRESSION_INTERVAL_DICT = {
    HEAT_EQUATION_1D: test_problem_1d
}

EXPRESSIONS_RECTANGLE_DICT = {
    HEAT_EQUATION_2D: test_problem_2d
}

EXPRESSIONS_TYPE_DICT = {
    INTERVAL: EXPRESSION_INTERVAL_DICT,
    RECTANGLE: EXPRESSIONS_RECTANGLE_DICT,
}
