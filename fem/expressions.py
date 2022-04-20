from typing import Union
import dolfin as df
import numpy as np

from services.constants import *


class Expression:
    """
    Class for representing user-defined expressions (with regard to the time-stepping loop).
    """

    def __init__(self, math_str: Union[str, float], expression_type: str, t: float = None, degree: int = None) -> None:
        """
        Constructor of this class.
        Will be called when an object is created from this class and
        allows the class to initialize the attributes of the class.

        :param math_str:            String expression.
        :param expression_type:     Determines whether the expression is time dependent, time independent or constant.
        :param t:                   Value of the time variable.
        :param degree:              Degree of the element.
        """
        self.math_str = math_str
        self.expression_type = expression_type
        self.t = t
        self.degree = degree

        # Define two expressions (for single-step methods) per class instance for time-stepping loop
        # In general, N-step methods need N + 1 expressions
        if self.expression_type == TIME_DEPENDENT:
            self.expression = df.Expression(self.math_str, t=self.t, degree=self.degree)
            self.expression2 = df.Expression(self.math_str, t=self.t, degree=self.degree)
        elif self.expression_type == TIME_INDEPENDENT:
            self.expression = df.Expression(self.math_str, degree=self.degree)
            self.expression2 = self.expression
        elif self.expression_type == CONSTANT:
            self.expression = df.Constant(self.math_str)
            self.expression2 = self.expression
        else:
            raise ValueError("No valid expression type")


def test_problem_1d():
    """
    Contains the inhomogeneity of the heat equation, the initial condition,
    the inhomogeneities of the boundary conditions and the solution
    of the 1D test problem, see Section 5.1.

    :return:    Data of the test problem as expressions or dict with key part of boundary and expressions as values.
    """
    degree = 2

    f = Expression(0, expression_type=CONSTANT)

    initial = Expression(
        f"sin(2*{np.pi}*x[0])",
        expression_type=TIME_INDEPENDENT, degree=degree)


    dirichlet = {
        (0, ALL): Expression(0, expression_type=CONSTANT)
    }

    solution = Expression(f"exp(-4*{np.pi * np.pi}*t)*sin(2*{np.pi}*x[0])",
                          expression_type=TIME_DEPENDENT, t=0, degree=degree)

    return f, initial, dirichlet, {}, {}, {}, solution


def test_problem_2d():
    """
    Contains the inhomogeneity of the heat equation, the initial condition,
    the inhomogeneities of the boundary conditions and the solution
    of the 2D test problem, see Section 5.2.

    :return:    Data of the test problem as expressions or dict with key part of boundary and expressions as values.
    """
    degree = 2

    pi_squared = np.pi * np.pi

    f = Expression(
        f"4*{pi_squared}*exp(-4*{pi_squared}*t)*cos(2*{np.pi}*x[0])*cos(2*{np.pi}*x[1])",
        expression_type=TIME_DEPENDENT, t=0, degree=degree)

    initial = Expression(
        f"cos(2*{np.pi}*x[0])*cos(2*{np.pi}*x[1])",
        expression_type=TIME_INDEPENDENT, degree=degree)

    dirichlet = {
        (0, LEFT_RIGHT): Expression(
            f"exp(-4*{pi_squared}*t)*cos(2*{np.pi}*x[1])",
            expression_type=TIME_DEPENDENT, t=0, degree=degree
        )
    }

    neumann = {
        (1, TOP_BOTTOM): Expression(0, expression_type=CONSTANT)
    }

    solution = Expression(f"exp(-4*{pi_squared}*t)*cos(2*{np.pi}*x[0])*cos(2*{np.pi}*x[1])",
                          expression_type=TIME_DEPENDENT, t=0, degree=degree)

    return f, initial, dirichlet, neumann, {}, {}, solution


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
