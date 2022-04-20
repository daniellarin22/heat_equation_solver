import numpy as np

from services.constants import *


class Boundary:
    """
    Class for holding python functions that specifies which points belong to which part of the boundary for PINN.
    """

    def __init__(self, domain_type: str, boundary_type: str) -> None:
        """
        Constructor of this class.
        Will be called when an object is created from this class and
        allows the class to initialize the attributes of the class.

        :param domain_type:         Name of the domain type (e.g. interval or rectangle).
        :param boundary_type:       Name of the part of the boundary.
        """
        self.boundary = self.BOUNDARY_TYPE_DICT[domain_type][boundary_type]

    @staticmethod
    def boundary_all(x, on_boundary):
        return on_boundary

    @staticmethod
    def rectangle_left_right(x, on_boundary):
        return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 1))

    @staticmethod
    def rectangle_top_bottom(x, on_boundary):
        return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


    # Dicts holding functions for different domain types and parts of the boundary
    BOUNDARY_INTERVAL_DICT = {
        ALL: boundary_all.__func__,
    }

    BOUNDARY_RECTANGLE_DICT = {
        ALL: boundary_all.__func__,
        TOP_BOTTOM: rectangle_top_bottom.__func__,
        LEFT_RIGHT: rectangle_left_right.__func__,
    }

    BOUNDARY_TYPE_DICT = {
        INTERVAL: BOUNDARY_INTERVAL_DICT,
        RECTANGLE: BOUNDARY_RECTANGLE_DICT,
    }
