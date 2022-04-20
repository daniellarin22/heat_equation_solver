import dolfin as df

from services.constants import *


BOUNDARY_EPS = 1e-3


class Boundary:
    """
    Class for holding subdomain subclasses that specifies which points belong to which part of the boundary for FEM.
    """

    def __init__(self, domain_type: str, boundary_type: str) -> None:
        """
        Constructor of this class.
        Will be called when an object is created from this class and
        allows the class to initialize the attributes of the class.

        :param domain_type:         Name of the domain type (e.g. interval or rectangle).
        :param boundary_type:       Name of the part of the boundary.
        """
        self.boundary = self.BOUNDARY_TYPE_DICT[domain_type][boundary_type]()

    class BoundaryAll(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class RectangleLeftRight(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (df.near(x[0], 0, BOUNDARY_EPS) or df.near(x[0], 1, BOUNDARY_EPS))

    class RectangleTopBottom(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (df.near(x[1], 0, BOUNDARY_EPS) or df.near(x[1], 1, BOUNDARY_EPS))


    # Dicts holding functions for different domain types and parts of the boundary
    BOUNDARY_INTERVAL_DICT = {
        ALL: BoundaryAll
    }

    BOUNDARY_RECTANGLE_DICT = {
        ALL: BoundaryAll,
        TOP_BOTTOM: RectangleTopBottom,
        LEFT_RIGHT: RectangleLeftRight,
    }

    BOUNDARY_TYPE_DICT = {
        INTERVAL: BOUNDARY_INTERVAL_DICT,
        RECTANGLE: BOUNDARY_RECTANGLE_DICT,
    }
