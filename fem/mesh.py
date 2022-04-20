from typing import Dict
import dolfin as df

from services.constants import *


def set_mesh_interval(args: Dict[str, int]) -> df.Mesh:
    """
    Creates a mesh of the interval with endpoints 0 and 1.
    The total number of intervals will be NX and the total number of vertices will be NX + 1.

    :param args:    Dict with key NX and number of cells as value.
    :return:        The mesh.
    """
    return df.UnitIntervalMesh(args[NX])


def set_mesh_rectangle(args: Dict[str, int]) -> df.Mesh:
    """
    Creates a triangular mesh of the rectangle with bottom left corner [0, 0] and top right corner [1, 1].
    The total number of triangles will be 2 * NX * NY and the total number of vertices will be (NX + 1) * (NY + 1).
    
    :param args:    Dict with keys NX and NY with number of cells in horizontal and vertical direction as values.
    :return:        The mesh.
    """
    return df.UnitSquareMesh(args[NX], args[NY])


# Dict holding mesh functions for different domain types
DOMAIN_TYPE_DICT = {
    INTERVAL: set_mesh_interval,
    RECTANGLE: set_mesh_rectangle,
}
