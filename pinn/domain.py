import deepxde as dde

from services.constants import *


def set_domain_interval() -> dde.geometry:
    """
    Defines the interval with endpoints 0 and 1 as the computational geometry.

    :return:    The computational geometry.
    """
    return dde.geometry.Interval(0, 1)


def set_domain_rectangle() -> dde.geometry:
    """
    Defines the rectangle with bottom left corner [0, 0] and top right corner [1, 1] as the computational geometry.

    :return:    The computational geometry.
    """
    return dde.geometry.Rectangle([0, 0], [1, 1])


# Dict holding geometry functions for different domain types
DOMAIN_TYPE_DICT = {
    INTERVAL: set_domain_interval,
    RECTANGLE: set_domain_rectangle,
}
