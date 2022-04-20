from typing import List, Tuple
import itertools
import random


def build_hyper_parameter_space(
    maximal_trials: int,
    hyper_parameter_lists: List[List[float]]
) -> List[Tuple[float]]:
    """
    Creates hyper parameter product space for tuning model architecture, training process, etc.

    :param maximal_trials:              Total number of trials to run during the search.
    :param hyper_parameter_lists:       List of lists containing the different parameters to search from
    :return:                            List of tuples containing the different parameters for the model
    """
    hyper_parameter_product_space = list(itertools.product(*hyper_parameter_lists))
    print(f"Dimension of hyper_parameter_product_space = {len(hyper_parameter_product_space)}")

    # Shuffle the product space and picking a few trials because searching the whole space is not feasible
    random.seed(0)
    random.shuffle(hyper_parameter_product_space)

    return hyper_parameter_product_space[:min(maximal_trials, len(hyper_parameter_product_space))]
