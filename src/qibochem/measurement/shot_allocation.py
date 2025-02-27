"""
Functions for allocating shots to the Hamiltonian terms
"""


def vmsa_shot_allocation(total_shots, n_trial_shots, variance_values):
    """
    Allocate shots for each term in a Hamiltonian based on the computed sample variances of each term. Unlike in VPSR,
    all shots will be allocated.

    Returns:
        list: List of integers corresponding to the allocation of the remaining shots
    """
    n_groups = len(variance_values)
    remaining_shots = total_shots - n_groups * n_trial_shots
    std_dev_values = [_var**0.5 for _var in variance_values]
    # Calculate everything as floats first, then convert to ints
    allocated_shots = [int(std_dev * remaining_shots / sum(std_dev_values)) for std_dev in std_dev_values]
    # Throw any leftover shots into the last term (arbitrarily)
    allocated_shots[-1] += remaining_shots - sum(allocated_shots)
    return allocated_shots


def vpsr_shot_allocation(total_shots, n_trial_shots, variance_values):
    """
    Allocate shots for each term in a Hamiltonian based on the computed sample variances of each term. Unlike in VMSA,
    the total number of shots allocated will be < total_shots.

    Returns:
        list: List of integers corresponding to the allocation of the remaining shots
    """
    n_groups = len(variance_values)
    remaining_shots = total_shots - n_groups * n_trial_shots
    std_dev_values = [_var**0.5 for _var in variance_values]
    _eta = sum(std_dev_values) ** 2 / (n_groups * sum(variance_values))  # eta in Equation 17 of the reference paper
    # Calculate everything as floats first, then convert to ints
    allocated_shots = [int(_eta * std_dev * remaining_shots / sum(std_dev_values)) for std_dev in std_dev_values]
    return allocated_shots
