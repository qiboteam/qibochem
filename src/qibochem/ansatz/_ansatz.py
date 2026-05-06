"""Helper functions for the `ansatz` module"""


# HF helper functions
def _bk_matrix_power2(dims: int) -> np.ndarray:
    """Build the Bravyi-Kitaev matrix of dimension ``dims`` :math:`d = 2^{n}` recursively"""
    if dims < 1:
        raise_error(ValueError, "Dimension of Bravyi-Kitaev matrix must be at least 1")
    # Base case
    elif dims == 1:
        return np.ones((1, 1), dtype=np.int8)

    # Recursive definition
    smaller_bk_matrix = _bk_matrix_power2(dims - 1)
    top_right = np.zeros((2 ** (dims - 2), 2 ** (dims - 2)), dtype=np.int8)
    top_half = np.concatenate((smaller_bk_matrix, top_right), axis=1)

    bottom_left = np.concatenate(
        (
            np.zeros(((2 ** (dims - 2)) - 1, 2 ** (dims - 2)), dtype=np.int8),
            np.ones((1, 2 ** (dims - 2)), dtype=np.int8),
        ),
        axis=0,
    )
    bottom_half = np.concatenate((bottom_left, smaller_bk_matrix), axis=1)

    # Combine top and bottom half
    return np.concatenate((top_half, bottom_half), axis=0)


def _bk_matrix(dims: int) -> np.ndarray:
    """Exact Brayvi-Kitaev matrix of size dims, obtained by slicing a larger BK matrix with dimension 2**m > n

    TODO: Update the occupation number vector using the update, parity, and flip set instead?
        Not sure if necessary; i.e. size of BK matrix probably not comparable to the memory needed
        for a classical simulation?

    Args:
        dims (int): Size of BK matrix
    """
    if dims < 1:
        raise_error(ValueError, "Dimension of Bravyi-Kitaev matrix must be at least 1")
    # Build bk_matrix_power2(m), where 2**m > dims
    min_bk_size = int(np.ceil(np.log2(dims))) + 1
    min_bk_matrix = _bk_matrix_power2(min_bk_size)
    # Then use array slicing to get the actual BK matrix
    return min_bk_matrix[:dims, :dims]
