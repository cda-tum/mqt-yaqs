from __future__ import annotations
import concurrent.futures
import copy
import multiprocessing
import numpy as np
import opt_einsum as oe
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.core.data_structures.networks import MPS


def scalar_product(A: MPS, B: 'MPS', site: int=-1) -> complex:
    """ Calculates the scalar product of two Matrix Product States
        by contracting all positions vertically then horizontally.

    Args:
        A: first MPS
        B: second MPS

    Returns:
        result: Frobenius norm of A and B <A|B>
    """
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)
    for i, tensor in enumerate(A_copy.tensors):
        A_copy.tensors[i] = np.conj(tensor)

    # Contract all sites
    if site == -1:
        for site in range(A.length):
            tensor = oe.contract('abc, ade->bdce', A_copy.tensors[site], B_copy.tensors[site])
            if site == 0:
                result = tensor
            else:
                result = oe.contract('abcd, cdef->abef', result, tensor)
        result = np.squeeze(result)

    # Used for ignoring other tensors if MPS is in canonical form
    else:
        # Single site operators
        result = oe.contract('ijk, ijk', A_copy.tensors[site], B_copy.tensors[site])

    return result


def local_expval(state: MPS, operator: np.ndarray, site: int) -> float:
    """ Expectation value for a given MPS-MPO-MPS network

    Args:
        A: MPS
        B: MPO
        C: MPS

    Returns:
        E.real: real portion of calculated expectation value
    """
    # TODO: Could be more memory-efficient by not copying state

    # This loop assumes the MPS is in canonical form at the given site
    temp_state = copy.deepcopy(state)
    temp_state.tensors[site] = oe.contract('ab, bcd->acd', operator, temp_state.tensors[site])
    return scalar_product(state, temp_state, site)


def measure_single_shot(state) -> int:
    """
    Performs a single-shot measurement of the MPS state.
    Args:
        state ('MPS'): The MPS state to measure.
    Returns:
        int: The observed basis state as an integer.
    """
    temp_state = copy.deepcopy(state)
    bitstring = []
    for site, tensor in enumerate(temp_state.tensors):
        reduced_density_matrix = oe.contract('abc, dbc->ad', tensor, np.conj(tensor))
        probabilities = np.diag(reduced_density_matrix).real
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        bitstring.append(chosen_index)
        selected_state = np.zeros(len(probabilities))
        selected_state[chosen_index] = 1
        # Multiply state
        tensor = oe.contract('a, acd->cd', selected_state, tensor)
        # Multiply site into next site
        if site != state.length - 1:
            temp_state.tensors[site + 1] = 1 / np.sqrt(probabilities[chosen_index]) * oe.contract(
                'ab, cbd->cad', tensor, temp_state.tensors[site + 1])
    return sum(c << i for i, c in enumerate(bitstring))


def measure(state: MPS, shots: int) -> dict:
    """
    Measures an MPS state for a given number of shots.
    
    Args:
        state ('MPS'): The MPS state to measure.
        shots (int): Number of measurements (shots) to perform.
        
    Returns:
        dict: A dictionary mapping basis states to their observed counts.
    """

    if shots > 1:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar:
                results = {}
                futures = [executor.submit(measure_single_shot, copy.deepcopy(state)) for _ in range(shots)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results[result] = results.get(result, 0) + 1
                    except Exception as e:
                        print(f"Shot measurement failed with exception: {e}.")
                    finally:
                        pbar.update(1)
        return results
    else:
        results = {}
        basis_state = measure_single_shot(state)
        results[basis_state] = results.get(basis_state, 0) + 1
        return results
