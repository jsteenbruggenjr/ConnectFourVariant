import numpy as np
import timeit

def swap_1(board):
    return np.array([board[..., 1], board[..., 0]])

# Sample data for demonstration (replace this with your actual arrays)
# For example, let's say you have 3 arrays with different lengths
A_1 = np.array([[0.5, 0.2], [0.3, 0.4]])
A_2 = np.array([[0.7, 0.1], [0.2, 0.6], [0.9, 0.5]])
A_3 = np.array([[0.1, 0.3]])

# List of arrays
arrays = [A_1, A_2, A_3]

def solution_1(arrays):
    results = np.zeros((len(arrays), 2))
    for i, A_n in enumerate(arrays):
        first_value = 1 - np.prod(1 - A_n[:, 1])
        second_value = np.prod(A_n[:, 0])
        results[i] = [first_value, second_value]

def solution_2(arrays):
    max_length = max(array.shape[0] for array in arrays)
    padded_arrays = np.full((len(arrays), max_length, 2), np.nan)
    for i, A_n in enumerate(arrays):
        padded_arrays[i, :A_n.shape[0], :] = A_n
    first_value = 1 - np.nanprod(1 - padded_arrays[:, :, 1], axis=1)
    second_value = np.nanprod(padded_arrays[:, :, 0], axis=1)
    results = np.column_stack((first_value, second_value))

print(timeit.timeit("solution_1(arrays)", globals=globals(), number=10000))
print(timeit.timeit("solution_2(arrays)", globals=globals(), number=10000))

arr = np.random.randint(0, 2, size=(7,7,2))
print(np.argwhere(arr == 1))
print(arr[*[0, 0], 1])