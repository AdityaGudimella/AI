"""
Sudoku problem. Add link to description of problem.
"""
import warnings

import numpy as np


def arraydiff1d(a, b):
    """
    Finds the elements in a that are not present in b.
    a and b are both allowed to have repetitions, which will be handled properly.
    >>> arraydiff1d(np.repeat(np.arange(3), [3, 2, 1]), np.array([1, 1, 2, 3])
    array([0, 0, 0])
    :param a: 1d int array_like
    :param b: 1d int array_like
    :return:
    """
    # Vectorized way to do it
    a_unique, a_counts = np.unique(a, return_counts=True)
    b_unique, b_counts = np.unique(b, return_counts=True)
    common_unique_elements = np.intersect1d(a_unique, b_unique, assume_unique=True)
    c_u_e_idx_in_a = np.where(np.any(a_unique[None, ...] == common_unique_elements[..., None], axis=0))
    c_u_e_idx_in_b = np.where(np.any(b_unique[None, ...] == common_unique_elements[..., None], axis=0))
    final_counts = a_counts
    final_counts[c_u_e_idx_in_a] -= b_counts[c_u_e_idx_in_b]
    if np.any(final_counts < 0):
        warnings.warn('a should be a superset of b taking counts into consideration! Ignoring elements which occur in '
                      'b more times than they occur in a')
    final_counts[final_counts < 0] = 0
    return np.repeat(a_unique, final_counts)


def block_view(a, block=(3, 3)):
    """
    Provide a 2D block view to 2D array.
    No error checking made. Therefore meaningful (as implemented) only for blocks strictly compatible with the shape of
    a.
    """
    from numpy.lib.stride_tricks import as_strided
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (a.shape[0] / block[0], a.shape[1] / block[1]) + block
    strides = (block[0] * a.strides[0], block[1] * a.strides[1]) + a.strides
    return as_strided(a, shape=shape, strides=strides)


class Sudoku:
    def __init__(self, board=None, empty_value=-1):
        """

        :param board: 2D ndarray with dtype int
        :param empty_value: int Representation of an unfilled space
        """
        self.board = board.copy()
        self.previous_boards = [self.board.copy()]
        self.empty_value = empty_value
        pass

    def fill_random_values(self):
        """
        Fill the empty spaces in the board with random values.
        Updated board need not be a solution to the Sudoku problem.
        :return: None
        """
        empty_indices = np.where(self.board == self.empty_value)
        # for a 9x9 board, there will be a total of 9 1s, 9 2s, ... 9 9s.
        board_values = np.repeat(np.arange(1, 1 + 9), 9)
        # Since board is already filled with some values, remaining values are difference between two arrays
        fill_values = arraydiff1d(board_values, self.board[self.board != self.empty_value])
        self.board[self.board == empty_indices] = np.random.permutation(fill_values)
        pass

    def _calculate_score(self):
        # Todo: Get current value depending upon the board
        col_score, row_score, block_score = [0] * 3
        # Get number of collisions in columns
        for col in self.board.T:
            _, counts = np.unique(col, return_counts=True)
            col_score += np.sum(counts > 1)
        # Get number of collisions in rows
        for row in self.board:
            _, counts = np.unique(row, return_counts=True)
            row_score += np.sum(counts > 1)
        # Get number of collisions in each 3x3 block
        for block in block_view(self.board):
            _, counts = np.unique(block.ravel(), return_counts=True)
            block_score += np.sum(counts > 1)
        return col_score + row_score + block_score

    @property
    def current_value(self):
        return self._calculate_score()

    @property
    def neighborhood(self):
        # Todo:
        raise NotImplementedError('Still in todo')

    def update(self, board):
        self.previous_boards.append(self.board.copy())  # Just keeping a track of all states we've passed through
        self.board = board.copy()


class LocalSearch:
    def __init__(self, problem: Sudoku, strategy='random_walk', minimize=True):
        self.strategy = strategy
        self.problem = problem
        self.values = [problem.current_value]
        self.optimum = np.min if minimize else np.max
        pass

    def init_random_soln(self):
        # Todo: Create a random solution for the board
        raise NotImplementedError('Still in todo')

    @property
    def best_value(self):
        # Todo: Get max/min of (current value of the problem, best value found so far)
        return self.optimum(self.values)

    def choose_update(self, neighborhood):
        if self.strategy == 'random_walk':
            return np.random.choice(neighborhood)
        else:
            # Todo: choose an update/swap based on the Local Search strategy
            raise NotImplementedError

    def perform_search_step(self):
        # Todo: Do something and change the state in the problem
        best_update = self.choose_update(self.problem.neighborhood)
        self.problem.update(best_update)
        # Keep track of all the values found by the agent in the optimization
        self.values.append(self.problem.current_value)
        pass


if __name__ == '__main__':
    sudoku_problem = Sudoku(board=np.random.randint(0, 9 + 1, (9, 9), dtype=np.int8))
    search_agent = LocalSearch(sudoku_problem)
    for i in range(100):
        search_agent.perform_search_step()
    print(search_agent.best_value)
