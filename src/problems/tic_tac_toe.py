"""
Implements the tic-tac-toe game.
"""
from enum import Enum
from itertools import chain

import numpy as np


class Mark(Enum):
    Χ = 1
    Ο = -1
    Φ = 0


class TicTacToe:
    """
    Agent is represented by Ο
    Player is represented by Χ
    Unfilled space is represented by Φ
    """
    AGENT = Mark.Ο
    PLAYER = Mark.Χ
    EMPTY = Mark.Φ

    def __init__(self, board_size=3, board=None):
        self.board_size = board_size
        self.board = np.zeros(shape=(board_size, board_size), dtype=np.int8) if board is None else board.copy()
        self._game_over = False
        self._is_empty = True

    @property
    def game_over(self) -> bool:
        self._game_over = self._is_marker_complete(TicTacToe.AGENT) or self._is_marker_complete(TicTacToe.PLAYER)
        return self._game_over

    @game_over.setter
    def game_over(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError(f'value must be bool. Provided {value} of type {type(value)}')
        self._game_over = value
        pass

    @property
    def did_agent_win(self) -> bool:
        return self._is_marker_complete(TicTacToe.AGENT)

    def mark(self, pos, mark: Mark) -> None:
        try:
            self.board[pos]
        except IndexError as e:
            raise e
        if mark == TicTacToe.EMPTY:
            raise ValueError(f'Cannot set a position with Φ')
        elif mark not in Mark:
            raise ValueError(f'{mark} not in list of possible marks {[x.name for x in Mark]}')

        if Mark(self.board[pos]) == Mark.Φ:
            self.board[pos] = mark.value
        else:
            raise ValueError(f'Position {pos} has already been set to {self.board[pos]}')
        pass

    def make_move(self) -> None:
        empty_rows, empty_cols = self._get_indices_of(TicTacToe.EMPTY)
        for row, col in zip(empty_rows, empty_cols):
            temp_board = TicTacToe(board=self.board)
            temp_board.mark((row, col), TicTacToe.AGENT)
            if temp_board._is_marker_complete(TicTacToe.AGENT):  # Found a winning move. Game over!
                self.mark((row, col), TicTacToe.AGENT)
                self.game_over = True
                return
        random_idx = np.random.choice(len(empty_rows))
        random_idx = empty_rows[random_idx], empty_cols[random_idx]
        self.mark(random_idx, TicTacToe.AGENT)
        pass

    def request_move(self) -> None:
        empty_idc = np.c_[self._get_indices_of(TicTacToe.EMPTY)]
        print_msg = "\t".join("{}: {}" for _ in range(empty_idc.shape[0]))
        while True:
            print('Please choose a position from: ')
            ans = input(print_msg.format(*chain.from_iterable(enumerate(empty_idc))))
            try:
                idx = tuple(empty_idc[int(ans)])
                break
            except ValueError:
                print(f'Wrong input given. Please enter an integer in [0,{len(empty_idc)})')
        self.mark(idx, TicTacToe.PLAYER)
        pass

    def print_board(self) -> None:
        row = "\t".join('{}' for _ in range(self.board_size))
        board = "\n".join(row for _ in range(self.board_size))
        print(board.format(*map(lambda x: Mark(x).name, self.board.ravel())))
        pass

    def _is_marker_complete(self, marker: Mark) -> bool:
        row, column = 0, 1
        if np.all(np.diag(self.board) == marker.value):  # Diagonal is complete
            return True
        elif np.any(np.all(self.board == marker.value, axis=column)):
            return True
        elif np.any(np.all(self.board == marker.value, axis=row)):
            return True
        else:
            return False
        pass

    def _get_indices_of(self, mark: Mark) -> np.ndarray:
        return np.where(self.board == mark.value)

    pass


if __name__ == '__main__':
    ttt = TicTacToe(board_size=3)
    ttt.print_board()
    print('Ξ' * ((ttt.board_size - 1) * 4 + 1))
    print('START')
    while not ttt.game_over:
        print('Ξ' * ((ttt.board_size - 1) * 4 + 1))
        ttt.make_move()
        ttt.print_board()
        if ttt.game_over:
            break
        print('Ξ' * ((ttt.board_size - 1) * 4 + 1))
        ttt.request_move()
        ttt.print_board()
    if ttt.did_agent_win:
        print('You lost!')
    else:
        print('You won!')
    pass
