import numpy as np
import timeit
from scipy.signal import convolve2d
import random

BOARD_WIDTH = 7
BOARD_HEIGHT = 7
WIN_STREAK_LENGTH = 4
EDGE_MASK = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype=bool)
EDGE_MASK[0, :] = EDGE_MASK[BOARD_WIDTH - 1, :] = EDGE_MASK[:, 0] = EDGE_MASK[:, BOARD_HEIGHT - 1] = True
KERNEL_HORIZONTAL = np.array([[1, 1, 1, 1]])
KERNEL_VERTICAL = np.array([[1], [1], [1], [1]])
KERNEL_DIAG_1 = np.eye(WIN_STREAK_LENGTH)
KERNEL_DIAG_2 = np.fliplr(np.eye(WIN_STREAK_LENGTH))

# takes in a board of shape (7,7,2) and returns a boolean array of shape (7,7) indicating the valid moves
def get_valid_moves(A):
    A = A[:, :, 0] | A[:, :, 1]
    adj_cells = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype=bool)
    adj_cells[1:, :] = A[:-1, :]
    adj_cells[:-1, :] |= A[1:, :]
    adj_cells[:, 1:] |= A[:, :-1]
    adj_cells[:, :-1] |= A[:, 1:]
    return (A[:, :] == 0) & (EDGE_MASK | adj_cells)

# takes in board of shape (7,7,2) and returns true iff board is full
def is_board_full(board):
    return np.all(board[:, :, 0] | board[:, :, 1])

# takes in board of shape (7,7,2) and (x,y) and returns true if that move is legal
def is_valid_move(A: np.ndarray, x: int, y: int):
    if A[x, y, 0] != 0 or A[x, y, 1] != 0:
        return False
    if x == 0 or x == BOARD_WIDTH - 1 or y == 0 or y == BOARD_HEIGHT - 1:
        return True
    if x > 0 and (A[x-1, y, 0] == 1 or A[x-1, y, 1] == 1):
        return True
    if x < BOARD_WIDTH - 1 and (A[x+1, y, 0] == 1 or A[x+1, y, 1] == 1):
        return True
    if y > 0 and (A[x, y-1, 0] == 1 or A[x, y-1, 1] == 1):
        return True
    if y < BOARD_HEIGHT - 1 and (A[x, y+1, 0] == 1 or A[x, y+1, 1] == 1):
        return True
    return False

# takes in a single layer of the board (7,7) and returns true iff there are 4 ones in a row
def is_terminal(board):
    return np.any(convolve2d(board, KERNEL_HORIZONTAL, mode='valid') == WIN_STREAK_LENGTH) \
        or np.any(convolve2d(board, KERNEL_VERTICAL, mode='valid') == WIN_STREAK_LENGTH) \
        or np.any(convolve2d(board, KERNEL_DIAG_1, mode='valid') == WIN_STREAK_LENGTH) \
        or np.any(convolve2d(board, KERNEL_DIAG_2, mode='valid') == WIN_STREAK_LENGTH)

# returns matrix encoding the position after making move
# sets move coord to 1 on top and then swaps layer
def get_resulting_board(board, move):
    res = board.copy()[..., ::-1]
    res[*move, 1] = 1
    return res

def hash_board(board):
    return hash(board.tobytes())

class State:
    def __init__(self, board):
        self.board = board
        self.parents = []
        self.children = []
        self.expanded = False
        pass
        # need pwin, ploss, pdraw

    def __hash__(self):
        return hash_board(self.board)

# returns a number of random states in the game
def generate_random_boards(num_boards):
    random_boards = []
    while len(random_boards) < num_boards:
        board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, 2), dtype=bool)
        num_moves = random.randint(0, 40)
        while num_moves > 0:
            move = random.choice(np.argwhere(get_valid_moves(board)))
            board = get_resulting_board(board, move)
            num_moves -= 1
        if not (is_board_full(board) or is_terminal(board[:, :, 1])):
            random_boards.append(board)
    return random_boards

# uses the model to simulate simul random positions, with times playthroughs through each
def simulate(model, simul, times):
    # for each node on the frontierr, we need to select a child
    # for each node on the frontier whose children are not expanded, we need to predict using the model
    visited_states = {}
    random_states = []
    for board in generate_random_boards(simul):
        state = State(board)
        random_states.append(state)
        visited_states[state] = state
    
    for playthrough in range(times):
        print("playthrough", playthrough, "total playthrough progress", str(round(100 * playthrough / times, 0)) + "%")
        frontier = random_states.copy()
        terminals = [] # each playthrough ends in a terminal state, which is stored here
        while len(frontier) > 0:
            to_evaluate_hashes = []
            to_evaluate_boards = []
            # print(len(frontier), "states currently on the frontier")
            for state in frontier:
                if not state.expanded:
                    # state has not been expanded, so it has no children marked
                    state.expanded = True
                    moves = np.argwhere(get_valid_moves(state.board) == 1)
                    for move in moves:
                        result = get_resulting_board(state.board, move)
                        corresponding_hash = hash_board(result)
                        if not corresponding_hash in visited_states:
                            # new board position, need to create state for it
                            child = State(result)
                            visited_states[corresponding_hash] = child
                            to_evaluate_hashes.append(corresponding_hash) # needs to be evaluated
                            to_evaluate_boards.append(result) # needs to be evaluated
                            child.parents.append(state)
                            state.children.append(child)
                        else:
                            # somehow this state was reached even though this parent has not been expanded
                            # link these states
                            child = visited_states[corresponding_hash]
                            child.parents.append(state)
                            state.children.append(child)
            # compute evaluations for all necessary states
            print(len(to_evaluate_boards), "boards needing evaluation")
            if len(to_evaluate_boards) > 0:
                evaluations = model.predict(np.array(to_evaluate_boards), verbose=0)
                for index in range(len(to_evaluate_hashes)):
                    evaluation = evaluations[index]
                    state = visited_states[to_evaluate_hashes[index]]
                    state.pwin = evaluation[0]
                    state.ploss = evaluation[1]
                    state.pdraw = evaluation[2]
            # now for each state on the frontier, we need to select a child
            next_frontier = []
            for state in frontier:
                # get probabilities and convert into distribution with softmax
                probabilities = np.array([[child.pwin, child.ploss, child.pdraw] for child in state.children])
                exp_values = probabilities[:, 0] - probabilities[:, 1]
                exp_values = np.exp(exp_values)
                exp_values /= np.mean(exp_values)
                # pick random index according to distribution
                rand_num = random.uniform(0, 1)
                chosen_index = -1
                while rand_num > 0:
                    rand_num -= exp_values[chosen_index + 1]
                    chosen_index += 1
                next_state = state.children[chosen_index]
                if is_board_full(next_state.board) or is_terminal(next_state.board[..., 1]):
                    terminals.append(next_state)
                else:
                    next_frontier.append(next_state)
            frontier = next_frontier
        # time to back propagate probabilities
        # first set scores of terminals to true scores
        for terminal in terminals:
            if is_board_full(terminal.board):
                terminal.pwin = 0
                terminal.ploss = 0
                terminal.pdraw = 1
            else:
                terminal.pwin = 0
                terminal.ploss = 1
                terminal.pdraw = 0
        backprop_frontier = terminals
        # backprop using negamin and logic
        while len(backprop_frontier) > 0:
            next_backprop_frontier = []
            for state in backprop_frontier:
                # need to update parents
                for parent in state.parents:
                    probabilities = np.array([[child.pwin, child.ploss, child.pdraw] for child in parent.children])
                    parent.pwin = 1 - np.prod(1 - probabilities[:, 1])
                    parent.ploss = np.prod(probabilities[:, 0])
                    parent.pdraw = 1 - parent.pwin - parent.ploss
                    next_backprop_frontier.append(parent)
            backprop_frontier = next_backprop_frontier
    return random_states

# LETS ONLY STORE STATES IN THE HASHTABLE IF THEY END UP BEING TRAVERSED
# WE WILL STILL HAVE TO TRACK THE EVALUATIONS THOUGH
# MAYBE EACH STATE CREATED WILL HAVE A MATRIX STORING THE PROBABILITIES FOR THE CHILDREN
# OR INSTEAD WE COULD STORE THESE MATRICES IN A LIST DURING THE SIMULATION

            