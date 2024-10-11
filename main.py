import numpy as np
from learning import *
import timeit
import game
import random

def pretty_print_board(board):
    parity = np.sum(board) % 2
    str = ""
    for i in range(7):
        for j in range(7):
            if board[i][j][parity] == 1:
                str += "X"
            elif board[i][j][1 - parity] == 1:
                str += "O"
            else:
                str += "-"
        str += "\n"
    print(str)


model_path = "first_model.keras"

model = get_model(model_path)

GAMES_IN_FORWARD_PASS = 500
PLAYS_PER_GAME = 200

random_states = game.simulate(model, GAMES_IN_FORWARD_PASS, PLAYS_PER_GAME)
X_train = np.array([state.board for state in random_states])
Y_train = np.array([[state.pwin, state.ploss, state.pdraw] for state in random_states])
rand_state = random_states[0]

pretty_print_board(rand_state.board)
print("X" if np.sum(rand_state.board) % 2 == 0 else "O", "to play")
print("pwin", rand_state.pwin, "ploss", rand_state.ploss, "pdraw", rand_state.pdraw)
print(X_train.shape)
model.fit(X_train, Y_train, epochs=500, batch_size=20)

model.save(model_path, include_optimizer=False)