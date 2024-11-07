import random

class MENACE:
    def __init__(self):
        self.matchboxes = {}  # Dictionary to store state-action pairs
        self.learning_rate = 0.1

    def get_move(self, board):
        state = tuple(board)
        
        # Initialize matchbox if state is unseen, with each action having equal probability
        if state not in self.matchboxes:
            self.matchboxes[state] = [1] * 9

        # Exploration-exploitation balance: 10% exploration, 90% exploitation
        if random.random() < 0.1:  # 10% exploration
            return random.choice(range(9))
        else:
            return self.weighted_choice(self.matchboxes[state])

    def update(self, moves, reward):
        # Update matchboxes with incremental learning based on reward
        for state, action in moves:
            self.matchboxes[state][action] += self.learning_rate * reward

    def weighted_choice(self, weights):
        # Selects an action based on weighted probabilities
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            if upto + w >= r:
                return i
            upto += w

# Initialize MENACE instance
menace = MENACE()

# Define a sample empty Tic-Tac-Toe board
board = [0] * 9  # Representing an empty board with zeros

# Track moves taken by MENACE
moves = []
for _ in range(5):  # Simulate 5 moves
    move = menace.get_move(board)
    print(f"Selected move: {move}")
    
    # Record the board state and move for future update
    moves.append((tuple(board), move))
    
    # Simulate MENACE playing by marking the chosen position with '1'
    board[move] = 1

# Simulate a positive reward (e.g., a win) for demonstration
reward = 1
menace.update(moves, reward)

# Output updated matchbox probabilities to observe learning
print("\nUpdated matchboxes after reward:")
for state, actions in menace.matchboxes.items():
    print(f"State: {state} | Actions: {actions}")
