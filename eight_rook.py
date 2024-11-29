def setup_eight_rook_hopfield(n=8):
    """Set up the Hopfield network for the n-rook problem."""
    size = n * n
    weights = np.zeros((size, size))
    for i in range(n):
        for j in range(n):
            # Row constraint
            for k in range(n):
                if k != j:
                    weights[i*n + j, i*n + k] = -1
            # Column constraint
            for l in range(n):
                if l != i:
                    weights[i*n + j, l*n + j] = -1
    return weights

def solve_eight_rook(weights, n=8, max_iter=1000):
    """Solve the Eight-Rook problem."""
    size = n * n
    state = np.random.choice([1, -1], size=size)
    for _ in range(max_iter):
        state = np.sign(np.dot(weights, state))
    return state.reshape((n, n))

# Example usage
n = 8
weights = setup_eight_rook_hopfield(n)
solution = solve_eight_rook(weights, n)
print("Solution to the Eight-Rook problem:")
print(solution)
