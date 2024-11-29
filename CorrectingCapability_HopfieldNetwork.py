import numpy as np

def hopfield_train(patterns):
    """Train a Hopfield network using Hebbian learning."""
    num_neurons = patterns.shape[1]
    weights = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        weights += np.outer(p, p)
    np.fill_diagonal(weights, 0)  # No self-connections
    return weights / len(patterns)

def hopfield_update(state, weights):
    """Update the state of the Hopfield network."""
    return np.sign(np.dot(weights, state))

def test_error_correction(weights, patterns):
    """Test error correction capability."""
    errors_corrected = 0
    for pattern in patterns:
        noisy_pattern = pattern.copy()
        # Introduce random noise
        noise_idx = np.random.choice(len(pattern), size=len(pattern)//5, replace=False)
        noisy_pattern[noise_idx] *= -1
        updated_pattern = hopfield_update(noisy_pattern, weights)
        if np.array_equal(updated_pattern, pattern):
            errors_corrected += 1
    return errors_corrected / len(patterns)

# Example usage
patterns = np.array([[1, -1, 1, -1, 1], [-1, 1, -1, 1, -1]])  # Binary patterns
weights = hopfield_train(patterns)
correction_rate = test_error_correction(weights, patterns)
print(f"Error correction rate: {correction_rate * 100:.2f}%")
