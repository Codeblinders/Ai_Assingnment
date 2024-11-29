import numpy as np
import operator
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, step_reward=-0.04):
        # Set information about the gridworld
        self.height = 3
        self.width = 4
        self.grid = np.ones((self.height, self.width)) * step_reward
        
        # Set start location for the agent
        self.current_location = (2, 0)
        
        # Set locations for the terminal states
        self.negative_terminal = (1,3)  
        self.positive_terminal = (0,3)
        self.terminal_states = [self.negative_terminal, self.positive_terminal]
        
        # Set grid rewards for terminal states
        self.grid[self.negative_terminal[0], self.negative_terminal[1]] = -1
        self.grid[self.positive_terminal[0], self.positive_terminal[1]] = 1
        
        # Block state (1,1) is not accessible
        self.blocked_state = (1,1)
        
        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Set transition probabilities
        self.intended_prob = 0.8
        self.sideways_prob = 0.1  # 0.1 for each perpendicular direction
        
    def get_next_state(self, state, action):
        """Returns possible next states and their probabilities"""
        next_states = {}
        
        # Get intended next state
        intended_state = self.get_intended_next_state(state, action)
        next_states[intended_state] = self.intended_prob
        
        # Get perpendicular states
        if action in ['UP', 'DOWN']:
            left_state = self.get_intended_next_state(state, 'LEFT')
            right_state = self.get_intended_next_state(state, 'RIGHT')
            next_states[left_state] = self.sideways_prob
            next_states[right_state] = self.sideways_prob
        else:
            up_state = self.get_intended_next_state(state, 'UP')
            down_state = self.get_intended_next_state(state, 'DOWN')
            next_states[up_state] = self.sideways_prob
            next_states[down_state] = self.sideways_prob
            
        return next_states
    
    def get_intended_next_state(self, state, action):
        """Returns the intended next state for an action, accounting for walls and blocks"""
        x, y = state
        
        if action == 'UP':
            next_state = (max(x-1, 0), y)
        elif action == 'DOWN':
            next_state = (min(x+1, self.height-1), y)
        elif action == 'LEFT':
            next_state = (x, max(y-1, 0))
        elif action == 'RIGHT':
            next_state = (x, min(y+1, self.width-1))
            
        # If next state is blocked, stay in current state
        if next_state == self.blocked_state:
            return state
            
        return next_state

def value_iteration(env, gamma=0.99, theta=0.0001):
    """Performs value iteration to find optimal value function"""
    # Initialize value function
    V = np.zeros((env.height, env.width))
    
    while True:
        delta = 0
        # For each state
        for i in range(env.height):
            for j in range(env.width):
                if (i,j) in env.terminal_states or (i,j) == env.blocked_state:
                    continue
                    
                v = V[i,j]
                values = []
                
                # Try all actions
                for action in env.actions:
                    next_states = env.get_next_state((i,j), action)
                    value = 0
                    
                    # Calculate expected value for this action
                    for next_state, prob in next_states.items():
                        value += prob * (env.grid[next_state] + gamma * V[next_state])
                    
                    values.append(value)
                
                # Take maximum over all actions
                V[i,j] = max(values)
                delta = max(delta, abs(v - V[i,j]))
        
        if delta < theta:
            break
            
    return V

# Test with different step rewards
step_rewards = [-2, 0.1, 0.02, 1]
for reward in step_rewards:
    env = GridWorld(step_reward=reward)
    V = value_iteration(env)
    
    print(f"\nOptimal value function for step reward = {reward}:")
    print(V)
    
    plt.figure(figsize=(8,6))
    plt.imshow(V, cmap='RdYlGn')
    plt.colorbar()
    plt.title(f'Value Function (Step Reward = {reward})')
    plt.show()