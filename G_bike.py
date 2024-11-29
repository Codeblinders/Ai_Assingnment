import numpy as np
from scipy.stats import poisson
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import logging

@dataclass
class SimulationResult:
    """Data class to store simulation results"""
    profit: float
    rentals: List[int]
    requests: List[int]
    moves: int
    final_state: Tuple[int, int]

class GBikeRental:
    def __init__(
        self,
        max_bikes_per_location: int = 20,
        max_bikes_to_move: int = 5,
        moving_cost: float = 2.0,
        rental_revenue: float = 10.0,
        expected_requests: List[float] = None,
        expected_returns: List[float] = None,
        initial_bikes: List[int] = None,
        discount_rate: float = 0.9,
        poisson_truncate: int = 10
    ):
        """
        Initialize the bike rental system with configurable parameters.
        
        Args:
            max_bikes_per_location: Maximum number of bikes each location can hold
            max_bikes_to_move: Maximum number of bikes that can be moved between locations
            moving_cost: Cost per bike moved between locations
            rental_revenue: Revenue per bike rental
            expected_requests: Expected number of requests per location [loc1, loc2]
            expected_returns: Expected number of returns per location [loc1, loc2]
            initial_bikes: Initial number of bikes at each location [loc1, loc2]
            discount_rate: Discount rate for future rewards
            poisson_truncate: Truncation point for Poisson distribution calculations
        """
        # System parameters
        self.MAX_BIKES_PER_LOCATION = max_bikes_per_location
        self.MAX_BIKES_TO_MOVE = max_bikes_to_move
        self.MOVING_COST = moving_cost
        self.RENTAL_REVENUE = rental_revenue
        self.EXPECTED_REQUESTS = expected_requests or [3, 4]
        self.EXPECTED_RETURNS = expected_returns or [3, 2]
        self.DISCOUNT_RATE = discount_rate
        self.POISSON_TRUNCATE = poisson_truncate
        
        # Initialize bike distribution
        self.bikes_at_locations = list(initial_bikes) if initial_bikes else [10, 10]
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize value function and policy
        self._initialize_state_space()
        self.initialize_values()
        self.initialize_policy()

    def _initialize_state_space(self) -> None:
        """Initialize the state space and cache valid actions for each state"""
        self.state_space: Set[Tuple[int, int]] = {
            (i, j) 
            for i in range(self.MAX_BIKES_PER_LOCATION + 1)
            for j in range(self.MAX_BIKES_PER_LOCATION + 1)
        }
        
        # Cache valid actions for each state
        self._valid_actions_cache: Dict[Tuple[int, int], List[int]] = {}
        for state in self.state_space:
            self._valid_actions_cache[state] = self._compute_valid_actions(state)

    def initialize_values(self) -> None:
        """Initialize the value function for all states to 0"""
        self.values = {state: 0.0 for state in self.state_space}

    def initialize_policy(self) -> None:
        """Initialize the policy for all states to 0 (no bikes moved)"""
        self.policy = {state: 0 for state in self.state_space}

    @lru_cache(maxsize=None)
    def get_poisson_probability(self, n: int, lam: float) -> float:
        """
        Calculate and cache Poisson probability for n events with mean lam
        
        Args:
            n: Number of events
            lam: Expected value (mean) of the Poisson distribution
            
        Returns:
            float: Probability of exactly n events occurring
        """
        return poisson.pmf(n, lam)

    def _compute_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """
        Compute valid actions for a given state
        
        Args:
            state: Tuple of (bikes at location 1, bikes at location 2)
            
        Returns:
            List of valid actions (number of bikes that can be moved)
        """
        bikes_loc1, bikes_loc2 = state
        return [
            action for action in range(-self.MAX_BIKES_TO_MOVE, self.MAX_BIKES_TO_MOVE + 1)
            if (0 <= bikes_loc1 - action <= self.MAX_BIKES_PER_LOCATION and
                0 <= bikes_loc2 + action <= self.MAX_BIKES_PER_LOCATION)
        ]

    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """Get cached valid actions for a state"""
        return self._valid_actions_cache[state]

    def calculate_expected_return(self, state: Tuple[int, int], action: int) -> float:
      """
      Calculate expected return for a state-action pair using vectorized operations
      and probability thresholds for faster computation.
      """
      bikes_loc1, bikes_loc2 = state
      immediate_reward = -abs(action) * self.MOVING_COST
      
      # New state after moving bikes
      new_bikes_loc1 = bikes_loc1 - action
      new_bikes_loc2 = bikes_loc2 + action
      
      # Only consider the most likely request/return combinations
      # Use 99th percentile of Poisson distribution for truncation
      max_req1 = min(new_bikes_loc1 + 1, 
                     int(self.EXPECTED_REQUESTS[0] * 2 + 1))
      max_req2 = min(new_bikes_loc2 + 1, 
                     int(self.EXPECTED_REQUESTS[1] * 2 + 1))
      max_ret1 = int(self.EXPECTED_RETURNS[0] * 2 + 1)
      max_ret2 = int(self.EXPECTED_RETURNS[1] * 2 + 1)
      
      # Create probability arrays
      req1_probs = np.array([self.get_poisson_probability(n, self.EXPECTED_REQUESTS[0]) 
                            for n in range(max_req1)])
      req2_probs = np.array([self.get_poisson_probability(n, self.EXPECTED_REQUESTS[1]) 
                            for n in range(max_req2)])
      ret1_probs = np.array([self.get_poisson_probability(n, self.EXPECTED_RETURNS[0]) 
                            for n in range(max_ret1)])
      ret2_probs = np.array([self.get_poisson_probability(n, self.EXPECTED_RETURNS[1]) 
                            for n in range(max_ret2)])
      
      expected_reward = 0.0
      
      # Only process combinations with significant probability
      for req1, prob_req1 in enumerate(req1_probs):
          if prob_req1 < 1e-4:  # Increased threshold for faster computation
              continue
              
          actual_rentals_loc1 = min(new_bikes_loc1, req1)
          
          for req2, prob_req2 in enumerate(req2_probs):
              prob_reqs = prob_req1 * prob_req2
              if prob_reqs < 1e-4:
                  continue
                  
              actual_rentals_loc2 = min(new_bikes_loc2, req2)
              reward = (actual_rentals_loc1 + actual_rentals_loc2) * self.RENTAL_REVENUE
              
              # Combine returns processing
              for ret1, prob_ret1 in enumerate(ret1_probs[:max_ret1]):
                  if prob_ret1 < 1e-4:
                      continue
                      
                  next_bikes_loc1 = min(
                      self.MAX_BIKES_PER_LOCATION,
                      new_bikes_loc1 - actual_rentals_loc1 + ret1
                  )
                  
                  for ret2, prob_ret2 in enumerate(ret2_probs[:max_ret2]):
                      prob = prob_reqs * prob_ret1 * prob_ret2
                      if prob < 1e-4:
                          continue
                          
                      next_bikes_loc2 = min(
                          self.MAX_BIKES_PER_LOCATION,
                          new_bikes_loc2 - actual_rentals_loc2 + ret2
                      )
                      
                      next_state = (next_bikes_loc1, next_bikes_loc2)
                      expected_reward += prob * (reward + self.DISCOUNT_RATE * self.values[next_state])
      
      return immediate_reward + expected_reward

    def policy_iteration(self, max_iterations: int = 20, theta: float = 0.1) -> None:
        """
        Perform policy iteration with early stopping and fewer iterations
        """
        for i in range(max_iterations):
            # Policy evaluation with fewer iterations
            delta = 0
            for _ in range(3):  # Reduced number of evaluation iterations
                for state in self.state_space:
                    v = self.values[state]
                    action = self.policy[state]
                    self.values[state] = self.calculate_expected_return(state, action)
                    delta = max(delta, abs(v - self.values[state]))

            # Policy improvement
            policy_stable = True
            for state in self.state_space:
                old_action = self.policy[state]
                actions = self.get_valid_actions(state)

                # Calculate returns for all actions
                action_returns = np.array([
                    self.calculate_expected_return(state, action)
                    for action in actions
                ])

                best_action = actions[np.argmax(action_returns)]

                if best_action != old_action:
                    self.policy[state] = best_action
                    policy_stable = False

            if policy_stable or delta < theta:
                self.logger.info(f"Policy converged after {i+1} iterations")
                break

    def simulate_day(self) -> SimulationResult:
        """Optimized simulation of one day"""
        state = tuple(self.bikes_at_locations)
        action = self.policy.get(state, 0)

        # Move bikes overnight
        self.bikes_at_locations[0] -= action
        self.bikes_at_locations[1] += action
        moving_cost = abs(action) * self.MOVING_COST

        # Use truncated Poisson for faster simulation
        requests = [
            min(
                np.random.poisson(lam),
                int(lam * 2)  # Truncate at 2x mean
            ) for lam in self.EXPECTED_REQUESTS
        ]
        returns = [
            min(
                np.random.poisson(lam),
                int(lam * 2)  # Truncate at 2x mean
            ) for lam in self.EXPECTED_RETURNS
        ]

        # Process rentals
        rentals = [min(self.bikes_at_locations[i], requests[i]) for i in range(2)]
        rental_revenue = sum(rentals) * self.RENTAL_REVENUE

        # Update bike counts
        for i in range(2):
            self.bikes_at_locations[i] = min(
                self.MAX_BIKES_PER_LOCATION,
                self.bikes_at_locations[i] - rentals[i] + returns[i]
            )

        return SimulationResult(
            profit=rental_revenue - moving_cost,
            rentals=rentals,
            requests=requests,
            moves=action,
            final_state=tuple(self.bikes_at_locations)
        )

def run_simulation(days: int = 30) -> None:
    """
    Run the bike rental simulation for a specified number of days
    
    Args:
        days: Number of days to simulate
    """
    # Initialize system
    gbike = GBikeRental()
    
    # Perform policy iteration
    logging.info("Computing optimal policy using policy iteration...")
    gbike.policy_iteration()
    logging.info("Optimal policy computed!")
    
    # Simulate for specified days
    total_profit = 0
    logging.info(f"\nStarting simulation for {days} days...")
    
    for day in range(days):
        result = gbike.simulate_day()
        total_profit += result.profit
        
        logging.info(f"\nDay {day + 1}:")
        logging.info(f"Bikes moved overnight: {abs(result.moves)} "
                    f"({'1->2' if result.moves > 0 else '2->1' if result.moves < 0 else 'none'})")
        logging.info(f"Requests: Location 1: {result.requests[0]}, Location 2: {result.requests[1]}")
        logging.info(f"Actual rentals: Location 1: {result.rentals[0]}, Location 2: {result.rentals[1]}")
        logging.info(f"Current bikes: Location 1: {result.final_state[0]}, Location 2: {result.final_state[1]}")
        logging.info(f"Daily profit: INR {result.profit:.2f}")
    
    logging.info(f"\nTotal profit over {days} days: INR {total_profit:.2f}")
    logging.info(f"Average daily profit: INR {total_profit/days:.2f}")

if __name__ == "__main__":
    run_simulation()