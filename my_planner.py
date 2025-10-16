import numpy as np
from multi_drone import MultiDroneUnc
import time

class MyPlanner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        self._a_param = a_param
        self._b_param = b_param

        # Initialising the values for each state
        # {All states have value 0 initially}
        self._V = {}

        # Initialising the policy
        self._policy = {}
    
    def _get_greedy_action(self, state: np.ndarray, num_actions: int, discount_factor: float):
        # Finding action a that maximises Q(s, a)
        a_greedy, Q_max, next_s_greedy = None, None, None
        for a in range(num_actions):
            next_s, reward, _, _ = self._env.simulate(state, a)

            # Initialising the value of the next state to 0 if it doesn't already have a value
            if tuple(next_s.flatten()) not in self._V.keys():
                self._V[tuple(next_s.flatten())] = 0
            
            cur_Q = reward + discount_factor * self._V[tuple(next_s.flatten())]
            # Updating the maximum Q(s,a)
            if Q_max == None or cur_Q > Q_max:
                Q_max = cur_Q
                a_greedy = a
                next_s_greedy = next_s
        return a_greedy, Q_max, next_s_greedy

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # represented by integer 0.
        s = current_state
        discount_factor = self._env.get_config().discount_factor
        num_actions = self._env.num_actions
        # start_time = time.time()
        while(np.any(s[:, :3] != self._env.get_config().goal_positions)):
            # Finding action a that maximises Q(s, a)
            a_greedy, Q_max, next_s_greedy = self._get_greedy_action(s, num_actions, discount_factor)
            # Updating the value and policy
            self._V[tuple(s.flatten())] = Q_max
            self._policy[tuple(s.flatten())] = a_greedy
            s = next_s_greedy
        
        # Computing action for current state
        a_greedy, Q_max, _ = self._get_greedy_action(current_state, num_actions, discount_factor)
        return a_greedy
