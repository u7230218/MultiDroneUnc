import numpy as np
from multi_drone import MultiDroneUnc
import time
import random

class RTDPPlanner:
    def __init__(self, env: MultiDroneUnc, action_sample_size=None):
        self._env = env

        # Initialising the values for each state
        # {All states have value 0 initially}
        self._V = {}
        self.action_sample_size = action_sample_size
    
    def _get_greedy_action(self, state: np.ndarray, discount_factor: float):
        # Finding action a that maximises Q(s, a)
        a_greedy, Q_max, next_s_greedy, greedy_signal = None, None, None, None
        num_actions = self._env.num_actions

        if self.action_sample_size is not None and self.action_sample_size < num_actions:
            relevant_actions = random.sample(range(num_actions), self.action_sample_size)
        else:
            relevant_actions = range(num_actions)

        for a in relevant_actions:
            next_s, reward, terminal_signal, _ = self._env.simulate(state, a)

            # Initialising the value of the next state to 0 if it doesn't already have a value
            if tuple(next_s.flatten()) not in self._V.keys():
                self._V[tuple(next_s.flatten())] = 0
            
            cur_Q = reward + discount_factor * self._V[tuple(next_s.flatten())]
            # Updating the maximum Q(s,a)
            if Q_max == None or cur_Q > Q_max:
                Q_max = cur_Q
                a_greedy = a
                next_s_greedy = next_s
                greedy_signal = terminal_signal
        return a_greedy, Q_max, next_s_greedy, greedy_signal

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # represented by integer 0.

        # Getting starting positions
        s = current_state

        discount_factor = self._env.get_config().discount_factor
        start_time = time.time()
        while(time.time() - start_time <= planning_time_per_step):
            # Finding action a that maximises Q(s, a)
            a_greedy, Q_max, next_s_greedy, greedy_signal = self._get_greedy_action(s, discount_factor)
            # Updating the value and policy
            self._V[tuple(s.flatten())] = Q_max
            s = next_s_greedy

            # Encountered a termina signal in best greedy action
            if greedy_signal:
                break
        
        # Computing action for current state
        a_greedy, _, _, _ = self._get_greedy_action(current_state, discount_factor)
        return a_greedy
