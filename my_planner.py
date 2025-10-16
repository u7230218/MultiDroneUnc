import numpy as np
from multi_drone import MultiDroneUnc

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

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # represented by integer 0.
        # TODO: Implement RTDP
        s = current_state
        while(np.any(s[:, :3] != self._env.get_config().goal_positions)):
            # Finding action a that maximises Q(s, a)
            a_greedy = None
            Q_max = None
            next_s_greedy = None
            for a in range(self._env.num_actions):
                next_s, reward, _, _ = self._env.simulate(s, a)

                # Initialising the value of the next state to 0 if it doesn't already have a value
                if tuple(next_s.flatten()) not in self._V.keys():
                    self._V[tuple(next_s.flatten())] = 0
                
                cur_Q = reward + self._env.get_config().discount_factor * self._V[tuple(next_s.flatten())]
                # Updating the maximum Q(s,a)
                if Q_max == None or cur_Q > Q_max:
                    Q_max = cur_Q
                    a_greedy = a
                    next_s_greedy = next_s
            # Updating the value and policy
            self._V[tuple(s.flatten())] = Q_max
            self._policy[tuple(s.flatten())] = a_greedy
            s = next_s_greedy
        return self._policy[tuple(current_state.flatten())]
