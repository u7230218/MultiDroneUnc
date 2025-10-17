import numpy as np
from multi_drone import MultiDroneUnc
import time
import math

class Node:
    def __init__(self, env: MultiDroneUnc, state: np.ndarray, parent=None):
        self.state = state
        self.parent = parent
        self.children = {} # a -> [N(s,a), s']
        self.visits = 0
        self.value = 0.0
        self.remaining_actions = list(range(env.num_actions))
    
    def __str__(self):
        node_string = f"Node:(state='{self.state}', Value={self.value}), Visits={self.visits}, Remaining_actions={self.remaining_actions}, Children=["
        for a in self.children.keys():
            node_string += f"(action={a}, (n_s_a={self.children[a][0]}, {self.children[a][1]}))"
        node_string = node_string[:-2] + "]"
        return node_string

class MCTSPlanner:
    def __init__(self, env: MultiDroneUnc, c: float = 1.0, rollout_lookahead = 10):
        self._env = env
        self.c = c
        self.rollout_lookahead = rollout_lookahead

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # representen by integer 0.
        root = Node(self._env, current_state)
        discount_factor = self._env.get_config().discount_factor
        num_actions = self._env.num_actions

        start_time = time.time()
        while (time.time() - start_time <= planning_time_per_step):
            cur_node = root
            action_history=[]

            # Selection
            prev_node = None
            while cur_node:
                max_ucb = None # The greatest upper confidence bound
                selected_a = None # Selected action
                cur_node_previous_actions = cur_node.children.keys()
                for a in range(num_actions):
                    n_s_a = 0
                    child_value = 0
                    if a in cur_node_previous_actions:
                        n_s_a = cur_node.children[a][0]
                        child_value = cur_node.children[a][1].value
                    ucb = child_value + self.c * math.sqrt(math.log(cur_node.visits + 1) / (n_s_a + 1))
                    if max_ucb == None or ucb > max_ucb:
                        max_ucb = ucb
                        selected_a = a
                action_history.append(selected_a)
                prev_node = cur_node
                if selected_a in cur_node_previous_actions:
                    cur_node = cur_node.children[selected_a][1] # Reached a terminal action node in Monte Carlo search tree
                else:
                    cur_node = None


            # Expansion
            action = action_history[-1]
            next_s, reward, _, _ = self._env.simulate(prev_node.state, action)
            cur_node = Node(self._env, next_s, prev_node)
            prev_node.children[action] = [0, cur_node]

            # Rollout
            estimated_total_reward = reward
            cur_state = cur_node.state
            cur_discount = discount_factor
            for _ in range(self.rollout_lookahead):
                a_greedy, next_s_greedy, greedy_signal = None, None, None
                a_greedy = np.random.randint(self._env.num_actions)
                next_s_greedy, reward, greedy_signal, _ = self._env.simulate(cur_state, a_greedy)
                estimated_total_reward += reward + cur_discount * estimated_total_reward

                if greedy_signal:
                    break
                cur_discount *= discount_factor
                cur_state = next_s_greedy
            
            # Backpropagation
            while cur_node is not None:
                cur_node.value += estimated_total_reward
                cur_node.visits += 1
                if len(action_history) > 0:
                    action = action_history.pop()
                    cur_node.parent.children[action][0] += 1
                cur_node = cur_node.parent
            # print("\n")
            # print(root)
            # print("\n")
        
        # Picking best action to perform
        best_a = None
        best_value = None
        for a in root.children.keys():
            child = root.children[a][1]
            if best_value == None or child.value > best_value:
                best_a = a
                best_value = child.value
        return best_a
