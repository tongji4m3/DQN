
# from DQN.hierarchical_dqn.dqn import DqnAgent
from DQN.RL_brain import SumDQN
from DQN.my_env import Env
import numpy as np
import tensorflow as tf

class HDqnAgent:
    INTRINSIC_STEP_COST = -1  # Step cost for the controller.固有的每部的花销
    env = Env()
    def __init__(
            self,
            # env.n_actions, env.n_features,
            # learning_rate = 0.01,
            # reward_decay = 0.9,
            # e_greedy = 0.9,
            # replace_target_iter = 200,
            # memory_size = 10000,
            # double_q = True,
            # prioritized = True,
            # dueling = True,
            # meta_controller_state_fn=None,
            # check_subgoal_fn=None
    ):
        self.subgoals = np.array([[3, 3], [5, 5], [7, 7]]),
        self.num_subgoals = 0,
        self.epsilon = 0.9,
        self.sess = None,

        self.env = Env()
        self._meta_controller = SumDQN(self.env.n_actions, self.env.n_features,
                                       learning_rate=0.1,
                                       reward_decay=0.9,
                                       e_greedy=0.9,
                                       replace_target_iter=200,
                                       memory_size=10000,
                                       double_q=True,
                                       prioritized=True,
                                       dueling=True)

        tf.reset_default_graph()

        self._controller = SumDQN(self.env.n_actions, self.env.n_features,
                                       learning_rate=0.1,
                                       reward_decay=0.9,
                                       e_greedy=0.9,
                                       replace_target_iter=200,
                                       memory_size=10000,
                                       double_q=True,
                                       prioritized=True,
                                       dueling=True)

        self._subgoals = self.subgoals
        self.subgoal=np.array([3,3])
        self._num_subgoals = self.num_subgoals
        self.state=self.env.position
        self.terminal=np.array([9,9])
        self.epsilon=self.epsilon

        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())


    def get_subgoal(self):
        self.state=self.env.position
        if self.state[0]<3 or self.state[1]<3:
            self.subgoal=self._subgoals[0]

        elif self.state[0]>3 or self.state[1]>3:
            self.subgoal=self._subgoals[1]
        elif self.state[0]>5 or self.state[1]>5:
            self.subgoal=self._subgoals[2]
        else:
            self.subgoal=self.terminal

        return self.subgoal

    def up_reward(self,reward):
        reward+=50
        return reward


    def check_get_subgoal(self,state,reward):
        if state[0]==self.subgoal[0]:
            self.get_subgoal()
            reward=self.up_reward(reward)
            return True,reward
        else:
            return False,reward


    def choose_action(self,observation):
        action=self._controller.choose_action(observation)
        return action

    def store_transition(self, s, a, r, s_):
        self._controller.store_transition(s,a,r,s_)

    def learn(self):
        self._controller.learn()