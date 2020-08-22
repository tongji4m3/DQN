
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

        self.subgoals = self.env.minRoadMaze
        self.epsilon = 0.9,
        self.sess = None,

        self.env = Env()
        self._meta_controller = SumDQN(self.env.n_actions, self.env.n_features,
                                       self.subgoals,
                                       isMeta=True,
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
                                       self.subgoals,
                                       isMeta=False,
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
        self.state=self.env.position
        self.terminal=np.array([9,9])
        self.epsilon=self.epsilon

        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())


    def get_subgoal(self,observation):
        sub_goal=self._meta_controller.choose_subgoal(observation)

        sub_goal=np.array(sub_goal)
        return sub_goal



    def check_get_subgoal(self,observation,sub_goal):
        observation = observation[np.newaxis, :]
        sub_goal= sub_goal[np.newaxis, :]
        if observation[0].any()==sub_goal[0].any():
            return True,
        else:
            return False


    def choose_action(self,observation,sub_goal):
        action=self._controller.h_choose_action(observation,sub_goal)
        return action

    def store_transition(self, s,sg,a, r, s_):
        self._controller.Inh_store_transition(s,sg,a,r,s_)

    def meta_store_transition(self,s,sg,r,s_):
        self._meta_controller.Exh_store_transition(s,sg,r,s_)

    def learn(self):
        self._controller.learn()

    def meta_learn(self):
        self._meta_controller.learn()