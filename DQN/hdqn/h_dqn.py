
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
        self.subgoals1=([0.25,0.25],[1.25,1.25])

        self.epsilon = 0.9,
        self.sess = None,

        self.env = Env()
        self.controller = SumDQN(self.env.n_actions, self.env.n_features,
                                       self.subgoals,
                                       learning_rate_h=0.1,
                                       learning_rate_l=0.0025,
                                       reward_decay=0.9,
                                       e_greedy=0.9,
                                       replace_target_iter=200,
                                       memory_size=10000,
                                       double_q=True,
                                       prioritized=True,
                                       dueling=True)

        tf.reset_default_graph()

        # self._controller = SumDQN(self.env.n_actions, self.env.n_features,
        #                                self.subgoals,
        #                                isMeta=False,
        #                                learning_rate=0.1,
        #                                reward_decay=0.9,
        #                                e_greedy=0.9,
        #                                replace_target_iter=200,
        #                                memory_size=10000,
        #                                double_q=True,
        #                                prioritized=True,
        #                                dueling=True)

        self._subgoals = self.subgoals
        self.subgoal=np.array([3,3])
        self.state=self.env.position
        self.terminal=np.array([9,9])
        self.epsilon=self.epsilon

        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())


    def get_subgoal(self,observation):
        # sub_goal=self._meta_controller.choose_subgoal(observation)
        sub_goal = self.controller.choose_subgoal(observation)
        sub_goal=np.array(sub_goal)
        return sub_goal

    def get_subgoal1(self,observation):
        if observation[0]<0.25 and observation[1]<0.25:
            sub_goal=self.subgoals1[0]
        elif observation[0]<1.25 and observation[1]<1.25:
            sub_goal = self.subgoals1[1]
        else :
            sub_goal=[1.75,1.75]

        sub_goal=np.array(sub_goal)
        return sub_goal


    def check_get_subgoal(self,observation,sub_goal):

        if observation[0]==sub_goal[0] and observation[1]==sub_goal[1]:
            return True
        else:
            return False


    def choose_action(self,observation,sub_goal):
        # action=self._controller.h_choose_action(observation,sub_goal)
        action = self.controller.h_choose_action(observation, sub_goal)

        return action

    def store_transition(self, s,sg,a, r, s_):
        self.controller.Inh_store_transition(s,sg,a,r,s_)

    def meta_store_transition(self,s,sg,r,s_):
        self.controller.Exh_store_transition(s,sg,r,s_)

    def learn(self,sub_goal):
        sub_goal=sub_goal[np.newaxis, :]
        self.controller.learn(sub_goal)

    def meta_learn(self,sub_goal):
        sub_goal=sub_goal[np.newaxis, :]
        self.controller.meta_learn(sub_goal)