"""
Hierarchical DQN implementation as described in Kulkarni et al.
https://arxiv.org/pdf/1604.06057.pdf
@author: Saurabh Kumar
"""

# from DQN.hierarchical_dqn.dqn import DqnAgent
from DQN.RL_brain import SumDQN
from DQN.my_env import Env
import numpy as np


class HierarchicalDqnAgent(object):
    INTRINSIC_STEP_COST = -1    # Step cost for the controller.固有的每部的花销

    def __init__(self,
                 learning_rates=[0.1, 0.00025],
                 state_sizes=[0, 0],
                 subgoals=np.array([5,5],[7,7]),
                 num_subgoals=0,
                 num_primitive_actions=0,
                 meta_controller_state_fn=None,
                 check_subgoal_fn=None):

        """Initializes a hierarchical DQN agent.

           Args:
            learning_rates: learning rates of the meta-controller and controller agents.
            state_sizes: state sizes of the meta-controller and controller agents.
                         State sizes are assumed to be 1-dimensional.
            subgoals: array of subgoals for the meta-controller.
            num_subgoals: the action space of the meta-controller.
            num_primitive_actions: the action space of the controller.
            meta_controller_state_fn: function that returns the state of the meta-controller.
            check_subgoal_fn: function that checks if agent has satisfied a particular subgoal.
            learning_rates：元控制器和控制器代理的学习率。
             state_sizes：元控制器和控制器代理的状态大小。
                          状态大小假定为一维。
             subgoals：元控制器的子目标数组。
             num_subgoals：元控制器的操作空间。
             num_primitive_actions：控制器的动作空间。
             meta_controller_state_fn：返回元控制器状态的函数。
             check_subgoal_fn：用于检查代理是否满足特定子目标的函数。
        """
        env = Env()
        self._meta_controller = SumDQN(env.n_actions, env.n_features,
        learning_rate = learning_rates[0],
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 200,
        memory_size = 10000,
        double_q = True,
        prioritized = True,
        dueling = True)

        self._controller = SumDQN(env.n_actions, env.n_features,
        learning_rate = learning_rates[1],
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 200,
        memory_size = 10000,
        double_q = True,
        prioritized = True,
        dueling = True)

        self._subgoals = subgoals
        self._num_subgoals = num_subgoals

        self._meta_controller_state_fn = meta_controller_state_fn
        self._check_subgoal_fn = check_subgoal_fn

        self._meta_controller_state = None
        self._curr_subgoal = None
        self._meta_controller_reward = 0
        self._intrinsic_time_step = 0
        self._episode = 0


    # 获取状态 meta控制子目标 controller控制动作
    def get_meta_controller_state(self, state):
        returned_state = state
        if self._meta_controller_state_fn:
            returned_state = self._meta_controller_state_fn(state, self._original_state)

        return np.copy(returned_state)

    # 获取
    def get_controller_state(self, state, subgoal_index):
        # Concatenates the environment state with the current subgoal.将环境状态与当前子目标并置。

        # curr_subgoal is a 1-hot vector indicating the current subgoal selected by the meta-controller.
        # curr_subgoal是1-hot向量，指示由元控制器选择的当前子目标。
        curr_subgoal = np.array(self._subgoals[subgoal_index])

        # Concatenate the environment state with the subgoal.将环境状态和子目标连接在一起
        controller_state = np.array(state)
        controller_state = np.concatenate((controller_state, curr_subgoal), axis=0)

        return np.copy(controller_state)

    # 针对低层的controller
    def intrinsic_reward(self, state, subgoal_index):
        # Intrinsically rewards the controller - this is the critic in the h-DQN algorithm.
        if self.subgoal_completed(state, subgoal_index):
            return 1
        else:
            return self.INTRINSIC_STEP_COST

    def subgoal_completed(self, state, subgoal_index):
        # Checks whether the controller has completed the currently specified subgoal.
        # 检查控制器是否已完成当前指定的子目标。
        if self._check_subgoal_fn is None:
            return state == self._subgoals[subgoal_index]
        else:
            return self._check_subgoal_fn(state, subgoal_index)

    # 每次store大概都获得下一次的subgoal
    def store(self, state, action, reward, next_state, terminal, eval=False):
        """Stores the current transition in replay memory.
           The transition is stored in the replay memory of the controller.
           If the transition culminates in a subgoal's completion or a terminal state, a
           transition for the meta-controller is constructed and stored in its replay buffer.

           Args:
            state: current state
            action: primitive action taken
            reward: reward received from state-action pair
            next_state: next state
            terminal: extrinsic terminal (True or False)
            eval: Whether the current episode is a train or eval episode.
            将当前过渡存储在重播内存中。
            转换存储在控制器的重放存储器中。
            如果过渡最终达到子目标的完成或最终状态，则
            元控制器的过渡已构建并存储在其重播缓冲区中。

             状态：当前状态
             行动：采取的原始行动
             奖励：从国家行动对获得的奖励
             next_state：下一个状态
             终端：外部终端（真或假）
             评估：当前情节是火车还是评估情节。
        """

        # Compute the controller state, reward, next state, and terminal.计算控制器状态，奖励，下一个状态和终端。
        # 现有状态和下一个状态
        intrinsic_state = np.copy(self.get_controller_state(state, self._curr_subgoal))
        intrinsic_next_state = np.copy(self.get_controller_state(next_state, self._curr_subgoal))
        # 现有的奖励
        intrinsic_reward = self.intrinsic_reward(next_state, self._curr_subgoal)
        # 全员布尔
        subgoal_completed = self.subgoal_completed(next_state, self._curr_subgoal)
        # 整体目标是否到达，terminal指的是最终目标点
        intrinsic_terminal = subgoal_completed or terminal

        # Store the controller transition in memory.存储控制器传递
        # dqn里的store
        self._controller.store_transition(intrinsic_state, action,
            intrinsic_reward, intrinsic_next_state)

        self._meta_controller_reward += reward

        if terminal and not eval:
            self._episode += 1

        # 如果子目标到达或者到达最终的重点
        if subgoal_completed or terminal:

            # 存进去
            # Store the meta-controller transition in memory.
            meta_controller_state = np.copy(self._meta_controller_state)
            next_meta_controller_state = np.copy(self.get_meta_controller_state(next_state))
            
            self._meta_controller.store_transition(meta_controller_state, self._curr_subgoal,
                self._meta_controller_reward, next_meta_controller_state,)

            # 重新初始
            # Reset the current meta-controller state and current subgoal to be None
            # since the current subgoal is finished. Also reset the meta-controller's reward.
            self._meta_controller_state = None
            self._curr_subgoal = None
            self._meta_controller_reward = 0
            self._intrinsic_time_step = 0

    # 采样
    def sample(self, state):
        """Samples an action from the hierarchical DQN agent.
           Samples a subgoal if necessary from the meta-controller and samples a primitive action原语
           from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: a sampled primitive action.
            从分层DQN代理中采样操作。
            如有必要，从元控制器中采样一个子目标，并采样一个原始动作
            从控制器。

            参数：
             状态：当前环境状态。

            返回值：
             动作：采样的原始动作。
        """
        self._intrinsic_time_step += 1

        # If the meta-controller state is None, it means that either this is a new episode 
        # or a subgoal has just been completed.
        if self._meta_controller_state is None:
            self._meta_controller_state = self.get_meta_controller_state(state)
            # metacontroller根据当前位置选择了一个子目标
            self._curr_subgoal = self._meta_controller.sample([self._meta_controller_state])

        # 把当前位置和子目标同时考虑
        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.choose_action(controller_state)

        return action

    # def best_action(self, state):
    #     """Returns the greedy action from the hierarchical DQN agent.
    #        Gets the greedy subgoal if necessary from the meta-controller and gets
    #        the greedy primitive action from the controller.
    #
    #        Args:
    #         state: the current environment state.
    #
    #        Returns:
    #         action: the controller's greedy primitive action.
    #     """
    #
    #     # If the meta-controller state is None, it means that either this is a new episode
    #     # or a subgoal has just been completed.
    #     if self._meta_controller_state is None:
    #         self._meta_controller_state = self.get_meta_controller_state(state)
    #         self._curr_subgoal = self._meta_controller.best_action([self._meta_controller_state])
    #
    #     controller_state = self.get_controller_state(state, self._curr_subgoal)
    #     action = self._controller.best_action(controller_state)
    #     return action

    # def update(self):
    #     self._controller.update()
    #     # Only update meta-controller right after a meta-controller transition has taken place,
    #     # which occurs only when either a subgoal has been completed or the agnent has reached a
    #     # terminal state.
    #     if self._meta_controller_state is None:
    #         self._meta_controller.update()
