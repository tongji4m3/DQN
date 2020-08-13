import numpy as np
import random

# 我们在不断地在系统中训练的过程中，会产生大量的训练数据。
# 虽然这些数据并不是应对当时环境最优的策略，但是是通过与环境交互得到的经验，
# 这对于我们训练系统是有非常大的帮助的。所以我们设置一个replay_buffer，
# 获得新的交互数据，抛弃旧的数据，
# 并且每次从这个replay_buffer中随机取一个batch，来训练我们的系统

# 在训练我们的Network之前，我们首先要进行多次实验，
# 将其state, action, reward, next_state, terminal这组数据存入Buffer中，
# 因为存的数据次序具有序列相关性，所以我们在训练的时候为了让每个trajectory相对独立，
# 于是便对存储的数据进行随机采样进行训练，这个思想就是Replay Buffer。
class ReplayBuffer(object):

    def __init__(self, max_size, init_size, batch_size):
        self.max_size = max_size
        self.init_size = init_size
        self.batch_size = batch_size

        self.states = np.array([None] * self.max_size)
        self.actions = np.array([None] * self.max_size)
        self.rewards = np.array([None] * self.max_size)
        self.next_states = np.array([None] * self.max_size)
        self.terminals = np.array([None] * self.max_size)

        self.curr_pointer = 0
        self.curr_size = 0

    # state，表示当时系统所面临的状态
    # action，表示我们的agent面临系统的状态时所做的行为
    # reward，表示agent做出了选择的行为之后从环境中获得的收益
    # next_state，表示agent做出了选择的行为，系统转移到的另外一个状态
    # terminal，表示这个epsiode有没有结束
    def add(self, state, action, reward, next_state, terminal):
        self.states[self.curr_pointer] = np.squeeze(state)
        self.actions[self.curr_pointer] = action
        self.rewards[self.curr_pointer] = reward
        self.next_states[self.curr_pointer] = np.squeeze(next_state)
        self.terminals[self.curr_pointer] = terminal

        self.curr_pointer += 1
        self.curr_size = min(self.max_size, self.curr_size + 1)
        # If replay buffer is full, set current pointer to be at the beginning of the buffer.
        if self.curr_pointer >= self.max_size:
            self.curr_pointer -= self.max_size

    #将五个数组随机化后输出,让每个trajectory相对独立,于是便对存储的数据进行随机采样进行训练
    def sample(self):
        if self.curr_size < self.init_size:
            return [], [], [], [], []
        sample_indices = []

        # 随机选取成为一个数组
        sample_indices.append(self.curr_pointer - 1)
        for i in range(self.batch_size - 1):
            sample_indices.append(random.randint(0, self.curr_size - 1))

        returned_states = []
        returned_actions = []
        returned_rewards = []
        returned_next_states = []
        returned_terminals = []

        for i in range(len(sample_indices)):
            index = sample_indices[i]
            returned_states.append(self.states[index])
            returned_actions.append(self.actions[index])
            returned_rewards.append(self.rewards[index])
            returned_next_states.append(self.next_states[index])
            returned_terminals.append(self.terminals[index])

        return np.array(returned_states), np.array(returned_actions), np.array(
            returned_rewards), np.array(returned_next_states), np.array(returned_terminals)