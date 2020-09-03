import random
import numpy as np
from HDQNtest.sumTree import Memory
import tensorflow as tf

default_meta_epsilon = 1.0
default_meta_memory_size = 1000000  # 高层经验池容量
default_meta_batch_size = 64  # 高层经验采样个数
default_meta_replace_target_iter = 1  # 高层网络参数更新频率

default_gamma = 0.95
default_tau = 0.001

# default_actor_epsilon = [1.0] * 6
default_actor_epsilon = [1.0] * 5
default_memory_size = 1000000  # 低层经验池容量
default_batch_size = 640  # 低层经验采样个数
default_replace_target_iter = 1   # 低层网络参数更新频率


class HDQN_TF:
    def __init__(self, n_actions=4, n_features=12, n_goals=5,
            learning_rate_h=0.0001, learning_rate_l=0.0001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=default_replace_target_iter, replace_target_iter_meta=default_meta_replace_target_iter, tau = 0.001,
            memory_size=default_memory_size, batch_size=default_batch_size,
            meta_memory_size=default_meta_memory_size, meta_batch_size=default_meta_batch_size,
                 actor_epsilon=default_actor_epsilon, meta_epsilon=default_meta_epsilon,
            e_greedy_increment=None, output_graph=False,
            double_q=True, prioritized=True, dueling=True, sess=None, soft_replacement=True, restore=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_goals = n_goals

        self.lr_h = learning_rate_h
        self.lr_l = learning_rate_l
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.target_tau = tau
        self.replace_target_iter = replace_target_iter
        self.replace_target_iter_meta = replace_target_iter_meta
        self.learn_step_counter = 0
        self.learn_step_counter_meta = 0

        self.goal_selected = np.ones(5)
        self.goal_success = np.zeros(5)

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.meta_memory_size = meta_memory_size
        self.meta_batch_size = meta_batch_size

        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.meta_epsilon = meta_epsilon
        self.actor_epsilon = actor_epsilon

        self.double_q = double_q  # decide to use double q or not
        self.prioritized = prioritized  # decide to use Prioritized Replay or not
        self.dueling = dueling      # decide to use dueling DQN or not
        
        self.restore = restore
        self._build_net()
        self._build_meta_net()
        
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        t_params_meta = tf.get_collection('meta_target_net_params')
        e_params_meta = tf.get_collection('meta_eval_net_params')

        self.soft_replacement = soft_replacement
        if self.soft_replacement:
            self.replace_target_op = [tf.assign(t, (1-self.target_tau)*t + self.target_tau*e) for t, e in zip(t_params, e_params)]
            self.replace_target_op_meta = [tf.assign(t, (1 - self.target_tau) * t + self.target_tau * e) for t, e in zip(t_params_meta, e_params_meta)]
        else:
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            self.replace_target_op_meta = [tf.assign(t, e) for t, e in zip(t_params_meta, e_params_meta)]

        # 根据 prioritized 来构建记忆库
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
            self.meta_memory = Memory(capacity=meta_memory_size)
        else:
            self.memory = np.zeros((self.memory_size, (self.n_features + self.n_goals) * 2 + 2))
            self.meta_memory = np.zeros((self.meta_memory_size, self.n_features * 2 + self.n_goals + 1))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []
        self.q = []  # 此处可以忽略
        # self.total_reward = 0  # 回合累计奖励
        
        if self.restore:
            print("loading parameters from exist model...")
            self.restore_model()

    def restore_model(self):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("尝试读取已有模型的参数")
        saver.restore(self.sess, "./HDQNtest/model2/hdqn_exp1.ckpt")
        print("读取模型参数")
        w1_info = tf.get_default_graph().get_tensor_by_name('eval_net/l1/w1:0')
        print(self.sess.run(w1_info))
        w1_meta_info = tf.get_default_graph().get_tensor_by_name('meta_eval_net/meta_l1/meta_w1:0')
        print(self.sess.run(w1_meta_info))


    # 构建低层DQN双网络
    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features+self.n_goals, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names, trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling: # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names, trainable=trainable)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names, trainable=trainable)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names, trainable=trainable)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names, trainable=trainable)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names, trainable=trainable)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names, trainable=trainable)
                    out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ 低层估计网络 Evaluate Net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features+self.n_goals], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr_l).minimize(self.loss)

        # ------------------ 低层目标网络 Target Net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features+self.n_goals], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)


    # 构建高层DQN双网络
    def _build_meta_net(self):
        def build_meta_layers(s_meta, c_names_meta, n_l1_meta, w_initializer, b_initializer, trainable):
            with tf.variable_scope('meta_l1'):
                w1 = tf.get_variable('meta_w1', [self.n_features, n_l1_meta], initializer=w_initializer, collections=c_names_meta, trainable=trainable)
                b1 = tf.get_variable('meta_b1', [1, n_l1_meta], initializer=b_initializer, collections=c_names_meta, trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s_meta, w1) + b1)

            if self.dueling: # Dueling DQN
                with tf.variable_scope('meta_Value'):
                    w2 = tf.get_variable('meta_w2', [n_l1_meta, 1], initializer=w_initializer, collections=c_names_meta, trainable=trainable)
                    b2 = tf.get_variable('meta_b2', [1, 1], initializer=b_initializer, collections=c_names_meta, trainable=trainable)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('meta_Advantage'):
                    w2 = tf.get_variable('meta_w2', [n_l1_meta, self.n_goals], initializer=w_initializer, collections=c_names_meta, trainable=trainable)
                    b2 = tf.get_variable('meta_b2', [1, self.n_goals], initializer=b_initializer, collections=c_names_meta, trainable=trainable)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('meta_Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('meta_Q'):
                    w2 = tf.get_variable('meta_w2', [n_l1_meta, self.n_goals], initializer=w_initializer, collections=c_names_meta, trainable=trainable)
                    b2 = tf.get_variable('meta_b2', [1, self.n_goals], initializer=b_initializer, collections=c_names_meta, trainable=trainable)
                    out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ 高层估计网络 Meta Evaluate Net ------------------
        self.s_meta = tf.placeholder(tf.float32, [None, self.n_features], name='s_meta')  # 当前状态
        self.q_target_meta = tf.placeholder(tf.float32, [None, self.n_goals], name='Q_target_meta')  # Q-真实值
        if self.prioritized:
            self.ISWeights_meta = tf.placeholder(tf.float32, [None, 1], name='IS_weights_meta')
        with tf.variable_scope('meta_eval_net'):
            c_names_meta, n_l1_meta, w_initializer, b_initializer = ['meta_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                                                          tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            self.q_eval_meta = build_meta_layers(self.s_meta, c_names_meta, n_l1_meta, w_initializer, b_initializer, True)  # 由网络得出Q-预测值

        with tf.variable_scope('meta_loss'):
            if self.prioritized:
                self.abs_errors_meta = tf.reduce_sum(tf.abs(self.q_target_meta - self.q_eval_meta), axis=1)
                self.loss_meta = tf.reduce_mean(self.ISWeights_meta * tf.squared_difference(self.q_target_meta, self.q_eval_meta))
            else:
                self.loss_meta = tf.reduce_mean(tf.squared_difference(self.q_target_meta, self.q_eval_meta))
        with tf.variable_scope('meta_train'):
            self._train_op_meta = tf.train.RMSPropOptimizer(self.lr_h).minimize(self.loss_meta)

        # ------------------ 高层目标网络 Meta Target Net ------------------
        self.s_meta_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_meta_')  # 下一状态的值
        with tf.variable_scope('meta_target_net'):
            c_names_meta = ['meta_target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next_meta = build_meta_layers(self.s_meta_, c_names_meta, n_l1_meta, w_initializer, b_initializer, False)  # 由网络得出的下一时刻的Q-预测值

    # 低层根据目标选动作
    def select_move(self, state, goal, goal_value):
        if random.random() < self.actor_epsilon[goal_value - 1]:
            action = random.choice([i for i in range(self.n_actions)])
        else:
            vector = np.concatenate([state, goal], axis=1)  # vector:(1,18)
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: vector})
            action = np.argmax(actions_value)
        return action

    # 高层根据状态选目标
    def select_goal(self, state):
        if random.random() < self.meta_epsilon:
            goal = random.choice([i+1 for i in range(self.n_goals)])
        else:
            vector = np.array(state).reshape((1, self.n_features))  # vector:(1,12)
            goals_value = self.sess.run(self.q_eval_meta, feed_dict={self.s_meta: vector})
            goal = np.argmax(goals_value) + 1
        return goal


    # 保存经验
    def store(self, experience, meta=False):
        # 说明：传入的experience.state等参数均为二维向量，size=(1,x)，故使用hstack前需要先降维
        if self.prioritized:
            if meta:  # 高层
                transition = np.hstack((np.squeeze(experience.state), np.squeeze(experience.goal), [experience.reward], np.squeeze(experience.next_state)))  # 一维向量的水平方向堆叠，最后得到一个一维向量
                self.meta_memory.store(transition)
            else:  # 低层
                transition = np.hstack((np.squeeze(experience.state), np.squeeze(experience.goal), [experience.action, experience.reward], np.squeeze(experience.next_state), np.squeeze(experience.goal)))
                self.memory.store(transition)
        else:
            if meta:
                if not hasattr(self, 'meta_memory_counter'):
                    self.meta_memory_counter = 0
                    transition = np.hstack((np.squeeze(experience.state), np.squeeze(experience.goal),
                                            [experience.reward], np.squeeze(experience.next_state)))
                meta_index = self.meta_memory_counter % self.meta_memory_size
                self.meta_memory[meta_index, :] = transition
                self.meta_memory_counter += 1
            else:
                if not hasattr(self, 'memory_counter'):
                    self.memory_counter = 0
                    transition = np.hstack((np.squeeze(experience.state), np.squeeze(experience.goal),
                                            [experience.action, experience.reward], np.squeeze(experience.next_state),
                                            np.squeeze(experience.goal)))
                index = self.emory_counter % self.memory_size
                self.memory[index, :] = transition
                self.memory_counter += 1

    # 低层更新参数
    def learn(self):
        # 判断是否需要更新低层 target 网络
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # 采样经验池
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        # 计算部分
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -(self.n_features+self.n_goals):],  # next observation
                       self.s: batch_memory[:, -(self.n_features+self.n_goals):]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :(self.n_features+self.n_goals)]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_action_index = batch_memory[:, self.n_features + self.n_goals].astype(int)   # 获取action
        reward = batch_memory[:, self.n_features + self.n_goals + 1]  # 获取 reward

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)  # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        # print(q_target[batch_index,:])  # 输出batch_size条，每条长度为n_actions，其实就是为每个action评分
        q_target[batch_index, eval_action_index] = reward + self.gamma * selected_q_next

        # 训练，优化损失函数
        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features + self.n_goals],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)  # 误差曲线
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max  # epsilon递增，此处不需要
        self.learn_step_counter += 1


    # 高层更新参数
    def meta_learn(self):
        # 判断是否需要更新高层 Target 网络
        if self.learn_step_counter_meta % self.replace_target_iter_meta == 0:  # 高层网络更新速度应该慢一些？
            self.sess.run(self.replace_target_op_meta)
            # print('\nmeta_target_params_replaced\n')

        # 采样经验池
        if self.prioritized:
            tree_idx_meta, batch_memory_meta, ISWeights_meta = self.meta_memory.sample(self.meta_batch_size)
        else:
            sample_index_meta = np.random.choice(self.meta_memory_size, size=self.meta_batch_size)
            batch_memory_meta = self.meta_memory[sample_index_meta, :]

        # 计算部分
        q_next_meta, q_eval4next_meta = self.sess.run(
            [self.q_next_meta, self.q_eval_meta],
            feed_dict={self.s_meta_: batch_memory_meta[:, -self.n_features:],  # next observation
                       self.s_meta: batch_memory_meta[:, -self.n_features:]})  # next observation
        q_eval_meta = self.sess.run(self.q_eval_meta, {self.s_meta: batch_memory_meta[:, :self.n_features]})

        q_target_meta = q_eval_meta.copy()

        batch_index_meta = np.arange(self.meta_batch_size, dtype=np.int32)
        eval_action_index_meta = batch_memory_meta[:, self.n_features].astype(int)  # 获取goal，即高层的action
        reward_meta = batch_memory_meta[:, self.n_features + 1]  # 获取 reward

        if self.double_q:
            max_act4next_meta = np.argmax(q_eval4next_meta, axis=1)  # the action that brings the highest value is evaluated by q_eval
            selected_q_next_meta = q_next_meta[batch_index_meta, max_act4next_meta]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next_meta = np.max(q_next_meta, axis=1)  # the natural DQN

        q_target_meta[batch_index_meta, eval_action_index_meta] = reward_meta + self.gamma * selected_q_next_meta

        # 训练，优化损失函数
        if self.prioritized:
            _, abs_errors_meta, self.cost_meta = self.sess.run([self._train_op_meta, self.abs_errors_meta, self.loss_meta],
                                                     feed_dict={
                                                         self.s_meta: batch_memory_meta[:, :self.n_features],
                                                         self.q_target_meta: q_target_meta,
                                                         self.ISWeights_meta: ISWeights_meta})
            self.meta_memory.batch_update(tree_idx_meta, abs_errors_meta)  # update priority
        else:
            _, self.cost_meta = self.sess.run([self._train_op_meta, self.loss_meta],
                                         feed_dict={self.s_meta: batch_memory_meta[:, :self.n_features],
                                                    self.q_target_meta: q_target_meta})

        self.cost_his.append(self.cost_meta)  # 误差曲线
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max  # epsilon递增，此处不需要
        self.learn_step_counter_meta += 1


    def calc_cost(self):
        avg_cost = np.mean(self.cost_his)
        self.cost_his.clear()
        return avg_cost
        
    
    def save_net(self, filename):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filename)
        print("Save to path: ", save_path)
        w1_info = tf.get_default_graph().get_tensor_by_name('eval_net/l1/w1:0')
        print(self.sess.run(w1_info))
        w1_meta_info = tf.get_default_graph().get_tensor_by_name('meta_eval_net/meta_l1/meta_w1:0')
        print(self.sess.run(w1_meta_info))
        
        
        
        
