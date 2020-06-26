import numpy as np
import time
import sys
import matplotlib.pyplot as plt


class Env:
    def __init__(self):
        self.n_actions = 4  # 共有四个动作:上下左右 0132
        self.n_features = 2  # 二维的
        # 二维迷宫

        self.maze = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        )
        self.m = self.maze.shape[0]
        self.n = self.maze.shape[1]

        self.position = np.array([0, 0])  # 起始位置
        self.fresh_time = 0.01  # 刷新速度
        self.array_x = []
        self.array_y = []
        self.array_z = []

    # 重新初始化位置参数
    def reset(self):
        self.position = np.array([0, 0])  # 起始位置

        self.maze = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        )

        return np.array([-0.5, -0.5])  # observation采取的坐标不太一样

    # def create_maze(self):
    #
    #     maze1 = np.array(
    #         [
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 0]
    #         ]
    #     )
    #     string=""
    #     for i in range(len(self.maze)):
    #         for j in range(len((self.maze[0]))):
    #             if (self.maze[i][j] == -1):
    #
    #                 string = string+"#"
    #
    #             elif (self.maze[i][j] == 1):
    #                 # print("R", end=" ")
    #
    #                 string = string + "R"
    #             elif (self.maze[i][j] == 2):
    #                 # print("*", end=" ")
    #
    #                 string = string + "*"
    #             else:
    #                 # print("0", end=" ")
    #
    #                 string = string + "0"
    #         string = string+" "
    #         #print()
    #         # sys.stdout.write('\r'+'\n')
    #         # sys.stdout.flush()
    #     #print()
    #     # sys.stdout.write('\r'+'\n')
    #     # sys.stdout.flush()
    #     return string
    #
    # def print_ground(self,string):
    #
    #     for i in range(len(string)):
    #         # 方式1
    #         if(i==19):
    #            print('\r' + string[i], end=" ", flush=True)
    #
    #         else:
    #            print(string[i],end=" ")

    def update_env(self, episode, step_counter, done, reward):

        if done:
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), '')
            # if reward==1:
            #     print("find reward!")
            #     self.Reward+=0.3
            # else:
            #     print("not find!")
            #     self.Reward-=0.3

            print('\r                           ', '')

            time.sleep(self.fresh_time)
        # if(episode>200):
        #     for i in range(len(self.maze)):
        #         for j in range(len((self.maze[0]))):
        #             if(self.position[0]==i and self.position[1]==j):
        #                 print("0", end=" ")
        #             elif (self.maze[i][j] == 0):
        #                 print("_", end=" ")
        #             elif (self.maze[i][j] == 1):
        #                 print("*", end=" ")
        #             else:
        #                 print(self.maze[i][j], end=" ")
        #         print()
        #     print()

        # 遍历打印二维数组
        # string1=self.create_maze()
        # self.print_ground(string1)
        # for i in range(len(self.maze)):
        #     for j in range(len((self.maze[0]))):
        #         if (self.maze[i][j] == -1):
        #             sys.stdout.write("#")
        #         elif (self.maze[i][j] == 1):
        #             #print("R", end=" ")
        #             sys.stdout.write("R")
        #         elif (self.maze[i][j] == 2):
        #             #print("*", end=" ")
        #             sys.stdout.write("*")
        #         else:
        #             #print("0", end=" ")
        #             sys.stdout.write("0")
        #
        #     print()
        #     #sys.stdout.write('\r'+'\n')
        #     #sys.stdout.flush()
        # print()
        # #sys.stdout.write('\r'+'\n')
        # #sys.stdout.flush()

    # 更新位置,获取奖励

    def step(self, action):
        # 0 上   1 下  3 左  2 右
        # 注意左右的定义不同
        # 根据动作,更新当前的位置 并且注意越界问题
        # 目前的位置变为空了
        # self.maze[self.position[0]][self.position[1]]=0
        if action == 0:
            if self.position[0] > 0:
                self.position[0] -= 1
        elif action == 1:
            if self.position[0] < self.m - 1:
                self.position[0] += 1
        elif action == 3:
            if self.position[1] > 0:
                self.position[1] -= 1
        elif action == 2:
            if self.position[1] < self.n - 1:
                self.position[1] += 1

        # 根据情况判断获得的奖励 以及是否结束
        if self.maze[self.position[0]][self.position[1]] == 1:
            reward = 1
            done = True
        # elif  self.maze[self.position[0]][self.position[1]]==-1:
        #     reward = -1
        #     done = True
        elif self.maze[self.position[0]][self.position[1]] <= -1:
            reward = self.maze[self.position[0]][self.position[1]] / 2
            done = False
        else:
            reward = 0
            done = False

        # 当前位置 之后再更新,不然会覆盖奖励
        # self.maze[self.position[0]][self.position[1]] = 2

        # -0.5 -0.5 0 0.25   m
        # -0.5
        #   0
        # 0.25
        #   n

        # 从position到observation_的坐标转换
        observation_ = np.array([-0.5 + self.position[1] * 0.25, -0.5 + self.position[0] * 0.25])
        return observation_, reward, done

    def Reward_memory(self, iterations, reward_sum, total_step):
        # 记录每一个点
        self.array_x.append(iterations)
        self.array_y.append(reward_sum)
        self.array_z.append(total_step)
        # self.array_y.append(self.Reward)
        self.Reward = 0  # 重置

    def draw(self):

        # x=np.linspace(0,300,3)
        # y=self.Reward/x 20回合的奖励

        # for i in range(len(self.array_x)):
        #     plt.scatter(self.array_x[i],self.array_y[i],color='black')

        # 画奖励曲线
        plt.plot(self.array_x, self.array_y)
        plt.xlim((0, 100))
        plt.ylim((-1.5, 1.5))
        plt.ylabel('Avg Reward')
        plt.xlabel('Iterations')
        plt.show()

        # # 画步数
        # plt.plot(self.array_x, self.array_z)
        # plt.xlim((0, 50))
        # plt.ylim((0,100))
        # plt.ylabel('Avg Steps')
        # plt.xlabel('Iterations')
        # plt.show()
