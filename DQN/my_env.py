import numpy as np
import time
import random
import matplotlib.pyplot as plt
import DQN.utils.matrix as mt

class Env:
    def __init__(self):
        self.flag=2   #判断是否要更改地图
        self.mt1=mt.Matrix()#导入图类
        self.n_actions = 4  # 共有四个动作:上下左右 0132
        self.n_features = 2  # 二维的
        # 二维迷宫
        self.maze=self.mt1.get_maze()
        # self.maze = np.array(
        #     [
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        #     ]
        # )
        # 距离矩阵
        self.distinction=self.mt1.get_distinction()

        # 人流量矩阵
        self.people=self.mt1.get_people()

        self.m = self.maze.shape[0]
        self.n = self.maze.shape[1]

        self.position = np.array([0, 0])  # 起始位置
        self.pre_position = np.array([0, 0])  # 上一位置
        self.fresh_time = 0.001  # 刷新速度
        # 用来画图存数据的数组
        self.array_x = []
        self.array_y = []
        self.array_z = []
        self.min_road_reward = []
        # 计算奖励用的函数参数
        self.distinction_weight = 2
        self.people_weight = 2
        self.target_reward = 30

        # 最短路径
        self.minRoadMaze=self.mt1.get_minRoadMaze()
        # self.minRoadMaze = np.array(
        #     [
        #         [0, 0],
        #         [0, 1],
        #         [1, 1],
        #         [2, 1],
        #         [3, 1],
        #         [3, 2],
        #         [4, 2],
        #         [4, 3],
        #         [5, 3],
        #         [6, 3],
        #         [7, 3],
        #         [8, 3],
        #         [8, 4],
        #         [8, 5],
        #         [8, 6],
        #         [8, 7],
        #         [8, 8],
        #         [8, 9],
        #         [9, 9],
        #     ]
        # )
        # self.minRoadMaze_n = self.minRoadMaze.shape[0]

        self.minRoadMaze_n = 10

    # 重新初始化位置参数
    def reset(self):
        self.position = np.array([0, 0])  # 起始位置

        self.maze=self.mt1.get_maze()

        return np.array([-0.5, -0.5])  # observation采取的坐标不太一样

    def resetPeople(self):
        a = random.uniform(0.95, 1.05)
        N = self.m * self.n
        for i in range(self.people.size):
            s = self.people.getVertex(i)
            eList = s.getConnections()
            for e in eList:
                x = random.randint(0, 4) - 2
                temp = self.people.getTwoPointsEdge(s.id, e.id) * a + x
                print(temp)
                if temp > 0:
                    self.people.addEdge(s.id,e.id,temp)
                else:
                    self.people.addEdge(s.id, e.id, 0)

    def get_speed(self, people):
        speed=5
        if (people>=0) and (people<10):
            speed=8
        elif (people>=10) and (people<20):
            speed=5
        elif (people >= 20) and (people < 30):
            speed = 3
        elif (people>=30) and (people<=50):
            speed=1
        elif (people>50):
            speed=0.1

        # speed = (200 - people) * 0.5
        # if speed <= 0:
        #     speed = 0.1
        return speed

    def update_env(self, episode, step_counter, done, reward , sub_goal,epoch):
        # print(self.position[0],self.position[1])
        # if epoch>50: return
        # if self.flag>=1:
        #     # np.array([-0.5 + self.position[0] * 0.25, -0.5 + self.position[1] * 0.25])
        #     sub_goal=(sub_goal+0.5)/0.25
        #
        #     temp=self.distinction.getVertex(sub_goal[0]*self.n+sub_goal[1])
        #     eList=temp.getConnections()
        #     for e in eList:
        #
        #         temp1=temp.getWeight(e)
        #         if temp1 > 1:
        #             temp1 = temp.getWeight(e)*0.96
        #             self.distinction.addEdge(temp.id, e.id, temp1)
        #         else:
        #             self.distinction.addEdge(temp.id, e.id, temp1*0.99)
        #
        #         ex_List=e.getConnections()
        #         for ex in ex_List:
        #             temp2=e.getWeight(ex)
        #             print(temp2)
        #             if temp2 > 1:
        #
        #                 self.distinction.addEdge(e.id, ex.id, temp2*0.96)
        #             else:
        #                 self.distinction.addEdge(e.id, ex.id, temp1*0.99)
        #
        #     if self.flag>0:
        #        self.flag-=1



        if done:
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), '')
            print('\r                           ', '')
            self.flag=1
            # time.sleep(self.fresh_time)
            # print(self.people)

    # 更新位置,获取奖励
    def minRoad(self):
        minRoadReward = 0
        for i in range(100):
            minRoadReward = 0

            if (i % 2 == 0):
                self.resetPeople()

            # self.resetPeople()
            for i in range(self.minRoadMaze_n - 1):
                temp_x = self.minRoadMaze[i][0] * self.n + self.minRoadMaze[i][1]
                temp_y = self.minRoadMaze[i + 1][0] * self.n + self.minRoadMaze[i + 1][1]
                # print(temp_x, temp_y)
                # reward现在拟化为时间，公式为:reward=-1*(路径距离/人流量值对应速度）
                temp = -1 * (self.distinction.getTwoPointsEdge(temp_x,temp_y)/ self.get_speed(self.people.getTwoPointsEdge(temp_x,temp_y)))

                # temp = -1 * (self.distinction_weight * self.distinction[temp_x][temp_y] + self.people_weight *
                #              self.people[temp_x][temp_y])

                minRoadReward += temp
            minRoadReward += self.target_reward
            self.min_road_reward.append(minRoadReward)
            # print(minRoadReward)
        return minRoadReward

    def judge(self,sub_goal,observation,observation_next):
        distinction=pow((sub_goal[0]-observation[0]),2)+pow((sub_goal[1]-observation[1]),2)
        distinction1=pow((sub_goal[0]-observation_next[0]),2)+pow((sub_goal[1]-observation_next[1]),2)

        print(distinction,distinction1)

        return (distinction-distinction1)*0.05


    def step(self, action,sub_goal):
        # 0 上   1 下  3 左  2 右
        # 注意左右的定义不同
        # 根据动作,更新当前的位置 并且注意越界问题
        # 目前的位置变为空了
        # self.maze[self.position[0]][self.position[1]]=0
        pre_position_x = self.position[0]
        pre_position_y = self.position[1]
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
            transformed_position_x = pre_position_x * self.n + pre_position_y
            transformed_position_y = self.position[0] * self.n + self.position[1]
            # reward=-1*A*当前距离+B*人流量
            # reward现在拟化为时间，公式为:reward=-1*(路径距离/人流量值对应速度）

            observation = [pre_position_x, pre_position_y]
            observation_next = [self.position[0], self.position[1]]

            reward = -1 * (self.distinction.getTwoPointsEdge(transformed_position_x,transformed_position_y)/ self.get_speed(self.people.getTwoPointsEdge(transformed_position_x,transformed_position_y)))+self.judge(sub_goal,observation,observation_next)
            print("reward1", self.judge(sub_goal, observation, observation_next), "reward2", reward)
            # reward = -1 * (self.distinction[transformed_position_x][
            #                    transformed_position_y] / self.get_speed(self.people[transformed_position_x][
            #                                                                 transformed_position_y]))
            # reward = -1 * (self.distinction_weight * self.distinction[transformed_position_x][
            #     transformed_position_y] + self.people_weight * self.people[transformed_position_x][
            #                    transformed_position_y])
            # print(reward)
            reward += self.target_reward
            done = True
        # elif  self.maze[self.position[0]][self.position[1]]==-1:
        #     reward = -1
        #     done = True
        else:
            # 转化为邻接矩阵的坐标
            transformed_position_x = pre_position_x * self.n + pre_position_y
            transformed_position_y = self.position[0] * self.n + self.position[1]
            # print(pre_position_x,pre_position_y,self.position[0],self.position[1])
            # reward现在拟化为时间，公式为:reward=-1*(路径距离/人流量值对应速度）

            observation = [pre_position_x, pre_position_y]
            observation_next = [self.position[0], self.position[1]]

            reward = -1 * (self.distinction.getTwoPointsEdge(transformed_position_x,
                                                             transformed_position_y) / self.get_speed(
                self.people.getTwoPointsEdge(transformed_position_x, transformed_position_y)))+self.judge(sub_goal,observation,observation_next)

            print("reward1",self.judge(sub_goal,observation,observation_next),"reward2",reward)

            # reward = -1 * (self.distinction[transformed_position_x][
            #                    transformed_position_y] / self.get_speed(self.people[transformed_position_x][
            #                                                                 transformed_position_y]))
            # reward=-1*A*当前距离+B*人流量
            # reward = -1 * (self.distinction_weight * self.distinction[transformed_position_x][
            #     transformed_position_y] + self.people_weight * self.people[transformed_position_x][
            #                    transformed_position_y])
            # print(reward)
            done = False


            # print("reward:",reward)
            # print("self.distinction:",self.distinction[transformed_position_x][transformed_position_y])
            # print("self.get_speed:",self.get_speed(self.people[transformed_position_x][transformed_position_y]))
            # print()



        # 当前位置 之后再更新,不然会覆盖奖励
        # self.maze[self.position[0]][self.position[1]] = 2

        # -0.5 -0.5 0 0.25   m
        # -0.5
        #   0
        # 0.25
        #   n

        # 从position到observation_的坐标转换
        observation_ = np.array([-0.5 + self.position[0] * 0.25, -0.5 + self.position[1] * 0.25])
        # if reward!=0:
        #     print(reward)
        return observation_, reward, done

    def step1(self, actions_value, step_counter):
        # 0 上   1 下  3 左  2 右
        # 注意左右的定义不同
        # 根据动作,更新当前的位置 并且注意越界问题
        # 目前的位置变为空了
        # self.maze[self.position[0]][self.position[1]]=0
        pre_position_x = self.position[0]
        pre_position_y = self.position[1]
        # print(self.pre_position[0], self.pre_position[1])
        # print(self.position[0], self.position[1])
        # 循环选择，保证值最大，且不往返，不撞墙
        for i in range(len(actions_value[1])):
            action = actions_value[0][i]
            # print(action)
            if action == 0 and (self.position[0] - 1 != self.pre_position[0]):
                # if self.position[0] > 0 & (self.position[0] - 1 != self.pre_position[0]):
                if self.position[0] > 0:
                    self.pre_position[0] = self.position[0]
                    self.position[0] -= 1
                    break
            elif action == 1 and (self.position[0] + 1 != self.pre_position[0]):
                if self.position[0] < self.m - 1:
                    self.pre_position[0] = self.position[0]
                    self.position[0] += 1
                    break
            elif action == 3 and (self.position[1] - 1 != self.pre_position[1]):
                if self.position[1] > 0:
                    self.pre_position[1] = self.position[1]
                    self.position[1] -= 1
                    break
            elif action == 2 and (self.position[1] + 1 != self.pre_position[1]):
                if self.position[1] < self.n - 1:
                    self.pre_position[1] = self.position[1]
                    self.position[1] += 1
                    break

        # if action == 0:
        #     if self.position[0] > 0:
        #         self.position[0] -= 1
        # elif action == 1:
        #     if self.position[0] < self.m - 1:
        #         self.position[0] += 1
        # elif action == 3:
        #     if self.position[1] > 0:
        #         self.position[1] -= 1
        # elif action == 2:
        #     if self.position[1] < self.n - 1:
        #         self.position[1] += 1

        # 根据情况判断获得的奖励 以及是否结束
        if self.maze[self.position[0]][self.position[1]] == 1:
            transformed_position_x = pre_position_x * self.n + pre_position_y
            transformed_position_y = self.position[0] * self.n + self.position[1]
            # print(pre_position_x, pre_position_y, self.position[0], self.position[1])
            # reward=-1*A*当前距离+B*人流量
            reward = -1 * (self.distinction_weight * self.distinction[transformed_position_x][
                transformed_position_y] + self.people_weight * self.people[transformed_position_x][
                               transformed_position_y])
            # print(reward)
            reward += self.target_reward
            done = True
        # elif  self.maze[self.position[0]][self.position[1]]==-1:
        #     reward = -1
        #     done = True
        else:
            # 转化为邻接矩阵的坐标
            transformed_position_x = pre_position_x * self.n + pre_position_y
            transformed_position_y = self.position[0] * self.n + self.position[1]
            # print(pre_position_x,pre_position_y,self.position[0],self.position[1])
            # reward=-1*A*当前距离+B*人流量
            reward = -1 * (self.distinction_weight * self.distinction[transformed_position_x][
                transformed_position_y] + self.people_weight * self.people[transformed_position_x][
                               transformed_position_y])
            # print(reward)
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
        # if reward!=0:
        #     print(reward)
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

        minReward = self.minRoad()
        # print(minReward)
        # self.min_road_reward=minReward
        # for i in range(100):
        #     self.min_road_reward.append(minReward)

        # 画奖励曲线
        plt.plot(self.array_x, self.array_y)
        plt.plot(self.array_x, self.min_road_reward)
        plt.xlim((0, 100))
        # plt.ylim((-50, 50))
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
