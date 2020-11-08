import numpy as np
import random
import DQN.utils.graph as graph
import sys

class Matrix:
    def __init__(self):
        self.m = 10
        self.n = 10
        self.size = self.m * self.n
        self.maze = np.zeros([self.m, self.n])
        self.maze[9][9] = 1  # 终点为1
        self.random_value = 5
        self.random_const = 5
        self.distinction = graph.Graph(self.size)
        self.people = graph.Graph(self.size)

    def get_maze(self):
        return self.maze

    def get_people(self):
        for i in range(self.m):
            for j in range(self.n):
                postion = i * self.n + j
                if j + 1 < self.n:
                    self.people.addEdge(postion, postion + 1, random.randint(0, self.random_value) + self.random_const)
                if j - 1 >= 0:
                    self.people.addEdge(postion, postion - 1, random.randint(0, self.random_value) + self.random_const)
                if i - 1 >= 0:
                    self.people.addEdge(postion, postion - self.n, random.randint(0, self.random_value) + self.random_const)
                if i + 1 < self.n:
                    self.people.addEdge(postion, postion + self.n, random.randint(0, self.random_value) + self.random_const)
        return self.people

    def get_distinction(self):
        for i in range(self.m):
            for j in range(self.n):
                postion = i * self.n + j
                if j + 1 < self.n:
                    temp = random.randint(0, self.random_value) + self.random_const
                    self.distinction.addEdge(postion, postion + 1, temp)
                    self.distinction.addEdge(postion + 1, postion, temp)
                if i + 1 < self.n:
                    temp = random.randint(0, self.random_value) + self.random_const
                    self.distinction.addEdge(postion, postion + self.n, temp)
                    self.distinction.addEdge(postion + self.n, postion, temp)
        return self.distinction

    # 用Dijkstra算法得到最短路径
    def get_minRoadMaze(self):
        N = self.distinction.numVertices  # 节点个数
        MAX = sys.maxsize  # 初始为最大值


        dists = [MAX for i in range(N)]  # 让左上角的开始点到所有点距离初始为最大值
        nodes = set()  # 存放已经算出最短距离的点,初始集合为空
        parents = [i for i in range(N)]  # 记录路径 初始都为他们本身

        dists[0] = 0  # 到自己距离为0
        min_point = 0  # 距离左上角最近的点,初始为空

        while (len(nodes) < N):  # 最短距离的节点数量没满就继续循环
            nodes.add(min_point)  # 将当前最短距离加入集合
            # 遍历与最短边直接相连的节点
            for board_road in enumerate(self.distinction.getEdge(min_point)):
                w=board_road[1][0] #相邻路径顶点
                weight=board_road[1][1] #相邻路径权重
                if w not in nodes and weight > 0:  # 更新不在最短距离集,且可达的点 的距离
                    if (dists[min_point] + self.distinction.getTwoPointsEdge(min_point, w) < dists[w]):  # 值得放缩
                        dists[w] = dists[min_point] + self.distinction.getTwoPointsEdge(min_point, w)
                        parents[w] = min_point

            # 选出不在最短距离节点集的,但是到根节点距离最小的点作为下一个最小节点
            min_dist = MAX
            for w, weight in enumerate(dists):
                if w not in nodes and weight > 0 and weight < min_dist:
                    min_dist = weight
                    min_point = w

        #parents里面已经记录了最短路径点
        #初始化为最后那个点
        location=self.size - 1
        paths = [location]
        while parents[location] != 0:
            paths.append(parents[location])
            location = parents[location]
        paths.append(0)
        # 反转路径,得到从根路径开始的最短路径经过的点
        paths = paths[::-1]

        #将图的顶点映射到矩阵 如 99转为(9,9)
        minRoadMaze = np.zeros([len(paths), 2], dtype=int)
        for w in range(len(paths)):
            minRoadMaze[w][0] = paths[w] // self.n
            minRoadMaze[w][1] = paths[w] % self.n

        # 对最短路径上的点人为设置拥堵路段
        for i in range(len(minRoadMaze)):
            # 对最短路径每隔三个十字路口次设置一次拥堵点
            if i % 3 == 0 and i + 1 < len(minRoadMaze):
                v = minRoadMaze[i][0] * self.n + minRoadMaze[i][1]
                w = minRoadMaze[i + 1][0] * self.n + minRoadMaze[i + 1][1]
                self.people.addEdge(v, w, random.randint(0, 100) + 50)

        return minRoadMaze
        # print(minRoadMaze)