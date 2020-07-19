import numpy as np
import time
import random
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
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        # 距离矩阵
        self.distinction = np.array(
            [
                [0, 9, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 0, 9, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 9, 0, 8, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 8, 0, 7, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 7, 0, 7, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 7, 0, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, -1, -1, -1, -1, -1, 0, 9, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 9, -1, -1, -1, -1, 9, 0, 7, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 8, -1, -1, -1, -1, 7, 0, 8, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 6, -1, -1, -1, -1, 8, 0, 6, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 8, -1, -1, -1, -1, 6, 0, 8, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 8, 0, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, 0, 9, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 9, 0, 9, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, 9, 0, 6, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 6, 0, 8, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, 8, 0, 7, -1, -1, -1, -1, 5, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 7, 0, -1, -1, -1, -1, -1, 9, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, 0, 5, -1, -1, -1, -1, 7, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 5, 0, 9, -1, -1, -1, -1, 7, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 9, 0, 9, -1, -1, -1, -1, 9,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, 9, 0, 6, -1, -1, -1, -1,
                 6, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 6, 0, 5, -1, -1, -1,
                 -1, 8, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, 5, 0, -1, -1,
                 -1, -1, -1, 6, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, 0, 6,
                 -1, -1, -1, -1, 9, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 6, 0, 5,
                 -1, -1, -1, -1, 6, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, 5,
                 0, 7, -1, -1, -1, -1, 7, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1,
                 7, 0, 8, -1, -1, -1, -1, 8, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1,
                 -1, 8, 0, 8, -1, -1, -1, -1, 6, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1,
                 -1, -1, 8, 0, -1, -1, -1, -1, -1, 5],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1,
                 -1, -1, -1, -1, 0, 7, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,
                 -1, -1, -1, -1, 7, 0, 8, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 7, -1, -1, -1, -1, 8, 0, 5, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, 8, -1, -1, -1, -1, 5, 0, 8, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, 6, -1, -1, -1, -1, 8, 0, 6],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, 5, -1, -1, -1, -1, 6, 0],

            ]
        )
        # 人流量矩阵
        self.people = np.array(
            [
                [0, 9, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 0, 9, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 6, 0, 8, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 8, 0, 8, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 7, 0, 7, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 9, 0, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [7, -1, -1, -1, -1, -1, 0, 5, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 8, -1, -1, -1, -1, 8, 0, 7, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 8, -1, -1, -1, -1, 6, 0, 7, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 5, -1, -1, -1, -1, 8, 0, 5, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 7, -1, -1, -1, -1, 9, 0, 9, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 6, 0, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, 0, 8, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 5, 0, 7, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, 5, 0, 5, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 5, 0, 6, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 8, 0, 8, -1, -1, -1, -1, 8, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, 9, 0, -1, -1, -1, -1, -1, 7, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, 0, 6, -1, -1, -1, -1, 9, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 7, 0, 9, -1, -1, -1, -1, 6, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 7, 0, 7, -1, -1, -1, -1, 5,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, 8, 0, 6, -1, -1, -1, -1,
                 5, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, 7, 0, 5, -1, -1, -1,
                 -1, 9, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, 7, 0, -1, -1,
                 -1, -1, -1, 5, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, 0, 5,
                 -1, -1, -1, -1, 7, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, 6, 0, 9,
                 -1, -1, -1, -1, 7, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 7,
                 0, 9, -1, -1, -1, -1, 9, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1,
                 9, 0, 7, -1, -1, -1, -1, 5, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1,
                 -1, 5, 0, 5, -1, -1, -1, -1, 6, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1,
                 -1, -1, 7, 0, -1, -1, -1, -1, -1, 9],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1,
                 -1, -1, -1, -1, 0, 7, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8,
                 -1, -1, -1, -1, 9, 0, 6, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 7, -1, -1, -1, -1, 5, 0, 59, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, 97, -1, -1, -1, -1, 90, 0, 111, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, 6, -1, -1, -1, -1, 5, 0, 8],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, 5, -1, -1, -1, -1, 7, 0],

        ]
        )
        self.m = self.maze.shape[0]
        self.n = self.maze.shape[1]

        self.position = np.array([0, 0])  # 起始位置
        self.pre_position = np.array([0, 0])#上一位置
        self.fresh_time = 0.001  # 刷新速度
        # 用来画图存数据的数组
        self.array_x = []
        self.array_y = []
        self.array_z = []
        self.min_road_reward = []
        # 计算奖励用的函数参数
        self.distinction_weight = 2
        self.people_weight = 2
        self.target_reward = 160

        # 最短路径
        self.minRoadMaze = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [3, 1],
                [4, 1],
                [4, 2],
                [5, 2],
                [5, 3],
                [5, 4],
                [5, 5],
            ]
        )
        self.minRoadMaze_n = self.minRoadMaze.shape[0]

    # 重新初始化位置参数
    def reset(self):
        self.position = np.array([0, 0])  # 起始位置

        self.maze = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return np.array([-0.5, -0.5])  # observation采取的坐标不太一样

    # 实现人流量动态改变
    def resetPeople(self):
        # move = random.randint(0, 1)  # 这个训练回合什么都不做
        move = 1
        N = self.m * self.n
        nn = self.n
        if move:
            # 改变放在新人流量矩阵上,每次整体更新,防止有人走了多步
            newPeople = self.people[:]
            for i in range(N):
                # 如果横着的边在实际中是存在的,并且边上有人
                if (i + 1 < N and (self.people[i][i + 1] > 0 or self.people[i + 1][i] > 0)):
                    # 对路径上的每个人分别随机走,共4种选择(不一定没种都能走)
                    # 不动,上，下，左/右
                    # 0   1   2  3

                    # 对于i+1→i
                    for pathPeople in range(self.people[i + 1][i]):
                        change = False  # 代表这个人是否已经走了
                        while change != True:
                            direction = random.randint(0, 3)
                            if direction == 0:
                                change = True
                            if direction == 1:
                                # 如果左上路径确实存在
                                if i - self.n >= 0 and self.people[i][i - self.n] > -1:
                                    newPeople[i][i - self.n] += 1
                                    newPeople[i + 1][i] -= 1
                                    change = True
                            if direction == 2:
                                # 如果左下路径确实存在
                                if i + self.n < N and self.people[i][i + self.n] > -1:
                                    newPeople[i][i + self.n] += 1
                                    newPeople[i + 1][i] -= 1
                                    change = True
                            if direction == 3:
                                # 如果左路径确实存在
                                if i - 1 >= 0 and self.people[i][i - 1] > -1:
                                    newPeople[i][i - 1] += 1
                                    newPeople[i + 1][i] -= 1
                                    change = True
                    # 对于i→i+1
                    for pathPeople in range(self.people[i][i + 1]):
                        change = False
                        while change != True:
                            direction = random.randint(0, 3)
                            if direction == 0:
                                change = True
                            if direction == 1:
                                # 如果右上
                                if i + 1 - self.n >= 0 and self.people[i + 1][i + 1 - self.n] > -1:
                                    newPeople[i + 1][i + 1 - self.n] += 1
                                    newPeople[i][i + 1] -= 1
                                    change = True
                            if direction == 2:
                                # 如果右下
                                if i + 1 + self.n < N and self.people[i + 1][i + 1 + self.n] > -1:
                                    newPeople[i + 1][i + 1 + self.n] += 1
                                    newPeople[i][i + 1] -= 1
                                    change = True
                            if direction == 3:
                                if i + 2 < N and self.people[i + 1][i + 2] > -1:
                                    newPeople[i + 1][i + 2] += 1
                                    newPeople[i][i + 1] -= 1
                                    change = True

                if (i + nn < N and (self.people[i][i + nn] > 0 or self.people[i + nn][i] > 0)):
                    # 对路径上的每个人分别随机走,共4种选择(不一定没种都能走)
                    # 不动,左，右，上/下
                    # 0   1   2  3
                    # 对于i+nn→i
                    for pathPeople in range(self.people[i + nn][i]):
                        change = False
                        while change != True:
                            direction = random.randint(0, 3)
                            if direction == 0:
                                change = True
                            if direction == 1:
                                if i - 1 >= 0 and self.people[i][i - 1] > -1:
                                    newPeople[i][i - 1] += 1
                                    newPeople[i + nn][i] -= 1
                                    change = True
                            if direction == 2:
                                if i + nn < N and self.people[i][i + 1] > -1:
                                    newPeople[i][i + 1] += 1
                                    newPeople[i + nn][i] -= 1
                                    change = True
                            if direction == 3:
                                if i - nn >= 0 and self.people[i][i - nn] > -1:
                                    newPeople[i][i - nn] += 1
                                    newPeople[i + nn][i] -= 1
                                    change = True
                    # 对于i→i+nn
                    for pathPeople in range(self.people[i][i + nn]):
                        change = False
                        while change != True:
                            direction = random.randint(0, 3)
                            if direction == 0:
                                change = True
                            if direction == 1:
                                if self.people[i + nn][i + nn - 1] > -1:
                                    newPeople[i + nn][i + nn - 1] += 1
                                    newPeople[i][i + nn] -= 1
                                    change = True
                            if direction == 2:
                                if i + nn + 1 < N and self.people[i + nn][i + nn + 1] > -1:
                                    newPeople[i + nn][i + nn + 1] += 1
                                    newPeople[i][i + nn] -= 1
                                    change = True
                            if direction == 3:
                                if i + 2 * nn < N and self.people[i + nn][i + 2 * nn] > -1:
                                    newPeople[i + nn][i + 2 * nn] += 1
                                    newPeople[i][i + nn] -= 1
                                    change = True

            # count=0
            # for i in range(self.m * self.n):
            #     for j in range(self.m * self.n):
            #         # count+=newPeople[i][j]
            #         print(newPeople[i][j], end=" ")
            #     print()
            # print(count)

            self.people = newPeople[:]

    def update_env(self, episode, step_counter, done, reward):
        # print(self.position[0],self.position[1])

        if done:
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), '')
            print('\r                           ', '')

            time.sleep(self.fresh_time)

    # 更新位置,获取奖励
    def minRoad(self):
        minRoadReward = 0
        for i in range(100):
            minRoadReward = 0
            # self.resetPeople()
            for i in range(self.minRoadMaze_n - 1):

                temp_x = self.minRoadMaze[i][0] * self.n + self.minRoadMaze[i][1]
                temp_y = self.minRoadMaze[i + 1][0] * self.n + self.minRoadMaze[i + 1][1]
                print(temp_x, temp_y)

                temp = -1 * (self.distinction_weight * self.distinction[temp_x][temp_y] + self.people_weight *
                             self.people[temp_x][temp_y])
                print(self.people[temp_x][temp_y])
                minRoadReward += temp
            minRoadReward += self.target_reward
            self.min_road_reward.append(minRoadReward)
            # print(minRoadReward)
        return minRoadReward

    def step(self, action):
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
            print(pre_position_x, pre_position_y, self.position[0], self.position[1])
            # reward=-1*A*当前距离+B*人流量
            reward = -1 * (self.distinction_weight * self.distinction[transformed_position_x][
                transformed_position_y] + self.people_weight * self.people[transformed_position_x][
                               transformed_position_y])
            print(reward)
            reward += self.target_reward
            done = True
        # elif  self.maze[self.position[0]][self.position[1]]==-1:
        #     reward = -1
        #     done = True
        else:
            # 转化为邻接矩阵的坐标
            transformed_position_x = pre_position_x * self.n + pre_position_y
            transformed_position_y = self.position[0] * self.n + self.position[1]
            print(pre_position_x,pre_position_y,self.position[0],self.position[1])
            # reward=-1*A*当前距离+B*人流量
            reward = -1 * (self.distinction_weight * self.distinction[transformed_position_x][
                transformed_position_y] + self.people_weight * self.people[transformed_position_x][
                               transformed_position_y])
            print(reward)
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

    def step1(self, actions_value, step_counter):
        # 0 上   1 下  3 左  2 右
        # 注意左右的定义不同
        # 根据动作,更新当前的位置 并且注意越界问题
        # 目前的位置变为空了
        # self.maze[self.position[0]][self.position[1]]=0
        pre_position_x = self.position[0]
        pre_position_y = self.position[1]
        print(self.pre_position[0], self.pre_position[1])
        print(self.position[0], self.position[1])
        #循环选择，保证值最大，且不往返，不撞墙
        for i in range(len(actions_value[1])):
            action = actions_value[0][i]
            print(action)
            if action == 0 and (self.position[0] - 1 != self.pre_position[0]):
                # if self.position[0] > 0 & (self.position[0] - 1 != self.pre_position[0]):
                if self.position[0] > 0 :
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
        plt.ylim((-1000, 50))
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
