import numpy as np

# M*N 代表的是点的个数
if __name__ == "__main__":
    M = 6
    N = 6
    root = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        )

    # 当前点到自身的距离为0,到不可达点为-1
    distance = np.zeros((M * N, M * N), dtype=np.int)
    for i in range(M * N):
        for j in range(M * N):
            if i!=j:
                distance[i][j] = -1

    # 根据实际二维数组,


    for i in range(M * N):
        for j in range(M * N):
            print(distance[i][j], end=" ")
        print()
