import DQN.utils.matrix as matrix

if __name__ == '__main__':
    matrix=matrix.Matrix()
    #print(matrix.get_maze())
    people=matrix.get_people()
    distinction=matrix.get_distinction()

    #print(matrix.distinction.numVertices)
    print(matrix.get_minRoadMaze())

    # for weight in enumerate(distinction.getEdge(0)):
    #     print(weight[1][0])
    #print(len(distinction.vertList))
    # for i in range(100):
    #     print(distinction.getEdge(i))
    #     print()


