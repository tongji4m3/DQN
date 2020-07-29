#定义顶点类
class Vertex(object):
    #初始化顶点
    def __init__(self,key):
        self.id=key
        self.connectTo={}

    #添加邻居顶点
    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr]=weight

    #获取顶点的所有邻居顶点的键
    def getConnections(self):
        return self.connectedTo.keys()

    #获取顶点的键
    def getId(self):
        return self.id

    #获取到邻居的权重
    def getWeight(self,nbr):
        return self.connectedTo[nbr]

#定义图类
class Graph(object):
    #初始化图类
    def __init__(self):
        self.vertList={}
        self.numVertices=0

    #添加顶点
    def addVertex(self,key):
        newVertex = Vertex(key)  # 创建顶点
        self.vertList[key] = newVertex  # 将新顶点添加到邻接表中
        self.numVertices = self.numVertices + 1  # 邻接表中顶点数+1
        return newVertex

    #获取顶点
    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    #添加边，参数s为起点，e为终点，cost为权重
    def addEdge(self,s,e,cost=0):
        if s not in self.vertList:
            self.addVertex(s)
        if e not in self.vertLsit:
            self.addVertex(e)
        self.vertList[s].addNeighboor(self.vertList[e],cost)

    #获取邻接表所有顶点的键
    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

people=Graph()