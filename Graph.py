#Adjacency List
class Graph:
    def __init__(self,num_nodes,edges):
        self.num_nodes = num_nodes
        self.data = [[] for i in range(num_nodes)]
        for n1,n2 in edges:
            #Insert the dege into the right list
            self.data[n1].append(n2)
            self.data[n2].append(n1)
    #add new edge
    def add_dege(self,edge):
        n1,n2=edge[0],edge[1]
        print(n1,n2)
        self.data[n1].append(n2)
        self.data[n2].append(n1)
    # delete a edge
    def delete_edge(self,edge):
        n1,n2=edge[0],edge[1]
        if n2 in self.data[n1]:
            self.data[n1].remove(n2)
            self.data[n2].remove(n1)
        else:
            print("Edge Not Exist")
    def __repr__(self):
        return "\n".join(["{}:{}".format(n,neighbores) for n,neighbores in enumerate(self.data)])
    def __str__(self):
        return self.__repr__()

num_nodes = 5
edges = [(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4)]
graph1 = Graph(num_nodes,edges)
# print(graph1.data)
# print(["{}:{}".format(n,neighbores) for n,neighbores in enumerate(graph1.data)])
print(graph1)
# graph1.add_dege((2,0))
# print(graph1)
# graph1.delete_edge((2,0))
# print(graph1)
# graph1.delete_edge((2,0))
# print(graph1)
print("Adjacency Matrix")
#Adjency Matrix
class Adjacency_Matrix_Graph():
    def __init__(self,num_nodes,edges):
        self.num_nodes = num_nodes
        self.data = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n1,n2 in edges:
            self.data[n1][n2]=1
            self.data[n2][n1]=1
    def add_node(self,edge):
        n1,n2 = edge[0],edge[1]
        self.data[n1][n2]=1
        self.data[n2][n1]=1
    def delete_dege(self,edge):
        n1,n2 = edge[0],edge[1]
        if self.data[n1][n2]!=1:
            print("Edge not exist")
        else:
            self.data[n1][n2]=0
            self.data[n2][n1]=0
    # def __repr__(self):
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             return " "+str(self.data[i][j])
    #         return "\n"
    # def __str__(self):
    #     return self.__repr__()
    def __repr__(self):
        return repr(self.data)
graph2 = Adjacency_Matrix_Graph(num_nodes,edges)
print(graph2)
graph2.add_node((0,2))
print(graph2)
graph2.delete_dege((0,2))
print(graph2)

