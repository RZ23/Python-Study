from collections import deque
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
        if n2 in self.data[n1]:
            print("Edge Exist")
        else:
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
print("Add Edge")
graph1.add_dege((2,0))
print(graph1)
print("Add Exist Edge")
graph1.add_dege((2,0))
print("Delete Edge")
graph1.delete_edge((2,0))
print(graph1)
print("Delete Not Exist Edge")
graph1.delete_edge((2,0))
print("------------------------------------")
print("Adjacency Matrix")
#Adjacency Matrix
class Adjacency_Matrix_Graph():
    def __init__(self,num_nodes,edges):
        self.num_nodes = num_nodes
        self.data = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        for n1,n2 in edges:
            self.data[n1][n2]=1
            self.data[n2][n1]=1
    def add_node(self,edge):
        n1,n2 = edge[0],edge[1]
        if self.data[n1][n2]==1:
            print("Edge Exist")
        self.data[n1][n2]=1
        self.data[n2][n1]=1
    def delete_dege(self,edge):
        n1,n2 = edge[0],edge[1]
        if self.data[n1][n2]!=1:
            print("Edge not exist")
        else:
            self.data[n1][n2]=0
            self.data[n2][n1]=0
    def __repr__(self):
        return repr(self.data)
graph2 = Adjacency_Matrix_Graph(num_nodes,edges)
print(graph2)
print("Add Edge")
graph2.add_node((0,2))
print(graph2)
print("Add the same edge")
graph2.add_node((0,2))
print("Delete a edge")
graph2.delete_dege((0,2))
print(graph2)
print("Delete the same edge")
graph2.delete_dege((0,2))

class Adjacency_List_Dict():
    def __init__(self,num_node,edges):
        self.num_node = num_node
        self.data = {}
        for node1,node2 in edges:
            if node1 in self.data:
                self.data[node1].append(node2)
            else:
                self.data[node1]=list(node2)
            if node2 in self.data:
                self.data[node2].append(node1)
            else:
                self.data[node2] = list(node1)
    def add_dege(self,edge):
        if len(edge)==1:
            node = edge
            # print("Add single Node")
            if node not in self.data.keys():
                self.data[node] = []
            return
        else:
            node1,node2 = edge[0],edge[1]
            if node1==node2:
                # print("Add Circle Edge")
                if node1 not in self.data.keys():
                    self.data[node1] = [node1]
                elif node1 in self.data.keys():
                    self.data[node1].append(node1)
                return
            if node1 in self.data:
                if node2 in self.data[node1]:
                    print("Edge Exist")
                    return
                else:
                    self.data[node1].append(node2)
            elif node1 not in self.data:
                self.data[node1] = list(node2)
            if node2 in self.data:
                if node1 in self.data[node2]:
                    print("Edge Exist")
                    return
                else:
                    self.data[node2].append(node1)
            elif node2 not in self.data:
                self.data[node2] = list(node1)
    def del_edge(self,edge):
        node1,node2 = edge[0],edge[1]
        if node1 not in self.data.keys() or node2 not in self.data.keys():
            print("Node Not Exist")
        elif node2 not in self.data[node1] or node1 not in self.data[node2]:
            print("Edge Not Exist")
        else:
            self.data[node1].remove(node2)
            self.data[node2].remove(node1)

num_nodes = 6
edges = (("a","b"),("a","c"),("b","d"),("c","e"),("d","f"))
print("---------------------------------------------------")
print("Adjacency List with Dict")
dict_graph = Adjacency_List_Dict(num_nodes,edges)
def print_Adjacency_list_Dict(graph):
    for key in graph.data.keys():
        s = key.upper()+":"
        for value in graph.data[key]:
            if value is None:
                s = " "+s
            else:
                s = " "+s+" "+value.upper()
        print(s,end = " ")
    print()
print_Adjacency_list_Dict(dict_graph)
print("Add New Edge: (E,F)")
new_dege = ("e","f")
exist_edge = ("a","c")
dict_graph.add_dege(new_dege)
print_Adjacency_list_Dict(dict_graph)
print("Add Exist Edge: (A,C)")
dict_graph.add_dege(exist_edge)
print_Adjacency_list_Dict(dict_graph)
print("Add Circle X,X")
dict_graph.add_dege(("x","x"))
print_Adjacency_list_Dict(dict_graph)
print("Add single node y")
dict_graph.add_dege("y")
print_Adjacency_list_Dict(dict_graph)
print("Delete Edge (A,C)")
dict_graph.del_edge(("a","c"))
print_Adjacency_list_Dict(dict_graph)
print("Delete Not Exist Edge (A,C)")
dict_graph.del_edge(("a","c"))
print_Adjacency_list_Dict(dict_graph)
print("Delete Not Exist Node (X,Y)")
dict_graph.del_edge(("x","y"))
print_Adjacency_list_Dict(dict_graph)

#Deep Frist Search
def dfs(graph,start_node):
    s = []
    seq = ""
    status = {}
    for key in graph.data.keys():
        status[key] = "UV"
    current = start_node
    status[current] = "V"
    s.append(current)
    while len(s)>0:
        current = s.pop()
        for neighbor in graph.data[current]:
            if status[neighbor] == "UV":
                s.append(neighbor)
        status[current] = "V"
        seq= seq+current.upper()
    return seq
print("Deep First Search")
print(dfs(dict_graph,"a"))

def bfs(graph,start_node):
    status = {}
    for key in graph.data.keys():
        status[key] = "UV"
    current = start_node
    s = deque()
    seq = ""
    s.append(start_node)
    status[start_node] = "V"
    while len(s)>0:
        current = s.popleft()
        for neighbor in graph.data[current]:
            if status[neighbor]=="UV":
                s.append(neighbor)
                status[neighbor]="V"
        seq = seq+current.upper()
    return seq

print(bfs(dict_graph,"a"))

