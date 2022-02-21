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
        return " ".join(["{}:{}".format(n,neighbores) for n,neighbores in enumerate(self.data)])
        # return "\n".join(["{}:{}".format(n,neighbores) for n,neighbores in enumerate(self.data)])
    def __str__(self):
        return self.__repr__()

num_nodes = 5
edges = [(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4)]
graph1 = Graph(num_nodes,edges)
# print(graph1.data)
# print(["{}:{}".format(n,neighbores) for n,neighbores in enumerate(graph1.data)])
print("Adjacency List Display")
print(graph1)
print("Add Edge (2,0)")
graph1.add_dege((2,0))
print(graph1)
print("Add Exist Edge (2,0)")
graph1.add_dege((2,0))
print("Delete Edge (2,0)")
graph1.delete_edge((2,0))
print(graph1)
print("Delete Not Exist Edge (2,0)")
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
print("---------------------BFS AND DFS------------------------")
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
print("Breadth First Search")
print(bfs(dict_graph,"a"))

# Create an Graph class for the weighted graph to calculate the shortest path
class weight_graph():
    def __init__(self, edges):
        self.weight={}
        for start,end,weight in edges:
            start_node = start
            end_node = end
            path_weight = weight
            self.weight[start_node,end_node]=weight
            self.weight[end_node,start_node] = weight
        self.data={}
        for start,end,weight in edges:
            start_node = start
            end_node = end
            path_weight = weight
            if start_node not in self.data.keys():
                if end_node not in self.data.keys():
                    self.data[start_node]=list(end_node)
                    self.data[end_node] = list(start_node)
                else:
                    self.data[start_node] = list(end_node)
                    self.data[end_node].append(start_node)
            else:
                if end_node not in self.data.keys():
                    self.data[start_node].append(end_node)
                    self.data[end_node]=list(start_node)
                else:
                    self.data[start_node].append(end_node)
                    self.data[end_node].append(start_node)
def display(graph):
    for start,end in graph.weight:
        print("("+start+","+end+")："+str(graph.weight[(start,end)]))

edges = [
    ('X', 'A', 7),
    ('X', 'B', 2),
    ('X', 'C', 3),
    ('X', 'E', 4),
    ('A', 'B', 3),
    ('A', 'D', 4),
    ('B', 'D', 4),
    ('B', 'H', 5),
    ('C', 'L', 2),
    ('D', 'F', 1),
    ('F', 'H', 3),
    ('G', 'H', 2),
    ('G', 'Y', 2),
    ('I', 'J', 6),
    ('I', 'K', 4),
    ('I', 'L', 4),
    ('J', 'L', 1),
    ('K', 'Y', 5),
]


weight_graph = weight_graph(edges)
# display(weight_graph)

def dijsktra(graph,start,end):
    shortest_paths= {start:(None,0)}
    current = start
    visited = set()
    while current !=end:
        visited.add(current)
        # destination = set(graph.data[current]).difference_update(visited)
        destination = [node for node in graph.data[current] if node not in visited]
        weight_to_current_node = shortest_paths[current][1]
        # update the weight for each node
        # print(graph.data[current])
        for neighbor_node in destination:
            if neighbor_node not in visited:
                # print("current = "+current," neighbor="+neighbor_node)
                # print(graph.weight[current,neighbor_node])
                # print(graph.weight.keys())
                edge = tuple([current, neighbor_node])
                # print(edge)
                # if edge in graph.weight.keys():
                #     print("yes")
                # make sure there is the edge between current edge and neighbor
                if edge in graph.weight.keys():
                    weight = graph.weight[current,neighbor_node]+weight_to_current_node
                    if neighbor_node not in shortest_paths:
                        shortest_paths[neighbor_node]=(current,weight)
                    else:
                        current_short_wight = shortest_paths[neighbor_node][1]
                        if current_short_wight>weight:
                            shortest_paths[neighbor_node]=(current,weight)
        # print(shortest_paths)
        next_destinations = {node:shortest_paths[node] for node in shortest_paths if node not in visited}
        # next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        # print(next_destinations)
        if not next_destinations:
            return "Route Not Possible"
        current = min(next_destinations,key = lambda k:next_destinations[k][1])
    path = []
    weight = 0
    while current is not None:
        path.append(current)
        next = shortest_paths[current][0]
        weight = shortest_paths[current][1]+weight
        current =next
    # print(path)
    path = path[::-1]
    print("Shortest Path:"+str(weight))
    return path

print(dijsktra(weight_graph, 'X', 'Y'))

# Has Path Algorithem
# Unweighted Directed Graph
class unweight_direct_graph():
    def __init__(self,edges):
        self.data = {}
        for start_node,dest_node in edges:
            if start_node not in self.data.keys():
                self.data[start_node] = list(dest_node)
            else:
                self.data[start_node].append(dest_node)

    def display(self):
        node_list = set(self.data.keys())
        for node in node_list:
            print(str(node)+":"+str(self.data[node]))
has_path_node = 6
edges=(("f","g"),("f","i"),("g","h"),("i","g"),("j","i"),("i","k"))

unweight_direct_graph = unweight_direct_graph(edges)
# unweight_direct_graph.display()

# Has Path Function
def has_path(graph, start, destination):
    node_list = set(graph.data.keys())
    # check if found the destination node
    if start==destination:
        return True
    # check if node has 0 neighbor node(s)
    elif start not in node_list:
        return False
    # if the node not the destination node
    # and has the neighbor node, then search the neighbor node
    else:
        for node in graph.data[start]:
            if has_path(graph,node,destination):
                return True
    return False
print(has_path(unweight_direct_graph,"f","h"))
print(has_path(unweight_direct_graph,"i","j"))

class unweight_undirect_graph():
    def __init__(self,edges):
        self.data = {}
        for start,destination in edges:
            if start not in set(self.data.keys()):
                if destination not in set(self.data.keys()):
                    self.data[start] = list(destination)
                    self.data[destination] = list(start)
                else:
                    self.data[start]=list(destination)
                    self.data[destination].append(start)
            else:
                if destination not in set(self.data.keys()):
                    self.data[start].append(destination)
                    self.data[destination] = list(start)
                else:
                    self.data[start].append(destination)
                    self.data[destination].append(start)
    def display(self):
        node_list = set(self.data.keys())
        for node in node_list:
            print(str(node)+":"+str(self.data[node]))

number_of_component_unweighted_undirect_edge = (("1","2"),("4","6"),("5","6"),("6","8"),("6","7"),("3","3"))
unweighted_undirect_graph = unweight_undirect_graph(number_of_component_unweighted_undirect_edge)

def number_of_component(graph):
    visted = []
    count = 0
    node_list =set(graph.data.keys())
    for node in node_list:
        if explor(graph,node,visted) :
            count = count+1
    return count
def explor(graph,current_node,visted):
    if current_node in visted:
        return False
    visted.append(current_node)
    for node in graph.data[current_node]:
        explor(graph,node,visted)
    return True
print("There are "+str(number_of_component(unweighted_undirect_graph))+" Coponent(s) in th graph")

largest_component_unweight_undirect_edges=(("1","0"),("8","0"),("5","0"),
("2","3"),("2","4"))
largest_component_unweight_undirect_graph = unweight_undirect_graph(largest_component_unweight_undirect_edges)
# largest_component_unweight_undirect_graph.display()

def largestComponent(graph):
    visted = []
    largest = 0
    node_list = graph.data.keys()
    for node in node_list:
        size = explorSize(graph,node,visted)
        if size> largest:
            largest = size
    return largest
def explorSize(graph,node,visted):
    if node in visted:
        return 0
    visted.append(node)
    size = 1
    for neighbhor in graph.data[node]:
        size = size+explorSize(graph,neighbhor,visted)
    return size
print("The Largest component has "+str(largestComponent(largest_component_unweight_undirect_graph))+" node(s)")