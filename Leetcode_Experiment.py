# from binarytree import Node, tree
from binarytree import Node, tree
from collections import deque
from collections import Counter
import collections
def print_matrix(dp):
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if isinstance(dp[i][j],int):
                print(format(dp[i][j],'02d'), end="|")
            else:
                print(dp[i][j], end="|")
        print()
import binarytree


def generate_tree_from_list(root):
    node_list = []
    # generate node for each item in the list
    for i in range(len(root)):
        if root[i] is not None:
            node_list.append(Node(root[i]))
        else:
            node_list.append(None)
    # Set the Left/Right child for each node
    for i in range(len(node_list)//2):
        if node_list[i] is not None:
            left_child =2*i+1
            right_child = 2*i+2
            if left_child<len(node_list):
                node_list[i].left = node_list[left_child]
            if right_child<len(node_list):
                node_list[i].right = node_list[right_child]
    return node_list[0]
class Graph_Node():
    def __init__(self,value = 0,neighbors = None):
        self.value = value
        self.neighbors = neighbors if neighbors is not None else []
def display_graph(node):
    if node is None:
        return []
    display_order = {}
    q= deque()
    q.append(node)
    visit = set()
    visit.add(node)
    result = []
    while len(q)>0:
        cur_node =q.popleft()
        print("{}".format(cur_node.value),end = ":")
        sub_result = []
        for neighbor in cur_node.neighbors:
            if neighbor not in visit:
                q.append(neighbor)
                visit.add(neighbor)
            print(neighbor.value,end = ",")
            sub_result.append(neighbor.value)
        print()
        # result.append(sub_result)
        display_order[cur_node.value] = sub_result
    for i in range(1,len(display_order.keys())+1):
        result.append(display_order[i])
    return result
def generate_graph(graph_node_list):
    if len(graph_node_list)==0:
        return None
    graph = []
    graph_map = {}
    for i in range(len(graph_node_list)):
        index = str(i+1)
        graph.append(Graph_Node(i+1))
        graph_map[index] = graph[i]
    for i in range(len(graph)):
        for adj in graph_node_list[i]:
            graph[i].neighbors.append(graph_map[str(adj)])
    return graph[0]
print("---------------------297. Serialize and Deserialize Binary Tree-------------------------")
print("***** Method Two: Pre-Order and DFS *****")
class TreeNode(binarytree.Node):
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def generate_tree_from_list(root):
    node_list = []
    # generate node for each item in the list
    for i in range(len(root)):
        if root[i] is not None:
            node_list.append(TreeNode(root[i]))
        else:
            node_list.append(None)
    # Set the Left/Right child for each node
    for i in range(len(node_list)//2):
        if node_list[i] is not None:
            left_child =2*i+1
            right_child = 2*i+2
            if left_child<len(node_list):
                node_list[i].left = node_list[left_child]
            if right_child<len(node_list):
                node_list[i].right = node_list[right_child]
    return node_list[0]
def serialize(root):
    result = []
    def dfs(node):
        if not node:
            result.append("N")
            return
        result.append(str(node.value))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(result)
def deserialize(data):
    vals=data.split(",")
    print(f"The result of split is {vals}")
    global i
    i=0
    def dfs():
        global i
        if vals[i]=="N":
            i = i+1
            return None
        node = TreeNode(int(vals[i]))
        i = i+1
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()
test_case = [[1,2,3,None,None,4,5],[]]
for tree_list in test_case:
    if len(tree_list)==0:
        print("[]")
        break
    else:
        root = generate_tree_from_list(tree_list)
        print(f"Original Tree: {root}")
    se = serialize(root)
    print(f"The Result of Serialize is {se}")
    print(f"The result of Deserialize is {deserialize(se)}")