from binarytree import tree,Node
import binarytree

class TreeNode(binarytree.Node):
    def __init__(self,values):
        self.val = values
        self.right = None
        self.left = None
def generate_tree_from_list(root):
    if len(root)==0:
        return None
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
print("---------------------110. Balanced Binary Tree-------------------------")
print("***** Method One: DFS *****")
def isBalanced(root):
    def dfs(root):
        # if it is not the Node, then it is the Ture balanced, and height is 0
        if not root:
            return [True,0]
        # if it is the node
        left = dfs(root.left)
        right = dfs(root.right)
        balanced = (left[0] and right[0] and abs(left[1]-right[1])<=1)
        return [balanced,1+max(left[1],right[1])]
    return dfs(root)[0]
test_case =[[3,9,20,None,None,15,7],[1,2,2,3,3,None,None,4,4],[]]
for tree_node_list in test_case:
    root = generate_tree_from_list(tree_node_list)
    print(f"The Tree {root} is balanced: {isBalanced(root)}")
print("***** Method Two: Recursive *****")
def isBalanced_recurssive(root):
    def max_Height(root):
        if not root:
            return 0
        else:
            return 1+max(max_Height(root.left),max_Height(root.right))
    if root is None:
        return True
    else:
        return abs(max_Height(root.left)-max_Height(root.right))<=1 and isBalanced_recurssive(root.left) and isBalanced_recurssive(root.right)
for tree_node_list in test_case:
    root = generate_tree_from_list(tree_node_list)
    print(f"The Tree {root} is balanced: {isBalanced_recurssive(root)}")
print("---------------------1448. Count Good Nodes in Binary Tree-------------------------")
def goodNodes(root):
    def dfs(node,max_value):
        if not node:
            return 0
        result = 1 if node.val>=max_value else 0
        max_value = max(max_value,node.val)
        result = result+dfs(node.left,max_value)
        result = result+dfs(node.right,max_value)
        return result
    return dfs(root,root.val)
test_case = [[3,1,4,3,None,1,5],[3,3,None,4,2],[1]]
for nodes_list in test_case:
    print(generate_tree_from_list(nodes_list))
    print(f"There is(are) {goodNodes(generate_tree_from_list(nodes_list))} goodnode(s) in the tree")
