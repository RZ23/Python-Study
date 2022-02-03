class TreeNode:
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None

tree_tuple = ((1,3,None),2,((None,3,4),5,(6,7,8)))
def parse_tuple (data):
    if isinstance(data,tuple) and len(data)==3:
        node= TreeNode(data[1])
        node.left = parse_tuple(data[0])
        node.right = parse_tuple(data[2])
    elif data == None:
        node = None
    else:
        node = TreeNode(data)
    return node

# create the turple based on the tree
def parse_node_to_turple(node):
    if node is None:
        return None
    if node.left is None and node.right is None:
        return node.key
    return (parse_node_to_turple(node.left),node.key,parse_node_to_turple(node.right))

tree2 = parse_tuple(tree_tuple)
print(tree2.key)
# inorder travel the tree
def inOrder(node):
    if node is None:
        return[]
    return (inOrder(node.left)+[node.key]+inOrder(node.right))
# Preorder travel the tree
def PreOrder(node):
    if node is None:
        return []
    return([node.key]+PreOrder(node.left)+PreOrder(node.right))
# Postorder travel the tree
def PostOrder(node):
    if node is None:
         return []
    return(PostOrder(node.left)+PostOrder(node.right)+[node.key])

# Calculate the height of the tree
def tree_height(node):
    if node is None:
        return 0
    else:
        return 1 +max(tree_height(node.left),tree_height(node.right))
#Calculate the node of a tree
def tree_node(tree):
    if tree is None:
        return 0
    else:
        return 1+tree_node(tree.left)+tree_node(tree.right)

print("InOrder Traverse,",end = " ")
print(inOrder(tree2))
print("PreOrder Traverse,",end = " ")
print(PreOrder(tree2))
print("PostOrder Traverse,",end = " ")
print(PostOrder(tree2))
print("Parse to turple:", end=" ")
print(parse_node_to_turple(tree2))
print("Calculate the Height of a tree",end = " ")
print(tree_height(tree2))
print("Calculate the nodes of a tree",end = " ")
print(tree_node(tree2))