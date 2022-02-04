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

def display_key(tree,space = "\t",level = 0):
    if tree is None:
        print(space*level+"@")
        return
    if tree.left is None and tree.right is None:
        print(space*level+str(tree.key))
        return
    display_key(tree.right, space, level+1)
    print(space*level+str(tree.key))
    display_key(tree.left, space, level+1)
# Determinate if it is the Binary Search Tree (BST)
def remove_None(nums):
    return[x for x in nums if x is not None]
def is_bst(tree):
    if tree is None:
        return True, None, None

    is_bst_l, min_l, max_l = is_bst(tree.left)
    is_bst_r, min_r, max_r = is_bst(tree.right)

    is_bst_node = is_bst_l and is_bst_r and (max_l is None or max_l< tree.key) and (min_r is None or min_r >tree.key)
    min_key = min(remove_None([min_l,tree.key,min_r]))
    max_key = max(remove_None([max_l,tree.key,max_r]))
    # print(tree.key,min_key,max_key)
    return  is_bst_node, min_key, max_key

# print("InOrder Traverse,",end = " ")
# print(inOrder(tree2))
# print("PreOrder Traverse,",end = " ")
# print(PreOrder(tree2))
# print("PostOrder Traverse,",end = " ")
# print(PostOrder(tree2))
# print("Parse to turple:", end=" ")
# print(parse_node_to_turple(tree2))
# print("Calculate the Height of a tree",end = " ")
# print(tree_height(tree2))
# print("Calculate the nodes of a tree",end = " ")
# print(tree_node(tree2))
# print("Display the Tree:")
# display_key(tree2," ")
# print(is_bst(tree2))

class Airport():
    def __init__(self,Airport_Code,Airport_Location):
        self.Airport_Code = Airport_Code
        self. Airport_Location=Airport_Location
class BSTNode():
    def __init__(self,key,value = None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

def insert(node, key,value):
    if node is None:
        node = BSTNode(key,value)
    elif key<node.key:
        node.left = insert(node.left,key,value)
        node.left.parent = node
    elif key> node.key:
        node.right = insert(node.right,key,value)
        node.right.partent = node
    return node
def find(node,key):
    if node is None:
        return None
    if key==node.key:
        return node
    if key<node.key:
        return find(node.left,key)
    if key>node.key:
        return find(node.right,key)

PEK = Airport("PEK","Beijing")
DEN = Airport("DEN","Denver")
DIK = Airport("DIK","Dickinson")
LAX = Airport("LAX","los Angeles")
CKG = Airport("CKG","Chongqing")
SEA = Airport("SEA","Seattle")
Airport_tree = BSTNode(PEK.Airport_Code,PEK)
Airport_tree. left= BSTNode(DEN.Airport_Code,DEN)
Airport_tree.right =BSTNode(DIK.Airport_Code,DIK)
# display_key(Airport_tree)
Airport_Tree2= insert(None, PEK.Airport_Code,PEK)
insert(Airport_Tree2,CKG.Airport_Code,CKG)
insert(Airport_Tree2,DIK.Airport_Code,DIK)
insert(Airport_Tree2,LAX.Airport_Code,LAX)
insert(Airport_Tree2,DEN.Airport_Code,DEN)
insert(Airport_Tree2,SEA.Airport_Code,SEA)
display_key(Airport_Tree2)
find_node = find(Airport_Tree2,"LAS")
find_node2 = find(Airport_Tree2,"LAX")
print(find_node2.key)
find_result = []
find_result.append(find_node)
find_result.append(find_node2)
for item in find_result:
    if item is None:
        print("Not Find",end =" ")
    else:
        print("find: "+item.value.Airport_Location,end = " ")
