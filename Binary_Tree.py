from collections import deque
# Declare TreeNode Class
# Include key, left and right attribute
class TreeNode:
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None
# Create a turple and a funtion based on the turple to create a Binary Tree
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
# Create a fundtion to display a tree
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
print("Display the Tree:")
display_key(tree2," ")
print(is_bst(tree2))

# Create an Airport class to test a tree, it includes the Airpot_Code and Airport Location
class Airport():
    def __init__(self,Airport_Code,Airport_Location):
        self.Airport_Code = Airport_Code
        self. Airport_Location=Airport_Location
# Create a class for the Binary Search Tree (BST)
# BST has all the attribute as the regular tree but has a parent attribute
class BSTNode():
    def __init__(self,key,value = None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
# Insert a node with the key to the BST
def insert(node, key,value=None):
    if node is None:
        node = BSTNode(key,value)
    elif key<node.key:
        node.left = insert(node.left,key,value)
        node.left.parent = node
    elif key> node.key:
        node.right = insert(node.right,key,value)
        node.right.partent = node
    return node
# Find a node with the key from a BST, return the node
def find(node,key):
    if node is None:
        return None
    if key==node.key:
        return node
    if key<node.key:
        return find(node.left,key)
    if key>node.key:
        return find(node.right,key)
#Test cases for the Airport BST

PEK = Airport("PEK","Beijing")
DEN = Airport("DEN","Denver")
DIK = Airport("DIK","Dickinson")
LAX = Airport("LAX","los Angeles")
CKG = Airport("CKG","Chongqing")
SEA = Airport("SEA","Seattle")
TOL = Airport("TOL","Toledo")
EWR = Airport("EWR","Newark")
# Create the root node
Airport_tree = BSTNode(PEK.Airport_Code,PEK)
# Insert other node(s)
Airport_tree. left= BSTNode(DEN.Airport_Code,DEN)
Airport_tree.right =BSTNode(DIK.Airport_Code,DIK)
# Create a tree, insert the first node the tree without create the root node manually
Airport_Tree2= insert(None, PEK.Airport_Code,PEK)
insert(Airport_Tree2,CKG.Airport_Code,CKG)
insert(Airport_Tree2,DIK.Airport_Code,DIK)
insert(Airport_Tree2,LAX.Airport_Code,LAX)
insert(Airport_Tree2,DEN.Airport_Code,DEN)
insert(Airport_Tree2,SEA.Airport_Code,SEA)
insert(Airport_Tree2,TOL.Airport_Code,TOL)
insert(Airport_Tree2,EWR.Airport_Code,EWR)
# Display the airport BST
display_key(Airport_Tree2)
# Loop search the tree
# while True:
#     airport = input("Please input the Airport Code to Search:").upper()
#     if find(Airport_Tree2,airport) is None:
#         print("There is no Airport with the code: "+airport)
#         break
#     else:
#         print("Found " + find(Airport_Tree2,airport).key + " at " + find(Airport_Tree2,airport).value.Airport_Location)
# find_node2 = find(Airport_Tree2,"LAX")
# print(find_node2.key)
# find_result = []
# find_result.append(find_node)
# find_result.append(find_node2)
# for item in find_result:
#     if item is None:
#         print("Not Find")
#     else:
#         print("Found "+item.key+" at "+item.value.Airport_Location)

#  create a balanced BST from a sorted list/array of key-value pairs

# def make_balanced_BST(list,low = 0,high = None,parent = None):
#     if high is None:
#         high = len(list)-1
#     if high<low:
#         return None
#     mid = (low+high)//2
#     key,value = list[mid]
#     root = BSTNode(key,value)
#     root.parent = parent
#     root.left = make_balanced_BST(list,low,mid-1,root)
#     root.right = make_balanced_BST(list,mid+1,high,root)
#     return root

# Create a Balanced BST with a sorted list
def make_balanced_BST(list,low = 0,high = None,parent = None):
    if high is None:
        high = len(list)-1
    if high<low:
        return None
    mid = (low+high)//2
    key= list[mid]
    root = BSTNode(key)
    root.parent = parent
    root.left = make_balanced_BST(list,low,mid-1,root)
    root.right = make_balanced_BST(list,mid+1,high,root)
    return root


list = [1,2,3,4,5,6,7,8]
balanced_node = make_balanced_BST(list)

# Create a Unbalanced BST
unbalanced_node=BSTNode(0)
for item in list:
    insert(unbalanced_node,item)
print("Unbalanced Tree")
display_key(unbalanced_node,"   ")

# Create a Balanced BST
print("Balanced Tree")
list = [0,1,2,3,4,5,6,7,8]
balanced_node = make_balanced_BST(list)
# display_key(balanced_node,"   ")

# Delete a node from the BST, return the sorted list
# using the In-order, if foudn the node, set to None and remove the None from the final return list
def delete_from_bst(node,key):
    if node is None:
        return []
    if node.key == key:
        node.key=None
    return (remove_None((delete_from_bst(node.left,key))+[node.key]+(delete_from_bst(node.right,key))))
# Customized funtion to print the item(s) in a list
def print_a_list(list):
    for list_item in list:
        print(list_item,end = " ")
    print()
print("Updated List")
delete_node = int(input("please input the key need to be deleted: "))
# print_a_list(delete_from_bst(unbalanced_node,delete_node))
# display_key(make_balanced_BST(delete_from_bst(unbalanced_node,delete_node)),"   ")

def MaxvalueNode(node):
    current = node
    while (current.left is not None):
        current = current.left
    return current
def deleteNode(node,key):
    # base case, if the node is None
    if node is None:
        return None
    # if the node to delete is smaller than the root's key, then it is lies in left
    if key<node.key:
        node.left = deleteNode(node.left,key)
    # if the node to delete is greater than the root's key, then it is lies in left
    elif key>node.key:
        node.right = deleteNode(node.right,key)
    # if found the key, three cases, no child node, one child note, and two child node
    else:
        # if has right only child
        if node.left is None:
            temp= node.right
            node = None
            return temp
        elif node.right is None:
            temp=node.left
            node = None
            return temp
        # has both left and right child, find the max node from the right child
        temp = MaxvalueNode(node.right)
        node.key = temp.key
        node.right = deleteNode(node.right,temp.key)
    return node

display_key(deleteNode(unbalanced_node,delete_node)," ")
deleted_list =inOrder(deleteNode(unbalanced_node,delete_node))
display_key(make_balanced_BST(deleted_list))

# Algorithm for Binary Tree

# Define Class

class Node():
    def __init__(self,value):
        self.value = value
        self. left = None
        self.right = None
node_list = ["a","b","c","d","e","f"]
a= Node("a")
b= Node("b")
c=Node("c")
d= Node("d")
e = Node("e")
f= Node("f")
# print(a.value)
a.left= b
a.right = c
b.left=d
b.right=e
c.right=f

#Breadth_First_Search
def Breadth_First_Search(Node):
    if Node is None:
        return []
    path = []
    queue= deque()
    queue.append(Node)
    while len(queue)>0:
        node = queue.popleft()
        path.append(node.value)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return path
print(Breadth_First_Search(a))

# Deep First Search
def Deep_First_Search(Node):
    path = []
    if Node is None:
        return []
    else:
        path.append(Node.value)
        left = Deep_First_Search(Node.left)
        right= Deep_First_Search(Node.right)
        return path+left+right

print(Deep_First_Search(a))

