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

print("---------------------226. Invert Binary Tree-------------------------")
def invertTree(root):
    if not root:
        return None
    temp = root.left
    root.left = root.right
    root.right = temp
    invertTree(root.left)
    invertTree(root.right)
    return root
test_case = [[4,2,7,1,3,6,9],[2,1,3],[]]
for nodes_list in test_case:
    print(generate_tree_from_list(nodes_list))
    print(f"There invert Tree is {invertTree(generate_tree_from_list(nodes_list))}")
print("---------------------617. Merge Two Binary Trees-------------------------")
def mergeTrees(root1,root2):
    if not root1 and not root2:
        return None
    v1 = root1.val if root1 else 0
    v2 = root2.val if root2 else 0
    root = TreeNode(v1+v2)
    root.left = mergeTrees(root1.left if root1 else None,root2.left if root2 else None)
    root.right = mergeTrees(root1.right if root1 else None,root2.right if root2 else None)
    return root
test_case = [[1,3,2,5],[2,1,3,None,4,None,7]],[[1],[1,2]]
for nodes_list1,nodes_list2 in test_case:
    root1 = generate_tree_from_list(nodes_list1)
    root2 = generate_tree_from_list(nodes_list2)
    print("Original Tree one:")
    print(root1)
    print("Original Tree two:")
    print(root2)
    print(f"The merged Tree is {mergeTrees(root1,root2)}")
print("---------------------108. Convert Sorted Array to Binary Search Tree-------------------------")
def sortedArrayToBST(nums):
   def helper(l,r):
       if l>r:
           return None
       mid = (l+r)//2
       root = TreeNode(nums[mid])
       root.left = helper(l,mid-1)
       root.right = helper(mid+1,r)
       return root
   return helper(0,len(nums)-1)
test_case = [[-10,-3,0,5,9],[1,3]]
for nodes_list in test_case:
    print(f"The BST build based on {nodes_list} is {sortedArrayToBST(nodes_list)}")
print("---------------------98. Validate Binary Search Tree-------------------------")
def isValidBST(root):
   def valid(root,left_bounday,right_bounday):
       if not root:
           return True
       if (root.val<left_bounday) or (root.value>right_bounday):
           return False
       return valid(root.left,left_bounday,root.val) and valid(root.right,root.val,right_bounday)
   return valid(root,float("-inf"),float("inf"))
test_case = [[2,1,3],[5,1,4,None,None,3,6],[5,1,6,None,None,3,8],[5,1,7,None,None,6,8]]
for nodes_list in test_case:
    root = generate_tree_from_list(nodes_list)
    print(f"{root}")
    print(f"The tree is a BST: {isValidBST(root)}")
print("---------------------543. Diameter of Binary Tree-------------------------")
def diameterOfBinaryTree(root):
    result = [0]
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        result[0] = max(result[0],2+left-1+right-1)
        return 1+max(left,right)
    dfs(root)
    return result[0]
test_case = [[1,2,3,4,5],[1,2]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The Diameter of tree is {diameterOfBinaryTree(root)}")
print("---------------------120. Triangle-------------------------")
print("***** Method One: Bottom to Top *****")
def minimumTotal(triangle):
    dp = [0]*(len(triangle)+1)
    for row in triangle[::-1]:
        for i,num in enumerate(row):
            dp[i]=num+min(dp[i],dp[i+1])
    return dp[0]
test_case = [[[2],[3,4],[6,5,7],[4,1,8,3]],[[-10]]]
for triangle in test_case:
    print(f"The minimum path sum from top to bottom of {triangle} is {minimumTotal(triangle)}")
print("***** Method Two: Top to Bottom *****")
def minimumTotal_to_to_bottom(triangle):
   if not triangle:
       return 0
   if len(triangle)==1:
       return triangle[0][0]
   leng = len(triangle)
   for i in range(1,leng):
       # update the boundary
       # left boundary
       triangle[i][0]=triangle[i][0]+triangle[i-1][0]
       # right boundary
       triangle[i][-1]=triangle[i][-1]+triangle[i-1][-1]
   # updated the middle items
   for i in range(2,leng):
       for j in range(1,i):
           triangle[i][j] = triangle[i][j]+min(triangle[i-1][j],triangle[i-1][j-1])
   print(triangle)
   # return the min item of bottom row
   return min(triangle[-1])
for triangle in test_case:
    print(f"The minimum path sum from top to bottom of {triangle} is {minimumTotal_to_to_bottom(triangle)}")