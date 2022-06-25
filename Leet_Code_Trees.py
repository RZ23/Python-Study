from binarytree import tree,Node
import binarytree
from collections import deque
import sys

class TreeNode(binarytree.Node):
    def __init__(self,values=0,right=None,left = None):
        self.val = values
        self.right = right
        self.left = left
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
def tree_level_to_list(root):
    if not root:
        return [None]
    result = []
    q = deque()
    q.append(root)
    while q:
        node = q.popleft()
        if node is None:
            result.append(None)
        else:
            result.append(node.val)
            q.append(node.left)
            q.append(node.right)
    i=len(result)-1
    while i>0 and result[i] is None:
        i=i-1
    return result[:i+1]
def tree_level_to_list_with_queue_and_null_value(root):
    if not root:
        return []
    result = []
    q= deque()
    q.append(root)
    while q:
        node = q.popleft()
        if node is None:
            result.append(None)
        else:
            result.append(node.val)
            q.append(node.left)
            q.append(node.right)
    return result
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
print("***** Method One: ")
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
print("***** Method Two *****")
def diameterOfBinaryTree_me2(root):
    max_diameter=[0]
    def dfs(root):
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        max_diameter[0] = max(max_diameter[0],left+right)
        return 1+max(left,right)
    dfs(root)
    return max_diameter[0]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The Diameter of tree is {diameterOfBinaryTree_me2(root)}")
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

print("---------------------129. Sum Root to Leaf Numbers-------------------------")
def sumNumbers(root):
    def dfs(cur_node,num):
        if not cur_node:
            return 0
        num = num*10+cur_node.val
        if not cur_node.left and not cur_node.right:
            return num
        return dfs(cur_node.left,num)+dfs(cur_node.right,num)
    return dfs(root,0)
test_case = [[1,2,3],[4,9,0,5,1]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The Sum Root to Leaf Numbers is {sumNumbers(root)}")
print("---------------------96. Unique Binary Search Trees-------------------------")
def numTrees(n):
  """
    numsTree[4] = numsTree[0]*numsTree[3]+
                  numsTree[1]*numsTree[2]+
                  numsTree[2]*numsTree[1]+
                  numsTree[4]*numsTree[0]
  """
  numTree = [1]*(n+1)
  # o node is 1 tree
  # 1 nnode is 1 tree
  for node in range(2,n+1):
      total = 0
      for root in range(1,node+1):
          left = root-1
          right = node-root
          total=total+numTree[left]*numTree[right]
      numTree[node]=total
  return numTree[n]
for i in range(1,20):
    print(f"For {i} node(s), it could construct {numTrees(i)} unique BST")
print("---------------------199. Binary Tree Right Side View-------------------------")
print("***** Method One: BFS *****")
def rightSideView_bfs(root):
    result = []
    q = deque()
    q.append(root)
    while q:
        rightSide = None
        qLen = len(q)
        for i in range(qLen):
            node=q.popleft()
            if node:
                rightSide=node
                q.append(node.left)
                q.append(node.right)
        if rightSide:
            result.append(rightSide.val)
    return result
test_case = [[1,2,3,None,5,None,4],[1,None,3],[],[2,1,3],[5,1,4,None,None,3,6]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print("The tree is ")
    print(root)
    print(f"And the Right View is {rightSideView_bfs(root)}")
print("***** Method One: BFS *****")
def rightSideView_dfs(root):
    """
    using result to identify the length, if the length of the result is less than the depth variable
    means need to add it the final result, else, means this level already scanned, and move to next level
    """
    def rightSideViewDFS(node,depth,result):
        if not node:
            return
        if depth>len(result):
            result.append(node.val)
        rightSideViewDFS(root.right,depth+1,result)
        rightSideViewDFS(root.left,depth+1,result)
    result = []
    rightSideViewDFS(root,1,result)
    return result
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print("The tree is ")
    print(root)
    print(f"And the Right View is {rightSideView_bfs(root)}")
print("---------------------230. Kth Smallest Element in a BST-------------------------")
print("***** Method One: in-order travel and return *****")
def kthSmallest_in_order_travel(root,k):
    def in_order_travel(root):
        if not root:
            return []
        else:
            return in_order_travel(root.left)+[root.val]+in_order_travel(root.right)
    result = in_order_travel(root)
    return result[k-1]
test_case = [[3,1,4,None,2],1],[[5,3,6,2,4,None,None,1],3]
for node_list,k in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The {k}th smallest element in this BST is {kthSmallest_in_order_travel(root,k)}")
print("***** Method Two: In-Order Travel and without recursive *****")
def kthSmallest_in_order_without_array(root,k):
   result = []
   def search(root,result):
       if not root:
           return result
       search(root.left,result)
       result.append(root.val)
       search(root.right,result)
   search(root,result)
   return result[k-1]
for node_list,k in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The {k}th smallest element in this BST is {kthSmallest_in_order_without_array(root,k)}")
print("***** Method Three: Iteration *****")
def kthSmallest_recurssive(root,k):
    n=0
    stack=[]
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        n=n+1
        if n==k:
            return cur.val
        cur=cur.right
for node_list,k in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The {k}th smallest element in this BST is {kthSmallest_recurssive(root,k)}")
print("---------------------337. House Robber III-------------------------")
"""
each node has a tuple variable [with_this_node, without_this_node]
"""
def rob(root):
    def dfs(root):
        if not root:
            return [0,0]
        left_part = dfs(root.left)
        right_part = dfs(root.right)
        withRoot = root.val+left_part[1]+right_part[1]
        withoutRoot = max(left_part)+max(right_part)
        return [withRoot,withoutRoot]
    return max(dfs(root))
test_case = [[3,2,3,None,3,None,1],[3,4,5,1,3,None,1]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print("The Tree is:")
    print(root)
    print(f"Maximum amount of money the thief can rob is {rob(root)}")
print("---------------------102. Binary Tree Level Order Traversal-------------------------")
def levelOrder(root):
    if root is None:
        return []
    q = deque()
    result = []
    if root:
        q.append(root)
    while q:
        level = []
        len_q = len(q)
        for item in range(len_q):
            cur = q.popleft()
            level.append(cur.val)
            if cur.left:
                q.append(cur.left)
            if cur.right:
                q.append(cur.right)
        result.append(level)
    return result
test_case = [[3,9,20,None,None,15,7],[1],[]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"The Level order is {levelOrder(root)}")
print("---------------------105. Construct Binary Tree from Preorder and Inorder Traversal-------------------------")
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    ind = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:ind+1],inorder[:ind])
    root.right = buildTree(preorder[ind+1:],inorder[ind+1:])
    return root
test_case = [[[3,9,20,15,7], [9,3,15,20,7]],[[-1],[-1]]]
for preorder,inorder in test_case:
    root = buildTree(preorder,inorder)
    print(f"The tree based on preorder {preorder} and inorder {inorder} is :")
    print(root)
print("---------------------100. Same Tree-------------------------")
def isSameTree(p,q):
    if not p and not q:
        return True
    if not p or not q or p.val!=q.val:
        return False
    return isSameTree(p.left,q.left) and isSameTree(p.right,q.right)
test_case = [[[1,2,3],[1,2,3]],[[1,2],[1,None,2]],[[1,2,1],[1,1,2]]]
for p_node_list,q_node_list in test_case:
    p=generate_tree_from_list(p_node_list)
    q=generate_tree_from_list(q_node_list)
    print(p)
    print(q)
    print(f"The trees are same: {isSameTree(p,q)}")
print("---------------------1993. Operations on Tree-------------------------")
class lockingTree():
    def __init__(self,parent):
        self.parent = parent
        self.locked = [None]*len(parent)
        self.child= {i:[] for i in range(len(parent))}
        for i in range(1,len(parent)):
            self.child[parent[i]].append(i)
    def lock(self,num,user):
        if self.locked[num]:return False
        self.locked[num]=user
        return True
    def unlock(self,num,user):
        if self.locked[num]!=user:return False
        self.locked[num]=None
        return True
    def upgrade(self,num,user):
        # up to find the ancestors
        i = num
        while i!=-1:
            if self.locked[i]:
                return False
            i=self.parent[i]
        lockedCount,q = 0,deque([num])
        while q:
            n = q.popleft()
            if self.locked[n]:
                self.locked[n]=None
                lockedCount = lockedCount+1
            q.extend(self.child[n])
        if lockedCount>0:
            self.locked[num]=user
        return lockedCount>0

parent = [-1, 0, 0, 1, 1, 2, 2]
obj = lockingTree(parent)
print(f"Lock the node 2 with user 2: {obj.lock(2,2)}")
print(f"Unlock the node 2 with user 3: {obj.unlock(2,3)}")
print(f"Unlock the node 2 with user 2: {obj.unlock(2,2)}")
print(f"Lock the node 4 with user 5: {obj.lock(4,5)}")
print(f"Upgrade the node 0 with user 1: {obj.upgrade(0,1)}")
print(f"Lock the node 0 with user 1: {obj.lock(0,1)}")
print("---------------------114. Flatten Binary Tree to Linked List-------------------------")
print("***** Method One: Pre-order and generate the tree *****")
def flatten(root):
    def pre_order_generate_list(root):
        if not root:
            return []
        else:
            return [root.val]+pre_order_generate_list(root.left)+pre_order_generate_list(root.right)
    node_list = pre_order_generate_list(root)
    if len(node_list)==0:
        return []
    root = TreeNode(node_list[0])
    temp_root = root
    for i in range(1,len(node_list)):
        temp_root.left = None
        temp_root.right = TreeNode(node_list[i])
        temp_root = temp_root.right
    return root
test_case = [[1,2,5,3,4,None,6],[],[0]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print("Original Tree:")
    print(root)
    print(f"The flatten tree is :")
    root = flatten(root)
    print(root)
    print(f"The Flatten List is {tree_level_to_list(root)}")
    # print(tree_level_to_list_with_queue_and_null_value(root))
print("***** Method Two: Modify the tree in-place *****")
def flatten_in_place(root):
    def dfs(root):
        if not root:
            return None
        left_tail= dfs(root.left)
        right_tail = dfs(root.right)
        if root.left:
            left_tail.right = root.right
            root.right = root.left
            root.left = None
        last = right_tail or left_tail or root
        return last
    dfs(root)
    return root
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print("Original Tree:")
    print(root)
    print(f"The flatten tree is :")
    root = flatten_in_place(root)
    print(root)
    print(f"The Flatten List is {tree_level_to_list(root)}")
    # print(tree_level_to_list_with_queue_and_null_value(root))
print("---------------------894. All Possible Full Binary Trees-------------------------")
'''A full binary tree is a binary tree where each node has exactly 0 or 2 children'''

def allPossibleFBT(n):
    if n%2==0:
        return []
    """
    using hashmap to reduce the computing time
    """
    dp ={0:[],1:[TreeNode(0,None,None)]}
    # Return the list of full-binary tree
    def backtracking(n):
        if n in dp:
            return dp[n]
        res = []
        # since it is from 0 to n-1, and for the n nodes, there are n-1 children nodes without the root
        for l in range(n):
            r = n-1-l
            leftTrees, rightTrees= backtracking(l),backtracking(r)

            for t1 in leftTrees:
                for t2 in rightTrees:
                    res.append(TreeNode(0,left = t1,right = t2))
        dp[n] = res
        return res
    return backtracking(n)
test_case = [7,3]
for n in test_case:
    print(f"All the possible of Full Binary Tree(s) of the {n} nodes:")
    root_list = allPossibleFBT(n)
    for i in range(len(root_list)):
        root = root_list[i]
        print(root)
        print(tree_level_to_list_with_queue_and_null_value(root))
print("***** Methond Two: Print the result with Null value")
def allPossibleFBT_format_result(n):
    # travel the tree and add all the left and right children in the result list
    def display(node,result_list):
        if not node:
            return
        if node.left:
            result_list.append(node.left.val)
        else:
            result_list.append(-sys.maxsize)
        if node.right:
            result_list.append(node.right.val)
        else:
            result_list.append(-sys.maxsize)
        display(node.left,result_list)
        display(node.right,result_list)
    # hashmap to reduce the computing time
    hm = {}
    def allPossibleFBT(n):
        if n not in hm:
            list = []
            if n==1:
                list.append(TreeNode(0,None,None))
            elif (n%2==1):
                for left_side in range(n):
                    right_side = n-left_side-1
                    leftTrees = allPossibleFBT(left_side)
                    rightTrees =allPossibleFBT(right_side)
                    for left in range(len(leftTrees)):
                        for right in range(len(rightTrees)):
                            node = TreeNode(0,None,None)
                            node.left = leftTrees[left]
                            node.right =rightTrees[right]
                            list.append(node)
            hm[n] = list
        return hm[n]
    list = allPossibleFBT(n)
    for root in range(len(list)):
        al = []
        al.append(list[root].val)
        display(list[root],al)
        print("[",end = "")
        for i in range(len(al)):
            if (i!=len(al)-1):
                if(al[i]==-sys.maxsize):
                    print("null, ",end = "")
                else:
                    print(al[i],end = ",")
            else:
                if(al[i]==-sys.maxsize):
                    print("null]",end = "")
                else:
                    print(al[i],end = "]")
        print()
for n in test_case:
    print(f"{n} nodes:")
    allPossibleFBT_format_result(n)
print("---------------------513. Find Bottom Left Tree Value-------------------------")
print("***** Method One: Using Left Side View and Return the last level's first item *****")
def findBottomLeftValue(root):
    result =[]
    q = deque()
    q.append(root)
    while q:
        left_node = None
        qLen = len(q)
        for i in range(qLen):
            node = q.popleft()
            if node:
                left_node=node
                q.append(node.right)
                q.append(node.left)
        if left_node:
            result.append(left_node.val)
    print(result)
    return result[-1]

test_case = [[2,1,3],[1,2,3,4,None,5,6,None,None,None,None,7]]
for nodes_list in test_case:
    root = generate_tree_from_list(nodes_list)
    print(root)
    print(f"The Bottom left node is  {findBottomLeftValue(root)}")
print("***** Method Two: Right to Left Level Travel ******")
def findBottomLeftValue_right_left_level_order(root):
    q = deque()
    q.append(root)
    result=[0]
    while q:
        deq_len = len(q)
        for i in range(deq_len):
            node = q.popleft()
            if node:
                q.append(node.right)
                q.append(node.left)
                result[0]=node.val
    return result[0]
for nodes_list in test_case:
    root = generate_tree_from_list(nodes_list)
    print(root)
    print(f"The Bottom left node is  {findBottomLeftValue_right_left_level_order(root)}")
print("***** Method Three: Right to Left Level Order, without len() in function *****")
def findBottomLeftValue_right_left_level_order_without_len(root):
    q = deque()
    q.append(root)
    while q:
        node = q.popleft()
        if node.right:
            q.append(node.right)
        if node.left:
            q.append(node.left)
    return node.val
for nodes_list in test_case:
    root = generate_tree_from_list(nodes_list)
    print(root)
    print(f"The Bottom left node is  {findBottomLeftValue_right_left_level_order_without_len(root)}")
print("---------------------669. Trim a Binary Search Tree-------------------------")
print("***** Method One: Handle Low and High separately *****")
def trimBST(root,low,high):
    # print(low,high)
    def trimLeft(root,low):
        if root and root.val==low:
            root.left = None
            return root
        elif root and root.val<low:
            return trimLeft(root.right,low)
        elif root and root.val>low:
            if root.left:
                if root.left.val<low:
                    root.left =trimLeft(root.left.right,low)
                elif root.left.val>=low:
                    root.left = trimLeft(root.left,low)
        return root
    def trimRight(root,high):
        if root and root.val==high:
            root.right = None
            return root
        if root and root.val>high:
            return trimRight(root.left,high)
        if root and root.val<high:
            if root.right:
                if root.right.val>high:
                    root.right = trimRight(root.right.left,high)
                elif root.right.val<=high:
                    root.right = trimRight(root.right,high)
        return root
    new_root = trimLeft(root,low)
    return trimRight(new_root,high)
test_case = [[[3,0,4,None,2,None,7,None,None,1,None,None,None,5,10,None,None,None,None,None,None,None,None,None,None,None,None,None,6,9],4,10],[[1,0,2],1,2],[[3,2,4,1],2,4]]
for nodes_list,low,high in test_case:
    root = generate_tree_from_list(nodes_list)
    print(root)
    print(f"Trim the tree with boundry {low} to {high} is:")
    print(trimBST(root,low,high))
print("***** Method Two : Recursive handle low and high at the same time  *****")
def trimBST_Recursive(root,low,high):
    if not root:
        return None
    if root.val >high:
        return trimBST_Recursive(root.left,low,high)
    if root.val<low:
        return trimBST_Recursive(root.right,low,high)
    root.left = trimBST_Recursive(root.left,low,high)
    root.right=trimBST_Recursive(root.right,low,high)
    return root
for nodes_list,low,high in test_case:
    root = generate_tree_from_list(nodes_list)
    print(root)
    print(f"Trim the tree with boundary {low} to {high} is:")
    print(trimBST_Recursive(root,low,high))
print("---------------------112. Path Sum-------------------------")
def hasPathSum(root,targetSum):
    if not root:
        return False
    if root.val==targetSum and root.left is None and root.right is None:
        return True
    return hasPathSum(root.left,targetSum-root.val) or hasPathSum(root.right,targetSum-root.val)
test_case = [[[5,4,8,11,None,13,4,7,2,None,None,None,1],22],[[1,2,3],5],[[],0],[[-2,None,-3],-5]]
for node_list,targetSum in test_case:
    root = generate_tree_from_list(node_list)
    print(root)
    print(f"There has the path sum to target {targetSum}:{hasPathSum(root,targetSum)}")
