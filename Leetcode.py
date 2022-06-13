import binarytree
from binarytree import Node,tree
from binarytree import Node, tree
from collections import deque
from collections import Counter
from collections import defaultdict
import collections
import heapq
# Helper Function
def print_matrix(matrix):
    row = len(matrix)
    col = len(matrix[0])
    for r in range(row):
        for c in range(col):
            print(matrix[r][c],end = "|")
        print()
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
    return node_list
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
class ListNode():
    def __init__(self,val=0,next = None):
        self.val = val
        self.next = next
def generate_linked_list_with_val(lst):
    if len(lst)==0:
        return None
    if len(lst)==1:
        return ListNode(lst[0])
    linked_list = []
    for i in range(len(lst)):
        linked_list.append(ListNode(lst[i]))
    for i in range(len(lst)-1):
        linked_list[i].next = linked_list[i+1]
    return linked_list[0]
def display_linked_list_with_val(node):
    if not node:
        print("[]")
    while node is not None:
        if node.next:
            print(node.val,end = "->")
        else:
            print(node.val)
        node = node.next
    # print()

def find_last_link_list(node):
    while node.next:
        node=node.next
    return node
def find_kth_linked_node(node,k):
    if k<0:
        return None
    while k>0:
        node = node.next
        k =k-1
    return node
#Roman to Int
print("---------------------Translate Roman Numbers to Integer Number -------------------------")
def romanToInt(s):
    dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    sum = 0
    s = list(s)
    if len(s)>15 or len(s)<1:
        return -1
    for char in s:
        if char not in dict.keys():
            return -1
    i =0
    while i<len(s):
        if i+1<(len(s)):
            if dict[s[i+1]]>dict[s[i]]:
                sum = sum+dict[s[i+1]]-dict[s[i]]
                i=i+2
                # print(sum,i)
            else:
                sum = sum+dict[s[i]]
                i = i+1
                # print(sum)
        else:
            sum = sum + dict[s[i]]
            i=i+1
            # print(sum)
    return sum
str_list= ["III","LVIII","MCMXCIV"]
for item in str_list:
    print("Translate {} to int is {}".format(item, romanToInt(item)))
import random
# Int to Roman with calc
print("---------------------Translate Integer Numbers to Roman Number -------------------------")
print("Method one: calculate the numbers with while loop")
def intToRoman_calc(num):
    dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    key_lst = list(dict.keys())
    value_lst = list(dict.values())
    operation = [2,5]
    s = ""
    operation_index = 1
    index = 3
    while(num>0):
        k = num//pow(10,index)
        if k==4:
            s= s+key_lst[value_lst.index(pow(10,index))]+key_lst[value_lst.index((k+1)*(pow(10,index)))]
        elif k<4 and k>0:
            s = s + (key_lst[value_lst.index(pow(10, index))] * k)
        elif k==9:
            s= s+key_lst[value_lst.index(pow(10,index))]+key_lst[value_lst.index((k+1)*(pow(10,index)))]
        elif k>5 and k<9:
            s= s+key_lst[value_lst.index(5*pow(10,index))]+key_lst[value_lst.index(pow(10,index))]*(k-5)
        elif k==5:
            s = s + (key_lst[value_lst.index(k*pow(10, index))])
        elif k!=0:
            s = s + (key_lst[value_lst.index(pow(10,index))] * k)
        num = num % pow(10, index)
        index = index - 1
    return s
num_list = [3,58,1994,3999]
for i in num_list:
    print("The Int {} translate to Roman Number is {}".format(i,intToRoman_calc(i)))
# i=0
# count = 0
# while count<3:
#     i = random.randint(1*pow(10,count),9*(pow(10,count)))
#     print("The Int {} translate to Roman Number is {}".format(i,intToRoman_calc(i)))
#     count = count+1
# num = random.randint(1000,3999)
# print("The Int {} translate to Roman Number is {}".format(num,intToRoman_calc(num)))
# dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# key_lst = list(dict.keys())
# value_lst = list(dict.values())
# print(key_lst[value_lst.index(1000)])

# Int to Roman with List
print("Method Two: With Build-In list")
def intToRoman_list(num):
    M = ["","M","MM","MMM"]
    D = ["","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"]
    X=["","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"]
    I = ["","I","II","III","IV","V","VI","VII","VIII","IX"]
    return M[num//1000]+D[(num%1000)//100]+X[num%100//10]+I[num%10]
num_list = [3,58,1994]
for i in num_list:
    print("The Int {} translate to Roman Number is {}".format(i,intToRoman_list(i)))
# two sum
print("---------------------Sum two list with target number -------------------------")
print("Method One: with Dictionary:")
def two_sum_with_dict(lst,target):
    dict = {}
    remain = 0
    for i in range(len(lst)):
        remain = target-lst[i]
        if remain in dict.keys():
            return dict[remain],i
        dict[lst[i]] = i
nums = [2,7,11,15]
target = 9
test_case = [([2,7,11,15],9),([3,2,4],6),([3,3],6)]
for item in test_case:
    print("index numbers to get the target number {} within the list {} are {} ".format(target,item[0],two_sum_with_dict(item[0],item[1])))
print("Method Two: Within two loops：")
def two_sum_iteration(lst,target):
    for i in range(len(lst)):
        for j in range(i+1,len(lst)):
            if lst[i]+lst[j]==target:
                return i,j
for item in test_case:
    print("index numbers to get the target number {} within the list {} are {}".format(target,item[0],two_sum_iteration(item[0],item[1])))

print("---------------------Add two linked list -------------------------")


def addTwoNumbers(l1,l2):
    i=0
    j=0
    listNode_lst = []
    new_value_high_digit = 0
    while i<len(l1) and j<len(l2):
        new_value = l1[i]+l2[j]+new_value_high_digit
        i=i+1
        j=j+1
        new_value_low_digit = new_value%10
        new_value_high_digit = new_value//10
        new_node = ListNode(new_value_low_digit,)
        # print("new_value:{}, new_value_low_digit:{},new_value_high_digit:{}".format(new_value,new_value_low_digit,new_value_high_digit))
        listNode_lst.append(new_node)
    if i==len(l1) and j<len(l2):
        while j<len(l2):
            new_value = l2[j] + new_value_high_digit
            j=j+1
            new_value_low_digit = new_value % 10
            new_value_high_digit = new_value // 10
            new_node = ListNode(new_value_low_digit,)
            # print("new_value:{}, new_value_low_digit:{},new_value_high_digit:{}".format(new_value, new_value_low_digit,new_value_high_digit))
            listNode_lst.append(new_node)
    if i<len(l1) and j==len(l2):
        while i<len(l1):
            new_value = l1[i] + new_value_high_digit
            i = i+1
            new_value_low_digit = new_value % 10
            new_value_high_digit = new_value // 10
            new_node = ListNode(new_value_low_digit,)
            # print("new_value:{}, new_value_low_digit:{},new_value_high_digit:{}".format(new_value, new_value_low_digit, new_value_high_digit))
            listNode_lst.append(new_node)
    if new_value_high_digit!=0:
        new_node = ListNode(new_value_high_digit,)
        listNode_lst.append(new_node)
    # print(listNode_lst)
    for i in range(len(listNode_lst)-1):
        # print(type(listNode_lst[i]))
        listNode_lst[i].next = listNode_lst[i+1]
        # print(type(listNode_lst[i]))
    return listNode_lst[0]
def print_linkList(listnode):
    lst = []
    while listnode is not None:
        lst.append(listnode.val)
        listnode = listnode.next
    return lst
test_case = [([2,4,3],[5,6,4]),([0],[0]),([9,9,9,9,9,9,9],[9,9,9,9])]
for item in test_case:
    print("the sum of linked list {} and {} is {}".format(item[0],item[1],print_linkList(addTwoNumbers(item[0],item[1]))))

print("---------------------Reversal Linked List -------------------------")
class LinkedListNode():
    def __init__(self,value,next = None):
        self.value = value
        self.next = next
head = [1,2,3,4,5]
linked_list = []
for item in head:
    linked_list.append(LinkedListNode(item))
for i in range(len(linked_list)-1):
    linked_list[i].next = linked_list[i+1]
def reverse_linked_list(node):
    current = node
    prev = None
    while current is not None:
        next = current.next
        current.next = prev
        prev = current
        current = next
    return prev
def reverse_linked_list_recurssive(node,prev = None):
    if node is None:
        return prev
    next_node = node.next
    node.next = prev
    return reverse_linked_list_recurssive(next_node,node)
def print_linked_list(node):
    print_out_list = []
    while node is not None:
        print_out_list.append(node.value)
        node = node.next
    print(print_out_list)
def print_linked_list_with_return(node):
    print_out_list = []
    while node is not None:
        print_out_list.append(node.value)
        node = node.next
    return print_out_list
print_linked_list(reverse_linked_list(linked_list[0]))
print_linked_list(reverse_linked_list_recurssive(linked_list[-1]))
print("---------------------Right Side View the Binary Tree -------------------------")

# root = [1,2,3,None,5,None,4]
# root = [1,None,3]
# root = [0,1,2,3,None,None,None,4]
# root = [1,2,2,3,3,None,None,4,4]
root = [3,9,20,None,None,15,7]
#root = [1,2,2,3,None,None,3,4,None,None,4]
# root = [None]
# root = [1]
# root = []
# root = [3,9,20,None,None,15,7]
# Generate the Node
node_list = []
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
    return node_list
node_list=generate_tree_from_list(root)
# for i in range(len(root)):
#     if root[i] is not None:
#         node_list.append(Node(root[i]))
#     else:
#         node_list.append(None)
# # Set the Left/Right child for each node
# for i in range(len(node_list)//2):
#     if node_list[i] is not None:
#         left_child =2*i+1
#         right_child = 2*i+2
#         if left_child<len(node_list):
#             node_list[i].left = node_list[left_child]
#         if right_child<len(node_list):
#             node_list[i].right = node_list[right_child]
# Print the tree for the refrence
print(node_list[0])
print("Method One: With recurssive")
def rightSideView(node,result_list=[],level=1):
    if not node:
        return
    if level > len(result_list):
        result_list.append(node.value)
    rightSideView(node.right,result_list,level+1)
    rightSideView(node.left,result_list,level+1)
    return result_list
print(rightSideView(node_list[0]))

print("Method Two: using the while loop")
def rightSideView_Iteration(root):
    if root is None:
        return []
    result,current = [],[root]
    while current:
        next_level = []
        for node in current:
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        result.append(node.value)
        current = next_level
    return result
print(rightSideView_Iteration(node_list[0]))

print("---------------------104. Maximum Depth of Binary Tree-------------------------")
print("***** Method One: Recursive *****")
def max_depth(node):
    if node is None:
        return 0
    else:
        return max(max_depth(node.left),max_depth(node.right))+1
print("The depth of the tree is {}".format(max_depth(node_list[0])))
print("***** Method Two: Iterative and DFS *****")
def max_depth_iterative(node):
    if not node:
        return 0
    level = 0
    q = deque()
    q.append(node)
    while len(q)>0:
        for i in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level = level+1
    return level
print("The depth of the tree is {}".format(max_depth_iterative(node_list[0])))
print("***** Method Three: BFS and Stack *****")
def max_depth_bfs_stack(node):
    if not node:
        return None
    stack = [[node,1]]
    result = 0
    while stack:
        node,depth = stack.pop()
        if node:
            result = max(result,depth)
            stack.append([node.left,depth+1])
            stack.append([node.right,depth+1])
    return result
print("The depth of the tree is {}".format(max_depth_bfs_stack(node_list[0])))
print("---------------------Balanced Binary Tree-------------------------")
def isBalanced(node):
    if node is None:
        return True
    # must check both the left subtree and right subtree are balanced Binary Tree
    return abs(max_depth(node.left)-max_depth(node.right))<=1 and isBalanced(node.left) and isBalanced(node.right)
print("The tree is the balanced binary tree:{}".format(isBalanced(node_list[0])))

print("---------------------Palindrome Number-------------------------")
def Palindrome_Number(num):
    if num<0:
        return False
    else:
        num_str = str(num)
        last_index = -1
        for i in range(len(num_str)//2):
            if num_str[i]!=num_str[last_index-i]:
                return False
    return True
num_list = [121,-121,10]
for num in num_list:
    print("the number {} is Palindrome Number:{}".format(num,Palindrome_Number(num)))

print("---------------------Longest Common Prefix-------------------------")
def longestCommonPrefix(lst):
    result = lst[0]
    for i in range(1,len(lst)):
        result_size = len(result)
        current=lst[i]
        boundry = min(result_size,len(current))
        s = ""
        for j in range(boundry):
            if result[j]==current[j]:
                s =s+result[j]
            else:
                break
        result = s
    if result=="":
        return ""
    else:
        return result
lst = ["flower","flow","flight"]
# lst = ["dog","racecar","car"]
lst = ["cir","car"]
print("The Longest Common Prefix of {} is {}".format(lst,longestCommonPrefix(lst)))
print("---------------------Binary Tree Level Order Traversal-------------------------")
def levelOrder(node):
    if node is None:
        return []
    if node.left is None and node.right is None:
        return [node.value]
    dq = deque()
    result = []
    dq.append(node)
    while dq:
        level = len(dq)
        current_node =dq.popleft()
        result.append(current_node.value)
        if current_node.left is not None:
            dq.append(current_node.left)
        if current_node.right is not None:
            dq.append(current_node.right)

    return result
print("Without []")
print(levelOrder(node_list[0]))

def leverOrder_array(node):
    if node is None:
        return []
    result=[]
    result_dict= levelOrder_dictionary(node,1)
    for i in result_dict.keys():
        result.append(result_dict[i])
    return result

def levelOrder_dictionary(node,level,dict={}):
    if level not in dict.keys():
        dict[level] = [node.value]
    else:
        dict[level].append(node.value)
    if node.left is not None:
        levelOrder_dictionary(node.left,level+1,dict)
    if node.right is not None:
        levelOrder_dictionary(node.right,level+1,dict)
    return dict
print("Methond One: with Recurssive")
print(leverOrder_array(node_list[0]))
def levelOrder_BFS(node):
    if node is None:
        return []
    dq = deque()
    result = []
    if node:
        dq.append(node)
    while len(dq):
        level = []
        size_dq = len(dq)
        for item in range(size_dq):
            current_node = dq.popleft()
            level.append(current_node.value)
            if current_node.left:
                dq.append(current_node.left)
            if current_node.right:
                dq.append(current_node.right)
        result.append(level)
    return result
print("Method Two: While Loop")
print(levelOrder_BFS(node_list[0]))


print("---------------------Flip Equivalent Binary Trees-------------------------")
root1 = [1,2,3,4,5,6,None,None,None,7,8]
root2 = [1,3,2,None,6,4,5,None,None,None,None,None,None,8,7]
root1_tree = generate_tree_from_list(root1)
root2_tree=generate_tree_from_list(root2)
print(root1_tree[0])
print(root2_tree[0])

def flipEquiv(node1,node2):
    # if node1==node2:
    #     return True
    if not node1 and not node2:
        return True
    if not node1 or not node2:
        return False
    if node1.value!=node2.value:
        return False
    return (flipEquiv(node1.left,node2.right) and
            flipEquiv(node1.right,node2.left)) or \
            (flipEquiv(node1.left,node2.left)
             and flipEquiv(node1.right,node2.right))

print(flipEquiv(root1_tree[0],root2_tree[0]))

print("---------------------Longest Substring Without Repeating Characters-------------------------")

def lengthOfLongestSubstring(s):
    result = []
    result_dict={}
    if len(s)==1:
        return 1
    max =0
    for i in range(len(s)):
        if s[i] not in result:
            result.append(s[i])
            if len(result)>max:
                max=len(result)
        else:
            repeat_index = result.index(s[i])
            new_result = result[repeat_index+1:]
            if len(result)>max:
                max = len(result)
            result = new_result
            result.append(s[i])
    return max
s_list= ["abcabcbb","bbbbb","pwwkew","au","dvdf"]
for s in s_list:
    print("the size of longest substring without repeating character of {} is {}".format(s,str(lengthOfLongestSubstring(s))))

print("---------------------5.Longest Palindromic Substring-------------------------")
print("***** Method One: Helper Function *****")
def longestPalindrome(s):
    longest = ""
    for i in range(len(s)):
        s1 = findlongest(s,i,i)
        if len(s1)>len(longest):
            longest = s1
        s2 = findlongest(s,i,i+1)
        if len(s2)>len(longest):
            longest = s2
    return longest

def findlongest(s,l,r):
    while l>=0 and r<len(s) and s[l]==s[r]:
        l = l-1
        r = r+1
    return s[l+1:r]
test_case = ['babad',"cbbd"]
for item in test_case:
    print("The Longest Palindromic Substring of {} is {}".format(item,longestPalindrome(item)))
print("***** Method Two: choose middle and search left and right (two pointers) *****")
def longestPalindrome_two_pointers(s):
    result = ""
    resLen = 0
    for i in range(len(s)):
        # odd string
        l,r = i,i
        while l>=0 and r<len(s) and s[l]==s[r]:
            if(r-l+1)>resLen:
                resLen = (r-l+1)
                result = s[l:r+1]
            l = l-1
            r = r+1
        # even string
        l,r = i,i+1
        while l>=0 and r<len(s) and s[l]==s[r]:
            if (r-l+1)>resLen:
                resLen = (r-l+1)
                result = s[l:r+1]
            l = l-1
            r = r+1
    return result
test_case = ['babad',"cbbd"]
for item in test_case:
    print("The Longest Palindromic Substring of {} is {}".format(item,longestPalindrome_two_pointers(item)))
print("---------------------20 Valid Parentheses-------------------------")
def isValid_1(s):
    if len(s)%2!=0:
        return False
    symbol_list = []
    dict_map={'{':'}','[':']','(':')'}
    for item in s:
        if item in dict_map.keys():
            symbol_list.append(item)
        else:
            if symbol_list==[]:
                return False
            a = symbol_list.pop()
            if item !=dict_map[a]:
                return False
    return symbol_list==[]
s_list = ["()","()[]{}","(]"]
for item in s_list:
    print("The string {} is {}".format(item,isValid_1(item)))


print("---------------------Merge Two Sorted Lists-------------------------")
class linkedNonde():
    def __init__(self,value = 0,next =None ):
        self.value = value
        self.next = next

lst1 = [1,2,4]
lst2 = [1,3,4]
def generate_linked_list(lst):
    if len(lst)==0:
        return None
    if len(lst)==1:
        return LinkedListNode(lst[0])
    linked_list = []
    for i in range(len(lst)):
        linked_list.append(LinkedListNode(lst[i]))
    for i in range(len(lst)-1):
        linked_list[i].next = linked_list[i+1]
    return linked_list[0]
LinkedList1= generate_linked_list(lst1)
LinkedList2 = generate_linked_list(lst2)
def  mergeTwoLists(node1,node2):
    head = LinkedListNode(0)
    current = head
    while node1 is not None and node2 is not None:
        if node1.value<node2.value:
            current.next= node1
            node1 = node1.next
        else:
            current.next = node2
            node2=node2.next
        current = current.next
    if node1 is not None:
        current.next = node1
    else:
        current.next = node2
    return head.next
def display_linked_list(node):
    while node is not None:
        print(node.value,end = " ")
        node = node.next
display_linked_list(mergeTwoLists(LinkedList1,LinkedList2))
print()
print("---------------------Zigzag Conversion-------------------------")
def Zigzag_convert(s,n):
    if n==1:
        return s
    result = [""] * n
    direct = 1
    rowNumber = 0
    for i in range(len(s)):
        result[rowNumber] = result[rowNumber]+s[i]
        if rowNumber<n-1 and direct ==1:
            rowNumber=rowNumber+1
        elif rowNumber>0 and direct==-1:
            rowNumber = rowNumber-1
        else:
            direct = direct*-1
            rowNumber=rowNumber+direct
    return ''.join(result)
input_list = {"s":"PAYPALISHIRING", "n":3},{"s":"PAYPALISHIRING","n":4}
for item in input_list:
    print("The Zigzag Convert of {} with {} rows is {}".format(item["s"],item["n"],Zigzag_convert(item["s"],item["n"])))

print("---------------------Median of Two Sorted Arrays-------------------------")
print("O(n+m)")
def findMedianSortedArrays_1(lst1,lst2):
    i = 0
    j = 0
    result = []
    while i<len(lst1) and j<len(lst2):
        if lst1[i]<=lst2[j]:
            result.append(lst1[i])
            i = i+1
        else:
            result.append(lst2[j])
            j = j+1
    result = result+lst1[i:]+lst2[j:]
    result_len = len(result)
    if result_len%2==1:
        index = result_len//2
        median = result[index]
    else:
        index = result_len//2
        index_1 = index-1
        median = (result[index_1]+result[index])/2
    return median
test_list=[{"lst1":[1,3], "lst2" :[2]},{"lst1":[1,2],"lst2":[3,4]},{"lst1":[-5, 3, 6, 12, 15],
        "lst2":[-12, -10, -6, -3, 4, 10]},{"lst1":[2,3,5,8],
        "lst2":[10, 12, 14, 16, 18, 20]}]
for item in test_list:
    print("The median of two sorted array {} and {} is {}".format(item["lst1"],item["lst2"],findMedianSortedArrays_1(item["lst1"],item["lst2"])))

print("O(log(n+m)): with binary search")
def findMedianSortedArrays_2(lst1,lst2):
    if len(lst1)>len(lst2):
        return findMedianSortedArrays_2(lst2,lst1)
    array_1 = lst1
    array_2 = lst2
    start = 0
    end = len(array_1)
    X = len(array_1)
    Y =len(array_2)
    while (start<=end):
        partitionX = int((start+end)/2)
        partitionY = int((X+Y+1)/2-partitionX)
        if(partitionX==0):
            X1 = float('-inf')
        else:
            X1 = array_1[partitionX-1]
        if(partitionX==len(array_1)):
            X2 = float('inf')
        else:
            X2 = array_1[partitionX]
        if partitionY==0:
            Y1 = float("-inf")
        else:
            Y1 =array_2[partitionY-1]
        if partitionY==len(array_2):
            Y2=float("-inf")
        else:
            Y2 = array_2[partitionY]
        if((X1<=Y2) and (Y1<=X2)):
            if((X+Y)%2==0):
                median = ((max(X1,Y1)+min(X2,Y2))/2)
                return median
            else:
                median = max(X1,Y1)
                return median
        elif(Y1>X2):
            start = partitionX+1
        else:
            end = partitionX-1
for item in test_list:
    print("The median of two sorted array {} and {} is {}".format(item["lst1"],item["lst2"],findMedianSortedArrays_2(item["lst1"],item["lst2"])))

print("---------------------Reverse Integer-------------------------")
print("Method One: Transfer to String")
def reverse_ingeter_to_string(num):
    if num<0:
        direct = -1
    else:
        direct = 1
    change_str = str(abs(num))
    order = []
    for char in change_str:
        order.append(char)
    s = ""
    for i in range(len(order)):
        char = order.pop()
        s= s+char
    final = int(s)
    final = final*direct
    return final
test_list=[123,-123,120,4002]
for item in test_list:
    print("Reverse the Interger {} is {}".format(item,reverse_ingeter_to_string(item)))
print("Method Two: Divide and Mode")
def reverse_ingeter_by_math(num):
    if num<0:
        direct = -1
    else:
        direct = 1
    result = 0
    while num:
        result = 10*result+num%10
        num = num/10
    if num<pow(2,31)-1 and num>-1*pow(2,31):
        return direct*result
    else:
        return 0
for item in test_list:
    print("Reverse the Interger {} is {}".format(item,reverse_ingeter_to_string(item)))

print("---------------------Remove Duplicates from Sorted Array-------------------------")
def removeDuplicates(lst):
    cur = 0
    pre = lst[0]
    k=0
    for i in range(1,len(lst)):
        while lst[i]!=pre and i <len(lst):
            # print("lst[{}]:{} is not equal {}".format(i, lst[i], pre))
            original_lst = lst.copy()
            new_index = cur + 1
            lst[new_index] = lst[i]
            pre = lst[i]
            cur = cur + 1
            # print(original_lst)
            # print(lst, cur, lst[cur])
        i = i+1
    return lst,cur+1
lst = [[0,0,1,1,1,2,2,3,3,4],[1,1,2]]
for item in lst:
    print(item)
    lst_ori=item.copy()
    print("The result of remove the duplicate sorted array {} is {}".format(lst_ori,removeDuplicates(item)))

print("---------------------String to Integer (atoi)-------------------------")
def myAtoi(s):
    direct = 1
    result = 0
    num_lst = ['1','2','3','4','5','6','7','8','9','0']
    last_digit = 0
    postive=False
    negative=False
    s = s.lstrip()
    for index in range(len(s)):
        while index<len(s) and s[index]=="+" and not postive:
            if negative:
                return 0
            if postive:
                return 0
            if s[index]=="+":
                postive = True
            index =index+1
        if index<len(s) and s[index]=="-" and not negative:
            if postive:
                return 0
            if negative:
                return 0
            else:
                direct = -1
                index = index+1
                negative = True
        if index<len(s) and not s[index] in num_lst:
            return 0
        if index<len(s) and s[index] in num_lst:
            while (index<len(s)) and (s[index] in num_lst):
                result = result*10
                result = result+int(s[index])
                index=index+1
            break
    result = result*direct
    if result<-pow(2,31):
        result = -pow(2,31)
    elif result>pow(2,31)-1:
        result = pow(2,31)-1
    return result
test_case = ["4193 with words","   -42","42","abc","Words and 987","-91283472332",
"+-12",".123","+",". ","+- "," ++1"]
for item in test_case:
    print("String {} to Int is {}".format(item,myAtoi(item)))


print("---------------------3Sum-------------------------")
print("Methond One: 3 loop")
def threeSum(lst):
    if len(lst)<=2:
        return []
    result = []
    dict = {}
    for i in range(len(lst)-2):
        for j in range(i+1,len(lst)-1):
            for k in range(j+1,len(lst)):
                if lst[i]+lst[j]+lst[k]==0:
                    tup = [lst[i],lst[j],lst[k]]
                    result.append(tup)
    final_result = []
    dup_check=[]
    for item in result:
        temp_item = sorted(item)
        # print(item,temp_item)
        if temp_item not in dup_check:
            final_result.append(item)
            dup_check.append(temp_item)
    return final_result
test_case = [[-1,0,1,2,-1,-4],[],[0]]
for item in test_case:
    print("the 3 sum of list {} is {}".format(item,threeSum(item)))
print("Method Two: 2 loops and Direct")
def threeSum_2(lst):
    if len(lst)<=2:
        return []
    lst.sort()
    dict = {}
    for i in range(len(lst)):
        dict[lst[i]] = i
    result = set()
    for i in range(len(lst)):
        if i!=0 and lst[i]==lst[i-1]:
            continue
        twoSum = -lst[i]
        for j in range(i+1,len(lst)):
            target = twoSum-lst[j]
            if target in dict and dict[target]>j:
                result.add((-twoSum,lst[j],target))
    return result
for item in test_case:
    print("the 3 sum of list {} is {}".format(item,threeSum_2(item)))

print("---------------------3Sum Closed-------------------------")
print("Method One: 3 Nest For Loop")
def threeSumClosest(lst,target):
    if len(lst)<3 or len(lst)>1000:
        return False
    diff = pow(10,4)
    result = []
    for i in range(len(lst)-2):
        for j in range(i+1,len(lst)-1):
            for k in range(j+1,len(lst)):
                if abs((lst[i]+lst[j]+lst[k])-target)<diff:
                    diff = abs((lst[i]+lst[j]+lst[k])-target)
                    tup = (i,j,k)
                    result.append(tup)
    a = result[-1][0]
    b=result[-1][1]
    c=result[-1][2]
    return lst[a]+lst[b]+lst[c]

test_case = [[[-1,2,1,-4],1],[[0,0,0],1],[[84,49,-47,-56,13,-3,62,-95,23,38,-97,92,34,68,30,90,41,24,-58,83,96,-99,-40,28,-18,-69,-78,95,-62,45,-66,-71,5,94,-42,-66,27,60,-90,-62,87,-22,56,7,-11,75,53,-16,-7,-19,17,18,-14,43,98,-11,0,80,-82,40,5,37,-94,-14,-62,-82,84,23,-9,-68,37,-23,10,26,-22,-52,14,18,-40,-74,-32,47,-87,-81,-68,34,60,75,93,-28,100,-42,0,-87,60,75,-47,7,-57,-61,-2,-96,-18,-98,-3,25,38,-83,60,-12,-62,78,-41,75,-5,89,-97,-1,87,92,57,93,-83,-67,-76,28,-98,-12,22,-2,54,-67,7,99,100,50,5,84,49,-96,-61,-62,-61,29,-59,43,55,30,-10,-22,50,-32,-81,-42,32,55,-94,84,-90,-71,-10,61,56,94,51,8,54,22,22,31],
82]]
for item in test_case:
    print(threeSumClosest(item[0],item[1]))

print("Method Two: For loop and While Loop")

def threeSumClosest_1(num,target):
    if len(num)<3:
        return False
    num.sort()
    print(num)
    diff = float("infinity")
    for i in range(len(num)-2):
        begin = i+1
        end = len(num)-1
        while begin<end:
            if abs(num[i]+num[begin]+num[end]-target)<diff:
                diff = abs(num[i]+num[begin]+num[end]-target)
                result = num[i]+num[begin]+num[end]
            if num[i]+num[begin]+num[end]>target:
                end = end-1
            else:
                begin = begin+1
    return result
test_case = [[[-1,2,1,-4],1]]
for item in test_case:
    print(threeSumClosest_1(item[0],item[1]))

print("---------------------1008. Construct Binary Search Tree from Preorder Traversal-------------------------")
print("print the tree by level")
def print_tree_by_level(node):
    if node is None:
        return None
    dq = deque()
    dq.append(node)
    result = []
    while len(dq)>0:
        # print(dq)
        current_node = dq.popleft()
        if current_node is None:
            result.append(None)
            current_node = dq.popleft()
            result.append(current_node.value)
        else:
            result.append(current_node.value)
        if current_node.left is not None:
            dq.append(current_node.left)
        if current_node.left is None and current_node.right is not None:
            dq.append(None)
        if current_node.right is not None:
            dq.append(current_node.right)
    return result
node_list = [8,5,10,1,7,None,2]
root = generate_tree_from_list(node_list)
print(root[0])
print(print_tree_by_level(root[0]))
print("Construct Binary Search Tree from Preorder Traversal")
def bstFromPreorder(lst):
    if lst is None:
        return None
    elif len(lst)==0:
        return None
    elif len(lst)==1:
        return Node(lst[0])
    else:
        left = []
        right=[]
        node = Node(lst[0])
        for i in range(1,len(lst)):
            if lst[i]<lst[0]:
                left.append(lst[i])
            else:
                right.append(lst[i])
        node.left=bstFromPreorder(left)
        node.right=bstFromPreorder(right)
        # print(node)
    return node
preorder = [8,5,1,7,10,12]
print(bstFromPreorder(preorder))
print(print_tree_by_level(bstFromPreorder(preorder)))

print("---------------------18. 4Sum-------------------------")
def fourSum(lst,target):
    dict = {}
    lst.sort()
    for i in range(len(lst)-1):
        for j in range(i+1,len(lst)):
            sum = lst[i]+lst[j]
            if sum not in dict.keys():
                dict[sum] =[[i,j]]
            else:
                dict[sum].append([i,j])
    # print(dict)
    result_set = set()
    for x in range(len(lst)-3):
        for y in range(x+1,len(lst)-2):
            dif = target - (lst[x]+lst[y])
            # print("dif:{}".format(dif))
            if dif in dict.keys():
                for item in dict[dif]:
                    # print(lst[x],lst[y],item)
                    if y<item[0]:
                        result_set.add((lst[x],lst[y],lst[item[0]],lst[item[1]]))
    # print(result_set)
    final_result = []
    for item in result_set:
        if item not in final_result:
            final_result.append(item)
    return final_result
test_case = [[[1,0,-1,0,-2,2],0],[[2,2,2,2,2],8],[[-5,5,4,-3,0,0,4,-2],4]]
for item in test_case:
    # print(item[0],item[1])
    print(fourSum(item[0],item[1]))
print("---------------------27. Remove Element-------------------------")
def removeElement(lst,val):
    count = 0
    for i in range(len(lst)):
        if lst[i]!=val:
            lst[count] = lst[i]
            count = count+1
    print(lst[:count])
    return count
test_case = [[[3,2,2,3],3],[[0,1,2,2,3,0,4,2],2],[[1,1,1],1]]
for item in test_case:
    print("the length of list {} after remove element {} is {}".format(item[0],item[1],removeElement(item[0],item[1])))
# print("the length of list {} after remove element {} is {}".format([0,1,2,2,3,0,4,2],2,removeElement([0,1,2,2,3,0,4,2],2)))

print("---------------------28. Implement strStr()-------------------------")
def strStr(lst1,lst2):
    lst2_size = len(lst2)
    if len(lst1)<len(lst2):
        return -1
    for i in range(0,len(lst1)-lst2_size+1):
        if lst1[i:i+lst2_size]==lst2:
            return i
    return -1
test_case = [["hello","ll"],["aaaaa","bba"],["a","a"]]
for item in test_case:
    print("the index of string {} in string {} is {}".format(item[1],item[0],strStr(item[0],item[1])))

print("---------------------136. Single Number-------------------------")
print("*** Method One: Sort and Compare ***")
def singleNumber_sort_compare(nums):
    if len(nums)==1:
        return nums[0]
    nums.sort()
    # print(nums)
    p = 0
    c =1
    while c<(len(nums)):
        if c<len(nums) and nums[c]!=nums[p] :
            return nums[p]
        c= c+2
        p=p+2
        # print("c:{},p:{}".format(c,p))
        if p==len(nums)-1:
            return nums[p]
        if c>len(nums):
            return nums[p]
test_case = [[2,2,1],[4,1,2,1,2],[1],[-336,513,-560,-481,-174,101,-997,40,-527,-784,-283,-336,513,-560,-481,-174,101,-997,40,-527,-784,-283,354]]
for item in test_case:
    print("the unduplicated item in array {} is {}".format(item,singleNumber_sort_compare(item)))

print("*** Method Two: Python Build_In Counter ***")
def singleNumber_counter(nums):
    counter_result = Counter(nums)
    # print(counter_result)
    for item in counter_result:
        if counter_result[item]==1:
            return item
for item in test_case:
    print("the unduplicated item in array {} is {}".format(item,singleNumber_counter(item)))

print("*** Method Three: Math ***")
def singleNumber_counter_math(nums):
    return 2*sum(set(nums))-sum(nums)
for item in test_case:
    print("the unduplicated item in array {} is {}".format(item,singleNumber_counter_math(item)))

print("---------------------1046. Last Stone Weight-------------------------")
def lastStoneWeight(stones):
    if len(stones)==0:
        return 0
    elif len(stones)==1:
        return stones[0]
    else:
        stones.sort()
        x = stones[-1]
        y = stones[-2]
        stones = stones[:-2]
        if x-y>0:
            new_weight = x-y
            stones.append(new_weight)
        return lastStoneWeight(stones)
stones_list = [[2,7,4,1,8,1],[1]]
for stones in stones_list:
    print("the last Stone Weight of {} is {}".format(stones,lastStoneWeight(stones)))

print("---------------------24. Swap Nodes in Pairs-------------------------")
def swapPairs(head):
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    while current.next is not None and current.next.next is not None:
        first = current.next
        second  = current.next.next
        first.next = second.next
        current.next = second
        current.next.next=first
        current=current.next.next
    return dummy.next
test_case = [[1,2,3,4],[],[1]]
for case in test_case:
    node_list = generate_linked_list(case)
    # print_linked_list(node_list)
    print("The result of Swap linked List: {} is {}".format(print_linked_list_with_return(node_list),print_linked_list_with_return(swapPairs(node_list))))

print("---------------------29. Divide Two Integers-------------------------")
def divide(dividend,divisor):
    negative = True
    if divisor ==0:
        return False
    if dividend>pow(2,31) or dividend<0-pow(2,31) or divisor>pow(2,31) or divisor<0-pow(2,31):
        return False
    if dividend==0:
        return 0
    if (abs(dividend)!=dividend and abs(divisor)!=divisor) or (abs(dividend)==dividend and abs(divisor)==divisor):
        negative = False
    dividend = abs(dividend)
    divisor =abs(divisor)
    count = -1
    while dividend>=0:
        dividend = dividend-divisor
        count = count+1
    if negative:
        count = 0-count
    if count>pow(2,31)-1:
        count = pow(2,31)-1
    if count<0-pow(2,31):
        count = 0-pow(2,31)
    return count
test_case = [{"dividend":10,"divisor":3},{"dividend":7,"divisor":-3},{"dividend":7,"divisor":0},{"dividend":1,"divisor":1},{"dividend":0,"divisor":12}]
for item in test_case:
    # print(item['dividend'],item["divisor"])
    print("The result of {} divide {} is {}".format(item['dividend'],item["divisor"],divide(item['dividend'],item["divisor"])))

print("---------------------35. Search Insert Position-------------------------")
def searchInsert(nums,target):
    if len(nums)==0:
        return False
    low = 0
    high = len(nums)-1
    while low<=high:
        mid = (low+high)//2
        if nums[mid]==target:
            return mid
        if nums[mid]<target:
            low = mid+1
        else:
            high = mid-1
    return low

test_case = [[[1,3,5,6],5],[[1,3,5,6],2],[[1,3,5,6],7]]
for item in test_case:
    print("The index of target {} in the sorted list {} should be {}".format(item[1],item[0],searchInsert(item[0],item[1])))

print("---------------------11. Container With Most Water-------------------------")
print("Method One: Two For Loops")
def maxArea(height):
    max = 0
    for i in range(len(height)-1):
        left_bound = i
        for j in range(i+1,len(height)):
            wide = j-i
            high = min(height[i],height[j])
            area = wide*high
            # print("wide={},high = {},area ={}".format(wide,high,area))
            if area > max:
                max = area
    return max
height_list = [[1,8,6,2,5,4,8,3,7],[1,1]]
for height in height_list:
    print("The Maxium Area of list {} is {} ".format(height,maxArea(height)))
print("Method Two: For Loop and While Loop ")

def maxArea_for_while_loop(height):
    max_area = 0
    left = 0
    right = len(height)-1
    while left<right:
        short_line = min(height[left],height[right])
        max_area = max(max_area,short_line*(right-left))
        if height[left]<height[right]:
            left = left+1
        else:
            right = right+1
    return max_area
for height in height_list:
    print("The Maxium Area of list {} is {} ".format(height,maxArea(height)))

print("---------------------17. Letter Combinations of a Phone Number-------------------------")
print("***** Method One: Hashmaping ***** ")
def letterCombinations(digits):
    if len(digits)==0:
        return []
    map ={"1":"","2":"abc","3":"def","4":"ghi","5":"jkl",
          "6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
    cur = ['']
    for d in digits:
        temp = []
        for c in map[d]:
            temp = temp+[r+c for r in cur]
        cur = temp
    return cur
test_case=["23","","2"]
for digits in test_case:
    print("The Combinations of {} are {}".format(digits,letterCombinations(digits)))
print("***** Method Two: Backtracking *****")
def letterCombinations_backtracking(digits):
    result = []
    digitToChart={
        "2":"abc","3":"def",
        "4":"ghi","5":"jkl",
        "6":"mno","7":"qprs",
        "8":"tuv","9":"wxyz"
    }
    def backtracking(i,curStr):
        if len(curStr)==len(digits):
            result.append(curStr)
            return
        for c in digitToChart[digits[i]]:
            backtracking(i+1,curStr+c)
    if digits:
        backtracking(0,"")
    return result
for digits in test_case:
    print("The Combinations of {} are {}".format(digits,letterCombinations_backtracking(digits)))
print("---------------------19. Remove Nth Node From End of List-------------------------")
print("Method One: Reverse, Remove and Reversed again")
def removeNthFromEnd(head,n):
    if head.next is None:
        return None
    revered_node = reverse_linked_list(head)
    revered_head = revered_node
    i = 1
    while i+1<n:
        revered_node.next = revered_node
        i = i+1
    revered_node.next=revered_node.next.next
    new_head = reverse_linked_list(revered_head)
    return new_head
test_case=[[[1,2,3,4,5],2],[[1,2],1],[[1],1]]
for linked_list in test_case:
    original_lst = linked_list.copy()
    processed_lst = linked_list.copy()
    linked_lst=generate_linked_list(original_lst[0])
    original_lst = generate_linked_list(original_lst[0])
    processed_lst = linked_lst=generate_linked_list(processed_lst[0])
    # print("The original Linked list is {}".format(linked_lst))
    # print(print_linked_list_with_return(removeNthFromEnd(linked_lst,linked_list[1])))
    print("The resulf or remove the {}th node from end for the {} is {}".format(linked_list[1],print_linked_list_with_return(original_lst),print_linked_list_with_return(removeNthFromEnd(processed_lst,linked_list[1]))))

print("Method Two: Two Pointers - Faster and Slower pointer")
def removeNthFromEnd_two_pointers(head,n):
    # Two Pointers, Fast and Slow
    fast = head
    slow = head
    # Move the faster pointer n steps ahead
    for i in range(n):
        if fast.next is None:
            # if n is equal to the number of the nodes, delete the head node
            if i==n-1:
                head = head.next
            return head
        fast = fast.next
        # loop until the Fast node reach the end
        # move both the fast and slow node
    while fast.next is not None:
        slow = slow.next
        fast = fast.next
    # find the nth node, and delink it
    if slow.next is not None:
        slow.next = slow.next.next
    return head

test_case=[[[1,2,3,4,5],2],[[1,2],1],[[1],1]]
for linked_list in test_case:
    linked_lst=generate_linked_list(linked_list[0])
    print(print_linked_list_with_return(removeNthFromEnd_two_pointers(linked_lst,linked_list[1])))

print("---------------------31. Next Permutation-------------------------")
def nextPermutation(nums):
    n = len(nums)
    # index of first element that smaller than the element to its right
    index = -1
    for i in range(n-1,0,-1):
        if nums[i]>nums[i-1]:
            index = i-1
            break
    # base condition
    if index==-1:
        reverse(nums,0,n-1)
        return nums
    j = n-1
    #swap from right to leef to find the first element
    #that is greater the above find element
    for i in range(n-1,index,-1):
        if nums[i]>nums[index]:
            j = i
            break
    nums[index],nums[j] = nums[j],nums[index]
    reverse(nums,index+1,n-1)
    return nums

def reverse(nums,i,j):
    while i<j:
        nums[i],nums[j]=nums[j],nums[i]
        i=i+1
        j=j-1
test_case_nums= [[1,2,3],[1,3,2],[4,3,2,1],[1,1,5]]
for nums in test_case_nums:
    original_num = nums.copy()
    print("The Next Permutation of {} is {}".format(original_num,nextPermutation(nums)))

print("---------------------33. Search in Rotated Sorted Array-------------------------")
def search(nums,target):
    if nums is None or len(nums)==0:
        return -1
    left,right = 0,len(nums)-1
    # find the pivot
    while left<right:
        # mid = left+(right-left)//2
        mid = (left+right)// 2
        if nums[mid]>nums[right]:
            left = mid+1
        else:
            right = mid
    # afte the loop, left will point to the pivot
    pivot = left
    left,right = 0,len(nums)-1
    if nums[pivot]<=target<=nums[right]:
        left = pivot
    else:
        right = pivot
    while left<=right:
        mid = (left+right)//2
        if nums[mid]==target:
            return mid
        elif nums[mid]<target:
            left = mid+1
        else:
            right = mid-1
    return -1
nums_test = [[[4,5,6,7,0,1,2],0],[[1,2,3,4,5,6,0],0],[[4,5,6,7,0,1,2],3],[[1],0],[[1,3],1],[[5,1,3],5]]
for test in nums_test:
    print("The index of {} in {} is {}".format(test[1],test[0],search(test[0],test[1])))

print("---------------------34. Find First and Last Position of Element in Sorted Array-------------------------")

def searchRange(nums,target):
    result = [-1,-1]
    low = 0
    high = len(nums)
    while low<high:
        mid = (high+low)//2
        if nums[mid]==target:
            result[0]=mid
            result[1]=mid
            high = mid
        elif nums[mid]<target:
            low = mid+1
        else:
            high = mid
    if result[0]==-1:
        return result
    low = result[0]+1
    high = len(nums)
    while low<high:
        mid = (low+high)//2
        if nums[mid]==target:
            result[1]=mid
            low = mid+1
        elif nums[mid]>target:
            high = mid
    return result

test_case = [[[5,7,7,8,8,10],8],[[5,7,7,8,8,10],6],[[],0],[[2,2,2,3,3,4,4,4,4,5,5,6],4],[[1,3],1]]
for item in test_case:
    print("the range of target number {} in sorted array {} is {}".format(item[1],item[0],searchRange(item[0],item[1])))

print("---------------------36. Valid Sudoku-------------------------")
def isValidSudoku(board):
    result = True
    for i in range(len(board)):
        result = checkrow(board,i) and result
    for j in range(len(board)):
        result = checkcol(board,j) and result
    for i in range(0,len(board),3):
        for j in range(0,len(board[i]),3):
            # print("new subsquare,i and j is {},{}".format(i,j))
            result = checksquare(board,i,j) and result
    return result
def checkrow(board,i):
    dict = {}
    for j in range(len(board[i])):
        if board[i][j]==".":
            continue
        elif board[i][j] not in dict.keys():
                dict[board[i][j]]=board[i][j]
        else:
                return False
    return True
def checkcol(board,j):
    dict = {}
    for i in range(len(board)):
        if board[i][j]==".":
            continue
        elif board[i][j] not in dict.keys():
            dict[board[i][j]] = board[i][j]
        else:
            return False
    return True
def checksquare(board,i,j):
    dict = {}
    for x in range(i,i+3):
        for y in range(j,j+3):
            # print("x = {},y={}".format(x,y))
            if board[x][y]==".":
                continue
            elif board[x][y] not in dict.keys():
                dict[board[x][y]]=board[x][y]
            else:
                return False
    return True
test_board = [[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
,[[".",".",".",".","5",".",".","1","."]
,[".","4",".","3",".",".",".",".","."]
,[".",".",".",".",".","3",".",".","1"]
,["8",".",".",".",".",".",".","2","."]
,[".",".","2",".","7",".",".",".","."]
,[".","1","5",".",".",".",".",".","."]
,[".",".",".",".",".","2",".",".","."]
,[".","2",".","9",".",".",".",".","."]
,[".",".","4",".",".",".",".",".","."]]]
for board in test_board:
    print("the Sudoku Board {} is {}".format(board,isValidSudoku(board)))

print("---------------------38. Count and Say-------------------------")
def countAndSay(n):
    if n==1:
        s= "1"
    else:
        item = countAndSay(n-1)
        s = ""
        i=0
        while i<len(item):
            count = 1
            while (i+1)<len(item) and item[i]==item[i+1]:
                count = count+1
                i = i+1
            #     print("i:{},count:{}".format(i,count))
            # print("i = {}".format(i))
            # print("s = {}".format(str(count)+item[i]))
            s = s+str(count)+item[i]
            i = i+1
    return s
for i in range(1,11):
    print(countAndSay(i))

print("---------------------39. Combination Sum-------------------------")
def combinationSum(candidates, target):
    result = []

    def dfs(i,cur,total):
        if total==target:
            result.append(cur.copy())
            return
        if i>=len(candidates) or total>target:
            return
        cur.append(candidates[i])
        dfs(i,cur,total+candidates[i])
        cur.pop()
        dfs(i+1,cur,total)
    dfs(0,[],0)
    return result
test_list = [[[2,3,6,7],7],[[2,3,5],8],[[2],1]]
for candidates,target in test_list:
    print("the combination of {} to get target number {} are {}".format(candidates,target,combinationSum(candidates,target)))

print("---------------------43. Multiply Strings-------------------------")
def multiply(num1,num2):
    # base condition if one of element is equal to 0
    if num1=="0" or num2=="0":
        return "0"
    # allocate a array for the result and set the size to len(num1+num2)
    result = [0]*(len(num1)+len(num2))
    # rverse the num1 and num2
    num1,num2 = num1[::-1],num2[::-1]
    for i in range(len(num1)):
        for j in range(len(num2)):
            digit = int(num1[i])*int(num2[j])
            result[i+j] = digit+result[i+j]
            result[i+j+1] = result[i+j+1]+(result[i+j])//10
            result[i+j] = result[i+j]% 10
    result = result[::-1]
    beg = 0
    while beg<len(result) and result[beg]==0:
        beg = beg+1
    result = map(str,result[beg:])
    return "".join(result)
test_case = [["2","3"],["123","456"]]
for num1,num2 in test_case:
    print("The result of multiply {} and {} is {}".format(num1,num2,multiply(num1,num2)))

print("---------------------40. Combination Sum II-------------------------")

def combinationSum2(candidates,target):
    # sort the array to help avoid the duplicated items
    candidates.sort()
    result = []
    def backtrack(cur,pos,target):
        if target==0:
            result.append(cur.copy())
        if target<=0:
            return
        prev = -1
        for i in range(pos,len(candidates)):
            if candidates[i]==prev:
                continue
            cur.append(candidates[i])
            backtrack(cur,i+1,target-candidates[i])
            cur.pop()
            prev = candidates[i]
    backtrack([],0,target)
    return result
test_case = [[[10,1,2,7,6,1,5],8],[[2,5,2,1,2],5]]
for candidates, target in test_case:
    print("the combination of {} to get target number {} are {}".format(candidates,target,combinationSum2(candidates,target)))
print("---------------------423. Reconstruct Original Digits from English-------------------------")
print("**** Method One: By Myself ****")
def originalDigits(s):
    letter_dict = {}
    for letter in s:
        if letter not in letter_dict.keys():
            letter_dict[letter] = 1
        else:
            letter_dict[letter]=letter_dict[letter]+1
    # print(letter_dict)
    num_list = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    num_str_int_map={"zero":"0","one":"1","two":"2","three":"3","four":"4",
                     "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
    s = ""
    while bool(letter_dict):
        for num in num_list:
            temp_letter_dict = letter_dict.copy()
            # print(type(temp_letter_dict))
            # print(temp_letter_dict)
            for i in range(len(num)):
                # print(i,num,num[i])
                if num[i] not in temp_letter_dict.keys() or temp_letter_dict[num[i]]==0:
                    break
                elif num[i] in temp_letter_dict.keys():
                    temp_letter_dict[num[i]] = temp_letter_dict[num[i]]-1
                    if temp_letter_dict[num[i]]==0:
                        del temp_letter_dict[num[i]]
                if i==len(num)-1:
                    num=num_str_int_map[num]
                    # s.append(num)
                    s = s+num
                    letter_dict = temp_letter_dict.copy()
        # result = set(s)
        # return "".join(result)
    return s


test_case = ["owoztneoer","fviefuro","zero","one","two","three","four","five","six","seven","eight","nine","zerozero"]
for test_str in test_case:
    print("the number(s) in the string {} are :{}".format(test_str,originalDigits(test_str)))
print()
print("**** Method One: Using Frequency ****")
def originalDigits_wt_fre(s):
    cnts = [Counter(_) for _ in ["zero","one","two","three","four","five",
                                "six","seven","eight","nine"]]
    # print(cnts)
    order = [0,2,4,6,8,1,3,5,7,9]
    unique_chars = ['z','o','w','t','u',
                    'f','x','s','g','n']
    cnt = Counter(list(s))
    # print(cnt)
    res = []
    for i in order:
        # print("cnt[unique_chars[i]]={}".format(cnt[unique_chars[i]]))
        while cnt[unique_chars[i]]>0:
            cnt -= cnts[i]
            # print(cnt)
            res.append(i)
    res.sort()
    return "".join(map(str,res))
# for str in test_case:
#     print("the number(s) in the string {} are :{}".format(str,originalDigits_wt_fre(str)))
print(originalDigits_wt_fre("owoztneoer"))

print("---------------------22. Generate Parentheses-------------------------")
def generateParenthesis(n):
    # only add open parenthesis if open<n
    # only add close parenthesis if close<open
    # valid if open==close==n
    stack= []
    res = []

    def backtrack(openN,closedN):
        if openN==closedN==n:
            res.append("".join(stack))
            return
        if openN<n:
            stack.append("(")
            backtrack(openN+1,closedN)
            stack.pop()
        if closedN<openN:
            stack.append(")")
            backtrack(openN,closedN+1)
            stack.pop()
    backtrack(0,0)
    return res
test_case=[3,1]
for n in test_case:
    print("All the combination of {} parenthesis are {}".format(n,generateParenthesis(n)))

print("---------------------51. N-Queens-------------------------")
print("***** Method One: Backtracking Method with multiple helping function *****")
class solveNQueens():
    """
    since no queen could at the same row or column
    using the 1d array instead of the 2d array.
    for example the solution(s)
    [1,3,0,2] presents
    [. Q . .]
    [. . . Q]
    [Q . . .]
    [. . Q .]
    or
    [2,0,3,1] presents
    [. . Q .]
    [Q . .  ]
    [. . . Q]
    [. Q . .]
    """
    # BACKTRACKING TEMPLATE
    # def is_valid_state(state):
    #     # CHECK if it is a valid solution
    #     return True
    # def get_candidates(state):
    #     return []
    # def search(state,solution):
    #     if is_valid_state(state):
    #         solution.append(state.copy())
    #         # return
    #     for candidate in get_candidates(state):
    #         state.add(candidate)
    #         search(state,solution)
    #         state.remove(candidate)
    # def solve():
    #     solution = []
    #     state = set()
    #     search(state,solution)
    #     return solution
    def solveNQueens(self,n):
        solutions= []
        state=[]
        self.search(state,solutions,n)
        return solutions

    def is_valid_state(self,state,n):
        # check if it is a valid solution
        # all the n queens placed on the board
        return len(state)==n
    def get_candidates(self,state,n):
        # if there is no queen on the board
        # placed it on all the possible positions
        if not state:
            return range(n)
        # find the next position in the state tp populate
        position = len(state)
        candidates = set(range(n))
        # prune down candidates that place the queen into attacks
        for row,col in enumerate(state):
            # discard the column index if it's occupied
            candidates.discard(col)
            dist = position-row
            # discard diagonals
            candidates.discard(col+dist)
            candidates.discard(col-dist)
        return candidates
    def search(self,state,solution,n):
        if self.is_valid_state(state,n):
            state_string = self.state_to_string(state,n)
            solution.append(state_string)
            return
        for candidate in self.get_candidates(state,n):
            # recursive
            state.append(candidate)
            self.search(state,solution,n)
            state.pop()
    def state_to_string(self,state,n):
        # ex. change [1,3,0,2]
        # output: [".Q..","...Q","Q...","..Q."]
        ret = []
        for i in state:
            string = '.'*i+'Q'+'.'*(n-i-1)
            ret.append(string)
        return ret

test_case = [4,1]
for n in test_case:
    s = solveNQueens()
    print(s.solveNQueens(n))

print("***** Method Two: Backtracking with DFS *****")
def solveNQueens_dfs(n):

    # Declare three Set() to store the Column, posDiag and negDig
    col = set()
    posDig = set() # postive diagonal (r+c) such as (2,2) and (3,1) slop is positive so called posDig
    negDig = set() # negative diagonal (r-c) such as (0,0) and (1,1)

    result = []
    board=[["X"]*n for _ in range(n)]
    def backtracking(r):
        # if scan the last row, then return the solution
        if r==n:
            copy=["".join(row) for row in board]
            result.append(copy)
            return
        #if not reach the last row
        for c in range(n):
            # if the position is not available
            if c in col or (r+c) in posDig or (r-c) in negDig:
                continue
            # else put the queen in this position and update the sets
            col.add(c)
            negDig.add(r-c)
            posDig.add(r+c)
            board[r][c] = "Q"
            # search for next row
            backtracking(r+1)
            # if last position is not available, remove all the updated
            col.remove(c)
            negDig.remove(r-c)
            posDig.remove(r+c)
            board[r][c] = "X"
    backtracking(0)
    return result
test_case = [4,1]
for n in test_case:
    print(f"For the {n}*{n} board, the solution is :")
    print_matrix(solveNQueens_dfs(n))

from itertools import product
print("---------------------37. Sudoku Solver-------------------------")
class solution():

    SHAPE = 9
    GRID = 3
    EMPTY = '.'
    DIGITS = set([str(num) for num in range(1,SHAPE+1)])
    def solveSudoku(self,board):
        '''
        Do not return anything, modify board in-place instead.
        '''
        self.search(board)
    def is_valid_state(self,board):
        for row in self.get_rows(board):
            if not set(row)==self.DIGITS:
                return False
        for col in self.get_cols(board):
            if not set(col)==self.DIGITS:
                return False
        for grid in self.get_grid(board):
            if not set(grid)==self.DIGITS:
                return False
        return True
    def get_candidates(self,board,row,col):
        used_digit = set()
        # remove digits used by the same row
        used_digit.update(self.get_kth_row(board,row))
        # remove digits used by the same column
        used_digit.update(self.get_kth_col(board,col))
        # remove digits used by 3*3 subbox
        used_digit.update(self.get_grid_at_row_col(board,row,col))
        used_digit-=set([self.EMPTY])
        candidates = self.DIGITS-used_digit
        return candidates
    def search(self,board):
        if self.is_valid_state(board):
            return True
        for row_idx,row in enumerate(board):
            for col_idx,elm in enumerate(row):
                if elm==self.EMPTY:
                    # find candidates to construct the next state
                    for candidate in self.get_candidates(board,row_idx,col_idx):
                        board[row_idx][col_idx]=candidate
                        is_solve = self.search(board)
                        if is_solve:
                            return True
                        else:
                            # undo the wrong guess and start new
                            board[row_idx][col_idx] = self.EMPTY
                    return False
        return True
    # helper functions for retrieving rows, cols, and grids
    def get_kth_row(self,board,k):
        return board[k]
    def get_rows(self,board):
        for i in range(self.SHAPE):
            yield board[i]
    def get_kth_col(self,board,k):
        return [
            board[row][k] for row in range(self.SHAPE)
        ]
    def get_cols(self,board):
        for col in range(self.SHAPE):
            ret = [
                board[row][col] for row in range(self.SHAPE)
            ]
            yield ret
    def get_grid(self,board):
        for row in range(0,self.SHAPE,self.GRID):
            for col in range(0,self.SHAPE,self.GRID):
                grid = [
                    board[r][c] for r,c in
                    product(range(row,row+self.GRID),range(col,col+self.GRID))
                ]
                yield grid
    def get_grid_at_row_col(self,board,row,col):
        row = row//self.GRID*self.GRID
        col = col//self.GRID*self.GRID
        return [
            board[r][c] for r,c in
            product(range(row,row+self.GRID),range(col,col+self.GRID))
        ]
    def Return_Board(self):
        return board
board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
solu = solution()
solu.solveSudoku(board)
print(solu.Return_Board())

print("---------------------45. Jump Game II-------------------------")
def jump(nums):
    '''
    using the BFS and each level of the tree is
    the items could arrival from the current item
    '''
    res =0
    l=r=0
    while r<len(nums)-1:
        farthest = 0
        for i in range(l,r+1):
            farthest = max(farthest,i+nums[i])
        l = r+1
        r = farthest
        res = res+1
    return res
test_case=[[2,3,1,1,4],[2,3,0,1,4]]
for nums in test_case:
    print("The Minimum Steps for Array {} from 1st to last item is {} ".format(nums,jump(nums)))

print("---------------------55. Jump Game-------------------------")
def canJump(nums):
    l = r = 0
    farthest = 0
    while r<len(nums)-1:
        for i in range(l,r+1):
            farthest = max(farthest,i+nums[i])
        if farthest==r:
            return False
        else:
            l = r
            r = farthest
    return True

test_case=[[2,3,1,1,4],[3,2,1,0,4]]
for nums in test_case:
    print("Is there a way for Array {} from 1st to last item? {} ".format(nums,canJump(nums)))

print("---------------------46. Permutations-------------------------")
print("***** Method One: Recursive *****")
def permute(nums):
    result = []
    if (len(nums)==1):
        # print(nums[:])
        return [nums[:]]
    for i in range(len(nums)):
        n = nums.pop(0)
        perms= permute(nums)
        for perm in perms:
            perm.append(n)
            # print("perm={}".format(perm))
        result.extend(perms)
        # print("nums={},n={}".format(nums,n))
        nums.append(n)
    # print("result = {}".format(result))
    return result
test_case=[[1,2,3],[0,1],[1]]
for nums in test_case:
     print("The Permute of array {} are {} ".format(nums,permute(nums)))
print("***** Method One: Recursive and Backtracking *****")
def permute_backtracking(nums):
    result = []
    len_n = len(nums)
    def backtracking(my_nums,use_nums):
        if len(my_nums)==len_n:
            result.append(my_nums)
            return
        for i in range(len(use_nums)):
            x = my_nums.copy()
            x.append(use_nums[i])
            backtracking(x,use_nums[:i]+use_nums[i+1:])
    backtracking([],nums)
    return result
for nums in test_case:
     print("The Permute of array {} are {} ".format(nums,permute_backtracking(nums)))

print("---------------------78. Subsets-------------------------")
def subsets(nums):
    result = []
    subset = []
    def dfs(i):
        if i>=len(nums):
            result.append(subset.copy())
            return
        # include the num[i]
        # print("i={},subset={}".format(i, subset))
        subset.append(nums[i])
        # print("i={},subset={}".format(i, subset))
        dfs(i+1)
        # exclude the num[i]
        # print("i={},subset={}".format(i, subset))
        subset.pop()
        # print("i={},subset={}".format(i,subset))
        dfs(i+1)
    dfs(0)
    return result

test_case=[[1,2,3],[0]]
for nums in test_case:
     print("The subset of array {} are {} ".format(nums,subsets(nums)))

print("---------------------66. Plus One-------------------------")
def plusOne(digits):
    sum = 0
    len_digits = len(digits)-1
    for i in range(len(digits)):
        sum = sum+digits[i]*pow(10,len_digits)
        len_digits=len_digits-1
    result = sum+1
    return list(str(result))
test_case = [[1,2,3],[4,3,2,1]]
for digits in test_case:
    print("the PlusOne result of {} is {}".format(digits,plusOne(digits)))

print("---------------------58. Length of Last Word-------------------------")
def lengthOfLastWord(s):
   len_s = len(s)
   last_char = len(s)-1
   while s[last_char]==' ':
       last_char = last_char-1
   # print(last_char)
   search_index = last_char
   while search_index>=0 and s[search_index]!=' ':
       search_index = search_index-1
   return last_char-search_index
test_case = ["Hello World","   fly me   to   the moon  ","luffy is still joyboy"," i i  i ","a"," a"]
for s in test_case:
    print("the length of {}'s last word is {}".format(s,lengthOfLastWord(s)))

print("---------------------47. Permutations II-------------------------")
print("***** Method One: Generate all the Permutation and remove the duplicated *****")
def permuteUnique(nums):
   result_set = set()
   subset = []
   result = []
   len_num = len(nums)
   def backtracking(my_num,use_num):
       if len(my_num)==len_num:
           result.append(my_num)
           return
       for i in range(len(use_num)):
           x = my_num.copy()
           x.append(use_num[i])
           backtracking(x,use_num[:i]+use_num[i+1:])

   backtracking([],nums)
   result_copy = []
   for item in result:
       # print(item)
       if item not in result_copy:
           result_copy.append(item)
   return result_copy
test_case = [[1,1,2],[1,2,3]]
for nums in test_case:
    print("the Permutation of {} are {}".format(nums,permuteUnique(nums)))
print("***** Method Two: Using hashing map *****")
def permuteUnique_hashmap(nums):
    result = []
    sub_perm = []
    count = {item:0 for item in nums}
    for item in nums:
        count[item] = count[item]+1
    def backtracking():
        # if the length of permutation equals to the length of nums
        # means find a avaliable permutation
        if len(sub_perm)==len(nums):
            result.append(sub_perm.copy())
            return
        # check all the items in the count
        for item in count:
            # the item is avaliable
            if count[item]>0:
                # brust force: include the current item
                sub_perm.append(item)
                count[item]=count[item]-1
                # print("count:{},sub_perm:{}".format(count, sub_perm))
                backtracking()
                # burst force: exclude the current item
                # print("after item {}".format(item))
                count[item] = count[item]+1
                sub_perm.pop()
                # print("after pop:count:{},sub_perm:{}".format(count,sub_perm))
    backtracking()
    return result
for nums in test_case:
    print("the Permutation of {} are {}".format(nums,permuteUnique_hashmap(nums)))
print("***** Method Three: Using backtracking and visted array*****")
def permuteUnique_backtracking_visted(nums):
    result =[]
    visted = [0] *len(nums)
    def backtracking_path(sub_perm):
        if len(sub_perm)==len(nums):
            result.append(sub_perm)
        else:
            for i in range(len(nums)):
                if not visted[i]:
                    visted[i]=1
                    backtracking_path(sub_perm+[nums[i]])
                    visted[i]=0
    backtracking_path([])
    return result
for nums in test_case:
    print("the Permutation of {} are {}".format(nums,permuteUnique_backtracking_visted(nums)))
print("---------------------50. Pow(x, n)-------------------------")
print("***** Method One: While Loop")
def myPow(x,n):
    if abs(x)>=100 or abs(n)>pow(2,31):
        return False
    integar=True
    if n<0:
        integar = False
        n=abs(n)
    result = 1
    while n>0:
        # print("result:{},x:{}".format(result,x))
        result = result*x
        n = n-1
    if integar:
        return result
    else:
        return 1/result
test_case =[(2.00000,10),(2.10000,3),(2.00000,-10),(23,0),(0,23)]
for x,n in test_case:
    print("the power of {} to {} is {}".format(x,n,myPow(x,n)))

print("***** Method Two: Divide and Conquer--Recuirsive")
def myPow_dq(x,n):
    def helper(x,n):
        if x==0:
            return 0
        if n==0:
            return 1
        result = helper(x,n//2)
        result = result*result
        return x*result if n%2 else result
    result = helper(x,abs(n))
    return result if n>=0 else 1/result
for x,n in test_case:
    print("the power of {} to {} is {}".format(x,n,myPow_dq(x,n)))

print("---------------------215. Kth Largest Element in an Array-------------------------")
print("***** Merge Sort and Find the Kth Largest Element")
def findKthLargest(nums,k):
    if len(nums)<1 or len(nums)>pow(10,4):
        return False
    def merge_sort(nums):
        if len(nums)<=1:
            return nums
        mid = len(nums)//2
        left = nums[:mid]
        right = nums[mid:]
        left_sorted = merge_sort(left)
        right_sorted = merge_sort(right)
        sorted_list = merge(left_sorted,right_sorted)
        return sorted_list
    def merge(left,right):
        merge_result = []
        i=0
        j=0
        while i<len(left) and j<len(right):
            if left[i]<right[j]:
                merge_result.append(left[i])
                i=i+1
            else:
                merge_result.append(right[j])
                j = j+1
        merge_result=merge_result+left[i:]+right[j:]
        return merge_result
    sorted_list = merge_sort(nums)
    print(sorted_list)
    return sorted_list[len(sorted_list)-k]
test_case = [[[3,2,1,5,6,4], 2],[[3,2,3,1,2,4,5,5,6],4],[[1],1]]
for nums,k in test_case:
    dict = {1: "st", 2: "nd", 3: "th", 4: "th", 5: "th", 6: "th", 7: "th", 8: "th", 9: "th", 0: "th"}
    print("The {} Largest Element in {} is {}".format(str(k) + dict[k % 10], nums,
                                                      findKthLargest(nums, k)))
print("***** Using Quick Sort and Find the Kth Largest Element")
def findKthLargest_quick_sort(nums,k):
    if len(nums)<1 or len(nums)>pow(10,4):
        return False
    def quick_sort(nums):
        if len(nums)<=1:
            return nums
        left = []
        right = []
        pivot = nums[-1]
        for i in range(len(nums)-1):
            if nums[i]<pivot:
                left.append(nums[i])
            else:
                right.append(nums[i])
        return quick_sort(left)+[pivot]+quick_sort(right)
    sorted_list = quick_sort(nums)
    return sorted_list[len(sorted_list)-k]
for nums, k in test_case:
    dict = {1: "st", 2: "nd", 3: "th", 4: "th", 5: "th", 6: "th", 7: "th", 8: "th", 9: "th", 0: "th"}
    print("The {} Largest Element in {} is {}".format(str(k) + dict[k % 10], nums,
                                                      findKthLargest_quick_sort(nums, k)))
print("***** Using Build-In Sort and Find the Kth Largest Element")
def findKthLargest_build_in_sorted(nums,k):
    sorted_nums =sorted(nums)
    return sorted_nums[len(sorted_nums)-k]
for nums,k in test_case:
    dict = {1:"st",2:"nd",3:"th",4:"th",5:"th",6:"th",7:"th",8:"th",9:"th",0:"th"}
    print("The {} Largest Element in {} is {}".format(str(k)+dict[k%10],nums,findKthLargest_build_in_sorted(nums,k)))

print("---------------------1985. Find the Kth Largest Integer in the Array-------------------------")
def kthLargestNumber(nums,k):
    num_list = []
    for item in nums:
        num_list.append(int(item))
    sorted_list = sorted(num_list)
    return str(sorted_list[len(nums)-k])
test_case = [[["3","6","7","10"], 4],[["2","21","12","1"],3],[["0","0"],2]]
for nums,k in test_case:
    dict = {1: "st", 2: "nd", 3: "th", 4: "th", 5: "th", 6: "th", 7: "th", 8: "th", 9: "th", 0: "th"}
    print("The {} Largest Element in {} is {}".format(str(k) + dict[k % 10], nums,
                                                      kthLargestNumber(nums, k)))

print("---------------------1945. Sum of Digits of String After Convert-------------------------")
def getLucky(s,k):
    string = ""
    for letter in s:
        string = string+str(ord(letter)-96)
    def letter_shift(st):
        temp = 0
        for letter in st:
            temp = temp+int(letter)
        return str(temp)
    while k>1:
        string = letter_shift(string)
        k = k-1
    string = int(letter_shift(string))
    return string
test_case = [["leetcode",2],["iiii",1],["zbax",2]]
for s,k in test_case:
    print("The result of transfer {} to numbers and shift {} times is {}".format(s,k,getLucky(s,k)))

print("---------------------48. Rotate Image-------------------------")
def rotate(matrix):
    left,right = 0,len(matrix)-1
    while left<right:
        for i in range(right-left):
            top,bottom = left,right
            # save the topleft
            topleft = matrix[top][left+i]
            # move the bottom left into top left
            matrix[top][left+i]=matrix[bottom-i][left]
            # move the bottom right into bottom left
            matrix[bottom-i][left] = matrix[bottom][right-i]
            # move the top right into bottom right
            matrix[bottom][right-i] = matrix[top+i][right]
            # move the top left into top right
            matrix[top+i][right] = topleft
        left = left+1
        right = right-1
        return matrix
test_case=[[[1,2,3],[4,5,6],[7,8,9]],[[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]]
for matrix in test_case:
    print("Original Image: {}".format(matrix))
    print_matrix(matrix)
    rotated_image = rotate(matrix)
    print("The Result of rotate is {} ".format(rotated_image))
    print_matrix(rotated_image)

print("---------------------73. Set Matrix Zeroes-------------------------")
print("***** Methond One: Using One Space *****")
def setZeroes(matrix):
    ROW,COL = len(matrix),len(matrix[0])
    rowZero=False
    for r in range(ROW):
        for c in range(COL):
            if matrix[r][c]==0:
                matrix[0][c]=0
                if r>0:
                    matrix[r][0]=0
                else:
                    rowZero= True
    for r in range(1,ROW):
        for c in range(1,COL):
            if matrix[0][c]==0 or matrix[r][0]==0:
                matrix[r][c]=0
    if matrix[0][0]==0:
        for r in range(ROW):
            matrix[r][0]=0
    if rowZero:
        for c in range(COL):
            matrix[0][c]=0
    return matrix


test_case = [[[1,1,1],[1,0,1],[1,1,1]],[[0,1,2,0],[3,4,5,2],[1,3,1,5]]]
for matrix in test_case:
    orig_matrix = matrix.copy()
    processed_matrix = matrix.copy()
    print("Original Matrix:")
    print_matrix(orig_matrix)
    print("Processed Matrix:{}".format(setZeroes(processed_matrix)))
    print_matrix(processed_matrix)
    print()

print("***** Methond Two: Using M+N Space *****")
def setZeroes_mn_space(matrix):
    col_index = [1 for _ in range(len(matrix[0]))]
    row_index = [1 for _ in range(len(matrix))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            # print("row={},col ={},matrix = {}".format(row,col,matrix[row][col]))
            if matrix[row][col]==0:
                col_index[col]=0
                row_index[row]=0
                # print(col_index,row_index)
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if col_index[col]==0 or row_index[row]==0:
                matrix[row][col]=0

    print_matrix(matrix)
test_case = [[[1,1,1],[1,0,1],[1,1,1]],[[0,1,2,0],[3,4,5,2],[1,3,1,5]]]
for matrix in test_case:
    orig_matrix = matrix.copy()
    processed_matrix = matrix.copy()
    print("Original Matrix:")
    print_matrix(orig_matrix)
    print("Processed Matrix:")
    setZeroes_mn_space(processed_matrix)
    print()

print("---------------------75. Sort Colors-------------------------")
def sortColors(nums):
    left,right = 0,len(nums)-1
    def swap_by_index(i,j):
        temp = nums[i]
        nums[i]=nums[j]
        nums[j]=temp
    i=0
    while i<=right:
        if nums[i]==0:
            swap_by_index(left,i)
            left = left+1
            i=i+1
        elif nums[i]==2:
            swap_by_index(right,i)
            right = right-1
        elif nums[i]==1:
            i=i+1
    return nums
test_case = [[2,0,2,1,1,0],[2,0,1],[2,2,1,2,1,1,0,0,0,]]
for nums in test_case:
    orig_nums = nums.copy()
    processed_nums = nums.copy()
    print("The result of sort {} is {}".format(orig_nums,sortColors(processed_nums)))



print("---------------------49. Group Anagrams-------------------------")
print("***** Method One: Sort and Dict")
def groupAnagrams(strs):
    dict = {}
    for item in strs:
        sorted_item= "".join(sorted(item))
        if sorted_item not in dict.keys():
            dict[sorted_item] = [item]
        else:
            dict[sorted_item].append(item)
    return dict.values()
test_case = [["eat","tea","tan","ate","nat","bat"],[""],["a"]]
for strs in test_case:
    print("The Anagrams of {} is {}".format(strs,groupAnagrams(strs)))
print("***** Method Two: count and dict (O(m*n))*****")
def groupAnagrams_count_dict(strs):
    result = defaultdict(list)
    for ch in strs:
        count=[0]*26
        for c in ch:
            count[ord(c)-ord("a")] = count[ord(c)-ord("a")]+1
        result[tuple(count)].append(ch)
    return result.values()
for strs in test_case:
    print("The Anagrams of {} is {}".format(strs,groupAnagrams_count_dict(strs)))
print("---------------------77. Combinations-------------------------")
def combine(n,k):
    result = []
    def backtracking(start,comb):
        if len(comb)==k:
            result.append(comb.copy())
            return
        for i in range(start,n+1):
            comb.append(i)
            backtracking(i+1,comb)
            comb.pop()
    backtracking(1,[])
    return result
test_case = [(4,2),(1,1),(5,3)]
for n,k in test_case:
    print("The combination of {} for {} is {}".format(k,n,combine(n,k)))

print("---------------------42. Trapping Rain Water-------------------------")
print("***** Method One: With Extra Array")
def trap_with_extra_array(height):
    water_trap = []
    for i in range(1,len(height)-1):
        leftMax = max(height[:i])
        rightMax = max(height[i+1:])
        min_height = min(leftMax,rightMax)
        water = min_height-height[i]
        if water>0:
            water_trap.append(water)
        else:
            water_trap.append(0)
    result = sum(water_trap)
    return result
test_case = [[4,2,0,3,2,5],[0,1,0,2,1,0,1,3,2,1,2,1]]
for height in test_case:
    print("The Trap of water in {} is {}".format(height,trap_with_extra_array(height)))
print("***** Method Two: In Place Calculate")
def trap(height):
    left,right = 0,len(height)-1
    leftMax, rightMax = height[left],height[right]
    result = 0
    while left<right:
        if leftMax<rightMax:
            left = left+1
            leftMax = max(leftMax,height[left])
            result = result+ (leftMax-height[left])
        else:
            right = right-1
            rightMax=max(rightMax,height[right])
            result = result+(rightMax-height[right])
    return result
test_case = [[4,2,0,3,2,5],[0,1,0,2,1,0,1,3,2,1,2,1]]
for height in test_case:
    print("The Trap of water in {} is {}".format(height,trap(height)))

print("---------------------54. Spiral Matrix-------------------------")
def spiralOrder(matrix):
    result = []
    left,right = 0,len(matrix[0])
    top,bottom = 0,len(matrix)
    while left<right and top<bottom:
        # get evey i in the top row
        for i in range(left,right):
            result.append(matrix[top][i])
        top = top+1
        # get evey i n the right col
        for i in range(top,bottom):
            result.append(matrix[i][right-1])
        right = right-1
        if not (left<right and top<bottom):
            break
        # get every i in the bottom row:
        for i in range(right-1,left-1,-1):
            result.append(matrix[bottom-1][i])
        bottom = bottom-1
        # get every i in the left col
        for i in range(bottom-1,top-1,-1):
            result.append(matrix[i][left])
        left = left+1
    return result
test_case= [[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]
for matrix in test_case:
    print_matrix(matrix)
    print("The Spiral Order is {}".format(spiralOrder(matrix)))
import sys
print("---------------------53. Maximum Subarray-------------------------")
print("***** Method One: One Loop")
def maxSubArray(nums):
    maxSub = nums[0]
    cursum = 0
    for n in nums:
        if cursum<0:
            cursum = 0
        cursum = cursum+n
        maxSub = max(maxSub,cursum)
    return maxSub
test_case=[[-2,1,-3,4,-1,2,1,-5,4],[1],[5,4,-1,7,8]]
for nums in test_case:
    print("The Maximum Sum of {} is {}".format(nums,maxSubArray(nums)))
print("***** Method Two: Two Loops")
def maxSubArray_two_loops(nums):
    maxSum = -sys.maxsize
    for i in range(len(nums)):
        total = 0
        for j in range(i,len(nums)):
            total = total+nums[j]
            if total>maxSum:
                maxSum = total
    return maxSum
for nums in test_case:
    print("The Maximum Sum of {} is {}".format(nums,maxSubArray_two_loops(nums)))

print("***** Method Three: Divide and Conquer")
def findmaxSubarray(nums,left=None,right=None):
    if not nums:
        return 0
    if left is None and right is None:
        left,right =0,len(nums)-1
    # the nums contains 0 or 1 element:
    if right==left:
        return nums[left]
    #find the mid element
    mid = (left+right)//2
    # find max sublist for the left
    leftMax = -sys.maxsize
    total = 0
    for i in range(mid,left-1,-1):
        total = total+nums[i]
        if total>leftMax:
            leftMax = total
    # find max sublist from right
    rightMax = -sys.maxsize
    total = 0
    for i in range(mid+1,right+1):
        total = total+nums[i]
        if total>rightMax:
            rightMax = total
    maxLeftRight = max(findmaxSubarray(nums,left,mid),findmaxSubarray(nums,mid+1,right))
    return max(maxLeftRight,leftMax+rightMax)
for nums in test_case:
    print("The Maximum Sum of {} is {}".format(nums,findmaxSubarray(nums)))

print("---------------------57. Insert Interval-------------------------")
def insert(intervals,newInterval):
    result = []
    for i in range(len(intervals)):
        if newInterval[1]<intervals[i][0]:
            result.append(newInterval)
            return result+intervals[i:]
        elif newInterval[0]>intervals[i][1]:
            result.append(intervals[i])
        else:
            newInterval = [min(intervals[i][0],newInterval[0]),max(intervals[i][1],newInterval[1])]
    result.append(newInterval)
    return result
test_case = [[[[1,3],[6,9]],[2,5]],[[[1,2],[3,5],[6,7],[8,10],[12,16]],[4,8]],[[],[5,7]]]
for intervals,newInterval in test_case:
    print("The result of inser {} into {} is {}".format(newInterval,intervals,insert(intervals,newInterval)))
print("---------------------56. Merge Intervals-------------------------")
def merge(intervals):
    intervals.sort(key= lambda i:i[0])
    result = [intervals[0]]
    for start, end in intervals[1:]:
        lastEnd = result[-1][1]
        lastStart = result[-1][0]
        if start<=lastEnd:
            # result[-1][1]=max(lastEnd,end)
            result.pop()
            result.append([lastStart,max(lastEnd,end)])
        else:
            result.append([start,end])
    return result
test_case = [[[1,3],[2,6],[8,10],[15,18]],[[1,4],[4,5]]]
for intervals in test_case:
    print("The result of merge {} is {}".format(intervals,merge(intervals)))
print("---------------------67. Add Binary-------------------------")
print("***** Method One: Build-In Function *****")
def addBinary(a,b):
    int_a = int(a,2)
    int_b = int(b,2)
    # print(int_a,int_b)
    return str(bin(int_a+int_b))[2:]
test_case = [["11","1"],["1010","1011"]]
for a,b in test_case:
    print("The result of add {} and {} is {}".format(a,b,addBinary(a,b)))
print("***** Method Two: with Loop *****")
def addBinary_loop(a,b):
    result = ""
    carry = 0
    final_len = max(len(a),len(b))
    a = "0"*(final_len-len(a))+a
    b = "0"*(final_len-len(b))+b
    for i in range(final_len-1,-1,-1):
        total = int(a[i])+int(b[i])+carry
        carry = total//2
        result = str(total%2)+result
    if carry==1:
        result = "1"+result
    return result
for a,b in test_case:
    print("The result of add {} and {} is {}".format(a,b,addBinary_loop(a,b)))

print("---------------------494. Target Sum-------------------------")
def findTargetSumWays(nums,target):
    dp= {}# (index, total) => # of ways
    def backtracking(i,total):
        if i==len(nums):
            return 1 if total==target else 0
        if (i,total) in dp:
            return dp[(i,total)]
        dp[(i,total)]=(backtracking(i+1,total+nums[i])+backtracking(i+1,total-nums[i]))
        return dp[(i,total)]
    return backtracking(0,0)
test_case = [[[1,1,1,1,1],3],[[1],1]]
for nums, target in test_case:
    print("There are {} ways for {} to get sum result {}".format(findTargetSumWays(nums,target),nums,target))
print("---------------------41. First Missing Positive-------------------------")
print("***** Method One: Loops *****")
def firstMissingPositive(nums):
    for i in range(len(nums)):
        if nums[i] < 0:
            nums[i] = 0
    for i in range(1, len(nums)+1):
        if i not in nums:
            return i
    if i == max(nums):
        return i + 1
test_case = [[1,2,0],[3,4,-1,1],[7,8,9,11,12]]
for nums in test_case:
    print("The First Missing Positive in {} is {}".format(nums,firstMissingPositive(nums)))
print("***** Method Two: Using array as Cashing *****")
def firstMissingPositive_array_as_caching(nums):
    for i in range(len(nums)):
        if nums[i]<0:
            nums[i]=0
    for i in range(len(nums)):
        val = abs(nums[i])
        # if the val is greated than 0, set the index value to negative
        if 1<=val<=len(nums):
            if nums[val-1]>0:
                nums[val-1] = nums[val-1]*-1
             # if the item's index value is 0, set to a default value, since -1*0=0
            elif nums[val-1]==0:
                nums[val-1] = -1*(len(nums)+1)
    for i in range(1,len(nums)+1):
        if nums[i-1]>=0:
            return i
    return len(nums)+1
test_case = [[1,2,0],[3,4,-1,1],[7,8,9,11,12]]
for nums in test_case:
    print("The First Missing Positive in {} is {}".format(nums,firstMissingPositive_array_as_caching(nums)))
print("***** Method Three: Hash Mapping *****")
def firstMissingPositive_hashing_mapping(nums):
    dict = {}
    for i in range(len(nums)):
        if nums[i]<0:
            nums[i]=0
    for i in range(len(nums)):
        if nums[i]>0:
            dict[nums[i]]=1
    for i in range(1,len(nums)+1):
        if i not in dict.keys():
            return i
    return len(nums)+1
test_case = [[1,2,0],[3,4,-1,1],[7,8,9,11,12]]
for nums in test_case:
    print("The First Missing Positive in {} is {}".format(nums,firstMissingPositive_hashing_mapping(nums)))
print("---------------------124. Binary Tree Maximum Path Sum-------------------------")
def maxPathSum(root):
    result = [root.val]
    #return the max path sum without split
    def dfs(root):
        # if there is no node, means the sum will be 0
        if not root:
            return 0
        leftMax = dfs(root.left)
        rightMax = dfs(root.right)
        # if the Max left sum or Max right sum is less then 0
        # then choose not to include it
        leftMax = max(leftMax,0)
        rightMax = max(rightMax,0)

        # compute max path sum with split, and store it to GLOBAL variable result
        result[0] = max(result[0],leftMax+rightMax+root.val)
        # return value for recursive dfs(), it shows the max sum for not split
        return root.val+max(leftMax,rightMax)
    dfs(root)
    return result[0]
test_case=[[1,2,3], [-10,9,20,None,None,15,7]]
for node_list in test_case:
    root = generate_tree_from_list(node_list)
    print(root[0])
    print("The Max Sum Path is {}".format(maxPathSum(root[0])))
print("---------------------127. Word Ladder-------------------------")
def ladderLength(beginWord,endWord,wordList):
    if endWord not in wordList:
        return 0
    # create the dict for each word pattern
    # like hot=> *ot,h*t,ho*
    # using default dict => if no key, shows None
    neighbor = collections.defaultdict(list)
    wordList.append(beginWord)
    for word in wordList:
        for j in range(len(word)):
            pattern = word[:j]+"*"+word[j+1:]
            neighbor[pattern].append(word)
    # using BFS algorith to find the path
    visited = set([beginWord])
    q= deque()
    q.append(beginWord)
    result = 1
    while len(q)>0:
        for i in range(len(q)):
            word=q.popleft()
            if word==endWord:
                return result
            for j in range(len(word)):
                pattern = word[:j]+"*"+word[j+1:]
                for neighbor_word in neighbor[pattern]:
                    if neighbor_word not in visited:
                        visited.add(neighbor_word)
                        q.append(neighbor_word)
        result = result+1
    return 0
test_case = [["hit", "cog", ["hot","dot","dog","lot","log","cog"]],["hit","cog",["hot","dot","dog","lot","log"]]]
for beginWord, endWord,wordlist in test_case:
    print("the word ladder from {} to {} is {}".format(beginWord,endWord,ladderLength(beginWord,endWord,wordlist)))
print("---------------------121. Best Time to Buy and Sell Stock-------------------------")
print("***** Method One: Using the Max function *****")
def maxProfit(prices):
    max_Profit = 0
    for i in range(len(prices)-1):
        max_sale_price=max(prices[i+1:])
        max_Profit = max(max_Profit,max_sale_price-prices[i])
    return max_Profit
test_case = [[7,1,5,3,6,4],[[7,6,4,3,1]]]
for prices in test_case:
    print("The max Profit of {} is {}".format(prices,maxProfit(prices)))
print("***** Method Two: using left and right pointer *****")
def maxProfit_two_pointer(prices):
    left,right = 0,1
    max_Porfit = 0
    while right<len(prices):
        if prices[left]<prices[right]:
            profit = prices[right]-prices[left]
            max_Porfit = max(max_Porfit,profit)
        else:
            left = right
        right = right+1
    return max_Porfit
for prices in test_case:
    print("The max Profit of {} is {}".format(prices,maxProfit_two_pointer(prices)))
print("---------------------217. Contains Duplicate-------------------------")
print("***** Method One: Compare length of list and set *****")
def containsDuplicate(nums):
    nums_set = set(nums)
    if len(nums_set)==len(nums):
        return False
    else:
        return True
test_case = [[1,2,3,1],[1,2,3,4],[1,1,1,3,3,4,3,2,4,2]]
for nums in test_case:
    print("There has the duplicated numbers in {}:{}".format(nums,containsDuplicate(nums)))
print("***** Method two: Hashing Table *****")
def containsDuplicate_hashtable(nums):
    hash=set()
    for item in nums:
        if item in hash:
            return True
        hash.add(item)
    return False
for nums in test_case:
    print("There has the duplicated numbers in {}:{}".format(nums,containsDuplicate_hashtable(nums)))
print("---------------------238. Product of Array Except Self-------------------------")
print("***** Method One: reconstruction list *****")
def productExceptSelf(nums):
    result =[]
    def product_list(lst):
        result = 1
        for i in range(len(lst)):
            result = result*lst[i]
        return result
    for i in range(len(nums)):
        result.append(product_list(nums[:i]+nums[i+1:]))
    return result
test_case = [[1,2,3,4],[-1,1,0,-3,3]]
for nums in test_case:
    print("The result of Product Except Self of {} is {}".format(nums,productExceptSelf((nums))))

print("***** Method Two : Store Pre an Post result in place *****")
def productExceptSelf_pre_post(nums):
    result = [1]*(len(nums))
    prefix = 1
    # store all the prefix result into the result list
    for i in range(len(nums)):
        result[i] = prefix
        prefix = prefix*nums[i]
    postfix = 1
    # calculate the postfix result and multiple the prefix result and store to the result list
    for i in range(len(nums)-1,-1,-1):
        result[i] = postfix*result[i]
        postfix = postfix*nums[i]
    return result
for nums in test_case:
    print("The result of Product Except Self of {} is {}".format(nums,productExceptSelf_pre_post((nums))))
print("***** Method Three : two additional list *****")
def productExceptSelf_two_pre_post(nums):
    prefix_result =[1]*(len(nums))
    postfix_result = [1]*(len(nums))
    result = []
    prefix = 1
    postfix = 1
    # create prefix list
    for i in range(len(nums)):
        prefix_result[i]=prefix
        prefix = prefix*nums[i]
    # create postfix list
    for i in range(len(nums)-1,-1,-1):
        postfix_result[i]=postfix
        postfix = postfix*nums[i]
    for i in range(len(nums)):
        result.append(prefix_result[i]*postfix_result[i])
    return result
for nums in test_case:
    print("The result of Product Except Self of {} is {}".format(nums,productExceptSelf_two_pre_post((nums))))
print("---------------------152. Maximum Product Subarray-------------------------")
def maxProduct(nums):
    result = max(nums)
    curMin,curMax = 1,1
    for n in nums:
        if n==0:
            curMin,curMax = 1,1
            continue
        temp = curMax
        curMax = max(n*curMax,n*curMin,n)
        curMin = min(n*curMin,n*temp,n)
        result = max(result,curMax)
    return result
test_case=[[2,3,-2,4],[-2,0,-1],[-4,-3,-2]]
for nums in test_case:
    print("The Max Product of {} is {} ".format(nums,maxProduct(nums)))
print("---------------------153. Find Minimum in Rotated Sorted Array-------------------------")
def findMin(nums):
    l,r = 0,len(nums)-1
    result = nums[0]
    while l<=r:
        # the array is sorted
        if nums[l]<nums[r]:
            result = min(nums[l],result)
            break
        # the array is rotated
        mid = (l+r)//2
        result = min(result,nums[mid])
        if nums[mid]>=nums[l]:
            l = mid+1
        else:
            r = mid-1
    return result
test_case = [[3,4,5,1,2],[4,5,6,7,0,1,2]]
for nums in test_case:
    print("The Minimun in Rotated Sorted Array {} is {}".format(nums,findMin(nums)))
print("***** Method Two: min function, o(n) *****")
def findMin_min_fund(nums):
    return min(nums)
for nums in test_case:
    print("The Minimun in Rotated Sorted Array {} is {}".format(nums,findMin_min_fund(nums)))

print("---------------------338. Counting Bits-------------------------")
def countBits(n):
    # using dynamic programming.
    '''
    base:
    offset =1, i = 0 dp[0] = 0
    start loop:
    i=1, offset = 1 dp[1] = 1+dp[1-1=0] = 1
    i=2, offset*2 = i so dp[2] = 1+dp[2-offst=2-2=0]=1
    i=3, offset = 2, dp[3] = 1+dp[i-offset=3-2=1]=1+1=2
    i=4, offset*2 = 4, dp[4] = 1+dp[i-offset=4-4=0]=1
    ...
    '''
    dp=[0]*(n+1)
    offset = 1
    for i in range(1,n+1):
        if offset*2==i:
            offset = i
        dp[i] = 1+dp[i-offset]
    return dp
test_case=[2,5]
for n in test_case:
    print("The Counting Bit of {} is {}".format(n,countBits(n)))
print("---------------------268. Missing Number-------------------------")
print("***** Method One: Sum of total -Sum of list *****")
def missingNumber(nums):
    total_sum = 0
    for i in range(len(nums)+1):
        total_sum = total_sum+i
    return total_sum-sum(nums)
test_case = [[3,0,1],[0,1],[9,6,4,2,3,5,7,0,1]]
for nums in test_case:
    print("The missing number in {} is {}".format(nums,missingNumber(nums)))
print("***** Method Two: In Place Calculate *****")
def missingNumber_in_place(nums):
    result = len(nums)
    for i in range(len(nums)):
        result = result+(i-nums[i])
    return result
for nums in test_case:
    print("The missing number in {} is {}".format(nums,missingNumber_in_place(nums)))
print("---------------------191. Number of 1 Bits-------------------------")
print("***** Method One: mode by 2 and shift *****")
def hammingWeight(n):
    result =0
    while n>0:
        result = result+(n%2)
        n = n>>1
    return result

test_case = [0b00000000000000000000000000001011,0b00000000000000000000000010000000]
for n in test_case:
    print("Thre are {}  bit 1 in the {}".format(hammingWeight(n),bin(n)))
print("***** Method Two: substract 1 and using '&' logic *****")
'''
101  101-1=100
101&100 = 100 result+1
100-1 = 011
100&011 = 000 result+1
'''
def hammingWeight_substrac_and(n):
    result = 0
    while n>0:
        n = n&(n-1)
        result = result+1
    return result
for n in test_case:
    print("Thre are {}  bit 1 in the {}".format(hammingWeight_substrac_and(n),bin(n)))
print("***** Method Three: Transfer to String and Count ")
def hammingWeight_transfer_to_string(n):
    bin_to_strig= str(bin(n))
    cnt = 0
    for c in bin_to_strig:
        if c=='1':
            cnt = cnt+1
    return cnt
for n in test_case:
    print("Thre are {}  bit 1 in the {}".format(hammingWeight_transfer_to_string(n),bin(n)))

print("---------------------190. Reverse Bits-------------------------")
print("***** Method One: Shift and Using '&' to get the last bit and using '|' to reverse to corresponding location *****")
def reverseBits(n):
    result = 0
    # there is 32 bit, so the loop should be 32
    for i in range(32):
        # get the bit of the last bit
        # for example=> 3rd bit
        # 10110: 10110>>3 = 101(00)
        # 101 &1 = 1
        # for example=> 2nd bit
        # 10110: 10110>>4 = 10(110)
        # 10&1 = 0
        bit =(n>>i)&1
        result =result|(bit<<(31-i))
    return result
test_case = [0b00000010100101000001111010011100,0b11111111111111111111111111111101]
for n in test_case:
    print("The reverse of {} is {} = {}".format(bin(n),bin(reverseBits(n)),reverseBits(n)))
print("***** Method Two : Transfer to String *****")
def reverseBits_transfer_to_string(n):
    b_t_s = str(bin(n))
    # remove the Ob
    b_t_s = b_t_s[2:]
    # make sure the length is 32
    b_t_s = '0'*(32-len(b_t_s))+b_t_s
    s=''
    for i in range(len(b_t_s)-1,-1,-1):
        s=s+b_t_s[i]
    return int(s,2)
test_case = [0b00000010100101000001111010011100,0b11111111111111111111111111111101]
for n in test_case:
    print("The reverse of {} is {} = {}".format(n,bin(reverseBits_transfer_to_string(n)),reverseBits_transfer_to_string(n)))
print("---------------------70. Climbing Stairs-------------------------")
print("***** Method One: Consider to downstair *****")
def climbStairs(n):
    '''
    consider climb to down from the nth floor
    from the nth, the solution is 1
    from the n-1th floor, the solution is 1
    for the n-2th floor, the solution is solution of n-1th floor <1 step> + the solution of nth floor <2 steps>
    '''
    top_step,top_minus_step = 1,1
    for i in range(n-1):
        temp = top_minus_step
        top_minus_step = top_minus_step+top_step
        top_step=temp
    return top_minus_step
test_case = [2,3]
for n in test_case:
    print("There are {} ways to climb to floor {}".format(climbStairs(n),n))
print("***** Method Two: Consider to one step and two steps with Dynamic Programming *****")
def climbStairs_two_stair(n):
    step = []
    step.append(1)
    step.append(2)
    for i in range(2,n):
        step.append(step[i-1]+step[i-2])
    return step[n-1]
for n in test_case:
    print("There are {} ways to climb to floor {}".format(climbStairs_two_stair(n),n))

print("---------------------322. Coin Change-------------------------")
print("***** Using Dynamic Programming *****")
def coinChange(coins,amount):
    dp=[sys.maxsize]*(amount+1)
    dp[0]=0
    #Fill the dp list
    for temp_amount in range(1,amount+1):
        for coin in coins:
            if temp_amount-coin>=0:
                dp[temp_amount] = min(dp[temp_amount],1+dp[temp_amount-coin])
    if dp[amount]<sys.maxsize:
        return dp[amount]
    else:
        return -1
test_case = [[[1,2,5],11],[[2],3],[[1],0]]
for coins, amount in test_case:
    print("The minimum coins to get {} from {} is {}".format(amount,coins,coinChange(coins,amount)))

print("---------------------300. Longest Increasing Subsequence-------------------------")
def lengthOfLIS(nums):
  LIS=[1]*len(nums)
  # fill the dp list from back to head
  for i in range(len(nums)-1,-1,-1):
      for j in range(i+1,len(nums)):
          if nums[i]<nums[j]:
              LIS[i] = max(LIS[i],1+LIS[j])
  return max(LIS)
test_case = [[10,9,2,5,3,7,101,18],[0,1,0,3,2,3],[7,7,7,7,7,7,7]]
for nums in test_case:
    print("The length of longest Increasing Subsequence in the {} is {}".format(nums,lengthOfLIS(nums)))
print("---------------------1143. Longest Common Subsequence-------------------------")
print("***** Using Dynamic Programming and Create a n*m matrix *****")
def longestCommonSubsequence(text1,text2):
    dp =[[0 for j in range(len(text2)+1)] for i in range(len(text1)+1)]
    '''
    fill the matrix from right bottom to left top
    if the character is equal, then dp[i][j] = 1+dp[i+1][j+1]
    else dp[i][j] = max(dp[i+1][j],dp[i][j+1])
    '''
    for i in range(len(text1)-1,-1,-1):
        for j in range(len(text2)-1,-1,-1):
            if text1[i]==text2[j]:
                dp[i][j]=1+dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j],dp[i][j+1])
    result = dp[0][0]
    def print_matrix(dp):
        row = list(text1)+['*']
        col = list(text2)+['*']
        col = ['*']+col
        for i in range(len(row)):
            dp[i] = [row[i]]+dp[i]
        dp = [col]+dp
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                print(dp[i][j],end = "|")
            print()
    print_matrix(dp)
    return result
test_case = [["abcde","ace"],["abc","abc"],["abc","def"]]
for text1,text2 in test_case:
    print("The longest common subsequence of {} and {} is {}".format(text1,text2,longestCommonSubsequence(text1,text2)))
print("---------------------91. Decode Ways-------------------------")
print("***** Method One: Using Dynamic Programming and Recursive *****")
def numDecodings(s):
    dp = {len(s):1}
    def dfs(i):
        if i in dp:
            return dp[i]
        if s[i]=="0":
            return 0
        res = dfs(i+1)
        #if the s[i] and s[i+1] looks like 1* or 21-26
        if (i+1<len(s) and (s[i]=="1" or s[i]=="2" and s[i+1] in "0123456")):
            res = res+dfs(i+2)
        dp[i] = res
        return res
    return dfs(0)
test_case=["12","226","02","06","1201234"]
for s in test_case:
    print("There are {} ways to decoding the string {}".format(numDecodings(s),s))

print("***** Method Two: Using Dynamic Programming and two variables *****")
'''
fill the dp from back to start 
'''
def numDecodings_two_var(s):
    dp = {len(s):1}
    for i in range(len(s)-1,-1,-1):
        if s[i]=="0":
            dp[i]=0
        else:
            dp[i] = dp[i+1]
        # check if 1* or 20-26
        if (i+1<len(s) and (s[i]=="1" or s[i]=="2" and s[i+1] in "0123456")):
            dp[i] = dp[i]+dp[i+2]
    return dp[0]
for s in test_case:
    print("Ther are {} ways to decoding the string {}".format(numDecodings(s),s))
print("---------------------139. Word Break-------------------------")
print("***** Method One: Using Dynamic Programming *****")
def wordBreak(s,wordDict):
    dp=[False]*(len(s)+1)
    dp[len(s)]=True
    for i in range(len(s)-1,-1,-1):
        for word in wordDict:
            if(i+len(word))<=len(s) and s[i:i+len(word)]==word:
                dp[i] = dp[i+len(word)]
            if dp[i] is True:
                break
    return dp[0]
test_case = [["leetcode",["leet","code"]],["applepenapple",["apple","pen"]],["catsandog",["cats","dog","sand","and","cat"]]]
for s,wordDict in test_case:
    print("The word {} could be break by {} ? Answer: {}".format(s,wordDict,wordBreak(s,wordDict)))

print("---------------------198. House Robber-------------------------")
def rob(nums):
    rob1,rob2 = 0,0
    for i in nums:
        newRob = max(rob1+i,rob2)
        rob1 = rob2
        rob2 = newRob
    return rob2
test_case = [[1,2,3,1],[2,7,9,3,1]]
for nums in test_case:
    print("The max profit in {} is {}".format(nums,rob(nums)))
print("---------------------213. House Robber II-------------------------")
def rob_circle(nums):
    def linear_rob(nums):
        rob1,rob2 = 0,0
        for i in nums:
            newRob = max(rob1+i,rob2)
            rob1 = rob2
            rob2 = newRob
        return rob2
    return max(nums[0],linear_rob(nums[1:]),linear_rob(nums[:-1]))
test_case = [[2,3,2],[1,2,3,1]]
for nums in test_case:
    print("The max profit in {} is {}".format(nums,rob_circle(nums)))
print("---------------------337. House Robber III-------------------------")
def rob_binary_tree(root):
    def dfs(root):
        if root is None:
            return [0,0]
        leftPair = dfs(root.left)
        rightPair = dfs(root.right)
        '''
        # for each node, there are two cases, so the return value will be [withNode, withoutNode]
        # for the withNode,it includes the node and plue and value of left and right children tree's withoutNode value
        # for the withOutNode, it euqals the max of left children plus the max of right children
        '''
        withRoot = root.value+leftPair[1]+rightPair[1]
        withoutRoot = max(leftPair)+max(rightPair)
        return [withRoot,withoutRoot]
    return max(dfs(root))
test_case = [[3,2,3,None,3,None,1],[3,4,5,1,3,None,1]]
for node_list in test_case:
    tree = generate_tree_from_list(node_list)
    print("The Map of Houses is {} and The Maximum Profit is {}".format(tree[0],rob_binary_tree(tree[0])))

print("---------------------362. Unique Paths-------------------------")
def uniquePaths(m,n):
    matrix = [[1 for _ in range(n)] for _ in range(m)]
    for row in range(1,m):
        for col in range(1,n):
            matrix[row][col] =matrix[row][col-1]+matrix[row-1][col]
    print_matrix(matrix)
    return matrix[m-1][n-1]
test_case = [[3,7],[3,2],[1,5],[5,1]]
for m,n in test_case:
    print("There are {} ways to get from left-top to right-bottom in the {} * {} grid".format(uniquePaths(m,n),m,n))
print("---------------------133. Clone Graph -------------------------")
def cloneGraph(node):
    '''
    Using hashtable to store the map of old and new Node
    using the DFS (Deep First Search) to find the adjacency node
    '''
    hash_map_new_old_map = {}
    def dfs(node):
        if node in hash_map_new_old_map:
            return hash_map_new_old_map[node]
        # copy the value of the old node
        copy = Graph_Node(node.value)
        hash_map_new_old_map[node] = copy
        for neighbor in node.neighbors:
            copy.neighbors.append(dfs(neighbor))
        return copy
    return dfs(node) if node else None
test_case = [[[2,4],[1,3],[2,4],[1,3]],[[]],[]]
for adjList in test_case:
    node = generate_graph(adjList)
    print("The Original Graph:")
    print(display_graph(node))
    print("The Result after Clone:")
    print(display_graph(cloneGraph(node)))

print("---------------------207. Course Schedule -------------------------")
def canFinish(numCourses,prerequisites):
    # map each course to  prerequise list
    Precours = {i:[] for i in range(numCourses)}
    for course, precourse in prerequisites:
        Precours[course].append(precourse)
    # visitSet = all course along the current coure
    visitSet =set()
    def dfs(crs):
        if crs in visitSet:
            return False
        if Precours[crs]==[]:
            return True
        visitSet.add(crs)
        for pre in Precours[crs]:
            if not dfs(pre):
                return False
        visitSet.remove(crs)
        Precours[crs] = []
        return True
    for crs in range(numCourses):
        if not dfs(crs):
            return False
    return True
test_case = [2,[[1,0]]],[2,[[1,0],[0,1]]]
for numCourses,prerequisites in test_case:
    print("The available of {} course(s) with {} requirement is {} ".format(numCourses,prerequisites,canFinish(numCourses,prerequisites)))
print("---------------------417. Pacific Atlantic Water Flow -------------------------")
def pacificAtlantic(heights):
    ROWS,COLS = len(heights),len(heights[0])
    pac,atl = set(),set()
    def dfs(r,c,visit,preHeight):
        # if the item is in the visit set
        # or if the item is out of bounder
        # or if the item is less than the previous height
        # "calculate from low to high, so the new height should greater than previous height"
        if (r,c) in visit or r<0 or c<0 or r==ROWS or c==COLS or heights[r][c]<preHeight:
            return
        # meet the requirement, add to visit set and using the dfs to search the round items
        visit.add((r,c))
        # down row
        dfs(r+1,c,visit,heights[r][c])
        # up row
        dfs(r-1,c,visit,heights[r][c])
        # left col
        dfs(r,c-1,visit,heights[r][c])
        # right col
        dfs(r,c+1,visit,heights[r][c])
    # create the visit set matrix
    # search for each column
    for c in range(COLS):
        # search from the first row
        # all the items on the top row could reach to the pacific
        dfs(0,c,pac,heights[0][c])
        # search for the bottom row
        # all the item on the bottom row could reach to the atlantic
        dfs(ROWS-1,c,atl,heights[ROWS-1][c])

    for r in range(ROWS):
        # search for the left column
        # all the items on the left column could reach to pacific
        dfs(r,0,pac,heights[r][0])
        # search for the right column
        # all the items on the right column could reach the atlantic
        dfs(r,COLS-1,atl,heights[r][COLS-1])
    result = []
    # visit each item to find the final result
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) in pac and (r,c) in atl:
                result.append([r,c])
    return result
test_case =[[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]],[[2,1],[1,2]]]
for heights in test_case:
    print("The Graphic:")
    print_matrix(heights)
    print("The list of water could flow to both Pacific and Atlantic is {} ".format(pacificAtlantic(heights)))
print("---------------------200. Number of Islands-------------------------")
'''
Using the DSF, after find the item, instead of put it in the set, just change it's value to "X"
The next round will not count it as the new item since only the value "1" will be count as a new island 
'''
def numIslands(grid):
    def dfs(grid,row,col,r,c):
        # if the item is out-of-boundary or not the island
        # fiish the search
        if (r<0 or c<0 or r>=row or c >=col or grid[r][c]!="1"):
            return
        grid[r][c]="X"
        # search the items around the current item
        dfs(grid,row,col, r-1,c)
        dfs(grid,row,col,r+1,c)
        dfs(grid,row,col,r,c-1)
        dfs(grid,row,col,r,c+1)
    if len(grid)==0 or len(grid[0]) ==0:
        return 0
    row,col = len(grid),len(grid[0])
    count =0
    for r in range(row):
        for c in range(col):
            if grid[r][c]=="1":
                count = count+1
                dfs(grid,row,col,r,c)
    print("After the Search: ")
    print_matrix(grid)
    return count
test_case =[[["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]],[
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]]
for grid in test_case:
    print("The Graphic:")
    print_matrix(grid)
    print("There is(are) {} island(s) on the ocean ".format(numIslands(grid)))

print("---------------------128. Longest Consecutive Sequence-------------------------")
'''
Using the DSF, after find the item, instead of put it in the set, just change it's value to "X"
The next round will not count it as the new item since only the value "1" will be count as a new island 
'''
print("***** Method One: Sort and Count O(nlgn) *****")
def longestConsecutive_nlgn(nums):
   nums.sort()
   max_count =1
   sub_count=1
   for i in range(len(nums)-1):
       if nums[i+1]==nums[i]+1:
           sub_count=sub_count+1
           max_count = max(sub_count,max_count)
       else:
           sub_count=1
   return max_count
test_case = [[100,4,200,1,3,2],[0,3,7,2,5,8,4,6,0,1]]
for nums in test_case:
    print("The Length of the Longest Consecutive of {} is {} ".format(nums, longestConsecutive_nlgn(nums)))
print("***** Method Two: Sort and Count O(n)*****")
def longestConsecutive(nums):
    numSet = set(nums)
    longest = 0
    for n in nums:
        # check is it is the star of a sequence
        if (n-1) not in numSet:
            length = 0
            while (n+length) in numSet:
                length = length+1
            longest = max(length,longest)
    return longest
for nums in test_case:
    print("The Length of the Longest Consecutive of {} is {} ".format(nums, longestConsecutive(nums)))

print("---------------------269. Alien Dictionary-------------------------")
'''
using postorder dfs and topologic
'''
def alienOeder(words):
    # set the hashmap for each character
    adj = {ch: set() for word in words for ch in word }
    for i in range(len(words)-1):
        w1,w2 = words[i],words[i+1]
        min_len = min(len(w1),len(w2))
        # if w1 = abc, w2 = ab, since the words are sorted
        # in this case, return false
        if len(w1)>len(w2) and w1[:min_len]==w2[:min_len]:
            return ""
        #fill the hashmap, by compare each character
        # find the first different character, and
        # append this character to first character's adj list
        for j in range(min_len):
            if w1[j]!=w2[j]:
                adj[w1[j]].add(w2[j])
                break
        # after finish the hashmap, use the DFS to find the cycle
        # if there is a cycle existed, reture " "
        visit = {} # False=visted, True = current path
        res = []
    def dfs(c):
        '''
        if character is visited, return the visit[c]'s value
        Ture:means character is on the current path, False: means character is just visited
        example: ab, and abc, using postorder dfs, a->b and a->c, after mark c as visited,
        when process the b->c, and found c in the visited, not means there is the cycle
        '''
        if c in visit:
            return visit[c]
        visit[c] = True
        for neighbor in adj[c]:
            if dfs(neighbor):
                return True
        visit[c] = False
        res.append(c)
    # get the result list
    for c in adj:
        if dfs(c):
            return ""
    res.reverse()
    return "".join(res)
test_case = [["caa", "aaa", "aab"],["wrt","wrf","er","ett","rftt"]]
for words in test_case:
    print("Based on the {}, the order is {}".format(words,alienOeder(words)))
print("---------------------261. Graph Valid Tree-------------------------")
'''
using postorder dfs and topologic
'''
def validTree(n,edges):
    # if there is no node, still is the tree
    if not n:
        return True
    adj={i:[] for i in range(n)}
    for n1,n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n1)
    visit = set()
    # using two variables, current node and previous node
    # if the node in the visit, not means there is a cycle
    # it also means the node is on the path too
    def dfs(i, prev):
        if i in visit:
            return False
        visit.add(i)
        for j in adj[i]:
            if j==prev:
                continue
            if not dfs(j,i):
                return False
        return True
    return dfs(0,-1) and n==len(visit)
n=5
edges = [[0,1],[0,2],[0,3],[1,4]]
print("based on the edges {}, these {} node(s) is the tree: {}".format(edges, n, validTree(n,edges)))
print("---------------------323. Number of Connected Components in an Undirected Graph-------------------------")
print("***** Method One: Using rank and Union Find Method *****")
def countComponent(n,edges):
    par = [i for i in range(n)]
    rank = [1]*n
    # find item's parent node
    def find(n1):
        res = n1
        while res!=par[res]:
            par[res] = par[par[res]]
            res = par[res]
        return res
    # find there is the connection of two nodes
    def union(n1,n2):
        p1,p2 = find(n1),find(n2)
        # if two nodes have the same parent node
        if p1==p2:
            return 0
        if rank[p2]>rank[p1]:
            par[p1] = p2
            rank[p2]=rank[p2]+rank[p1]
        else:
            par[p2] = p1
            rank[p1] = rank[p1]+rank[p2]
        return 1
    res = n
    for n1,n2 in edges:
        res = res-union(n1,n2)
    return res
test_case = [[5, [[0, 1], [1, 2], [3, 4]]],[5,[[0, 1], [1, 2], [2, 3], [3, 4]]]]
for n,edges in test_case:
    print("In this {} nodes graph with {} edges, there are {} Connected Components".format(n,edges, countComponent(n,edges)))
print("***** Method Two: Traditional DFS method *****")
def countComponent_dfs(n,edges):
    graph = {i:[] for i in range(n)}
    for n1,n2 in edges:
        graph[n1].append(n2)
        graph[n2].append(n1)
    visit = set()
    count = 0
    q = deque()
    for i in range(n):
        if i in visit:
            continue
        q.append(i)
        while len(q)>0:
            cur = q.popleft()
            if cur in visit:
                continue
            visit.add(cur)
            for nei in graph[cur]:
                q.append(nei)
        count = count+1
    return count
test_case = [[5, [[0, 1], [1, 2], [3, 4]]],[5,[[0, 1], [1, 2], [2, 3], [3, 4]]]]
for n, edges in test_case:
    print("In this {} nodes graph with {} edges, there are {} Connected Components".format(n, edges,
                                                                                               countComponent_dfs(n,
                                                                                                              edges)))
print("---------------------435. Non-overlapping Intervals-------------------------")
def eraseOverlapIntervals(intervals):
    intervals.sort()
    # print(intervals)
    count  = 0
    PrevEnd = intervals[0][1]
    for start,end in intervals[1:]:
        if start>=PrevEnd:
            PrevEnd = end
        else:
            count = count+1
            PrevEnd = min(PrevEnd,end)
    return count
test_case = [[[1,2],[2,3],[3,4],[1,3]],[[1,2],[1,2],[1,2]],[[1,2],[2,3]]]
for intervals in test_case:
    print("There is(are) {} overlapping intervals need to be remove from the {}".format(eraseOverlapIntervals(intervals),intervals))
print("---------------------920 · Meeting Rooms-------------------------")
def can_attend_meetings(intervals):
    intervals.sort(key = lambda i: i[0])
    for i in range(1,len(intervals)):
        i1 = intervals[i-1]
        i2 = intervals[i]
        if i1[1]>i2[0]:
            return False
        return True
test_case = [[(0,30),(5,10),(15,20)], [(5,8),(9,15)]]
for intervals in test_case:
    print("The meeting can be scheduled based on {}:{}".format(intervals,can_attend_meetings(intervals)))
print("---------------------919 · Meeting Rooms II-------------------------")
def min_meeting_rooms(intervals):
    start = sorted([i[0] for i in intervals])
    end = sorted([i[1] for i in intervals])
    s,e = 0,0
    count = 0
    result =0
    while s <len(start):
        if start[s]<end[e]:
            count = count+1
            s= s+1
        else:
            count = count-1
            e = e+1
        result = max(result,count)
    return result
test_case = [[(0,30),(5,10),(15,20)], [(2,7)]]
for intervals in test_case:
    print("For the meeting schedule {}, need to have at least {} meeting rooms".format(intervals,min_meeting_rooms(intervals)))
print("---------------------69. Sqrt(x)-------------------------")
'''
using binary search to find the y, where y*y ==x
'''
def mySqrt(x):
    start = 0
    end = x
    while start+1<end:
        mid = (start+end)//2
        if mid*mid==x:
            return mid
        elif mid*mid>x:
            end = mid
        else:
            start = mid
    if end*end==x:
        return end
    else:
        return start
test_case = [4,8]
for x in test_case:
    print("The square root of {} is {}".format(x,mySqrt(x)))

print("---------------------23. Merge k Sorted Lists-------------------------")
'''
using merge method
'''
def mergeKLists(lists):
    def mergeList(l1,l2):
        dummy = ListNode()
        tail = dummy
        while l1 and l2:
            if l1.val<l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail=tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next
    if not lists or len(lists)==0:
        return None
    while len(lists)>1:
        mergeLists=[]
        for i in range(0,len(lists),2):
            l1 = lists[i]
            l2 = lists[i+1] if (i+1)<len(lists) else None
            mergeLists.append(mergeList(l1,l2))
        lists = mergeLists
    return lists[0]

test_case = [[[1,4,5],[1,3,4],[2,6]],[],[[]]]
for linked_lists in test_case:
    link_list = []
    for lst in linked_lists:
        link_list.append(generate_linked_list_with_val(lst))
    print("The result of merger {} is:".format(linked_lists))
    display_linked_list_with_val(mergeKLists(link_list))
print("---------------------141. Linked List Cycle-------------------------")
print("***** Method One: Hashmap *****")
def hasCycle(head):
    hashmap = set()
    while head:
        if head in hashmap:
            return True
        else:
            hashmap.add(head)
            head = head.next
    return False
test_case = [[[3,2,0,-4],1],[[1,2],0],[[1],-1]]
for linked_lists,pos in test_case:
    head = generate_linked_list(linked_lists)
    tail = find_last_link_list(head)
    tail.next = find_kth_linked_node(head,pos)
    print("Is there a cycle in the linked list {},{}:{}".format(linked_lists,pos,hasCycle(head)))
print("***** Method Two: Tow pointer -> Fast Point and Slow Point ***** ")
def hasCycle_fast_slow_point(head):
    fast = head
    slow = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow==fast:
            return True
    return False
test_case = [[[3,2,0,-4],1],[[1,2],0],[[1],-1]]
for linked_lists,pos in test_case:
    head = generate_linked_list(linked_lists)
    tail = find_last_link_list(head)
    tail.next = find_kth_linked_node(head,pos)
    print("Is there a cycle in the linked list {},{}:{}".format(linked_lists,pos,hasCycle_fast_slow_point(head)))
print("---------------------143. Reorder List-------------------------")

def reorderList(head):
    '''
    using the fase and slow pointer to find the middle
    divide the linked list from the middle
    reversal the second half
    merge the first and second half
    '''
    fast,slow = head,head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # slow will be the middle
    # reverse the second half
    second = slow.next
    # break the linked list after slow
    slow.next = None
    # set prev as the None
    prev = slow.next
    while second:
        temp = second.next
        second.next = prev
        prev = second
        second = temp
    # merge two list
    first,second = head,prev
    while second:
        temp1,temp2 = first.next,second.next
        first.next = second
        second.next = temp1
        first,second = temp1,temp2
    return head
test_case = [[1,2,3,4,5],[1,2,3,4]]
for linked_list in test_case:
    head = generate_linked_list_with_val(linked_list)
    print("Reoder the liked list:",end ="")
    display_linked_list_with_val(head)
    print("The result is:",end="")
    display_linked_list_with_val(reorderList(head))
    print()

print("---------------------79. Word Search-------------------------")

def exist(board,word):
    '''
    using the backtracking method and DFS algortithm
    '''
    ROW,COLS = len(board),len(board[0])
    path = set()
    # r,c is the position and i is the index of the word
    def dfs(r,c,i):
        if i==len(word):
            return True
        if (r<0 or c<0 or r>=ROW or c>=COLS or word[i]!=board[r][c] or (r,c) in path):
            return False
        path.add((r,c))
        # search for the neighbor characters
        res= (dfs(r+1,c,i+1))or(dfs(r-1,c,i+1))or(dfs(r,c+1,i+1))or(dfs(r,c-1,i+1))
        # after search, remove the (r,c) from the set
        path.remove((r,c))
        return res
    for r in range(ROW):
        for c in range(COLS):
            if dfs(r,c,0):
                return True
    return False
test_case = [
[[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED"],
[[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE"],
[[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB"],
]
for board, word in test_case:
    print("There is the word {} exits in board {}: {}".format(word,board,exist(board,word)))

print("---------------------424. Longest Repeating Character Replacement-------------------------")
print("***** Method One: Regular Sliding windows *****")
def characterReplacement(s,k):
    '''
    using the sliding windows
    '''
    count = {}
    res = 0
    l = 0
    for r in range(len(s)):
        # length of the sliding window - max value of the letters > k
        # then the window is valid
        count[s[r]] = 1+count.get(s[r],0)
        # if not meet the requirement, then update the count and
        # and sliding the window
        while (r-l+1)-max(count.values())>k:
            count[s[l]]=count[s[l]]-1
            l = l+1
        res = max(res,r-l+1)
    return res

test_case = [["ABAB",2],["AABABBA",1]]
for s,k in test_case:
    print("The longest Repeating Character Replacement in {} with {} wide cards is {}".format(s,k,characterReplacement(s,k)))
print("***** Method Two: Sliding windows but without substract the count dict *****")
def characterReplacement_not_count_update(s,k):
    count = {}
    res = 0
    l = 0
    maxf = 0
    for r in range(len(s)):
        count[s[r]]=1+count.get(s[r],0)
        maxf = max(maxf,count[s[r]])
        while (r-l+1)-maxf>k:
            count[s[l]] = count[s[l]]-1
            l = l+1
        res = max(res,r-l+1)
    return res
for s,k in test_case:
    print("The longest Repeating Character Replacement in {} with {} wide cards is {}".format(s,k,characterReplacement_not_count_update(s,k)))

print("---------------------76. Minimum Window Substring-------------------------")
def minWindow(s,t):
    if t=="":
        return ""
    countT={}
    window = {}
    # set the target hashmap
    for c in t:
        countT[c]= 1+countT.get(c,0)
    # current count
    have = 0
    # need to count (count in the string t)
    need = len(countT)
    # result start and end index
    res=[-1,-1]
    # length of the result
    resLen = float("infinity")
    # left point
    l = 0
    for r in range(len(s)):
        c = s[r]
        window[c] = 1+window.get(c,0)
        if c in countT and window[c]==countT[c]:
            # meet the requirement for this character
            have = have+1
        while have == need:
            # update result, shrink the window to find the minimum length
            if (r-l+1)<resLen:
                resLen=(r-l+1)
                res = [l,r]
            # pop from the left of the window
            window[s[l]] = window[s[l]]-1
            if s[l] in countT and window[s[l]]<countT[s[l]]:
                have = have-1
            l = l+1
    l,r = res
    return s[l:r+1] if resLen!=float("infinity") else ""
test_case = [["ADOBECODEBANC", "ABC"],[ "a","a"],["a",  "aa"]]
for s,t in test_case:
    print(f"the minimum window substring in {s} of {t} is {minWindow(s,t)}")

print("---------------------125. Valid Palindrome-------------------------")
def isPalindrome(s):
    newStr = ""
    for c in s:
        if c.isalnum():
            newStr = newStr+c.lower()
    if newStr==newStr[::-1]:
        return True
    else:
        return False
test_case = ["A man, a plan, a canal: Panama", "race a car", " "]
for s in test_case:
    print(f"The phrase\" {s} \" is a palindrome: {isPalindrome(s)}")

print("---------------------647. Palindromic Substrings-------------------------")
def countSubstrings(s):
    result = 0
    # helper function to find the palindromic
    def countPali(l,r):
        temp_result = 0
        while l>=0 and r<len(s) and s[l]==s[r]:
            temp_result = temp_result+1
            l = l-1
            r= r+1
        return temp_result
    for i in range(len(s)):
        result = result+countPali(i,i)
        result = result+countPali(i,i+1)
    return result
test_case = ['abc',"aaa"]
for item in test_case:
    print(f"There is(are) {countSubstrings(item)} Palindromic Substrings in the {item}")

print("---------------------271. Encode and Decode Strings-------------------------")
print("***** Method One: Using List to store the count *****")
def encode_edcode_count_list(s):
    count_list = []
    updated_string= ""
    result = []
    for ch in s:
        count_list.append(len(ch))
        updated_string =  updated_string+ch
    j=0
    for i in range(len(count_list)):
        length = count_list[i]
        result.append( updated_string[j:j+length])
        j=j+length
    return result
test_case = [["lint","code","love","you"],["we", "say", ":", "yes"]]
for s in test_case:
    print(f"The encoding and decoing of {s} is {encode_edcode_count_list(s)}")
print("***** Method Two: In-Place Count *****")
def encode_edcode(s):
    updated_string = ""
    # add length and symbol for each item
    for ch in s:
        updated_string = updated_string+str(len(ch))+"#"
    # decoding the updated string
    result,i = [],0
    while i <len(updated_string):
        j = i
        while updated_string[j]!="#":
            j =j+1
        length = int(updated_string[i:j])
        result.append(updated_string[j+1:j+length+1])
        i = j+1+length
    return result
for s in test_case:
    print(f"The encoding and decoing of {s} is {encode_edcode_count_list(s)}")
print("---------------------100. Same Tree-------------------------")
class TreeNode(binarytree.Node):
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def generate_tree_from_list(root):
    #
    node_list = []
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
def isSameTree(p,q):
    if not p and not q:
        return True
    if not p or not q or p.value!=q.value:
        return False
    return (isSameTree(p.left,q.left)) and (isSameTree(p.right,q.right))

test_case = [[1,2,3], [1,2,3]],[[1,2], [1,None,2]],[[1,2,1], [1,1,2]]
for tree1_list,tree2_list in test_case:
    p = generate_tree_from_list(tree1_list)
    q = generate_tree_from_list(tree2_list)
    print(p,q)
    print(f"The two trees is same: {isSameTree(p,q)}")

print("---------------------226. Invert Binary Tree-------------------------")
# def generate_tree_from_list(root):
#     #
#     node_list = []
#     for i in range(len(root)):
#         if root[i] is not None:
#             node_list.append(Node(root[i]))
#         else:
#             node_list.append(None)
#     # Set the Left/Right child for each node
#     for i in range(len(node_list)//2):
#         if node_list[i] is not None:
#             left_child =2*i+1
#             right_child = 2*i+2
#             if left_child<len(node_list):
#                 node_list[i].left = node_list[left_child]
#             if right_child<len(node_list):
#                 node_list[i].right = node_list[right_child]
#     return node_list
def invertTree(root):
    if not root:
        return None
    temp_left = root.left
    root.left = root.right
    root.right = temp_left
    invertTree(root.left)
    invertTree(root.right)
    return root
test_case = [[4,2,7,1,3,6,9],[2,1,3],[]]
for tree_list in test_case:
    if len(tree_list)==0:
        print(None)
        break
    root = generate_tree_from_list(tree_list)
    print(f"The list to formate the tree is {tree_list}")
    if root:
        print(root)
        print("The result of Invert tree is: ")
        print(invertTree(root))
    if not root:
        print([])
print("---------------------297. Serialize and Deserialize Binary Tree-------------------------")
print("***** Method One: BFS and level scan *****")
def serialize(root):
    result = []
    q = deque()
    q.append(root)
    while len(q)>0:
        current = q.popleft()
        # if not isinstance(current,binarytree.Node):
        #     result.append(current)
        # else:
        #     result.append(current.values)
        result.append(current)
        if current:
            if current and current.left:
                q.append(current.left)
            else:
                q.append(None)
            if current.right:
                q.append(current.right)
            else:
                q.append(None)
    final_result =[]
    for i in range(len(result)):
        if isinstance(result[i],binarytree.Node):
            final_result.append(result[i].value)
        else:
            final_result.append(result[i])
    index = len(final_result)-1
    while final_result[index] is None:
        index = index-1
    return final_result[:index+1]
def deserialize(data):
    # return generate_tree_from_list(data)
    tree_node_list = []
    for i in range(len(data)):
        if data[i]==None:
            tree_node_list.append(None)
        else:
            tree_node_list.append(Node(data[i]))
    for i in range(len(tree_node_list)//2):
        if tree_node_list[i] is not None:
            left_index = i*2+1
            right_index =i*2+2
            if left_index<len(tree_node_list):
                tree_node_list[i].left = tree_node_list[left_index]
            if right_index<len(tree_node_list):
                tree_node_list[i].right = tree_node_list[right_index]
    return tree_node_list[0]

test_case = [[1,2,3,None,None,4,5],[]]
for tree_list in test_case:
    if not tree_list:
        print("None")
        break
    root = generate_tree_from_list(tree_list)
    print(f"Original Tree: {root}")
    data = serialize(root)
    print(f"Serialize Result: {data}")
    de_root = deserialize(data)
    print(f"Deserialize Result:{de_root}")
print("***** Method Two: Pre-Order and DFS *****")
def serialize_dfs_preorder(root):
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
def deserialize_dfs_preorder(data):
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
    se = serialize_dfs_preorder(root)
    print(f"The Result of Serialize is {se}")
    print(f"The result of Deserialize is {deserialize_dfs_preorder(se)}")
print("---------------------572. Subtree of Another Tree-------------------------")
def isSubtree(root, subroot):
    def sameTree(p,q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.value==q.value:
            return (sameTree(p.left,q.left)) and (sameTree(p.right,q.right))
    if not subroot:
        return True
    if not root:
        return False
    if sameTree(root,subroot):
        return True
    else:
        return isSubtree(root.left,subroot) or isSubtree(root.right,subroot)
test_case = [[[3,4,5,1,2], [4,1,2]],[[3,4,5,1,2,None,None,None,None,0], [4,1,2]],[[3,4,5,1,2,None,None,None,None,0], [4,1,2,None,None,0]]]
for root_list, subroot_list in test_case:
    root = generate_tree_from_list(root_list)
    subroot = generate_tree_from_list(subroot_list)
    print(f"Tree One is {root}")
    print(f"Tree Two is {subroot}")
    print(f"Tree TWO is the subtree of Tree One: {isSubtree(root,subroot)}")
print("---------------------105. Construct Binary Tree from Preorder and Inorder Traversal-------------------------")
def buildTree(preorder,inorder):
    if not preorder or not inorder:
        return None
    root=TreeNode(preorder[0])
    ind = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:ind+1],inorder[:ind])
    root.right=buildTree(preorder[ind+1:],inorder[ind+1:])
    return root
test_case = [[[3,9,20,15,7],[9,3,15,20,7]],[[-1],[-1]]]
for preorder, inorder in test_case:
    print(f"The Tree based on the Pre-Order {preorder} and In-Order {inorder} is {buildTree(preorder,inorder)}")

print("---------------------230. Kth Smallest Element in a BST-------------------------")
print("***** Method One: In-Order Scan and find kth smallest element *****")
def kthSmallest_in_order(root,k):
    if k<1:
        return False
    def in_order(root):
        if root is None:
            return []
        else:
            return in_order(root.left)+[root.value]+in_order(root.right)
    result = in_order(root)
    print(result)
    return result[k-1]

test_case = [[[3,1,4,None,2],1],[[5,3,6,2,4,None,None,1], 3]]
for tree_node_list,k in test_case:
    root =generate_tree_from_list(tree_node_list)
    print(f"The Tree is {root}")
    print(f"The {k} smallest element in BST is {kthSmallest_in_order(root,k)}")
print("***** Method One: using Stack *****")
def kthSmallest_stack(root,k):
    stack = []
    n = 0
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        n = n+1
        if n==k:
            return current.values[0]
        current = current.right
for tree_node_list,k in test_case:
    root =generate_tree_from_list(tree_node_list)
    print(f"The Tree is {root}")
    print(f"The {k} smallest element in BST is {kthSmallest_stack(root,k)}")

print("---------------------98. Validate Binary Search Tree-------------------------")
def isValidBST(root):
    def validBST(root,left_bound,right_bound):
        if not root:
            return True
        if not((root.value>left_bound) and (root.value<right_bound)):
            return False
        return validBST(root.left,left_bound,root.value) and validBST(root.right,root.value,right_bound)
    return validBST(root,float("-inf"),float("inf"))
test_case = [[2,1,3],[5,1,4,None,None,3,6]]
for tree_node_list in test_case:
    root =generate_tree_from_list(tree_node_list)
    print(f"The Tree  {root }is the valid BST: {isValidBST(root)}")
print("---------------------235. Lowest Common Ancestor of a Binary Search Tree-------------------------")
print("***** Method One: Recursive *****")
def lowestCommonAncestor(root,p,q):
    if p.value<root.value<q.value:
        return root.value
    elif p.value==root.value or q.value==root.value:
        if p==root:
            return p.value
        else:
            return q.value
    elif p.value<root.value and q.value<root.value:
        return lowestCommonAncestor(root.left,p,q)
    elif p>root.value and q>root.value:
        return lowestCommonAncestor(root.right,p,q)

test_case = [[[6,2,8,0,4,7,9,None,None,3,5],2,8],[[6,2,8,0,4,7,9,None,None,3,5],2,4],[[2,1],2,1]]
for tree_node_list,p,q in test_case:
    root =generate_tree_from_list(tree_node_list)
    # print(f"The Tree is {root}")
    p,q = min(p,q),max(p,q)
    p = TreeNode(p)
    q = TreeNode(q)
    print(f"The Lowest Common Ancestor of {p.value},{q.value} in {root} is {lowestCommonAncestor(root,p,q)}")
print("***** Method One: Iteration *****")
def lowestCommonAncestor_iteration(root,p,q):
    cur = root
    while cur:
        if p.value <cur.value and q.value<cur.value:
            cur=cur.left
        elif p.value >cur.value and q.value>cur.value:
            cur=cur.right
        else:
            return cur.value
for tree_node_list,p,q in test_case:
    root =generate_tree_from_list(tree_node_list)
    # print(f"The Tree is {root}")
    p,q = min(p,q),max(p,q)
    p = TreeNode(p)
    q = TreeNode(q)
    print(f"The Lowest Common Ancestor of {p.value},{q.value} in {root}is {lowestCommonAncestor_iteration(root,p,q)}")
print("---------------------208. Implement Trie (Prefix Tree)-------------------------")
class TriesNode():
    def __init__(self):
        self.children = {}
        self.EndofWord= False
class Trie():
    def __init__(self):
        self.root = TriesNode()
    def insert(self,word):
        current_node = self.root
        for ch in word:
            if ch not in current_node.children:
                current_node.children[ch]= TriesNode()
            current_node = current_node.children[ch]
        current_node.EndofWord=True
    def search(self,word):
        current = self.root
        for ch in word:
            if ch not in current.children:
                return False
            current = current.children[ch]
        return current.EndofWord
    def startsWith(self,prefix):
        cur = self.root
        for ch in prefix:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return True
search_word1 = "apple"
search_word2 = "app"
search_prefix = "app"
obj = Trie()
obj.insert(search_word1)
print(f"The result of search {search_word1} is {obj.search(search_word1)}")
print(f"The result of search {search_word2} is {obj.search(search_word2)}")
print(f"The result of search prefix {search_prefix} is {obj.startsWith(search_prefix)}")
obj.insert("app")
print(f"The result of search {search_word2} is {obj.search(search_word2)}")
print("---------------------211. Design Add and Search Words Data Structure-------------------------")
class WordDictionary():
    def __init__(self):
        self.root = TriesNode()
    def addWord(self, word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = TriesNode()
            cur = cur.children[ch]
        cur.EndofWord = True

    def search(self,word):
        def dfs(j,root):
            cur = root
            for i in range(j,len(word)):
                ch = word[i]
                if ch==".":
                    for child in cur.children.values():
                        if dfs(i+1,child):
                            return True
                    return False
                else:
                    if ch not in cur.children:
                        return False
                    cur = cur.children[ch]
            return cur.EndofWord
        return dfs(0,self.root)

obj = WordDictionary()
add_word_test_case = ["bad","dad","mad"]
search_word_test_case = ["pad","bad",".ad","b.."]
for word in add_word_test_case:
    obj.addWord(word)
for word in search_word_test_case:
    print(f"Word {word} is in the dictionary: {obj.search(word)}")

print("---------------------212. Word Search II-------------------------")
class TriesNode(TriesNode):
    def __init__(self):
        self.children = {}
        self.endWord = False
    def addword(self,word):
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = TriesNode()
            cur = cur.children[ch]
        cur.endWord = True
def findwords(board,words):
    root = TriesNode()
    for word in words:
        root.addword(word)
    ROWS,COLS = len(board),len(board[0])
    res,visit = set(),set()
    def dfs(r,c,node,word):
        if(r<0 or c<0 or r==ROWS or c==COLS or board[r][c] not in node.children or (r,c) in visit):
            return
        visit.add((r,c))
        node = node.children[board[r][c]]
        word = word+board[r][c]
        if node.endWord:
            res.add(word)
        dfs(r+1,c,node,word)
        dfs(r-1,c,node,word)
        dfs(r,c+1,node,word)
        dfs(r,c-1,node,word)
        visit.remove((r,c))
    for r in range(ROWS):
        for c in range(COLS):
            dfs(r,c,root,"")
    return list(res)

test_case = [[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],["oath","pea","eat","rain"]],\
            [[["a","b"],["c","d"]],["abcb"]]
for board,words in test_case:
    print(f"The board {board} contain words: {findwords(board,words)}")
print("---------------------347. Top K Frequent Elements-------------------------")
print("***** Method One: Sort Dict *****")
def topKFrequent(nums,k):
    cnt = {}
    for item in nums:
        if item not in cnt:
            cnt[item] = 1
        else:
            cnt[item] = cnt[item]+1
    result = []
    # print(cnt)
    cnt_1 = sorted(cnt.items(),key=lambda x:x[1],reverse = True)
    # print(cnt_1)
    for i in range(k):
        result.append(cnt_1[i][0])
    return result
test_case =[[[1,1,1,2,2,3],2],[[1],1]]
for nums,k in test_case:
    print(f"The Top {k} of elements in {nums} are {topKFrequent(nums,k)}")
print("***** Method One: Bucket Sort O(N)*****")
def topKFrequent_bucket(nums,k):
    count = {}
    freq = [[] for _ in range(len(nums)+1)]
    for n in nums:
        count[n] = 1+count.get(n,0)
    for n,c in count.items():
        freq[c].append(n)
    result = []
    for i in range(len(freq)-1,0,-1):
        for n in freq[i]:
            result.append(n)
            if len(result)==k:
                return result
for nums,k in test_case:
    print(f"The Top {k} of elements in {nums} are {topKFrequent_bucket(nums,k)}")
print("---------------------295. Find Median from Data Stream-------------------------")
print("***** Method One: Heap *****")
class MedianFinder:
    def __init__(self):
        # two heaps, large(atually works as min heap), small(works as max_heap)
        #Python using the min_heap
        self.small,self.large = [],[]
    def addNum(self,num):
        # add num to small heap, since it is the min heap,*-1 to put the max number to the top
        heapq.heappush(self.small,-1*num)
        # make sure the evey num small is <= every nums in large
        if(self.small and self.large and (-1*self.small[0])>self.large[0]):
            val = -1*heapq.heappop(self.small)
            heapq.heappush(self.large,val)
        # smake sure the size of two heaps are equal or 1 different
        if len(self.small)>len(self.large)+1:
            val=-1*heapq.heappop(self.small)
            heapq.heappush(self.large,val)
        if len(self.large)>len(self.small)+1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small,-1*val)
    def findMedian(self):
        if len(self.small)>len(self.large):
            return -1*self.small[0]
        if len(self.large)>len(self.small):
            return self.large[0]
        return (-1*self.small[0]+self.large[0])/2
med = MedianFinder()
nums = []
for item in [1,2]:
    med.addNum(item)
    nums.append(item)
print(f"The median of {nums} is {med.findMedian()}")
med.addNum(3)
nums.append(3)
print(f"The median of {nums} is {med.findMedian()}")
print("***** Method Two: in-order sort")
class MedianFinder_sorted:
    def __init__(self):
        self.nums = []
    def addNum(self,num):
        self.nums.append(num)
    def findMedian(self):
        self.nums.sort()
        if len(self.nums)%2==0:
            return (self.nums[len(self.nums)//2-1]+self.nums[(len(self.nums)//2)])/2
        else:
            return self.nums[len(self.nums)//2]
med = MedianFinder_sorted()
nums = []
for item in [1,2,3,-1,-2,-3,-4,-5]:
    med.addNum(item)
    nums.append(item)
    print(f"The median of {nums} (sorted {sorted(nums)}) is {med.findMedian()}")

print("---------------------242. Valid Anagram-------------------------")
print("***** Methond One: Using Counter")
def isAnagram(s,t):
    count_s = Counter(s)
    count_t = Counter(t)
    return count_t==count_s
test_case = [["anagram","nagaram"],["rat","car"]]
for s,t in test_case:
    print(f"{s} and {t} is anagram: {isAnagram(s,t)}")
print("***** Method Two: Sort the String *****")
def isAnagram_sort_string(s,t):
    sort_s = sorted(s)
    sort_t = sorted(t)
    return sort_t==sort_s
for s,t in test_case:
    print(f"{s} and {t} is anagram: {isAnagram_sort_string(s,t)}")
print("***** Method Three: Hashmap *****")
def isAnagram_hashmap(s,t):
    count_hash = {}
    for ch in s:
        count_hash[ch] = 1+count_hash.get(ch,0)
    for ch in t:
        if ch not in count_hash.keys():
            return False
        elif count_hash[ch]==1:
            count_hash.pop(ch)
        else:
            count_hash[ch] = count_hash[ch]-1
    if len(count_hash)==0:
        return True
    else:
        return False
for s,t in test_case:
    print(f"{s} and {t} is anagram: {isAnagram_hashmap(s,t)}")
print("---------------------371. Sum of Two Integers-------------------------")
def getSum(a,b):

    mask = 0xffffffff
    a = a & mask
    while b:
        sum = (a^b)&mask
        carry = ((a&b)<<1)&mask
        a = sum
        b= carry
    # if a is negative in 32 bits
    if (a>>31)&1:
        return ~(a^mask)
    return a
test_case = [[1,2],[2,3]]
for a,b in test_case:
    print(f"The sum of {a}+{b} is {getSum(a,b)}")
print("---------------------438. Find All Anagrams in a String-------------------------")
print("***** Methond One: Iteration and Sort *****")
def findAnagrams(s,p):
    sort_p =sorted(p)
    result = []
    for i in range(len(s)-len(p)+1):
        if sorted(s[i:i+len(p)])==sort_p:
            result.append(i)
    return result
test_case = [["cbaebabacd","abc"],["abab","ab"]]
for s,p in test_case:
    print(f"The start index of anagram {p} in {s} is {findAnagrams(s,p)}")
print("***** Method Two: Two Pointers *****")
def findAnagrams_two_pointer(s,p):
    if len(p)>len(s):
        return []
    pCount,sCount = {},{}
    for i in range(len(p)):
        pCount[p[i]] = 1+pCount.get(p[i],0)
        sCount[s[i]] = 1+sCount.get(s[i],0)
    res = [0] if sCount ==pCount else []
    l=0
    for r in range(len(p),len(s)):
        sCount[s[r]] = 1+sCount.get(s[r],0)
        sCount[s[l]]=sCount[s[l]]-1
        if sCount[s[l]]==0:
            sCount.pop(s[l])
        l = l+1
        if sCount==pCount:
            res.append(l)
    return res
for s,p in test_case:
    print(f"The start index of anagram {p} in {s} is {findAnagrams_two_pointer(s,p)}")
print("---------------------239. Sliding Window Maximum-------------------------")
print("***** Method One: Loop O(k*(n-k)) *****")
def maxSlidingWindow(nums,k):
    result = []
    for i in range(len(nums)-k+1):
        result.append(max(nums[i:i+k]))
    return result
test_case = [[1,3,-1,-3,5,3,6,7],3],[[1],1]
for nums,k in test_case:
    print(f"The Sliding Window Maximun is {nums} is {maxSlidingWindow(nums,k)}")
print("***** Method One: Dqueue and Monotonic Queue O(n) *****")
def maxSlidingWindow_monotonic(nums,k):
    result  = []
    q = deque()
    # sliding windows
    l=r=0
    while r<len(nums):
        # pop smaller values from q
        while q and nums[q[-1]]<nums[r]:
            q.pop()
        q.append(r)
        # remove left val from window
        if l>q[0]:
            q.popleft()
        # based on the size of k, add new value
        if (r+1)>=k:
            result.append(nums[q[0]])
            l = l+1
        r = r+1
    return result
for nums,k in test_case:
    print(f"The Sliding Window Maximun is {nums} is {maxSlidingWindow_monotonic(nums,k)}")

print("---------------------1838. Frequency of the Most Frequent Element-------------------------")
def maxFrequency(nums,k):
    # sort the list
    nums.sort()
    # using sliding windows
    l,r = 0,0
    # result store the final result, total calculates the sum of sliding windows
    result,total,final_result,updated_substring = 0,0,[],[]
    while r<len(nums):
        # calculate the sum of current sliding windows
        total = total+nums[r]
        # assume all the numbers in the sliding windows are same to get the max frequency
        # since the the array is sorted, the assumed sum should be nums[r]*length of sliding windows
        # if current sliding windows break the assumption, shrink the windows
        while nums[r]*(r-l+1)>total+k:
            # update the total
            total = total-nums[l]
            # left pointer move forward
            l = l + 1
        # now the current sliding windows meet the assumption, update the final result
        result = max(result,(r-l+1))
        final_result = nums[l:r + 1]
        updated_substring = nums[:l]+[nums[r] for _ in range(r - l + 1)]
        # move the right pointer to extent the sliding windows
        r = r+1
    # return final result
    return result,final_result,updated_substring
test_case= [[[1,2,4],5],[[1,4,8,13],5],[[3,9,6],2]]
for nums,k in test_case:
    print(f"The Maximum Frequency in {nums} with credit {k} is {maxFrequency(nums,k)[0]}, the sub_array is {maxFrequency(nums,k)[1]}"
          f" and updated substring is {maxFrequency(nums, k)[2]}" )
print("---------------------1888. Minimum Number of Flips to Make the Binary String Alternating-------------------------")
def minFlips(s):
    n = len(s)
    s = s+s
    alt1,alt2 = "",""
    for i in range(len(s)):
        alt1 += "0" if i%2 else "1"
        alt2 += "1" if i%2 else "0"
    res = len(s)
    diff1,diff2 =0,0
    l = 0
    for r in range(len(s)):
        if s[r]!=alt1[r]:
            diff1 = diff1+1
        if s[r]!=alt2[r]:
            diff2 = diff2+1
        if (r-l+1)>n:
            if s[l]!=alt1[l]:
                diff1 = diff1-1
            if s[l]!=alt2[l]:
                diff2 = diff2-1
            l = l+1
        if (r-l+1)==n:
            res= min(res,diff1,diff2)
    return res
test_case = ["111000","010","1110"]
for s in test_case:
    print(f"The Minimum flip in {s} is {minFlips(s)}")
print("---------------------1963. Minimum Number of Swaps to Make the String Balanced-------------------------")
'''
Using the Sliding window,with the CountClose variable. if character is [,CountClose minus one
if character is ], CountClose plus one, after each loop, using max() get the max of previous value 
and CountClose. the return value should be (max+1)//2. for example: ][[], CountClose = 2, 
just swap 1st and 2nd characters 
'''
def minSwaps(s):
   close,maxClose = 0,0
   for ch in s:
       if ch=="[":
           close = close-1
       else:
           close = close+1
       maxClose = max(close,maxClose)
   return (maxClose+1)//2
test_case = ["][][","]]][[[","[]","]][]"]
for s in test_case:
    print(f"Minimum Number of Swaps in {s} to Make the String Balanced is {minSwaps(s)}")
print("---------------------567. Permutation in String-------------------------")
print("***** Method One: Hashmap *****")
def checkInclusion(s1,s2):
    if len(s1)>len(s2):
        return False
    s1_hashmap,s2_hashmap = {},{}
    for i in range(len(s1)):
        s1_hashmap[s1[i]] = 1+s1_hashmap.get(s1[i],0)
        s2_hashmap[s2[i]] = 1+s2_hashmap.get(s2[i],0)
    l=0
    r = len(s1)
    # print(s1_hashmap,s2_hashmap)
    while r<len(s2) :
        if s1_hashmap==s2_hashmap:
            return True
        else:
            if s2_hashmap[s2[l]] ==1:
                s2_hashmap.pop(s2[l])
            else:
                s2_hashmap[s2[l]] = s2_hashmap[s2[l]]-1
            l=l+1
        s2_hashmap[s2[r]] = 1+s2_hashmap.get(s2[r],0)
        # print(s1_hashmap,s2_hashmap)
        r = r + 1
    return s1_hashmap==s2_hashmap
test_case = [["ab","eidbaooo"],["ab","eidboaoo"],["adc","dcda"]]
for s1,s2 in test_case:
    print(f"There is Permutation of {s1} in {s2}: {checkInclusion(s1,s2)}")
print("***** Method Two: Using 26 alphabet to calculate the Permutation *****")
'''
count the number of 26 alphabet characters. if the count of all the 26 characters are equal, return True
else sliding the windows, to continue check.
after sliding the right window, add the new character to the s2count, update the match variable. if the updated 
s2Count[i]==s1Count[i], match plus one, if s1Count[i]==s2Count[i] before the update, then match minus one.
end the loop, continue to check if match ==26 (means all the alphabet characters are equal)
'''
def checkInclusion_count26(s1,s2):
    if len(s1)>len(s2):
        return False
    s1Count,s2Count = [0]*26,[0]*26
    for i in range(len(s1)):
        s1Count[ord(s1[i])-ord("a")] = s1Count[ord(s1[i])-ord("a")]+1
        s2Count[ord(s2[i]) - ord("a")] = s2Count[ord(s2[i]) - ord("a")] + 1
    match =0
    for i in range(26):
        if s1Count[i]==s2Count[i]:
            match = match+1
    l = 0
    for r in range(len(s1),len(s2)):
        if match ==26:
            return True
        index = ord(s2[r])-ord("a")
        s2Count[index] = s2Count[index]+1
        if s1Count[index]==s2Count[index]:
            match = match+1
        elif s1Count[index]+1==s2Count[index]:
            match = match-1
        index = ord(s2[l])-ord("a")
        s2Count[index] = s2Count[index]-1
        if s1Count[index]==s2Count[index]:
            match = match+1
        elif s1Count[index]-1==s2Count[index]:
            match=match-1
        l = l+1
    return match==26

for s1,s2 in test_case:
    print(f"There is Permutation of {s1} in {s2}: {checkInclusion_count26(s1,s2)}")
print("---------------------209. Minimum Size Subarray Sum-------------------------")
'''
Using Sliding windows,if the total sum is greater than the target,
shrink the window, left pointer plus one and to check if the new window's total sum still
greater or equal to the target.
after each loop, update the right pointer and the total sum  
'''
def minSubArrayLen(target,nums):
    total,minSize = 0,float("inf")
    l= 0
    for r in range(len(nums)):
        total = total+nums[r]
        while total>=target:
            minSize = min(minSize,(r-l+1))
            total = total-nums[l]
            l = l+1
    return 0 if minSize ==float("inf") else minSize
test_case = [7, [2,3,1,2,4,3]],[4,[1,4,4]],[11,[1,1,1,1,1,1,1,1]]
for target,nums in test_case:
    print(f"The Minimum size of {nums} to get {target} is {minSubArrayLen(target,nums)}")
print("---------------------187. Repeated DNA Sequences-------------------------")
'''
Using Sliding windows and set(). Scan 10 characters each time, if it is in the visited set, add to the result set.
for each iteration, add the scanned character sequences to the visited set. Since it is the set, there will not be 
the duplicated sequency in it. formate the result to the list with list() function 
'''
def findRepeatedDnaSequences(s):
    seen,repeat = set(),set()
    for i in range(len(s)-9):
        cur = s[i:i+10]
        if cur in seen:
            repeat.add(cur)
        seen.add(cur)
    return list(repeat)
test_case = ["AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT","AAAAAAAAAAAAA"]
for s in test_case:
    print(f"The repeat DNA sequences in {s} is {findRepeatedDnaSequences(s)}")
print("---------------------131. Palindrome Partitioning-------------------------")
'''
Using Backtracking Algorithm. scan the characters one by one. if it meet the requirement, append to temp list and 
continue to scan next character
'''
def partition(s):
    result = []
    part=[]
    def isPali(s,l,r):
        while l<r:
            if s[l]!=s[r]:
                return False
            l,r = l+1,r-1
        return True
    def dfs(i):
        if i>=len(s):
            result.append(part.copy())
            return
        for j in range(i,len(s)):
            if isPali(s,i,j):
                part.append(s[i:j+1])
                dfs(j+1)
                part.pop()
    dfs(0)
    return result
test_case = ["aab","a"]
for s in test_case:
    print(f"The Palindrome Partitioning in \'{s}\'is {partition(s)}")
print("---------------------1332. Remove Palindromic Subsequences-------------------------")
def removePalindromeSub(s):
    # if s is None, means the string already empty, and return 0
    if not s:
        return 0
    # s is equals to s[::-1] (revers of the s) means the s is palindromic, then delete it and return 1
    elif s==s[::-1]:
        return 1
    # if s is not empty and not palindromic, since there are only two unique character is the s
    # the after delete first not palindromic items, then rest will be palindromic substring
    else:
        return 2
test_case = ["ababa","abb","baabb"]
for s in test_case:
    print(f"The minimum step to remove palindromic substring in '{s}' is {removePalindromeSub(s)} ")