from binarytree import Node,tree
from binarytree import Node, tree
from collections import deque
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
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

print("---------------------Maximum Depth of Binary Tree-------------------------")
def max_depth(node):
    if node is None:
        return 0
    else:
        return max(max_depth(node.left),max_depth(node.right))+1
print("The depth of the tree is {}".format(max_depth(node_list[0])))

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

print("---------------------Longest Palindromic Substring-------------------------")

string_list = ['babad',"cbbd"]
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
for item in string_list:
    print("The Longest Palindromic Substring of {} is {}".format(item,longestPalindrome(item)))

print("---------------------Valid Parentheses-------------------------")

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

