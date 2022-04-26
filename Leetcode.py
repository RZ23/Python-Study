from binarytree import Node,tree
from binarytree import Node, tree
from collections import deque
from collections import Counter

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
    linked_lst=generate_linked_list(linked_list[0])
    print(print_linked_list_with_return(removeNthFromEnd(linked_lst,linked_list[1])))

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