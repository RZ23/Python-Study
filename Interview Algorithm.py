import itertools
import math
import random
import collections
from collections import deque
from binarytree import tree
from collections import Counter
from binarytree import tree,Node
print("------------------Merge Sort-----------------------------")
lst = [random.randint(1,20) for x in range(10)]
print("Before Sorting")
print(lst)
def Merge_Sort(lst):
    if len(lst)<=1:
        return lst
    mid = len(lst)//2
    left_list = lst[:mid]
    right_list = lst[mid:]
    left = Merge_Sort(left_list)
    right = Merge_Sort(right_list)
    sorted_lst = merge(left,right)
    return sorted_lst
def merge(left,right):
    sorted_lst = []
    i=0
    j=0
    while i<len(left) and j<len(right):
        if left[i]<right[j]:
           sorted_lst.append(left[i])
           i=i+1
        else:
            sorted_lst.append(right[j])
            j=j+1
        left_left = left[i:]
        right_left = right[j:]
    return sorted_lst+left_left+right_left
print("After Sorting")
print(Merge_Sort(lst))
print("------------------Quick Sort-----------------------------")
lst1 = [random.randint(1,20) for x in range (10)]
print("Before Sorting")
print(lst1)
def Quick_Sort(lst):
    if len(lst)<=1:
        return lst
    pivot = lst[-1]
    left = []
    right = []
    for i in range(len(lst)-1):
        if lst[i]<=pivot:
            left.append(lst[i])
        else:
            right.append(lst[i])
    return Quick_Sort(left)+[pivot]+Quick_Sort(right)
print("After Sorting")
print(Quick_Sort(lst1))
print("------------------Valid Anagram-----------------------------")
print("------------------'Danger' and 'Garden' => True-------------")
def valid_anagrams(s1,s2):
    se1_dict={}
    se2_dict={}
    for item in s1:
        if item not in se1_dict:
            se1_dict[item]=1
        else:
            se1_dict[item] = se1_dict[item]+1
    for cha in s2:
        if cha not in se2_dict:
            se2_dict[cha]=1
        else:
            se2_dict[cha] = se2_dict[cha]+1
    # print(se1_dict,se2_dict)
    for item in se1_dict.keys():
        if item not in se2_dict:
            return False
        elif se1_dict[item]!=se2_dict[item]:
            return False
    return True
s1= "nameless"
s2 = "salesmen"
s3="Hello"
s4="Wolle"
print("Using Directionary Map")
print("{} and {} is anagram: {}".format(s1,s2,valid_anagrams(s1,s2)))
print("{} and {} is anagram: {}".format(s3,s4,valid_anagrams(s3,s4)))
def valid_anagram_sorted(s1,s2):
    sorted_s1 = sorted(s1)
    sorted_s2 = sorted(s2)
    if len(sorted_s1)!=len(sorted_s2):
        return False
    return sorted_s1 == sorted_s2
print("Using the Sorting")
print("{} and {} is anagram: {}".format(s1,s2,valid_anagram_sorted(s1,s2)))
print("{} and {} is anagram: {}".format(s3,s4,valid_anagram_sorted(s3,s4)))

def valid_anaram_count(s1,s2):
    print("Count of {} is {}, and Counter of {} is {}".format(s1,Counter(s1),s2,Counter(s2)) )
    size_1 = len(s1)
    size_2= len(s2)
    if size_1!=size_2:
        return False
    return Counter(s1)==Counter(s2)
print("Using the Collection Counter")
print("{} and {} is anagram: {}".format(s1,s2,valid_anaram_count(s1,s2)))
print("{} and {} is anagram: {}".format(s3,s4,valid_anaram_count(s3,s4)))
print("------------------First and Last Position-----------------------------")
def First_Last_Position(sequence,target):
    start=0
    for i in range(len(sequence)):
        if sequence[i]==target:
            start=i
            while sequence[i+1]==target and i<len(sequence):
                i = i+1
            return [start,i]
    return [-1,-1]
s =[2,4,5,5,5,5,5,7,9,9]
target = 5
print("Linear Search:")
print("The sequence is {} and the target is {}".format(s,target))
print("The First and End Position of the target {} is {}".format(target,First_Last_Position(s,target)))
print("Binary Search")
def First_and_End_Biary(s,target):
    if (len(s)==0) or (s[0]>target) or (s[len(s)-1]<target):
        return (-1,-1)
    return find_start(s,target),find_end(s,target)
def find_start(s,target):
    if s[0]==target:
        return 0
    low,high =0,len(s)-1
    while low<=high:
        mid = (low+high)//2
        if s[mid]==target and s[mid-1]<target:
            return mid
        elif s[mid]<target:
            low = mid+1
        else:
            high = mid-1
    return -1
def find_end(s,target):
    if s[0]==target:
        return 0
    low,high = 0,len(s)-1
    while low<=high:
        mid = (low+high)//2
        if s[mid]==target and s[mid+1]>target:
            return mid
        elif s[mid]>target:
            high = mid-1
        else:
            low = mid+1
    return -1

print("Linear Search:")
print("The sequence is {} and the target is {}".format(s,target))
print("The First and End Position of the target {} is {}".format(target,First_and_End_Biary(s,target)))
print("------------------Kth Largest Element-----------------------------")
def Kth_Element(s,k):
    for i in range(k-1):
        s.remove(max(s))
    return max(s)
s=[4,2,9,7,5,6,7,1,3]
s1 = s.copy()
k=4
print("Without Sorting")
print("the list is {} and the {}th larget element is {}".format(s1,k,Kth_Element(s,k)))
print("With Sorting")
def Kth_Element_Sorting(s,k):
    return sorted(s)[-k]
print("the list is {} and the {}th larget element is {}".format(s1,k,Kth_Element_Sorting(s1,k)))

print("------------------Symmetric Tree-----------------------------")
# Symmetric Tree
def Symmetric_Tree(root1,root2):
    if root1 is None and root2 is None:
        return True
    elif ((root1 is None) != (root2 is None)) or root1.value!=root2.value:
        return False
    return Symmetric_Tree(root1.left,root2.right) and Symmetric_Tree(root2.left,root1.right)
def Check_Symmetric_Tree(node):
    if node is None:
        return True
    return Symmetric_Tree(node.left,node.right)
# Customized the Node class set to be the subclass of the binarytree's Node class
class Node(Node):
    def __init__(self,value=None,left= None,right=None):
        self.value = value
        self.right = right
        self.left=left

node1 = Node(3)
node2 = Node(0)
node3 = Node(6)
node4 = Node(9,node1,node2)
node5 = Node(7,None,node3)
node6 = Node(2,node4,node5)
node7 = Node(1)
node8 = Node(8,node7,None)
node9 = Node(5,node6,node8)
node10 = Node(1)
node11 = Node(8,None,node10)
node12 = Node(6)
node13 = Node(0)
node14 = Node(3)
node15 = Node(7,node12,None)
node16 = Node(9,node13,node14)
node17 = Node(2,node15,node16)
node18 =Node(5,node11,node17)
Symmetric_Head = Node(10,node9,node18)
print(Symmetric_Head)
# Node_List = [Symmetric_Head,node1,node2,node3,node4,node5,node6,node7,node8,node9,
#              node10,node11,node12,node13,node14,node15,node16,node17, node18,node18]
#
# for item in Node_List:
#     if item.left is None and item.right is None:
#         print("Node {}'s Left Node is {} and the right Node is {}".format(item.value, None,None))
#     elif item.right is None:
#         print("Node {}'s Left Node is {} and the right Node is {}".format(item.value, item.left.value, None))
#     elif item.left is None:
#         print("Node {}'s Left Node is {} and the right Node is {}".format(item.value, None, item.right.value))
#     else:
#         print("Node {}'s Left Node is {} and the right Node is {}".format(item.value, item.left.value,item.right.value) )
print("The Tree is the symmetric Tree? {}".format(Check_Symmetric_Tree(Symmetric_Head)))
Un_node1 = Node(3)
Un_node2 = Node(0)
Un_node3 = Node(6)
Un_node4 = Node(9,Un_node1,Un_node2)
Un_node5 = Node(7,None,Un_node3)
Un_node6 = Node(2,Un_node4,Un_node5)
Un_node7 = Node(1)
Un_node8 = Node(8,Un_node7,None)
# Change left and right child to creat teh un-symmetric Tree
Un_node9 = Node(5,Un_node8,Un_node6)
Un_node10 = Node(1)
Un_node11 = Node(8,None,Un_node10)
Un_node12 = Node(6)
Un_node13 = Node(0)
Un_node14 = Node(3)
Un_node15 = Node(7,Un_node12,None)
Un_node16 = Node(9,Un_node13,Un_node14)
Un_node17 = Node(2,Un_node15,Un_node16)
Un_node18 =Node(5,Un_node11,Un_node17)
Un_Symmetric_Head = Node(10,Un_node9,Un_node18)
print(Un_Symmetric_Head)
print("The Tree is the symmetric Tree? {}".format(Check_Symmetric_Tree(Un_Symmetric_Head)))

#Gas Station Travel Question
print("------------------Gas Station Travel-----------------------------")
gas = [1,5,3,3,5,3,1,3,4,5]
cost = [5,2,2,8,2,4,2,5,1,2]
def can_travel(gas,cost,start):
    n = len(gas)
    remaining = 0
    i = start
    started = False
    while i!=start or not started:
        # Start the loop and stoped when back to start point
        started = True
        remaining = remaining + gas[i]-cost[i]
        if remaining<0:
            return False
        # need to conside the cycle, so that why the i should be (i=1)%n
        i = (i+1)%n
    return True
def gas_station_travel(gas,cost):
    for i in range(len(gas)):
        if can_travel(gas,cost,i):
            return i
    return -1
print("The Start Gas Station should be {}".format(gas_station_travel(gas,cost)))

def gas_station(gas,cost):
    remaining =0
    candidate = 0
    for i in range(len(gas)):
        remaining = remaining+gas[i]-cost[i]
        if remaining<0:
            candidate = i+1
            remaining = 0
    prev_remaining = sum(gas[:candidate])-sum(cost[:candidate])
    # No candicate since the cadicate is the end of the list
    if candidate == len(gas):
        return -1
    # Cannot travel the second part of the cycle
    elif candidate<len(gas) and remaining+prev_remaining<0:
        return -1
    else:
        return candidate
print("The Start Gas Station should be {}".format(gas_station_travel(gas,cost)))

# Course Schedule
print("------------------Course Schedule -----------------------------")
n = 6
prerequisties = [[3,0],[1,3],[2,1],[4,1],[4,2],[5,3],[5,4]]
print("0->3->1->2")
print("   |   |  ")
print("   5<- 4  ")
print("Deep First Search:")
def dfs(graph,node,path,order,visted):
    path.add(node)
    for neighbor in graph[node]:
        if neighbor in path:
            return False
        if neighbor not in visted:
            visted.add(neighbor)
            if not dfs(graph,neighbor,path,order,visted):
                return False
    path .remove(node)
    order.append(node)
    return True
def course_schedule(n,prerequisties):
    graph = [[] for i in range(n)]
    for pre in prerequisties:
        graph[pre[1]].append(pre[0])
    visted = set()
    path = set()
    order = []
    for course in range(n):
        if course not in visted:
            visted.add(course)
            if not dfs(graph,course,path,order,visted):
                return False
    return True
print("The {} Couses with Schedule {} is availiable? {} ".format(n,prerequisties,course_schedule(n,prerequisties)))
print("Breadth Search First:")
def course_schedule_bsf(n,prerequisties):
    graph = [[] for i in range(n)]
    indegree = [0 for i in range(n)]
    for pre in prerequisties:
        graph[pre[1]].append(pre[0])
        indegree[pre[0]] = indegree[pre[0]]+1
    order = []
    queue = deque([i for i in range(n) if indegree[i]==0])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] = indegree[neighbor]-1
            if indegree[neighbor]==0:
                queue.append(neighbor)
    return len(order)==n
print("The {} Couses with Schedule {} is availiable? {} ".format(n,prerequisties,course_schedule_bsf(n,prerequisties)))

class Course():
    def __init__(self,num_course,prequisties):
        self.num_course = num_course
        self. prequisties = {}
        for pre in prerequisties:
            if pre[1] in self.prequisties.keys():
                self.prequisties[pre[1]].append(pre[0])
            else:
                self.prequisties[pre[1]] = list(pre[0])
    def display(self):
        course_list = list(self.prequisties.keys())
        for course in course_list:
            print("The Course After {} is {}".format(str(course).upper(),str(self.prequisties[course]).upper()))
n = 6
prerequisties = [["d","a"],["b","d"],["c","b"],["e","b"],["e","c"],["f","d"],["f","e"]]
print("a->d->b->c")
print("   |   |  ")
print("   f<- e  ")
course = Course(n,prerequisties)
course.display()
# Kth Permutation
print("------------------Kth Permutation -----------------------------")
print("Regular Method: (build-in permutation function )")
def kth_permutation_build_in(n,k):
    permutations = list(itertools.permutations(range(1,n+1)))
    # for i in range (len(permutations)):
    #     print("{} is {}".format(i,permutations[i]))
    return "".join(map(str,permutations[k-1]))
print("the {}th of {} numbers permutation is {}".format(k,n,kth_permutation_build_in(4,16)))
print("Customized Function:")
def kth_permutation(n,k):
    permutation = []
    unused = list(range(1,n+1))
    print(unused)
    fact = [1]*(n+1)
    for i in range(1,n+1):
        fact[i] = i*fact[i-1]
    # the start is 0
    k = k-1
    while n>0:
        part_length = fact[n]//n
        i= k//part_length
        permutation.append(unused[i])
        unused.pop(i)
        print("part_length = {},i={},permuatation ={},unused = {}".format(part_length,i,permutation,unused))
        n = n-1
        k = k%part_length
    return "".join(map(str,permutation))
print("the {}th of {} numbers permutation is {}".format(k,n,kth_permutation(4,16)))

print("------------------Minimal Window Substring -----------------------------")
print("Regular Function")
def mini_window_regular(s,t):
    n,m = len(s),len(t)
    if n<m or m==0:
        return ""
    freqt =Counter(t)
    # create initial shortest substring
    shortest = " "*(n-1)
    for length in range(1,n+1):
        for i in range(n-length+1):
            sub= s[i:i+length]
            freqs = Counter(sub)
            if contain_all(freqs,freqt) and length <len(shortest):
                shortest= sub
    return shortest if len(shortest)<=n else " "
def contain_all(freq1,freq2):
    for ch in freq2:
        if freq1[ch]<freq2[ch]:
            return False
    return True
s= "ADCFEBECEABEBADFCDFCBFCBEAD"
t= "ABCA"
print("The shortest substring is of {} based on the {} is {}".format(t,s,mini_window_regular(s,t)))
print("Advanced Function (without extracting and storing substring, store the start and end index --sliding window )")
def min_window_slid_window(s,t):
    n,m = len(s),len(t)
    if n<m or t=="":
        return ""
    freqt = Counter(t)
    start,end = 0,n+1
    for length in range(1,n+1):
        freqs=Counter()
        satisfied  =0
        for ch in s[:length]:
            freqs[ch] = freqs[ch]+1
            if ch in freqt and freqt[ch]==freqs[ch]:
                satisfied = satisfied+1
        # find the first matched substring
        if satisfied==len(freqt) and length < end-start:
        # store the start and end index and then move to next character
            start,end = 0,length
        # move to next character one by one
        for i in range(1,n-length+1):
            # for the new move in item
            freqs[s[i+length-1]]= freqs[s[i+length-1]]+1
            if s[i+length-1] in freqt and freqs[s[i+length-1]]==freqt[s[i+length-1]]:
                satisfied = satisfied+1
            # for the new move out item
            if s[i-1] in freqt and freqs[s[i-1]]==freqt[s[i-1]]:
                satisfied = satisfied-1
            freqs[s[i-1]] = freqs[s[i-1]]-1
            if satisfied==len(freqt) and length<end-start:
                start,end = i,i+length
    return s[start:end] if end-start<=n else ""
print("The shortest substring is of {} based on the {} is {} with Sliding Windows".format(t,s,min_window_slid_window(s,t)))
print("Imporved Slinding Window")
def min_window(s,t):
    n,m = len(s),len(t)
    if n<m or m=="":
        return ""
    freqt = Counter(t)
    start,end = 0,n
    satisfied = 0
    freqs = Counter()
    left = 0
    for right in range(n):
        freqs[s[right]]=freqs[s[right]]+1
        if s[right] in freqt and freqs[s[right]]==freqt[s[right]]:
            satisfied=satisfied+1
        # first the first satisfied substring
        if satisfied==len(freqt):
            while s[left] not in freqt or freqs[s[left]]>freqt[s[left]]:
                freqs[s[left]] = freqs[s[left]]-1
                left = left +1
            if right-left+1<end-start+1:
                start,end = left,right
    # the index starts from 0, so must add 1 to make sure it is the substring
    return s[start:end] if end-start+1<=n else""
print("The shortest substring is of {} based on the {} is {} with Improvving Sliding Windows".format(t, s,min_window(s, t)))

print("------------------Largest Rectangle Area -----------------------------")
heights = [3,2,4,5,7,6,1,3,8,9,11,10,7,5,2,6]
def largest_rectangle(heights):
    max_area=0
    for i in range(len(heights)):
        # find the left
        left = i
        while left-1>0 and heights[left-1]>=heights[i]:
            left = left-1
        # find the right
        right = i
        while right+1<len(heights) and heights[right+1]>=heights[i]:
            right = right+1
        max_area = max(max_area,heights[i]*(right-left+1))
    return max_area
print("The Max Rectangle of {} is {}".format(heights,largest_rectangle(heights)))
print("Recursive Method:")
def rec(heights,low,high):
    if low>high:
        return 0
    elif low==high:
        return heights[low]
    else:
        minh = min(heights[low:high+1])
        pos_min = heights.index(minh,low,high+1)
        from_left = rec(heights,low,pos_min-1)
        from_right = rec(heights,pos_min+1,high)
        return max(from_left,from_right,minh*(high-low+1))
def largest_rectangle_recurssive(heights):
    return rec(heights,0,len(heights)-1)
print("The Max Rectangle of {} is {}".format(heights,largest_rectangle_recurssive(heights)))
print("Left and Right stack Method:")
def largest_rectangle_stack(heights):
    # add left and right point
    heights=[-1]+heights+[-1]
    from_left = [0]*len(heights)
    stack=[0]
    for i in range(1,len(heights)-1):
        while heights[stack[-1]]>=heights[i]:
            stack.pop()
        from_left[i] = stack[-1]
        stack.append(i)
    from_right = [0]*len(heights)
    stack = [len(heights)-1]
    for i in range(1,len(heights)-1)[::-1]:
        while heights[stack[-1]]>=heights[i]:
            stack.pop()
        from_right[i] = stack[-1]
        stack.append(i)
    max_area = 0
    for i in range(1,len(heights)-1):
        max_area = max(max_area,heights[i]*(from_right[i]-from_left[i]-1))
    return max_area
print("The Max Rectangle of {} is {}".format(heights,largest_rectangle_stack(heights)))

print("With Tuple Stack Method:")
def largest_rectangle_Tuple_Stack(heights):
    heights = [-1]+heights+[-1]
    max_area= 0
    # initial the start stack, (index, height)
    stack = [(0,-1)]
    for i in range(1,len(heights)):
        start=i
        while stack[-1][1]>heights[i]:
            top_index,top_height = stack.pop()
            max_area = max(max_area,top_height*(i-top_index))
            start=top_index
        stack.append((start,heights[i]))
    return max_area
print("The Max Rectangle of {} is {}".format(heights,largest_rectangle_Tuple_Stack(heights)))