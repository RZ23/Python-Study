from collections import deque
from binarytree import tree,Node
import binarytree
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
print("---------------------20. Valid Parentheses-------------------------")
def isValid(s):
    stack = []
    closeToken = {")":"(","]":"[","}":"{"}
    for c in s:
        if c in closeToken:
            if stack and stack[-1]==closeToken[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
    if len(stack)==0:
        return True
    else:
        return False
test_case = ["()","()[]{}","(]"]
for i,s in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"The Parentheses sequence '{s}' is valid: {isValid(s)} ")
print("---------------------71. Simplify Path-------------------------")
def simplifyPath(path):
    stack = []
    cur = ""
    for c in path+"/":
        if c == "/":
            if cur == "..":
                if len(stack)>0:
                    stack.pop()
            elif cur != "" and cur != ".":
                stack.append(cur)
            cur = ""
        else:
            cur = cur+c
    return "/"+"/".join(stack)
test_case = ["/home/","/../","/home//foo/"]
for i, path in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"The Canonical path of Absolute path {path} is {simplifyPath(path)}")
print("---------------------84. Largest Rectangle in Histogram-------------------------")
'''
Time Complexity 0(n)
Space Complexity 0(n)
'''
def largestRectangleArea(heights):
   maxArea = 0
   stack = [] # (index,height)
   for i, h in enumerate(heights):
       start = i
       while stack and stack[-1][1]>h:
           index,height = stack.pop()
           maxArea = max(maxArea,height*(i-index))
           start = index
       stack.append((start,h))
   for i,h in stack:
       maxArea = max(maxArea,h*(len(heights)-i))
   return maxArea
test_case = [[2,1,5,6,2,3],[2,4]]
for i,heights in enumerate(test_case):
    print(f"Test Case {i+1}:Heights List is {heights}")
    print(f"The area of the largest rectangle in the histogram is {largestRectangleArea(heights)}")
print("---------------------22. Generate Parentheses-------------------------")
def generateParenthesis(n):
    # only add open parenthesis if open <n
    # only add a closing parenthesis if close < open
    # valid iif open ==close ==n
    stack = []
    result = []
    def backtrack(openN,closedN):
        if openN==closedN==n:
            result.append("".join(stack))
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
    return result
test_case = [3,1]
for i,n in enumerate(test_case):
    print(f"For {n} pair(s) of Parentheses, it could generate {len(generateParenthesis(n))} combinations of well-formed Parentheses"
          f"the list is {generateParenthesis(n)}")
print("---------------------155. Min Stack-------------------------")
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []
    def push(self,val):
        self.stack.append(val)
        val = min(val,self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)
    def pop(self):
        self.stack.pop()
        self.minStack.pop()
    def top(self):
        return self.stack[-1]
    def getMin(self):
        return self.minStack[-1]
    def get_all_elements(self):
        All_Element = []
        for i in range(len(self.stack)):
            All_Element.append(self.stack[i])
        return All_Element
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(f"All the elements in the stack are {minStack.get_all_elements()}")
print(f"the min stack is {minStack.getMin()}")
print(f"the pop item is {minStack.pop()}")
print(f"the top item is {minStack.top()}")
print(f"the min stack is {minStack.getMin()}")
print("---------------------739. Daily Temperatures-------------------------")
print("***** Method One: Array *****")
def dailyTemperatures_array_list(temperatures):
    result = []
    for i in range(len(temperatures)-1):
        updated = False
        for j in range(i+1,len(temperatures)):
            if temperatures[j]>temperatures[i]:
                result.append(j-i)
                updated = True
                break
        if not updated:
            result.append(0)
    result.append(0)
    return result
test_case = [[73,74,75,71,69,72,76,73],[30,40,50,60],[30,60,90]]
for i, temperatures in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"Based on the temperatures list {temperatures},"
          f"the days from ith day to get warner day is {dailyTemperatures_array_list(temperatures)}")
print("***** Method One: Stack *****")
def dailyTemperatures(temperatures):
    result = [0]*len(temperatures)
    stack = [] # pair: [temp,index]
    for i, t in enumerate(temperatures):
        while stack and t>stack[-1][0]:
            stackT,stackInd = stack.pop()
            result[stackInd] = (i-stackInd)
        stack.append([t,i])
    return result
for i, temperatures in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"Based on the temperatures list {temperatures},"
          f"the days from ith day to get warner day is {dailyTemperatures(temperatures)}")
print("---------------------735. Asteroid Collision-------------------------")
print("***** Method One: By self *****")
def asteroidCollision(asteroids):
    stack = [asteroids[0]]
    for i in range(1,len(asteroids)):
        if len(stack)==0 or asteroids[i]*stack[-1]>0:
            stack.append(asteroids[i])
        if asteroids[i]>0 and stack[-1]<0:
            stack.append(asteroids[i])
        elif asteroids[i]*stack[-1]<0:
            stack.append(asteroids[i])
            while len(stack)>=2 and stack[-1]*stack[-2]<0 and (stack[-1]<0 and stack[-2]>0):
                last_asteroid_1 = stack.pop()
                last_asteroid_2 = stack.pop()
                if abs(last_asteroid_1)>abs(last_asteroid_2):
                    stack.append(last_asteroid_1)
                elif abs(last_asteroid_1)<abs(last_asteroid_2):
                    stack.append(last_asteroid_2)
    return stack
test_case = [[5,10,-5],[8,-8],[10,2,-5],[-2,-1,1,2]]
for i,asteroids in enumerate(test_case):
    print(f"Test Case: {i+1}:")
    print(f"Based on the asteroids list {asteroids}, the final result is {asteroidCollision(asteroids)}")
print("***** Method One: By NeetCode *****")
def asteroidCollision(asteroids):
    stack = []
    for a in asteroids:
        while stack and a<0 and stack[-1]>0:
            diff = a+stack[-1]
            if diff<0:
                stack.pop()
            elif diff >0:
                a=0
            else:
                a = 0
                stack.pop()
        if a:
            stack.append(a)
    return stack
for i,asteroids in enumerate(test_case):
    print(f"Test Case: {i+1}:")
    print(f"Based on the asteroids list {asteroids}, the final result is {asteroidCollision(asteroids)}")
print("---------------------1856. Maximum Subarray Min-Product-------------------------")
def maxSumMinProduct(nums):
    result = 0
    stack = []
    pre_sum = [0]
    for n in nums:
        pre_sum.append(pre_sum[-1]+n)
    for i,n in enumerate(nums):
        newStart = i
        while stack and stack[-1][1]>n:
            start,val = stack.pop()
            total = pre_sum[i]-pre_sum[start]
            result = max(result,total*val)
            newStart = start
        stack.append((newStart,n))
    for start,val in stack:
        total = pre_sum[len(nums)]-pre_sum[start]
        result = max(result,total*val)
    return result
test_case = [[1,2,3,2],[2,3,3,1,2],[3,1,5,6,4,2]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"For the nums list {nums}, The maximum min-product is {maxSumMinProduct(nums)}")
print("---------------------853. Car Fleet-------------------------")
def carFleet(target,position,speed):
    car_position = [[p,s] for p,s in zip(position,speed)]
    stack = []
    for p,s in sorted(car_position)[::-1]:
        stack.append((target-p)/s)
        if len(stack)>1 and stack[-1]<=stack[-2]:
            stack.pop()
    return len(stack)
test_case = [[12,[10,8,0,5,3],[2,4,1,1,3]],[10, [3], [3]],[100, [0,2,4],[4,2,1]]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"Based on the speed list {test[1]} and position list {test[2]}, to the target {test[0]},"
          f"there will be {carFleet(test[0],test[1],test[2])} car(s) fleets that will arrive at the destination")
print("---------------------225. Implement Stack using Queues-------------------------")
class MyStack:
    def __init__(self):
        self.q = deque()
    def push(self, x):
        self.q.append(x)
    def pop(self):
        for i in range(len(self.q)-1):
            self.q.append(self.q.popleft())
        return self.q.popleft()
    def top(self):
        return self.q[-1]
    def empty(self):
        return len(self.q)==0
    def display_myStack(self):
        if len(self.q)==0:
            print("The myStack is empty")
        else:
            print("Element(s) in the myStack:")
            for i in range(len(self.q)):
                print(self.q[i],end = ",")
            print()
myStack = MyStack()
myStack.push(1)
myStack.push(2)
myStack.push(3)
myStack.push(6)
myStack.display_myStack()
print(f"Top element of myStack: {myStack.top()}")
myStack.push(9)
myStack.push(12)
myStack.display_myStack()
print(f"pop from the myStack: {myStack.pop()}")
myStack.display_myStack()
print(f"The myStack is empty: {myStack.empty()}")
while not myStack.empty():
    myStack.pop()
myStack.display_myStack()
print(f"The myStack is empty: {myStack.empty()}")
print("---------------------901. Online Stock Span-------------------------")
print("***** Method One: Using Array")
class StockSpanner:
    def __init__(self):
        self.lst = []
    def next(self, price):
        self.lst.append(price)
        span = 1
        for i in range(len(self.lst)-2,-1,-1):
            if price>=self.lst[i]:
                span = span+1
            else:
                break
        return span
stockSpanner = StockSpanner()
span_list = [100,80,60,70,60,75,85]
for i,price in enumerate(span_list):
    print(f"For the {i+1} day, the span is {stockSpanner.next(price)}")

print("***** Method One: Using Stack")
class StockSpanner:
    def __init__(self):
        self.stack = []
    def next(self, price):
        span = 1
        while self.stack and self.stack[-1][0]<=price:
            span = self.stack[-1][1]+span
            self.stack.pop()
        self.stack.append((price,span))
        return span
stockSpanner = StockSpanner()
span_list = [100,80,60,70,60,75,85]
for i,price in enumerate(span_list):
    print(f"For the {i+1} day, the span is {stockSpanner.next(price)}")
print("---------------------150. Evaluate Reverse Polish Notation-------------------------")
def evalRPN(tokens):
    stack = []
    for t in tokens:
        if t =="+":
            stack.append(stack.pop()+stack.pop())
        elif t=="-":
            a,b = stack.pop(),stack.pop()
            stack.append(b-a)
        elif t=="*":
            stack.append(stack.pop()*stack.pop())
        elif t== "/":
            a,b = stack.pop(),stack.pop()
            stack.append(int(b/a))
        else:
            stack.append(int(t))
    return stack[0]
test_case = [["2","1","+","3","*"],\
            ["4","13","5","/","+"],\
            ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]]
for i, tokens in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"Based on the Token {tokens}, the result is {evalRPN(tokens)}")
print("---------------------402. Remove K Digits-------------------------")
def removeKdigits(num,k):
    if len(num)==k:
        return "0"
    stack = []
    for c in num:
        while k>0 and stack and stack[-1]>c:
            k = k-1
            stack.pop()
        stack.append(c)
    stack = stack[:len(stack)-k]
    result = ""
    for i in range(len(stack)):
        result = stack.pop()+result
    return str(int(result))
test_case = [["1432219",3],["10200",1],["10",2],["1234567890",9]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"For the nums {test[0]},remove {test[1]} digits, the smallest possible integer is {removeKdigits(test[0],test[1])}")
print("---------------------895. Maximum Frequency Stack-------------------------")
class FreqStack:
    def __init__(self):
        self.group_stack = {}
        self.maxCnt = 0
        self.cnt = {}
        self.display_list = []
    def push(self, val):
        valCnt = 1+self.cnt.get(val,0)
        self.cnt[val] = valCnt
        if valCnt>self.maxCnt:
            self.maxCnt = valCnt
            self.group_stack[valCnt] = []
        self.group_stack[valCnt].append(val)
        self.display_list.append(val)
        print(f"After push the {val}, The updated list is {self.display_list}")
    def pop(self):
        if self.maxCnt==0:
            print("The Stack is Empty, cannot run  the pop function")
            return
        result = self.group_stack[self.maxCnt].pop()
        self.cnt[result] = self.cnt[result]-1
        if not self.group_stack[self.maxCnt]:
            self.maxCnt = self.maxCnt-1
        display_list_cp = self.display_list.copy()
        display_list_cp.reverse()
        remove_index = len(self.display_list)-display_list_cp.index(result)-1
        updated_list = self.display_list[:remove_index]+self.display_list[remove_index+1:]
        self.display_list = updated_list
        print(f"After remove the most frequency item {result} from the stack, The updated list is {self.display_list}")
        return result
freqStack = FreqStack()
freqStack.push(5)
freqStack.push(7)
freqStack.push(5)
freqStack.push(7)
freqStack.push(4)
freqStack.push(5)
# print(f"pop the most frequency from the stack is {freqStack.pop()}")
# print(f"pop the most frequency from the stack is {freqStack.pop()}")
# print(f"pop the most frequency from the stack is {freqStack.pop()}")
# print(f"pop the most frequency from the stack is {freqStack.pop()}")
freqStack.pop()
freqStack.pop()
freqStack.pop()
freqStack.pop()
freqStack.pop()
freqStack.pop()
freqStack.pop()
print("---------------------496. Next Greater Element I-------------------------")
print("***** Method One: Multiple Loop 0(m*n) *****")
def nextGreaterElement(nums1, nums2):
    result = []
    updated = False
    for num in nums1:
        if num not in nums2:
            result.append(-1)
        else:
            updated = False
            num2_index = nums2.index(num)
            j = num2_index+1
            while j<len(nums2) and not updated:
                if nums2[j]>num:
                    result.append(nums2[j])
                    updated = True
                j = j+1
            if not updated:
                result.append(-1)
    return result
test_case = [[[4,1,2],[1,3,4,2]],[[2,4],[1,2,3,4]],[[4,1,2],[2,1,3,4]]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"For the list {test[0]}, the next greater element in the list {test[1]} is {nextGreaterElement(test[0],test[1])} ")
print("***** Method Two: Hashmap 0(m*n) *****")
def nextGreaterElement(nums1, nums2):
    num1Index = {n:i for i,n in enumerate(nums1) }
    result = [-1]*len(nums1)
    for i in range(len(nums2)):
        if nums2[i] not in num1Index:
            continue
        for j in range(i+1,len(nums2)):
            if nums2[j]>nums2[i]:
                index = num1Index[nums2[i]]
                result[index] = nums2[j]
                break
    return result
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"For the list {test[0]}, the next greater element in the list {test[1]} is {nextGreaterElement(test[0],test[1])} ")
print("***** Method Three: Stack 0(m+n) *****")
def nextGreaterElement(nums1, nums2):
    nums1Index = {n:i for i,n in enumerate(nums1)}
    result = [-1]*len(nums1)
    stack = []
    for i in range(len(nums2)):
        cur = nums2[i]
        while stack and cur>stack[-1]:
            val = stack.pop()
            index = nums1Index[val]
            result[index] = cur
        if cur in nums1Index:
            stack.append(cur)
    return result
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"For the list {test[0]}, the next greater element in the list {test[1]} is {nextGreaterElement(test[0],test[1])} ")
print("---------------------173. Binary Search Tree Iterator-------------------------")
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left
    def next(self):
        res = self.stack.pop()
        cur = res.right
        while cur:
            self.stack.append(cur)
            cur = cur.left
        return res.val
    def hasNext(self) -> bool:
        return len(self.stack)>0
root = generate_tree_from_list([7, 3, 15, None, None, 9, 20])
print(root)
obj = BSTIterator(root)
print(f"The Stack is:{obj.stack}, and the pointer is to {obj.stack[-1]}")
print(f"The value of obj.next() is {obj.next()}")
print(f"The value of obj.next() is {obj.next()}")
print(f"Has the next node: {obj.hasNext()}")
print(f"The value of obj.next() is {obj.next()}")
print(f"Has the next node: {obj.hasNext()}")
print(f"The value of obj.next() is {obj.next()}")
print(f"Has the next node: {obj.hasNext()}")
print("---------------------682. Baseball Game-------------------------")
def calPoints(ops):
    stack = []
    for c in ops:
        if c=="+":
            stack.append(stack[-1]+stack[-2])
        elif c=="D":
            stack.append(2*stack[-1])
        elif c=="C":
            stack.pop()
        else:
            stack.append(int(c))
    sum = 0
    while stack:
        sum = sum+stack.pop()
    return sum
test_case = [["5","2","C","D","+"], ["5","-2","4","C","D","9","+","+"],["1","C"]]
for i, ops in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"Based on the score record {ops}, the total sum is {calPoints(ops)}")
print("---------------------1209. Remove All Adjacent Duplicates in String II-------------------------")
def removeDuplicates(s, k):
    stack = []
    for c in s:
        if stack and stack[-1][0]== c:
            stack[-1][1] = stack[-1][1]+1
        else:
            stack.append([c,1])
        if stack[-1][1]==k:
            stack.pop()
    result = ""
    for char,count in stack:
        result = result+(char*count)
    return result
test_case = [["abcd",2],["deeedbbcccbdaa",3],["pbbcggttciiippooaais",2]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"Remove all {test[1]} adjacent duplicates in string {test[0]} is {removeDuplicates(test[0],test[1])} ")
print("---------------------456. 132 Pattern-------------------------")
print("***** Method One: Stack *****")
def find132pattern(nums):
    stack = [] # pair [val,minLeft]
    curMin = nums[0]
    for n in nums:
        while stack and n>=stack[-1][0]:
            stack.pop()
        if stack and n<stack[-1][0] and n>stack[-1][1]:
            return True
        stack.append([n,curMin])
        curMin = min(curMin,n)
    return False
test_case = [[1,2,3,4],[3,1,4,2],[-1,3,2,0]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"There is the subsquence in the {nums} meet the 132 pattern:{find132pattern(nums)}")
print("***** Method One: Nest Loop *****")
def find132pattern(nums):
    for i in range(len(nums)-1,1,-1):
        for j in range(i-1,0,-1):
            if nums[i]>=nums[j]:
                continue
            if nums[i]<nums[j]:
                for k in range(j-1,-1,-1):
                    if nums[k]<nums[i]:
                        return True
    return False
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"There is the subsquence in the {nums} meet the 132 pattern:{find132pattern(nums)}")