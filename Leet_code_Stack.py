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
