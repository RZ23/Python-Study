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
