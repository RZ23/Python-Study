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