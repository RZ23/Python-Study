print("---------------------1849. Splitting a String Into Descending Consecutive Values-------------------------")
'''
Using Backtracking Algorithm. Get one character and transfer into int, using backtracking algorithm, if the 
following is descent order and difference is one, then continue, if the scan all the characters, return True
if not, return False. 
'''
def splitString(s):
    def dfs(index,pre_value):
        if index ==len(s):
            return True
        for j in range(index,len(s)):
            val = int(s[index:j+1])
            if val+1==pre_value and dfs(j+1,val):
                return True
        return False
    for i in range(len(s)-1):
        val = int(s[:i+1])
        if dfs(i+1,val):
            return True
    return False
test_case= ["1234","050043","9080701"]
for s in test_case:
    print(f"The String '{s}' can be split into substring in descent order and the different is one: {splitString(s)}")