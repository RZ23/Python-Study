print("---------------------494. Target Sum -------------------------")
def findTargetSumWays(nums,target):
    dp = {} # (index,total): # of ways
    def backtrack(i,total):
        if i==len(nums):
            if total==target:
                return 1
            else:
                return 0
        if (i,total) in dp:
            return dp[(i,total)]
        dp[(i,total)] = (backtrack(i+1,total+nums[i]))+(backtrack(i+1,total-nums[i]))
        return dp[(i,total)]
    return backtrack(0,0)
test_case = [[[1,1,1,1,1],3],[[1],1]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"There are {findTargetSumWays(test[0],test[1])} ways tp get the sum {test[1]}"
          f" from the given list {test[0]}")
print("---------------------70. Climbing Stairs -------------------------")
print("***** Method One: iteration *****")
def climbStairs(n):
    if n==1:
        return 1
    if n==2:
        return 2
    else:
        return climbStairs(n-1)+climbStairs(n-2)
for i in range(1,11):
    print(f"For {i} stairs floor, there are {climbStairs(i)} to reach the top")
print("***** Method Two: Dynamic Programming *****")
def climbStairs(n):
    if n<3:
        return n
    dp = [0]*(n+1)
    dp[2],dp[1]=2,1
    for i in range(3,n+1):
        dp[i] = dp[i-1]+dp[i-2]
    return dp[n]
for i in range(1,11):
    print(f"For {i} stairs floor, there are {climbStairs(i)} to reach the top")
print("***** Method Three: Dynamic Programming II *****")
def climbStairs(n):
    one_step,two_step = 1,1
    for i in range(n-1):
        temp = one_step
        one_step = one_step+two_step
        two_step = temp
    return one_step
for i in range(1,11):
    print(f"For {i} stairs floor, there are {climbStairs(i)} to reach the top")
print("---------------------983. Minimum Cost For Tickets -------------------------")
print("***** Method One: Iteration *****")
def mincostTickets(days,costs):
    dp = {}
    def dfs(i):
        if i==len(days):
            return 0
        if i in dp:
            return dp[i]
        dp[i] = float("inf")
        for d,c in zip([1,7,30],costs):
            j = i
            while j<len(days) and days[j]<d+days[i]:
                j = j+1
            dp[i] = min(dp[i],c+dfs(j))
        return dp[i]
    return dfs(0)
test_case = [[[1,4,6,7,8,20], [2,7,15]],[[1,2,3,4,5,6,7,8,9,10,30,31],[2,7,15]]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}")
    print(f"Based on the days {test[0]} and cost {test[1]}"
          f", the Minimum cost is {mincostTickets(test[0],test[1])}")
print("***** Method Two: Dynamic Programming *****")
def mincostTickets(days,costs):
    dp =  {}
    for i in range(len(days)-1,-1,-1):
        dp[i] = float("inf")
        for d,c in zip([1,7,30],costs):
            j = i
            while j<len(days) and days[j]<d+days[i]:
                j = j+1
            dp[i] = min(dp[i],c+dp.get(j,0))
    return dp[0]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}")
    print(f"Based on the days {test[0]} and cost {test[1]}"
          f", the Minimum cost is {mincostTickets(test[0],test[1])}")
print("---------------------518. Coin Change 2 -------------------------")
print("***** Method One: 2D Array *****")
def change(amount, coins):
    dp = [[0]*(len(coins) + 1) for i in range(amount+1)]
    dp[0] = [1]*(len(coins)+1)
    for a in range(1,amount+1):
        for i in range(len(coins)-1,-1,-1):
            dp[a][i] = dp[a][i+1]
            if a-coins[i]>=0:
                dp[a][i] = dp[a][i]+dp[a-coins[i]][i]
    return dp[amount][0]
test_case = [[5,[1,2,5]],[3,[2]],[10,[10]]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"There are {change(test[0],test[1])} ways to use coins {test[1]} to get the amount {test[0]}")
print("***** Method Two: Iteration *****")
def change(amount, coins):
    cache = {}
    def dfs(i,a):
        if a==amount:
            return 1
        if a>amount:
            return 0
        if i==len(coins):
            return 0
        cache[(i,a)] = dfs(i,a+coins[i])+dfs(i+1,a)
        return cache[(i,a)]
    return dfs(0,0)
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"There are {change(test[0],test[1])} ways to use coins {test[1]} to get the amount {test[0]}")
print("---------------------139. Word Break -------------------------")
def wordBreak(s,wordDict):
    dp = [False]*(len(s)+1)
    dp[len(s)]=True
    for i in range(len(s)-1,-1,-1):
        for w in wordDict:
            if (i+len(w))<=len(s) and s[i:i+len(w)]==w:
                dp[i] = dp[i+len(w)]
            if dp[i]:
                break
    return dp[0]
test_case = [["leetcode",["leet","code"]],["applepenapple", ["apple","pen"]],["catsandog",["cats","dog","sand","and","cat"]]]
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"'{test[0]}' can be segmented into a space-separated sequence of one or more dictionary words in {test[1]}:{wordBreak(test[0],test[1])} ")