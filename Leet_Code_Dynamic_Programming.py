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

print("---------------------5. Longest Palindromic Substring -------------------------")
print("***** Method One: Dynamic Programming *****")
def longestPalindrome(s):
    result = ""
    resultLen = 0
    for i in range(len(s)):
        # for odd string
        l,r = i,i
        while l>=0 and r<len(s) and s[l]==s[r]:
            if (r-l+1)>resultLen:
                resultLen = (r-l+1)
                result = s[l:r+1]
            l = l-1
            r = r+1
        # for even string
        l,r = i,i+1
        while l>=0 and r<len(s) and s[l]==s[r]:
            if (r-l+1)>resultLen:
                resultLen = (r-l+1)
                result = s[l:r+1]
            l = l-1
            r = r+1
    return result
test_case = ["babad","cbbd","aba","abba","ccc","a","aacabdkacaa"]
for i,s in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"The Longest Palindromic Substring in {s} is {longestPalindrome(s)}")
print("***** Method Two: Brute Force *****")
def longestPalindrome(s):
    if len(s)==1:
        return s
    def check_palindrome(s):
        l = 0
        r = len(s)-1
        while l<=r:
            if s[l]!=s[r]:
                return False
            l = l+1
            r = r-1
        return True
    maxLen = 1
    result = s[0]
    #
    for i in range(len(s)):
        for j in range(i,len(s)):
            if check_palindrome(s[i:j+1]):
                if (j-i+1)>maxLen:
                    maxLen = (j-i+1)
                    result = s[i:j+1]
    # return check_palindrome(s)
    return result
for i,s in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print(f"The Longest Palindromic Substring in {s} is {longestPalindrome(s)}")
print("---------------------91. Decode Ways -------------------------")
print("***** Method One: Dynamic Programming *****")
def numDecodings(s):
    dp = {len(s):1}
    for i in range(len(s)-1,-1,-1):
        if s[i]=="0":
            dp[i] = 0
        else:
            dp[i]=dp[i+1]
        if i+1<len(s) and (s[i]=="1" or s[i]=="2" and s[i+1] in ("0123456")):
            dp[i] = dp[i]+dp[i+2]
    return dp[0]
test_case = ["12","226","06"]
for i,s in enumerate(test_case):
    print(f"Test Case {i+1}:"
          f"There are {numDecodings(s)} ways to decoding the string '{s}'")
print("***** Method Two: Iteration  *****")
def numDecodings(s):
    dp = {len(s):1}
    def dfs(i):
        if i in dp:
            return dp[i]
        if s[i]=="0":
            return 0
        result = dfs(i+1)
        if i+1<len(s) and (s[i]=="1" or (s[i]=="2" and s[i+1] in "0123456")):
            result = result+dfs(i+2)
        dp[i] = result
        return result
    return dfs(0)
test_case = ["12","226","06"]
for i,s in enumerate(test_case):
    print(f"Test Case {i+1}:"
          f"There are {numDecodings(s)} ways to decoding the string '{s}'")
print("---------------------198. House Robber -------------------------")
print("***** Method One: Dynamic Programming with list: Time:O(n), Space: O(n)*****")
def rob(nums):
    if len(nums) == 1:
        return nums[0]
    result = [0] * len(nums)
    result[0] = nums[0]
    result[1] = max(nums[0], nums[1])
    if len(nums) < 3:
        return max(result[0], result[1])
    for i in range(2, len(nums)):
        result[i] = max(result[i - 2] + nums[i], result[i - 1])
    return result[-1]
test_case = [[1,2,3,1],[2,7,9,3,1],[0],[2,1],[1,2]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: For the House layout {nums}, "
          f"the maximum amount of money you can rob tonight without alerting the police is {rob(nums)} ")
print("***** Method Two: Dynamic Programming with only 2 variables: Time:O(n), Space:O(1) *****")
def rob(nums):
    rob1,rob2 = 0,0
    for n in nums:
        temp = max(rob1+n,rob2)
        rob1 = rob2
        rob2 = temp
    return rob2
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: For the House layout {nums}, "
          f"the maximum amount of money you can rob tonight without alerting the police is {rob(nums)} ")
print("---------------------97. Interleaving String -------------------------")
print("***** Method One: Dynamic Programming with Iteration *****")
def isInterleave(s1, s2, s3):
    if len(s1)+len(s2)!=len(s3):
        return False
    dp= {}
    def dfs(i,j):
        if i==len(s1) and j== len(s2):
            return True
        if (i,j) in dp:
            return dp[(i,j)]
        if i<len(s1) and s1[i]==s3[i+j] and dfs(i+1,j):
            return True
        if j<len(s2) and s2[j]==s3[i+j] and dfs(i,j+1):
            return True
        dp[(i,j)] = False
        return False
    return dfs(0,0)
test_case = [["aabcc","dbbca","aadbbcbcac"],
             ["aabcc","dbbca","aadbbbaccc"],
             ["","",""],["","","a"]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:The String '{test[2]}' could be interleaved by string '{test[0]}'"
          f"and String '{test[1]}':{isInterleave(test[0],test[1],test[2])}")
print("***** Method One: Dynamic Programming with 2D Arrays *****")
def isInterleave(s1, s2, s3):
    if len(s1)+len(s2)!=len(s3):
        return False
    dp = [[False]*(len(s2)+1) for i in range(len(s1)+1)]
    dp[len(s1)][len(s2)]=True
    for i in range(len(s1),-1,-1):
        for j in range(len(s2),-1,-1):
            if i<len(s1) and s1[i]==s3[i+j] and dp[i+1][j]:
                dp[i][j] = True
            if j<len(s2) and s2[j]==s3[i+j] and dp[i][j+1]:
                dp[i][j]=True
    return dp[0][0]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}:The String '{test[2]}' could be interleaved by string '{test[0]}'"
          f"and String '{test[1]}':{isInterleave(test[0],test[1],test[2])}")
print("---------------------213. House Robber II -------------------------")
def rob(nums):
    def robber(house):
        house1,house2 = 0,0
        for h in house:
            temp = max(house1+h,house2)
            house1 = house2
            house2 = temp
        return house2
    # get max of [except first one], [except the last one] and,[first one]-> in case there is only one in the list
    return max(robber(nums[:-1]),robber(nums[1:]),nums[0])
test_case = [[2,3,2],[1,2,3,1],[1,2,3]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the house layout {nums}, the maximum amount of money can rob is {rob(nums)} ")
print("---------------------256. Paint House -------------------------")
def minCost(costs):
    if len(costs)==0:
        return 0
    if len(costs)==1:
        return min(costs[0])
    for i in range(1,len(costs)):
        costs[i][0] = costs[i][0]+min(costs[i-1][1],costs[i-1][2])
        costs[i][1] = costs[i][1] + min(costs[i - 1][0], costs[i - 1][2])
        costs[i][2] = costs[i][2] + min(costs[i - 1][0], costs[i - 1][1])
    return min(costs[-1])
test_case = [[17,2,17],[16,16,5],[14,3,19]]
print(f"The minimum cost of painting houses by the list {test_case} is {minCost(test_case)}")
print("---------------------300. Longest Increasing Subsequence -------------------------")
def lengthOfLIS(nums):
    if len(nums)==1:
        return nums[0]
    LIS = [1]*len(nums)
    for i in range(len(nums)-1,-1,-1):
        for j in range(i+1,len(nums)):
            if nums[i]<nums[j]:
                LIS[i] = max(LIS[i],1+LIS[j])
    return max(LIS)
test_case = [[10,9,2,5,3,7,101,18],[0,1,0,3,2,3],[7,7,7,7,7,7,7]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: The length of Longest Increasing Subsequence in {nums} "
          f"is {lengthOfLIS(nums)} ")