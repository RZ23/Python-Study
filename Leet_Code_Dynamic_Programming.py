from collections import Counter
def print_matrix(dp):
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if isinstance(dp[i][j],int):
                print(format(dp[i][j],'02d'), end="|")
            else:
                print(dp[i][j], end="|")
        print()
def print_int_matrix(matrix):
    print("Matrix:")
    row = len(matrix)
    col =len(matrix[0])
    for i in range(row):
        for j in range(col):
            print(format(matrix[i][j],"02d"),end = " ")
        print()
def print_matrix_in_triangle_format(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == float("inf"):
                break
            else:
                print(matrix[i][j],end = " ")
        print()
def print_triangle(triangle):
    temp_triangle = [[" " for i in range(len(triangle[-1]))] for j in range(len(triangle))]
    for i in range(len(triangle)):
        for j in range(len(triangle[i])):
            temp_triangle[i][j] = triangle[i][j]
    for i in range(len(temp_triangle)):
        for j in range(len(temp_triangle[i])):
            print(temp_triangle[i][j],end = " ")
        print()
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
print("---------------------1143. Longest Common Subsequence -------------------------")
def longestCommonSubsequence(text1, text2):
    dp = [[0 for i in range(len(text2)+1)] for j in range(len(text1)+1)]
    for i in range(len(text1)-1,-1,-1):
        for j in range(len(text2)-1,-1,-1):
            if text1[i]==text2[j]:
                dp[i][j] = 1+dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j],dp[i][j+1])
    return dp[0][0]
test_case = [["abcde","ace"],["abc","abc"],["abc","def"]]
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: The length of the longest Common Subsequence of "
          f"{test[0]} and {test[1]} is {longestCommonSubsequence(test[0],test[1])}")
print("---------------------152. Maximum Product Subarray -------------------------")
print("***** Method One:Brute Force *****")
def maxProduct(nums):
    result = [1]* len(nums)
    for i in range(len(nums)):
        curCont = nums[i]
        curMax = nums[i]
        for j in range(i+1,len(nums)):
            if nums[j]==0:
                curCont = 1
                continue
            else:
                curCont = curCont*nums[j]
                curMax = max(curMax,curCont)
        result[i] = curMax
    return max(result)
test_case = [[2,3,-2,4],[-2,0,-1],[-4,-3,-2]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: The Maximum Product Subarray of {nums} is {maxProduct(nums)} ")
print("***** Method Two:Dynamic Programming *****")
def maxProduct(nums):
    result = max(nums)
    curMin,curMax = 1,1
    for n in nums:
        if n==0:
            curMin,curMax = 1,1
            continue
        temp = curMax
        curMax = max(curMax*n,curMin*n,n)
        curMin = max(curMin * n, temp * n,n)
        result = max(result,curMax)
    return result
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: The Maximum Product Subarray of {nums} is {maxProduct(nums)} ")
print("---------------------322. Coin Change -------------------------")
print("***** Method One:Dynamic Programming *****")
def coinChange(coins, amount):
    dp = [float("inf")]*(amount+1)
    dp[0] = 0
    for temp_amount in range(1,amount+1):
        for coin in coins:
            if temp_amount-coin>=0:
                dp[temp_amount] = min(dp[temp_amount],1+dp[temp_amount-coin])
    return dp[amount] if dp[amount]!=float("inf") else -1
test_case = [[[1,2,5], 11],[[2],3],[[1],0],[[1,2,5],5],[[2],3],[[10],10]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the coins {test[0]},"
          f"the fewest number of coins to make up that amount {test[1]}"
          f" is {coinChange(test[0],test[1])}. ")
print("***** Method Two:Backtracking *****")
def coinChange(coins, amount):
    result = []
    # result = set()
    def dfs(remain,temp_arr):
        if remain == amount:
            result.append(sorted(temp_arr.copy()))
            # result.update(sorted(temp_arr.copy()))
            return True
        if remain>amount:
            return False
        for c in coins:
            temp_arr.append(c)
            dfs(remain+c,temp_arr)
            temp_arr.pop()
    dfs(0,[])
    min_coin = float("inf")
    unduplicated_result = []
    for lst in result:
        if lst not in unduplicated_result:
            unduplicated_result.append(lst)
    print(f"There are {len(unduplicated_result)} combinations to get the amount {amount}")
    print(f"Conbination: {unduplicated_result}")
    for item in unduplicated_result:
        if len(item)<min_coin:
            min_coin = len(item)
    return min_coin if min_coin!=float("inf") else -1
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}: ")
    print(f"Based on the coins {test[0]} the fewest number of coins to make up that amount {test[1]}"
          f" is {coinChange(test[0],test[1])}.")
print("---------------------221. Maximal Square -------------------------")
def maximalSquare(matrix):
    ROW,COLS = len(matrix),len(matrix[0])
    dp = {}
    def helper(r,c):
        if r>=ROW or c>=COLS:
            return 0
        if (r,c) not in dp:
            down = helper(r+1,c)
            right = helper(r,c+1)
            diag = helper(r+1,c+1)
            dp[(r,c)]=0
            if matrix[r][c]=="1":
                dp[(r,c)] = 1+min(down,right,diag)
        return dp[(r,c)]
    helper(0,0)
    return max(dp.values())**2
test_case = [[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]],
             [["0","1"],["1","0"]],[["0"]]]
for i,matrix in enumerate(test_case):
    print(f"Test Case {i+1}: The Matrix is ")
    print_matrix(matrix)
    print(f"The area of largest square is {maximalSquare(matrix)} ")
print("---------------------120. Triangle -------------------------")
print("***** Method One: Dynamic Programming Time Complexity O(n^2) Space Complexity O(n) *****")
def minimumTotal(triangle):
    dp = [0]*(len(triangle)+1)
    for row in triangle[::-1]:
        for i,n in enumerate(row):
            dp[i] = n+min(dp[i],dp[i+1])
    return dp[0]
test_case = [[[2],[3,4],[6,5,7],[4,1,8,3]],[[-10]]]
for i,triangle in enumerate(test_case):
    print(f"Test Case {i+1}: The triangle is:")
    print_triangle(triangle)
    print(f"The minimum path sum from top to bottom is {minimumTotal(triangle)}")
print("***** Method Two: Auxiliary 2d array Time Complexity O(n^2) Space Complexity O(n^2) *****")
def minimumTotal(triangle):
    support_array = [[float("inf") for i in range(len(triangle[-1]))] for j in range(len(triangle))]
    for j in range(len(triangle[-1])):
        support_array[-1][j] = triangle[-1][j]
    for i in range(len(triangle)-2,-1,-1):
        for j in range(len(triangle[i])):
            support_array[i][j] = triangle[i][j]+min(support_array[i+1][j],support_array[i+1][j+1])
    print(f"The result of support array is:")
    print_matrix_in_triangle_format(support_array)
    return support_array[0][0]
for i,triangle in enumerate(test_case):
    print(f"Test Case {i+1}: The triangle is:")
    print_triangle(triangle)
    print(f"The minimum path sum from top to bottom is {minimumTotal(triangle)}")
print("---------------------96. Unique Binary Search Trees -------------------------")
print("***** Time Complexity O(n^2) Space Complexity O(n) *****")
def numTrees(n):
    # numTree[4] = numTree[0]*numTree[3]+
    #              numTree[1]*numTree[2]+
    #              numTree[2]*numTree[1]+
    #              numTree[3]*numTree[0]
    numTree = [1]*(n+1)
    # numTree[1] =1
    # numTree[0] = 1
    for nodes in range(2,n+1):
        total = 0
        for root in range(1,nodes+1):
            left = root-1
            right = nodes-root
            total = total+(numTree[left]*numTree[right])
        numTree[nodes] = total
    return numTree[n]
for i in range(6):
    print(f"For {i} node(s),it could construct {numTrees(i)}  structurally unique BST")
print("---------------------10. Regular Expression Matching -------------------------")
def isMatch(s,p):
    def backtracking(i,j):
        # Both at the end of the string, return True
        if i>=len(s) and j>=len(p):
            return True
        # pattern reach the end, but string are not, return False
        if j>=len(p):
            return False
        match = i<len(s) and (s[i]==p[j] or p[j]==".")
        if (j+1)<len(p) and p[j+1]=="*":
            # if the next character in the p is "*"
            # two choices, not use previous character -> backtracking(i,j+2)
            #              use the previous character ->backtracking (i+1,j)
            return (backtracking(i,j+2) or ( match and backtracking(i+1,j)))
        if match:
            return backtracking(i+1,j+1)
    return backtracking(0,0)
test_case = [["aa","a"],["aa", "a*"],["ab",".*"]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}: the Pattern '{test[1]}' matchs the String '{test[0]}': {isMatch(test[0],test[1])} ")
print("---------------------55. Jump Game -------------------------")
print("***** Method One: Greedy *****")
def canJump(nums):
    goal = len(nums)-1
    for i in range(len(nums)-1,-1,-1):
        if i+nums[i]>=goal:
            goal = i
    if goal ==0:
        return True
    else:
        return False
test_case = [[2,3,1,1,4],[3,2,1,0,4]]
for i, nums in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the {nums}, it could jump from the 1st item to the last item: {canJump(nums)}")
print("***** Method Two: Dynamic Programming *****")
def canJump(nums):
    dp = [False]*len(nums)
    dp[len(nums)-1] = True
    for i in range(len(nums)-2,-1,-1):
        if nums[i]>0:
            for j in range(1,nums[i]+1):
                dp[i] = dp[i] or dp[i+j]
        # print(f"dp[{i}] = {dp[i]}")
    return dp[0]
for i, nums in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the {nums}, it could jump from the 1st item to the last item: {canJump(nums)}")
print("***** Method Three: Two Pointers *****")
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
for i, nums in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the {nums}, it could jump from the 1st item to the last item: {canJump(nums)}")
print("---------------------45. Jump Game II -------------------------")
print("***** Method One: Greedy *****")
def canJump(nums):
    l=r=0
    result = 0
    while r<len(nums)-1:
        farthest = 0
        for i in range(l,r+1):
            farthest = max(farthest,i+nums[i])
        l = r+1
        r = farthest
        result = result+1
    return result
test_case = [[2,3,1,1,4],[2,3,0,1,4]]
for i, nums in enumerate(test_case):
    print(f"Test Case {i+1}: For {nums}, the minimum step(s) to reach the"
          f"last index is {canJump(nums)} ")
print("***** Method Two: Dynamic Programming *****")
def canJump(nums):
    dp = [float("inf")]*len(nums)
    dp[len(nums)-1] = 0
    for i in range(len(nums)-2,-1,-1):
        for j in range(1,nums[i]+1):
            if i+j<len(nums):
                dp[i] = min(dp[i],1+dp[i+j])
    return dp[0]
for i, nums in enumerate(test_case):
    print(f"Test Case {i+1}: For {nums}, the minimum step(s) to reach the"
          f"last index is {canJump(nums)} ")
print("---------------------62. Unique Paths -------------------------")
print("***** Method One: Dynamic Programming *****")
def uniquePaths(m,n):
    row = [1]*n
    for i in range(m-1):
        newRow = [1]*n
        for j in range(n-2,-1,-1):
            newRow[j] = newRow[j+1]+row[j]
        row =newRow
    return row[0]
test_case = [[3,7],[3,2]]
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: For the {test[0]}*{test[1]} matrix,"
          f"there is(are) {uniquePaths(test[0],test[1])} path from left-top to bottom-right")
print("***** Method Two: Recursive *****")
def uniquePaths(m,n):
    def path(r,c):
        if r==m:
            return 1
        if c==n:
            return 1
        else:
            return path(r,c+1)+path(r+1,c)
    return path(1,1)
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: For the {test[0]}*{test[1]} matrix,"
          f"there is(are) {uniquePaths(test[0],test[1])} path from left-top to bottom-right")
print("***** Method Three: 2D Array Dynamic Programming *****")
def uniquePaths(m,n):
    dp = [[1 for i in range(n)] for j in range(m)]
    for row in range(1,m):
        for col in range(1,n):
            dp[row][col] = dp[row][col-1]+dp[row-1][col]
    return dp[-1][-1]
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: For the {test[0]}*{test[1]} matrix,"
          f"there is(are) {uniquePaths(test[0],test[1])} path from left-top to bottom-right")
print("---------------------279. Perfect Squares -------------------------")
def numSquares(n):
    dp = [n]*(n+1)
    dp[0] = 0
    for target in range(1,n+1):
        for s in range(1,target+1):
            square = s*s
            if target-square<0:
                break
            dp[target] = min(dp[target],1+dp[target-square])
            # if 1+dp[target-square]<dp[target]:
            #     dp[target] = 1+dp[target-square]
    return dp[n]
for i in range(1,16):
    print(f"The least number of perfect square numbers that sum to {i} is {numSquares(i)} ")
print("---------------------72. Edit Distance -------------------------")
def minDistance(word1,word2):
    dp = [[float("inf") for i in range(len(word2)+1)] for j in range(len(word1)+1)]
    for i in range(len(word1)+1):
        dp[i][len(word2)] = len(word1)-i
    for j in range(len(word2)+1):
        dp[len(word1)][j] = len(word2)-j
    for i in range(len(word1)-1,-1,-1):
        for j in range(len(word2)-1,-1,-1):
            if word1[i]==word2[j]:
                dp[i][j] = dp[i+1][j+1]
            else:
                dp[i][j] = 1+min(dp[i+1][j],dp[i][j+1],dp[i+1][j+1])
    return dp[0][0]
test_case = [["horse", "ros"],["intention","execution"]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}: The minimum number of operations required to convert '{test[0]}' to '{test[1]}' "
          f"is {minDistance(test[0],test[1])}")
print("---------------------115. Distinct Subsequences -------------------------")
def numDistinct(s, t):
    dp = {}
    def recursive(i,j):
        # at the end of the target string t
        if j==len(t):
            return 1
        # target string t is longer than the original string s
        if i==len(s):
            return 0
        if (i,j) in dp:
            return dp[(i,j)]
        if s[i]==t[j]:
            dp[(i,j)] = recursive(i+1,j+1)+recursive(i+1,j)
        else:
            dp[(i,j)] = recursive(i+1,j)
        return dp[(i,j)]
    return recursive(0,0)
test_case = [["rabbbit","rabbit"],["babgbag","bag"]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}: There is(are) {numDistinct(test[0],test[1])} distinct subsequences of "
          f"'{test[0]}' which equals '{test[1]}'. ")
print("-------------------- 377. Combination Sum IV -------------------------")
print("***** Method One: Dynamic Programming *****")
print("***** Time Complexity O(m*n) *****")
def combinationSum4(nums,target):
    dp={0:1}
    for total in range(1,target+1):
        dp[total] = 0
        for n in nums:
            dp[total] = dp[total]+dp.get(total-n,0)
    return dp[target]
test_case = [[[1,2,3], 4],[[9],3]]
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: based on the list {test[0]}, there is"
          f" {combinationSum4(test[0],test[1])} combination(s) to get target number"
          f" {test[1]}")
print("***** Method Two: Recursive *****")
def combinationSum4(nums,target):
    def countIt(cur):
        if cur == target:
            return 1
        if target<cur:
            return 0
        ans = 0
        for num in nums:
            ans = ans+countIt(cur+num)
        return ans
    return countIt(0)
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: based on the list {test[0]}, there is"
          f" {combinationSum4(test[0],test[1])} combination(s) to get target number"
          f" {test[1]}")
print("***** Method Three: Recursive with Programming *****")
def combinationSum4(nums,target):
    def countIt(cur,dp):
        if cur==target:
            return 1
        if cur>target:
            return 0
        if cur in dp:
            return dp[cur]
        ans = 0
        for num in nums:
            ans = ans+ countIt(cur+num,dp)
        dp[cur] = ans
        return ans
    return countIt(0,{})
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: based on the list {test[0]}, there is"
          f" {combinationSum4(test[0],test[1])} combination(s) to get target number"
          f" {test[1]}")
print("-------------------- 312. Burst Balloons -------------------------")
print("***** Method One: Dynamic Programming - DP Cache")
def maxCoins(nums):
    nums = [1]+nums+[1]
    dp = {}
    def dfs(l,r):
        if l>r:
            return 0
        if (l,r) in dp:
            return dp[(l,r)]
        dp[(l,r)] = 0
        for i in range(l,r+1):
            coins = nums[l-1]*nums[i]*nums[r+1]
            coins = coins+dfs(l,i-1)+dfs(i+1,r)
            dp[(l,r)] = max(dp[l,r],coins)
        return dp[(l,r)]
    return dfs(1,len(nums)-2)
test_case = [[3,1,5,8],[1,5]]
for i, nums in enumerate(test_case):
    print(f"Based on the nums {nums}, the maximum"
          f"coins to burst balloons is {maxCoins(nums)}")
print("***** Method Two: Dynamic Programming - 2D Cache")
def maxCoins(nums):
    N = len(nums)
    nums = [1]+nums+[1]
    dp = [[0 for i in range(len(nums)+2)] for j in range(len(nums)+2)]
    for length in range(1, N + 1):
        for left in range(1, N - length + 2):
            right = left+length-1
            for last in range(left,right+1):
                dp[left][right] = max(dp[left][right],dp[left][last-1]+nums[left-1]*nums[last]*nums[right+1]+dp[last + 1][right])
    return dp[1][N]
for i, nums in enumerate(test_case):
    print(f"Based on the nums {nums}, the maximum"
          f"coins to burst balloons is {maxCoins(nums)}")
print("-------------------- 1866. Number of Ways to Rearrange Sticks With K Sticks Visible -------------------------")
print("***** Method One: Dynamic Programming - Recursive")
def rearrangeSticks(n,k):
    dp = {}
    def dfs(N, K):
        if N == K:
            return 1
        if N == 0 or K == 0:
            return 0
        if (N, K) in dp:
            return dp[(N, K)]
        dp[(N, K)] = ((dfs(N - 1, K - 1) + (N - 1) * dfs(N - 1, K))) % (10 ** 9 + 7)
        return dp[(N, K)]
    return dfs(n, k)
test_case = [[3,2],[5,5],[20,11]]
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: There is(are) {rearrangeSticks(test[0],test[1])} ways for {test[0]} sticks and {test[1]} visiable")
print("***** Method Two: Dynamic Programming - Cache")
def rearrangeSticks(n,k):
    dp = {(1,1):1}
    for N in range(2,n+1):
        for K in range(1,k+1):
            dp[(N,K)] = (dp.get((N-1,K-1),0)+(N-1)*dp.get((N-1,K),0))
    return dp[(n,k)]% (10**9+7)
for i, test in enumerate(test_case):
    print(f"Test Case {i+1}: There is(are) {rearrangeSticks(test[0],test[1])} ways for {test[0]} sticks and {test[1]} visiable")
print("-------------------- 1911. Maximum Alternating Subsequence Sum -------------------------")
print("***** Method One: Recursive *****")
def maxAlternatingSum(nums):
    dp = {}
    # flag is true=> even, else odd
    def dfs(i,even):
        if i==len(nums):
            return 0
        if (i,even) in dp:
            return dp[(i,even)]
        total = nums[i] if even else (-1*nums[i])
        dp[(i,even)] = max(total+dfs(i+1, not even),dfs(i+1,even))
        return dp[(i,even)]
    return dfs(0,True)
test_case = [[4,2,5,3],[5,6,7,8],[6,2,1,2,4,5]]
for i,nums in enumerate(test_case):
    print(f"Test case {i+1}: for array {nums}, "
          f"the Maximum Alternating Subsequence Sum is {maxAlternatingSum(nums)}")
print("***** Method One: Dynamic Programming *****")
def maxAlternatingSum(nums):
    sumEven = 0
    sumOdd = 0
    for i in range(len(nums)-1,-1,-1):
        tempEven = max(sumOdd+nums[i],sumEven)
        tempOdd = max(sumEven-nums[i],sumOdd)
        sumOdd = tempOdd
        sumEven = tempEven
    return sumEven
for i,nums in enumerate(test_case):
    print(f"Test case {i+1}: for array {nums}, "
          f"the Maximum Alternating Subsequence Sum is {maxAlternatingSum(nums)}")
print("-------------------- 416. Partition Equal Subset Sum -------------------------")
def canPartition(nums):
    if sum(nums)%2:
        return False
    target = sum(nums)//2
    dp = set()
    dp.add(0)
    for i in range(len(nums)-1,-1,-1):
        TempDP= set()
        for t in dp:
            if t+nums[i]==target:
                return True
            TempDP.add(t+nums[i])
            TempDP.add(t)
        dp = TempDP
    return True if target in dp else False
test_case = [[1,5,11,5],[1,2,3,5]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: The array {nums} could be partitioned into two subsets "
          f"that the sum of elements in both subsets is equal: {canPartition(nums)}")
print("-------------------- 691. Stickers to Spell Word -------------------------")
def minStickers(stickers, target):
    stickCount = []
    for i,s in enumerate(stickers):
        stickCount.append({})
        for c in s:
            stickCount[i][c] = 1+stickCount[i].get(c,0)
    dp = {}
    def dfs(t,stick):
        if t in dp:
            return dp[t]
        res = 1 if stick else 0
        remainT =""
        for c in t:
            if c in stick and stick[c]>0:
                stick[c] = stick[c]-1
            else:
                remainT = remainT+c
        if remainT:
            used = float("inf")
            for s in stickCount:
                if remainT[0] not in s:
                    continue
                used = min(used,dfs(remainT,s.copy()))
            dp[remainT] = used
            res = res+used
        return res
    res = dfs(target,{})
    return res if res!=float("inf") else -1
test_case = [[["with","example","science"], "thehat"],[["notice","possible"], "basicbasic"]]
for i,test in enumerate(test_case):
    print(f"Test Case {i+1}: The minimum number of stickers to spell out {test[1]} based on the {test[0]}"
          f" is {minStickers(test[0],test[1])} ")
print("-------------------- 338. Counting Bits -------------------------")
print("***** Method One: Transfer to Bit and Count *****")
def countBits(n):
    def count1(s):
        cnt = 0
        for c in s:
            if c == "1":
                cnt = cnt + 1
        return cnt
    binary_result = []
    for i in range(n + 1):
        binary_result.append(str(bin(i))[2:])
    result = []
    for item in binary_result:
        result.append(count1(item))
    return result
test_case = [2,5]
for i, n in enumerate(test_case):
    print(f"Test Case {i+1}: For {n}, the numbers of '1' by each number in bit format is {countBits(n)}")
print("***** Method Two: Dynamic Programming *****")
def countBits(n):
    dp = [0]*(n+1)
    dp[0] = 0
    offset = 1
    for i in range(1,n+1):
        if offset*2==i:
            offset = i
        dp[i] = 1+dp[i-offset]
    return dp
for i, n in enumerate(test_case):
    print(f"Test Case {i+1}: For {n}, the numbers of '1' by each number in bit format is {countBits(n)}")
print("-------------------- 1553. Minimum Number of Days to Eat N Oranges -------------------------")
def minDays(n):
    dp = {0:0,1:1}
    def dfs(n):
        if n in dp:
            return dp[n]
        divide_by_two = 1+(n%2)+dfs(n//2)
        divide_by_three = 1+(n%3)+dfs(n//3)
        dp[n] = min(divide_by_two,divide_by_three)
        return dp[n]
    return dfs(n)
test_case = [10,6,1,56]
for i, n in enumerate(test_case):
    print(f"The minimum days to eat {n} oranges is {minDays(n)} days")
print("-------------------- 64. Minimum Path Sum -------------------------")
def minPathSum(grid):
    ROW = len(grid)
    COL = len(grid[0])
    path = [[0 for i in range(COL)] for j in range(ROW)]
    path[0][0] = grid[0][0]
    for i in range(1,COL):
        path[0][i] = grid[0][i]+path[0][i-1]
    for j in range(1,ROW):
        path[j][0] = grid[j][0]+path[j-1][0]
    for i in range(1,ROW):
        for j in range(1,COL):
            path[i][j] = min(grid[i][j]+path[i][j-1],grid[i][j]+path[i-1][j])
    print("Result Path Sum:")
    print_matrix(path)
    return path[-1][-1]
test_case = [[[1,3,1],[1,5,1],[4,2,1]],[[1,2,3],[4,5,6]]]
for i, grid in enumerate(test_case):
    print(f"Test Case {i+1}:")
    print_matrix(grid)
    print(f"The Minimum Path Sum is {minPathSum(grid)}")
print("-------------------- 118. Pascal's Triangle -------------------------")
def generate(numRows):
    result = [[1]]
    for i in range(1,numRows):
        temp_result = []
        temp_result.append(1)
        for j in range(1,i):
            temp_result.append(result[i-1][j-1]+result[i-1][j])
        temp_result.append(1)
        result.append(temp_result)
    return result
test_case = [5,1]
for i,numRows in enumerate(test_case):
    print(f"Test Case {i+1}: The Pascal's Triangle of {numRows} is {generate(numRows)}")
    print_triangle(generate(numRows))
print("---------------------894. All Possible Full Binary Trees-------------------------")
from collections import deque
import binarytree
from binarytree import Node,tree
class TreeNode(binarytree.Node):
    def __init__(self,values=0,right=None,left= None):
        self.val = values
        self.right = right
        self.left = left
def tree_level_to_list_with_queue_and_null_value(root):
    if not root:
        return []
    result = []
    q= deque()
    q.append(root)
    while q:
        node = q.popleft()
        if node is None:
            result.append(None)
        else:
            result.append(node.val)
            q.append(node.left)
            q.append(node.right)
    return result

'''A full binary tree is a binary tree where each node has exactly 0 or 2 children'''
def allPossibleFBT(n):
    if n%2==0:
        return []
    """
    using hashmap to reduce the computing time
    """
    dp ={0:[],1:[TreeNode(0,None,None)]}
    # Return the list of full-binary tree
    def backtracking(n):
        if n in dp:
            return dp[n]
        res = []
        # since it is from 0 to n-1, and for the n nodes, there are n-1 children nodes without the root
        for l in range(n):
            r = n-1-l
            leftTrees, rightTrees= backtracking(l),backtracking(r)

            for t1 in leftTrees:
                for t2 in rightTrees:
                    res.append(TreeNode(0,left = t1,right = t2))
        dp[n] = res
        return res
    return backtracking(n)
test_case = [7,3]
for n in test_case:
    print(f"All the possible of Full Binary Tree(s) of the {n} nodes:")
    root_list = allPossibleFBT(n)
    for i in range(len(root_list)):
        root = root_list[i]
        print(root)
        print(tree_level_to_list_with_queue_and_null_value(root))
print("---------------------746. Min Cost Climbing Stairs-------------------------")
print("***** Method One: Dynamic Programming with Array List")
def minCostClimbingStairs(cost):
    dp =[0]*(len(cost))
    dp[-1] = cost[-1]
    dp[-2] = cost[-2]
    for i in range(len(cost)-3,-1,-1):
        dp[i] = cost[i]+min(dp[i+1],dp[i+2])
    # print(dp)
    return min(dp[0],dp[1])
test_case = [[10,15,20],[1,100,1,1,1,100,1,1,100,1]]
for i,cost in enumerate(test_case):
    print(f"Test Case: {i+1}: Based on the list {cost}, the Min Cost Climbing Stairs is {minCostClimbingStairs(cost)}")
print("***** Method Two: Dynamic Programming without Additional Array")
def minCostClimbingStairs(cost):
    cost.append(0)
    for i in range(len(cost)-3,-1,-1):
        cost[i] = cost[i]+min(cost[i+1],cost[i+2])
    return min(cost[0],cost[1])
for i,cost in enumerate(test_case):
    print(f"Test Case: {i+1}: Based on the list {cost}, the Min Cost Climbing Stairs is {minCostClimbingStairs(cost)}")
print("---------------------877. Stone Game-------------------------")
def stoneGame(pile):
    dp = {} #dp[(l,r)] = max pile from l to r
    def dfs(l,r):
        if l>r:
            return 0
        if (l,r) in dp:
            return dp[(l,r)]
        even = True if (r-l)%2==1 else False
        left = pile[l] if even else 0
        right = pile[r] if even else 0
        dp[(l,r)] = max(left+dfs(l+1,r),right+dfs(l,r-1))
        return dp[l,r]
    first_player_score  = dfs(0,len(pile)-1)
    return first_player_score,first_player_score>(sum(pile)//2)
test_case = [[5,3,4,5],[3,7,2,3]]
for i, pile in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the Pile {pile},"
          f"The first player's maximum score is {stoneGame(pile)[0]} and could win the game: {stoneGame(pile)[1]}")
print("---------------------309. Best Time to Buy and Sell Stock with Cooldown-------------------------")
def maxProfit(prices):
    dp ={} # dp[i,buying] = max_value
    def dfs(i,buying):
        if i>=len(prices):
            return 0
        if (i,buying) in dp:
            return dp[(i,buying)]
        cooldown = dfs(i + 1, buying)
        if buying:
            buy = dfs(i+1,not buying)-prices[i]
            dp[(i,buying)] = max(buy,cooldown)
        else:
            sell = dfs(i+2,not buying)+prices[i]
            dp[(i,buying)] = max(sell,cooldown)
        return dp[(i,buying)]
    return dfs(0,True)
test_case = [[1,2,3,0,2],[1]]
for i,prices in enumerate(test_case):
    print(f"Test Case {i+1}: Based on the Price {prices},"
          f"The maximum value is {maxProfit(prices)}")
print("---------------------343. Integer Break-------------------------")
print("***** Method One: Dynamic Programming with Recursive *****")
def integerBreak(n):
    dp = {1:1}
    def dfs(num):
        if num in dp:
            return dp[num]
        dp[num] = 0 if num==n else num
        for i in range(1,num):
            val = dfs(i)*dfs(num-i)
            dp[num] = max(dp[num],val)
        return dp[num]
    return dfs(n)
test_case = [2,10]
for i, n in enumerate(test_case):
    print(f"For the int {n}, the maximum "
          f"product is {integerBreak(n)} ")
print("***** Method Two: Dynamic Programming without Recursive *****")
def integerBreak(n):
    dp = {1:1}
    for num in range(2,n+1):
        dp[num] = 0 if num== n else num
        for i in range(1,num):
            val = dp[i]*dp[num-i]
            dp[num] = max(dp[num],val)
    return dp[n]
for i, n in enumerate(test_case):
    print(f"For the int {n}, the maximum "
          f"product is {integerBreak(n)} ")
print("---------------------673. Number of Longest Increasing Subsequence-------------------------")
def findNumberOfLIS(nums):
    dp = {}
    lenLIS,cnt = 0,0
    for i in range(len(nums)-1,-1,-1):
        maxLen,maxCnt = 1,1
        for j in range(i+1,len(nums)):
            if nums[j]>nums[i]:
                length,count = dp[j]
                if length+1>maxLen:
                    maxLen,maxCnt = length + 1,count
                elif length + 1 ==maxLen:
                    maxCnt = maxCnt + count
        if maxLen>lenLIS:
            lenLIS,cnt = maxLen,maxCnt
        elif maxLen==lenLIS:
            cnt = cnt+maxCnt
        dp[i] = [maxLen,maxCnt]
    return cnt
test_case = [[1,3,5,4,7],[2,2,2,2,2]]
for i,nums in enumerate(test_case):
    print(f"Test Case {i+1}: There are {findNumberOfLIS(nums)} longest increasing subsequence"
          f" based on {nums} ")
print("---------------------740. Delete and Earn-------------------------")
def deleteAndEarn(nums):
    count = Counter(nums)
    nums = sorted(list(set(nums)))
    earn1,earn2 = 0,0
    for i in range(len(nums)):
        curEarn = nums[i]*count[nums[i]]
        #can't use both curEarn amd earn2
        if i>0 and nums[i]==nums[i-1]+1:
            temp = earn2
            earn2 = max(curEarn+earn1,earn2)
            earn1 = temp
        else:
            temp = earn2
            earn2 = curEarn+earn2
            earn1 = temp
    return earn2
test_case = [[3,4,2],[2,2,3,3,3,4]]
for i,nums in enumerate(test_case):
    print(f"Test Case :{i+1}: Based on the list {nums}, The maximum number of point is {deleteAndEarn(nums)}")
print("---------------------329. Longest Increasing Path in a Matrix-------------------------")
def longestIncreasingPath(matrix):
    dp = {}
    COL= len(matrix[0])
    ROW = len(matrix)
    def dfs(r,c,preVal):
        if (r<0 or r>=ROW or c<0 or c>=COL or matrix[r][c]<=preVal):
            return 0
        if (r,c) in dp:
            return dp[(r,c)]
        res = 1
        res = max(res, 1 + dfs(r + 1,c,matrix[r][c]))
        res = max(res, 1 + dfs(r - 1, c, matrix[r][c]))
        res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))
        res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))
        dp[(r,c)] = res
        return res
    for i in range(ROW):
        for j in range(COL):
            dfs(i,j,-1)
    return max(dp.values())
test_case = [[[9,9,4],[6,6,8],[2,1,1]],[[3,4,5],[3,2,6],[2,2,1]]]
for i, matrix in enumerate(test_case):
    print(f"Test Case {i+1}: ")
    print_int_matrix(matrix)
    print(f"The Length of longest Increasing Path is {longestIncreasingPath(matrix)} ")