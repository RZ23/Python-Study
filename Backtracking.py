import collections
import heapq
from collections import deque
from collections import Counter
import sys
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
print("---------------------1239. Maximum Length of a Concatenated String with Unique Characters-------------------------")
def findDifferentBinaryString(nums):
    strSet = { s for s in nums}
    def backtracking(i,cur):
        if i==len(nums):
            res="".join(cur)
            return None if res in strSet else res
        res = backtracking(i+1,cur)
        if res:return res
        cur[i]="1"
        res= backtracking(i+1,cur)
        if res:return res
    return backtracking(0,["0" for s in nums])
test_case = [["01","10"],["00","01"],["111","011","001"]]
for nums in test_case:
    print(f"The Different Binary String without {nums} is {findDifferentBinaryString(nums)}")
print("---------------------1239. Maximum Length of a Concatenated String with Unique Characters-------------------------")
def maxLength(arr):
    charSet = set()
    # check if the character is already in the string
    def overlap(charSet,s):
        c = Counter(charSet)+Counter(s)
        return max(c.values())>1
    def backtracking(i):
        if i==len(arr):
            return len(charSet)
        res = 0
        # if the characters in arr[i] not already in charSet
        if not overlap(charSet,arr[i]):
            # add all the characters in arr[i] to charSet
            for c in arr[i]:
                charSet.add(c)
            # move to next item in the arr list
            res = backtracking(i+1)
            for c in arr[i]:
                charSet.remove(c)
        return max(res,backtracking(i+1))
    return backtracking(0)
test_case = [["un","iq","ue"],["cha","r","act","ers"],["abcdefghijklmnopqrstuvwxyz"]]
for arr in test_case:
    print(f"The maximum length of Concatenated String with Unique Characters ins '{arr}' is {maxLength(arr)}")
print("---------------------93. Restore IP Addresses-------------------------")
def restoreIpAddresses(s):
    result = []
    if len(s)>12:
        return result
    def backtracking(i,dots,curIP):
        # 4 dots means meet the requirement of the IP address
        # and variable i equals the length of s means scan all the characters
        # so append it to result, except the last character in the curIP since it is "."
        if dots==4 and i==len(s):
            result.append(curIP[:-1])
            return
        # if dots greater than 4, means it exceeds the requirement of the IP address
        # then return nothing (back)
        if dots>4:
            return
        # start the backtracking, each loop is 3 characters or if it exceeds range, so
        #using the min(i+3,len(s))
        for j in range(i,min(i+3,len(s))):
            # make sure the numer is less than 255 or except the 0, there is no 0 leading numbers
            if int(s[i:j+1])<=255 and (i==j or s[i]!="0"):
                backtracking(j+1,dots+1,curIP+s[i:j+1]+".")
    backtracking(0,0,"")
    return result
test_case = ["25525511135","0000","101023"]
for s in test_case:
    print(f"The all possible valid IP addresses that can be formed by inserting dots into '{s}' are {restoreIpAddresses(s)}")

print("---------------------698. Partition to K Equal Sum Subsets-------------------------")
def canPartitionKSubsets(nums,k):
    used = [False] * len(nums)
    total_sum=sum(nums)
    # if the total sum cannot be divide by k, means it cannot be partitioned to
    # k equal sum subsets
    if total_sum%k!=0:
        return False
    target = sum(nums) // k
    # sorting the array in descending order, if the first value is greater than target
    # it will not be included in any subset, so will return False
    nums.sort(reverse=True)
    if nums[0]>target:
        return False

    # using dynamic programming to store the previous calculated paired value
    dp ={}
    def backtracking(i,k,rem):
        if tuple(used) in dp:
            return dp[tuple(used)]
        # k==0, means there are k equal subsets
        if k==0:
            return True
        # rem =0, means find one qualified subset
        # from the start point to find the rest subset
        # set k to k-1 and rem tp target
        if rem==0:
            partition = backtracking(0,k-1,target)
            dp[(tuple(used),rem)]=partition
            return partition
        # loop for each item in the array
        for j in range(i,len(nums)):
            # if the nums[j] not used and nums[j] less or equals to current rem
            if not used[j] and rem-nums[j]>=0:
                used[j] = True
                if backtracking(j+1,k,rem-nums[j]):
                    return True
                # if not qualify, reset the nums[j] to unused state
                used[j] = False
        # after each loop, if not qualify, memorize it to False
        dp[tuple(used)]=False
        return False
    return backtracking(0,k,target)
test_case =[[[4,3,2,3,5,2,1], 4],[[1,2,3,4], 3]]
for nums,k in test_case:
    print(f"The array {nums} can be divide to {k} equal sum subsets: {canPartitionKSubsets(nums,k)}")
print("---------------------473. Matchsticks to Square-------------------------")
print("***** Methond One: Backtracking without Dynamic Programming *****")
def makesquare(matchsticks):
    length = sum(matchsticks)//4
    sides = [0]*4
    if sum(matchsticks)%4!=0:
        return False
    matchsticks.sort(reverse = True)
    if matchsticks[0]>length:
        return False
    def backtracking(i):
        # scan all the matchsticks
        if i==len(matchsticks):
            return True
        # if not, the main body of the backtracking
        # calculate the length of each side
        for j in range(4):
            if sides[j]+matchsticks[i]<=length:
                sides[j] = sides[j]+matchsticks[i]
                if backtracking(i+1):
                    return True
                # if not qualify, must return back
                sides[j] = sides[j]-matchsticks[i]
        return False
    return backtracking(0)
test_case = [[1,1,2,2,2],[3,3,3,3,4]]
for matchsticks in test_case:
    print(f"The matchsticks {matchsticks} could make a square: {makesquare(matchsticks)}")
print("***** Methond One: Backtracking with Dynamic Programming *****")
def makesquare_with_dp(matchsticks):
    used = [False] * len(matchsticks)
    total_sum = sum(matchsticks)
    # if the total sum cannot be divide by 4, means it cannot be a square
    if total_sum % 4 != 0:
        return False
    target = sum(matchsticks) // 4
    # sorting the array in descending order, if the first value is greater than target
    # it will not be included in any subset, so will return False
    matchsticks.sort(reverse=True)
    if matchsticks[0] > target:
        return False
    # using dynamic programming to store the previous calculated paired value
    dp = {}
    def backtracking(i, k, rem):
        if tuple(used) in dp:
            return dp[tuple(used)]
        # k==0, means finish all the 4 sides
        if k == 0:
            return True
        # rem =0, means find one side is qualified
        # from the start point to find the rest subset
        # set k to k-1 and rem tp target
        if rem == 0:
            partition = backtracking(0, k - 1, target)
            dp[(tuple(used), rem)] = partition
            return partition
        # loop for each item in the array
        for j in range(i, len(matchsticks)):
            # if the nums[j] not used and nums[j] less or equals to current rem
            if not used[j] and rem - matchsticks[j] >= 0:
                used[j] = True
                if backtracking(j + 1, k, rem - matchsticks[j]):
                    return True
                # if not qualify, reset the nums[j] to unused state
                used[j] = False
        # after each loop, if not qualify, memorize it to False
        dp[tuple(used)] = False
        return False
    return backtracking(0, 4, target)
for matchsticks in test_case:
    print(f"The matchsticks {matchsticks} could make a square: {makesquare_with_dp(matchsticks)}")
print("---------------------90. Subsets II-------------------------")
def subsetsWithDup(nums):
    result = []
    nums.sort()
    def backtracking(i,subset):
        # basic condition, index i is equals to the length of the list
        # add the subset tot final result, since the subset is the global
        # variable, must add the copy of the subset
        if i==len(nums):
            result.append(subset.copy())
            return
        '''
        include the nums[i] into the subset
        '''
        subset.append(nums[i])
        # backtracking next item in the nums
        backtracking(i+1,subset)
        # After the backtracking, must remove the nums[i] from the subset
        subset.pop()

        '''
        not include the nums[i]
        '''
        # check if there is duplicated item in the list, if so, skip it
        while i+1<len(nums) and nums[i+1]==nums[i]:
            i=i+1
        # the first unused/unduplicated item
        backtracking(i+1,subset)
    backtracking(0,[])
    return result
test_case = [[1,2,2],[0]]
for nums in test_case:
    print(f"The all possible not duplicated subsets of {nums} is {subsetsWithDup(nums)}")



