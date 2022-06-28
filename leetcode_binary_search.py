print("---------------------410. Split Array Largest Sum-------------------------")
print("***** Method One: Binary Search")
"""
to split the array, 
the minimum value is the max(nums), it means there is only one item in this subarray
the maximum value is the sum(nums), it means all the items are in the subarray
so the range should be in [max(nums),sum(nums)]
using binary search, find the mid value, 
    if the array could be split with m or less than m subarrays and the largest sum is less or equal then mid value
    then means the mid value is too large, and move the right boundary from sum(nums) to mid-1
    otherwise, means the mid value is too small, and move the left boundary from max(nums) to mid+1
until l==r, means find the desired result and return it
"""
def splitArray(nums,m):
    def canSplit(largest):
        subarray = 0
        curSum = 0
        for n in nums:
            curSum = curSum+n
            if curSum>largest:
                subarray = subarray+1
                curSum = n
        return subarray+1<=m
    l =max(nums)
    r = sum(nums)
    res = r
    while l<=r:
        mid = (l+r)//2
        if canSplit(mid):
            res = mid
            r=mid-1
        else:
            l = mid+1
    return res
test_case = [[[7,2,5,10,8],2],[[1,2,3,4,5],2],[[1,4,4],3]]
for nums,m in test_case:
    print(f"Minimize the largest sum among these {m} subarrays for {nums} is {splitArray(nums,m)} ")
print("***** Method Two: Iteration and Dynamic Programming *****")
def splitArray_dp(nums,m):
    dp = {}
    def dfs(i,m):
        if m==1:
            return sum(nums[i:])
        if (i,m) in dp:
            return dp[(i,m)]
        result = float("inf")
        curSum=0
        for j in range(i,len(nums)-m+1):
            curSum = curSum+nums[j]
            maxSum = max(curSum,dfs(j+1,m-1))
            result = min(result,maxSum)
            if curSum>result:
                break
        dp[(i,m)] = result
        return result
    return dfs(0,m)
for nums,m in test_case:
    print(f"Minimize the largest sum among these {m} subarrays for {nums} is {splitArray_dp(nums,m)} ")