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
print("---------------------704. Binary Search-------------------------")
def search(nums,target):
    left,right= 0,len(nums)-1
    while left<=right:
        mid = (left+right)//2
        if nums[mid]==target:
            return mid
        elif nums[mid]<target:
            left = mid+1
        else:
            right=mid-1
    return -1
test_case = [[[-1,0,3,5,9,12],9],[[-1,0,3,5,9,12],2]]
for nums,target in test_case:
    print(f"The {target} is the {search(nums,target)}th in the {nums}")
print("---------------------367. Valid Perfect Square-------------------------")
print("***** Method One: Binary Search *****")
def isPerfectSquare(num):
    left,right= 1,num
    while left<=right:
        mid = (left+right)//2
        if mid*mid==num:
            return True
        if mid*mid<num:
            left = mid+1
        else:
            right =mid-1
    return False
test_case = [16,14,2000105819]
for num in test_case:
    print(f"{num} is the perfect Square: {isPerfectSquare(num)}")
print("***** Method Two: Iteration *****")
def isPerfectSquare_iteration(num):
    for i in range(num+1):
        if i*i==num:
            return True
        if i*i>num:
            return False
for num in test_case:
    print(f"{num} is the perfect Square: {isPerfectSquare_iteration(num)}")
print("---------------------441. Arranging Coins-------------------------")
def arrangeCoins(n):
    left = 0
    right = n
    res = 0
    while left<=right:
        mid = (left+right)//2
        coins = (mid/2)*(mid+1)
        if coins>n:
            right = mid-1
        else:
            left = mid+1
            res = max(mid,res)
    return res
for i in range(1,9):
    print(f"For {i} conins, it could construct {arrangeCoins(i)} completed level stair(s)")
print("***** Method Two: Brute Force *****")
def arrangeCoins_brut_forct(n):
    level=1
    while True:
        coins = (level*(level+1))/2
        if coins==n:
            return level
        elif coins>n:
            return level-1
        else:
            level = level+1
    return level
for i in range(1,9):
    print(f"For {i} conins, it could construct {arrangeCoins_brut_forct(i)} completed level stair(s)")
print("---------------------374. Guess Number Higher or Lower-------------------------")
def guessNumber(n,pick):
    def guess(num):
        if num > pick:
            return -1
        elif num < pick:
            return 1
        else:
            return 0
    left,right = 1,n
    count = 0
    while True:
        mid=(left+right)//2
        result = guess(mid)
        if result<0:
            right = mid-1
        elif result>0:
            left = mid+1
        else:
            return mid,count
        count = count+1
test_case = [[10,6],[1,1],[2,1]]
for num,pick in test_case:
    print(f"For the given number {num}, to guess {pick},and the result is {guessNumber(num,pick)[0]} with {guessNumber(num,pick)[1]} times.")
print("---------------------658. Find K Closest Elements-------------------------")
print("***** Method One: Binary Search *****")
def findClosestElements(arr,k,x):
    left = 0
    right = len(arr)-k
    while left<right:
        mid = (left+right)//2
        if x-arr[mid]>arr[mid+k]-x:
            left = mid+1
        else:
            right = mid
    return arr[left:left+k]
test_case = [[[1,2,3,4,5],4,3],[[1,2,3,4,5], 4,-1]]
for arr,k,x in test_case:
    print(f"the {k} closest integers to {x} in the {arr} is {findClosestElements(arr,k,x)}")
print("***** Method Two: Binary Search 2 *****")
def findClosestElements_bs(arr,k,x):
    left = 0
    right = len(arr)-1
    # find index of x or the close val to x
    val,idx = arr[0],0
    while left<=right:
        m=(left+right)//2
        curDiff,resDiff = abs(arr[m]-x),abs(val-x)
        if (curDiff<resDiff) or (curDiff==resDiff and arr[m]<val):
            val,idx = arr[m],m
        if arr[m]<x:
            left = m+1
        elif arr[m]>x:
            right = m-1
        else:
            break
    l=r= idx
    for i in range(k-1):
        if l==0:
            r = r+1
        elif r==len(arr)-1 or x-arr[l-1]<=arr[r+1]-x:
            l = l-1
        else:
            r = r+1
    return arr[l:r+1]
for arr,k,x in test_case:
    print(f"the {k} closest integers to {x} in the {arr} is {findClosestElements_bs(arr,k,x)}")
print("---------------------981. Time Based Key-Value Store-------------------------")
class TimeMap():
    def __init__(self):
        self.store = {}
    def set(self,key,value,timestamp):
        if key not in self.store:
            self.store[key]=[]
        self.store[key].append([value,timestamp])
    def get(self,key,timestamp):
        res = ""
        values = self.store.get(key,[])
        left = 0
        right = len(values)-1
        while left<=right:
            mid = (left+right)//2
            if values[mid][1]<=timestamp:
                res = values[mid][0]
                left = mid+1
            elif values[mid][1]>timestamp:
                right = mid-1
        return res
obj = TimeMap()
obj.set("foo","bar",1)
print(obj.get("foo",1))
print(obj.get("foo",3))
obj.set("foo","bar2",4)
print(obj.get("foo",4))
print(obj.get("foo",5))