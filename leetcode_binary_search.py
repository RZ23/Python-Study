import math
def print_int_matrix(matrix):
    row = len(matrix)
    col =len(matrix[0])
    for i in range(row):
        for j in range(col):
            print(format(matrix[i][j],"02d"),end = " ")
        print()
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
print("---------------------153. Find Minimum in Rotated Sorted Array-------------------------")
def findMin(nums):
    left = 0
    right = len(nums)-1
    result = nums[0]
    while left<=right:
        # subarray is sorted
        if nums[left]<nums[right]:
            result =min(result,nums[left])
            break
        mid = (left+right)//2
        result = min(result,nums[mid])
        if nums[mid]>=nums[left]:
            left = mid+1
        else:
            right = mid-1
    return result
test_case = [[3,4,5,1,2],[4,5,6,7,0,1,2],[11,13,15,17]]
for nums in test_case:
    print(f"The minimum in {nums} is {findMin(nums)}")
print("---------------------74. Search a 2D Matrix-------------------------")
print("***** Method One: Two Binary Search, find row and search the col *****")
def searchMatrix(matrix,target):
    row = len(matrix)
    col = len(matrix[0])
    top = 0
    bottom = len(matrix)-1
    while top<=bottom:
        row = (top+bottom)//2
        # target > last item of this row
        #move to next row
        if target>matrix[row][-1]:
            top = row+1
        # target < first item of this row
        # move to last row
        elif target<matrix[row][0]:
            bottom = row-1
        # target > the first item and < the last item in this row
        # the target is in the range of this row
        else:
            break
    # target is out of the range
    if not (top<=bottom):
        return False
    row = (top+bottom)//2
    left = 0
    right = col-1
    while left<=right:
        mid = (left+right)//2
        if target<matrix[row][mid]:
            right = mid-1
        elif target>matrix[row][mid]:
            left = mid+1
        else:
            return True
    return False
test_case = [[[[1,3,5,7],[10,11,16,20],[23,30,34,60]],3],[[[1,3,5,7],[10,11,16,20],[23,30,34,60]],13],[
[[1],[3]],1]]
for matrix, target in test_case:
    print("For matrix:")
    print_int_matrix(matrix)
    print(f"It has the {target} in it:{searchMatrix(matrix,target)}")
print("***** Method Two: From top to bottom scan the row and binary search the selected row *****")
def searchMatrix_scan(matrix,target):
    if target<matrix[0][0] or target>matrix[-1][-1]:
        return False
    i=0
    row = 0
    while i <len(matrix):
        if target<=matrix[i][-1]:
            row=i
            break
        else:
            i=i+1
    left=0
    right = len(matrix[0])
    while left<=right:
        mid = (left+right)//2
        if matrix[row][mid]>target:
            right = mid-1
        elif matrix[row][mid]<target:
            left = mid+1
        else:
            return True
    return False
for matrix, target in test_case:
    print("For matrix:")
    print_int_matrix(matrix)
    print(f"It has the {target} in it:{searchMatrix_scan(matrix,target)}")
print("---------------------1898. Maximum Number of Removable Characters-------------------------")
print("***** Method One: Binary Search *****")
def maximumRemovals(s,p,removable):
    def isSubstring(str,substr):
        idx1,idx2 = 0,0
        while idx1<len(str) and idx2 <len(substr):
            if idx1 in removed or str[idx1]!=substr[idx2]:
                idx1 = idx1+1
                continue
            idx1 = idx1+1
            idx2 = idx2+1
        return idx2==len(substr)
    result = 0
    left = 0
    right = len(removable)-1
    while left<=right:
        mid = (left+right)//2
        removed = set(removable[:mid+1])
        if isSubstring(s,p):
            result = max(result,mid+1)
            left = mid+1
        else:
            right = mid-1
    return result
test_case = [["abcacb","ab",[3,1,0]],["abcbddddd","abcd",[3,2,1,4,5,6]],["abcab","abc",[0,1,2,3,4]]]
for s,p,removable in test_case:
    print(f"The maximum slice of removable in {removable} to keep {p} as the substring of {s} is {maximumRemovals(s,p,removable)}")
print("***** Method Two: Brute Force *****")
def maximumRemovals_brute_force(s,p,removable):
    result = 0
    def issubStr(str,substr):
        idx1,idx2 = 0,0
        while idx1<len(str) and idx2<len(substr):
            if str[idx1]!=substr[idx2]:
                idx1 = idx1+1
            else:
                idx1 = idx1+1
                idx2 = idx2+1
        if idx2==len(substr):
            return True
        else:
            return False
    def remove_char(str,remove_list):
        updated_str = ""
        for i in range(len(str)):
            if i not in remove_list:
                updated_str = updated_str+str[i]
        return updated_str
    for i in range(len(removable)):
        removable_list = removable[:i+1]
        if issubStr(remove_char(s,removable_list),p):
            result = max(result,i+1)
    return result
for s,p,removable in test_case:
    print(f"The maximum slice of removable in {removable} to keep {p} as the substring of {s} is {maximumRemovals_brute_force(s,p,removable)}")
print("---------------------875. Koko Eating Bananas-------------------------")
print("***** Method One: Binary Search *****")
def minEatingSpeed(piles, h):
    left =1
    right = max(piles)
    result = max(piles)
    while left<=right:
        k = (left+right)//2
        hours = 0
        for p in piles:
            hours = hours+math.ceil(p / k)
        if hours<=h:
            result = min(result,k)
            right = k-1
        else:
            left = k+1
    return result
test_case = [[[3,6,7,11],8],[[30,11,23,4,20],5],[[30,11,23,4,20],6]]
for piles, h in test_case:
    print(f"The minimum speed to eat bananas in {h} hours with piles {piles} is {minEatingSpeed(piles,h)}")
print("***** Method Two: Binary Search *****")
def minEatingSpeed_brute_force(piles, h):
    for i in range(1,max(piles)+1):
        hours = 0
        for p in piles:
            hours = hours + math.ceil(p/i)
        if hours<=h:
            return i
for piles, h in test_case:
    print(f"The minimum speed to eat bananas in {h} hours with piles {piles} is {minEatingSpeed_brute_force(piles, h)}")
print("---------------------34. Find First and Last Position of Element in Sorted Array-------------------------")
print("***** Method One: Binary Search,Find left boundary and then right boundary *****")
def searchRange(nums,target):
    # binary search to find the left value
    left_most =-1
    left = 0
    right = len(nums)-1
    while left<=right:
        mid = (left+right)//2
        if nums[mid]==target:
            left_most=mid
            right=mid-1
        elif nums[mid]<target:
            left = mid+1
        elif nums[mid]>target:
            right = mid-1
    if left_most==-1:
        return [-1,-1]
    left = left_most+1
    right = len(nums)-1
    right_most=left_most
    while left<=right:
        mid = (left+right)//2
        if nums[mid]>target:
            right = mid-1
        elif nums[mid]==target:
            right_most = mid
            left = mid+1
    return [left_most,right_most]
test_case = [[[5,7,7,8,8,10],8],[[5,7,7,8,8,10],6],[[],0],[[2,2],2]]
for nums, target in test_case:
    print(f"The start and end position for target {target} in {nums} is {searchRange(nums,target)} ")
print("***** Method Two: Binary Search,Find left boundary and then right boundary with helper function *****")
def searchRange_helper_function(nums,target):
    def find_left_most(nums,target):
        result=-1
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid]==target:
                result = mid
                right = mid-1
            elif nums[mid]>target:
                right = mid-1
            elif nums[mid]<target:
                left = mid+1
        # print(f"The left most target index is {result}")
        return result
    def find_right_most(nums,target):
        result = -1
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid]>target:
                right = mid-1
            elif nums[mid]<target:
                left = mid+1
            else:
                result = mid
                left = mid+1
        # print(f"The right most target index is {result}")
        return result
    left = find_left_most(nums,target)
    right = find_right_most(nums,target)
    return [left,right]
for nums, target in test_case:
    print(f"The start and end position for target {target} in {nums} is {searchRange_helper_function(nums,target)} ")
print("***** Method Three: Binary Search,with one helper function *****")
def searchRange_one_helper_function(nums,target):
    def find_most(nums,target,flag):
        result = -1
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid]<target:
                left = mid+1
            elif nums[mid]>target:
                right = mid-1
            else:
                result = mid
                # find the left most
                if flag:
                    right=mid-1
                # find the right most
                else:
                    left = mid+1
        return result
    left = find_most(nums,target,True)
    right =find_most(nums,target,False)
    return [left,right]
for nums, target in test_case:
    print(f"The start and end position for target {target} in {nums} is {searchRange_one_helper_function(nums,target)} ")

print("---------------------4. Median of Two Sorted Arrays-------------------------")
print("***** Method One: Binary Search *****")
def findMedianSortedArrays(nums1,nums2):
    A,B=nums1,nums2
    # keep B the short array
    if len(A)>len(B):
        A,B=B,A
    total = len(nums1)+len(nums2)
    half = total//2
    left = -1
    right = len(A)
    while True:
        # find the middle of the short array
        short_mid = (left+right)//2
        # based on the value of short array
        # get the middle of the long array
        long_mid = half-short_mid-2
        Aleft = A[short_mid] if short_mid>=0 else float("-infinity")
        Aright = A[short_mid+1] if short_mid+1<len(A) else float("infinity")
        Bleft = B[long_mid] if long_mid>=0 else float("-infinity")
        Bright = B[long_mid+1] if long_mid+1<len(B) else float("infinity")
        # find the middle item(s), check the odd/even of the combined arrays
        if Aleft<=Bright and Bleft<=Aright:
            if total%2:
                return min(Aright,Bright)
            else:
                return (max(Aleft,Bleft)+min(Aright,Bright))/2
        elif Aleft>Bright:
            right = short_mid-1
        elif Aright<Bleft:
            left = short_mid+1
test_case = [[[1,3],[2]],[[1,2],[3,4]],[[],[1]]]
for nums1,nums2 in test_case:
    print(f"The median of {nums1} and {nums2} is {findMedianSortedArrays(nums1,nums2)}")
print("---------------------35. Search Insert Position-------------------------")
def searchInsert(nums,target):
    left = 0
    right = len(nums)-1
    while left<=right:
        mid = (left+right)//2
        if nums[mid]==target:
            return mid
        elif nums[mid]>target:
            right = mid-1
        else:
            left = mid+1
    return left
test_case = [[[1,3,5,6],5],[[1,3,5,6],2],[[1,3,5,6],7]]
for nums,target in test_case:
    print(f"The index of {target} in {nums} is {searchInsert(nums,target)}")
print("---------------------33. Search in Rotated Sorted Array-------------------------")
print("***** Method One: Binary Search Find the pivot and then find the target")
def search(nums,target):
    # find the pivot
    left = 0
    right = len(nums)-1
    while left<right:
        mid = (left+right)//2
        if nums[mid]<nums[right]:
            right=mid
        elif nums[mid]>nums[right]:
            left = mid+1
    pivot=left
    # print(pivot)
    if target>nums[-1]:
        left = 0
        right = pivot-1
        while left<=right:
            mid = (left+right)
            if target==nums[mid]:
                return mid
            elif target>nums[mid]:
                left = mid+1
            else:
                right = mid-1
        return -1
    elif target<=nums[-1]:
        left = pivot
        right = len(nums)
        while left<=right:
            mid = (left+right)//2
            if target ==nums[mid]:
                return mid
            elif target>nums[mid]:
                left = mid+1
            else:
                right = mid-1
        return -1
test_case = [[[4,5,6,7,0,1,2],0],[[4,5,6,7,0,1,2],3],[[1],0],[[3,1],3]]
for nums,target in test_case:
    print(f"the target {target} is {nums} is {search(nums,target)}")
print("***** Method Two: build-in function find the pivot and then find the target")
def search_with_build_in(nums,target):
    if target<min(nums)or target>max(nums):
        return -1
    else:
        pivot = nums.index(min(nums))
        if target > nums[-1]:
            left = 0
            right = pivot - 1
            while left <= right:
                mid = (left + right)
                if target == nums[mid]:
                    return mid
                elif target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        elif target <= nums[-1]:
            left = pivot
            right = len(nums)
            while left <= right:
                mid = (left + right) // 2
                if target == nums[mid]:
                    return mid
                elif target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
for nums,target in test_case:
    print(f"the target {target} is {nums} is {search_with_build_in(nums,target)}")