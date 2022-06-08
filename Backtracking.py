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