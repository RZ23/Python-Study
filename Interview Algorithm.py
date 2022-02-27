import math
import random
import collections
from collections import Counter
print("------------------Merge Sort-----------------------------")
lst = [random.randint(1,20) for x in range(10)]
print("Before Sorting")
print(lst)
def Merge_Sort(lst):
    if len(lst)<=1:
        return lst
    mid = len(lst)//2
    left_list = lst[:mid]
    right_list = lst[mid:]
    left = Merge_Sort(left_list)
    right = Merge_Sort(right_list)
    sorted_lst = merge(left,right)
    return sorted_lst
def merge(left,right):
    sorted_lst = []
    i=0
    j=0
    while i<len(left) and j<len(right):
        if left[i]<right[j]:
           sorted_lst.append(left[i])
           i=i+1
        else:
            sorted_lst.append(right[j])
            j=j+1
        left_left = left[i:]
        right_left = right[j:]
    return sorted_lst+left_left+right_left
print("After Sorting")
print(Merge_Sort(lst))
print("------------------Quick Sort-----------------------------")
lst1 = [random.randint(1,20) for x in range (10)]
print("Before Sorting")
print(lst1)
def Quick_Sort(lst):
    if len(lst)<=1:
        return lst
    pivot = lst[-1]
    left = []
    right = []
    for i in range(len(lst)-1):
        if lst[i]<=pivot:
            left.append(lst[i])
        else:
            right.append(lst[i])
    return Quick_Sort(left)+[pivot]+Quick_Sort(right)
print("After Sorting")
print(Quick_Sort(lst1))
print("------------------Valid Anagram-----------------------------")
print("------------------'Danger' and 'Garden' => True-------------")
def valid_anagrams(s1,s2):
    se1_dict={}
    se2_dict={}
    for item in s1:
        if item not in se1_dict:
            se1_dict[item]=1
        else:
            se1_dict[item] = se1_dict[item]+1
    for cha in s2:
        if cha not in se2_dict:
            se2_dict[cha]=1
        else:
            se2_dict[cha] = se2_dict[cha]+1
    # print(se1_dict,se2_dict)
    for item in se1_dict.keys():
        if item not in se2_dict:
            return False
        elif se1_dict[item]!=se2_dict[item]:
            return False
    return True
s1= "nameless"
s2 = "salesmen"
s3="Hello"
s4="Wolle"
print("Using Directionary Map")
print("{} and {} is anagram: {}".format(s1,s2,valid_anagrams(s1,s2)))
print("{} and {} is anagram: {}".format(s3,s4,valid_anagrams(s3,s4)))
def valid_anagram_sorted(s1,s2):
    sorted_s1 = sorted(s1)
    sorted_s2 = sorted(s2)
    if len(sorted_s1)!=len(sorted_s2):
        return False
    return sorted_s1 == sorted_s2
print("Using the Sorting")
print("{} and {} is anagram: {}".format(s1,s2,valid_anagram_sorted(s1,s2)))
print("{} and {} is anagram: {}".format(s3,s4,valid_anagram_sorted(s3,s4)))

def valid_anaram_count(s1,s2):
    print("Count of {} is {}, and Counter of {} is {}".format(s1,Counter(s1),s2,Counter(s2)) )
    size_1 = len(s1)
    size_2= len(s2)
    if size_1!=size_2:
        return False
    return Counter(s1)==Counter(s2)
print("Using the Collection Counter")
print("{} and {} is anagram: {}".format(s1,s2,valid_anaram_count(s1,s2)))
print("{} and {} is anagram: {}".format(s3,s4,valid_anaram_count(s3,s4)))
print("------------------First and Last Position-----------------------------")
def First_Last_Position(sequence,target):
    start=0
    for i in range(len(sequence)):
        if sequence[i]==target:
            start=i
            while sequence[i+1]==target and i<len(sequence):
                i = i+1
            return [start,i]
    return [-1,-1]
s =[2,4,5,5,5,5,5,7,9,9]
target = 5
print("Linear Search:")
print("The sequence is {} and the target is {}".format(s,target))
print("The First and End Position of the target {} is {}".format(target,First_Last_Position(s,target)))
print("Binary Search")
def First_and_End_Biary(s,target):
    if (len(s)==0) or (s[0]>target) or (s[len(s)-1]<target):
        return (-1,-1)
    return find_start(s,target),find_end(s,target)
def find_start(s,target):
    if s[0]==target:
        return 0
    low,high =0,len(s)-1
    while low<=high:
        mid = (low+high)//2
        if s[mid]==target and s[mid-1]<target:
            return mid
        elif s[mid]<target:
            low = mid+1
        else:
            high = mid-1
    return -1
def find_end(s,target):
    if s[0]==target:
        return 0
    low,high = 0,len(s)-1
    while low<=high:
        mid = (low+high)//2
        if s[mid]==target and s[mid+1]>target:
            return mid
        elif s[mid]>target:
            high = mid-1
        else:
            low = mid+1
    return -1

print("Linear Search:")
print("The sequence is {} and the target is {}".format(s,target))
print("The First and End Position of the target {} is {}".format(target,First_and_End_Biary(s,target)))
print("------------------Kth Largest Element-----------------------------")
def Kth_Element(s,k):
    for i in range(k-1):
        s.remove(max(s))
    return max(s)
s=[4,2,9,7,5,6,7,1,3]
s1 = s.copy()
k=4
print("Without Sorting")
print("the list is {} and the {}th larget element is {}".format(s1,k,Kth_Element(s,k)))
print("With Sorting")
def Kth_Element_Sorting(s,k):
    return sorted(s)[-k]
print("the list is {} and the {}th larget element is {}".format(s1,k,Kth_Element_Sorting(s1,k)))