import random
from binarytree import tree, Node

import binarytree
import math
# print("--------------String Reversal--------------")
# def string_reversal(string):
#     if len(string)==0:
#         return ""
#     else:
#         return string[-1]+string_reversal(string[:len(string)-1])
# string = input("Please input the string for the reversal: ")
# print("Recursive String Reversal string {} is {}:".format(string,string_reversal(string)))
#
# def string_reversal_iteration(string):
#     s = ""
#     for i in range(1,len(string)+1):
#         s=s+string[-i]
#     return s
# print("Iteration String Reversal String {} is {}: ".format(string,string_reversal_iteration(string)))
#
# print("--------------Number Reversal--------------")
# i = int(input("please input the number for reversal:"))
# print("Iteration: ")
# def number_reversal_iteration(i):
#     digits = len(str(i))
#     t= 0
#     s = 0
#     for index in range (0,digits)[::-1]:
#         s=int(i/pow(10,index))
#         i=int(i%pow(10,index))
#         t = t+s*pow(10,(digits-index-1))
#     return t
# print(number_reversal_iteration(i))
# print("Recursive:")
# def number_reversal_recursive(i):
#     digits = len(str(i))
#     if i/10==0:
#         return i
#     else:
#         mod = i%10
#         result = int(i/10)
#         return mod*pow(10,digits-1)+number_reversal_recursive(result)
# print(number_reversal_recursive(i))

# print("--------------Palindrome--------------")
# def palindrome(string):
#     if len(string)<=1:
#         return True
#     else:
#         if string[0]==string[-1]:
#             return palindrome(string[1:-1])
#     return False
# string = input("Please input the string for Palindorm check: ")
# print("{} is palindrome: {}".format(string,palindrome(string)))

# print("--------------Decimal to Binary--------------")
# decimal = int(input("Please input the number to convert to binary:"))
# print("Using the building fuunction to convert {} to binary is {}".format(decimal,bin(decimal)))
# print("Iteration:")
# def decimal_to_binary_iteration(i):
#     s = ""
#     while i!=0:
#         t = i%2
#         s =str(t)+s
#         i = i//2
#     return s
# print("Conert decimal {} to binary is {}".format(decimal,decimal_to_binary_iteration(decimal)))
#
#
# print("Recursive:")
# def decimal_to_binary_recursive(i,result=None):
#     if result is None:
#         result = ""
#     if i==0:
#         return result
#     result = str(i%2)+result
#     return decimal_to_binary_recursive(i//2,result)
# print("Conert decimal {} to binary is {}".format(decimal,decimal_to_binary_recursive(decimal)))

print("--------------Reveral Linked List--------------")
class List_Node():
    def __init__(self,value):
        self.value = value
        self.next = None
def print_linked_list(node):
    while node is not None:
        print(node.value,end = ",")
        node = node.next
node1 = List_Node(1)
node2 = List_Node(2)
node3 = List_Node(3)
node4= List_Node(4)
node5 = List_Node(5)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5
print("Original Linked List:")
print_linked_list(node1)
print()
def reversal_linked_list_recursice(node):
    if node is None or node.next is None:
        return node
    p=reversal_linked_list_recursice(node.next)
    node.next.next = node
    node.next = None
    return p
reversed = reversal_linked_list_recursice(node1)
print("Reversed Linked List:")
print_linked_list(reversed)

# def reversed_linked_list_2_parameters(node,prev = None):
#     if node is None:
#         return prev
#     next_node = node.next
#     node.next = prev
#     return reversed_linked_list_2_parameters(next_node,node)
# print()
# print_linked_list(reversed_linked_list(node5))
print()
print("--------------Merge Sorted Linked List --------------")
merge_node1 = List_Node(1)
merge_node2 = List_Node(8)
merge_node3 = List_Node(22)
merge_node4 = List_Node(40)
merge_node5 = List_Node(4)
merge_node6 = List_Node(11)
merge_node7 = List_Node(16)
merge_node8 = List_Node(20)
merge_node1.next = merge_node2
merge_node2.next=merge_node3
merge_node3.next=merge_node4
merge_node5.next = merge_node6
merge_node6.next = merge_node7
merge_node7.next = merge_node8
def merge_sorted_link(l1,l2):
    if l1 is None:
        return l2
    if l2 is None:
        return l1
    if l1.value<=l2.value:
        l1.next = merge_sorted_link(l1.next,l2)
        return l1
    else:
        l2.next = merge_sorted_link(l1,l2.next)
        return l2
print_linked_list(merge_sorted_link(merge_node1,merge_node5))
# print("--------------Sum of Natural Numbers --------------")
# number_length = int(input("Please input the last number for sum: "))
# def number_sum_recursive(num):
#     if num==0:
#         return 0
#     else:
#         return num+number_sum_recursive(num-1)
# print("The sum of {} natural numbers is/are {}".format(number_length,number_sum_recursive(number_length)))
# def number_sum_iteration(num):
#     sum=0
#     while num>0:
#         sum = num+sum
#         num=num-1
#     return num
# print("The sum of {} natural numbers is/are {}".format(number_length,number_sum_recursive(number_length)))
# print("--------------Binary Search --------------")
# binary_search_list_length = int(input("Please input the size of binary search list:"))
# binary_search_list = [random.randint(1,100) for i in range(binary_search_list_length)]
# sorted_binary_search_list = sorted(binary_search_list)
# print(sorted_binary_search_list)
# def binary_search(list,target,low ,high):
#     if low>high:
#         return -1
#     else:
#         mid = (low+high)//2
#         if list[mid]==target:
#             return mid
#         elif list[mid]>target:
#             return binary_search(list,target,low,mid-1)
#         else:
#             return binary_search(list,target,mid+1,high)
# target = int(input("Please input the target for the search:"))
# print("The index of target {} in the list is {}".format(target,binary_search(sorted_binary_search_list,target,0,len(sorted_binary_search_list)-1)))
# print("---------------Fibonacci Sequence---------------")
# def fib(n):
#     if n==1 or n==0:
#         return n
#     else:
#         return fib(n-1)+fib(n-2)
# fibonacci_size = int(input("Please input the nubmer to calculate Fibonacci:"))
# print("the {} if fibonacci is {}".format(fibonacci_size,fib(fibonacci_size)))
# print("Calculate Fibonacci Sequence with memorization")
# dict ={}
# def fib_with_mem(n,dict):
#     if n in dict.keys():
#         return dict["n"]
#     if n==1 or n==0:
#         return n
#     else:
#         value = fib(n-1)+fib(n-2)
#         dict["n"] = value
#         return value
# print("the {} if fibonacci is {}".format(fibonacci_size,fib_with_mem(fibonacci_size,dict)))
# print("---------------generate random list---------------")
# random_list = []
# length = int(input("Please input the length of the list:"))
# i = 0
# while i <length:
#     temp = random.randint(1,100)
#     if temp not in random_list:
#         random_list.append(temp)
#         i=i+1
# random_list=sorted(random_list)
# print(random_list)
#
# class tree_node(Node):
#     def __init__(self,value =0):
#         self.value = value
#         self.left = None
#         self.right = None
#     def add_value(self,value):
#         self.value = value
# tree_list = []
# for i in range(len(random_list)):
#     tree_list.append(tree_node())
# for i in range(len(tree_list)):
#     tree_list[i].add_value(random_list[i])
# # for i in range(len(tree_list)):
# #     print(tree_list[i].value,end =",")
#
# def create_bst(list):
#     if len(list)==1:
#         return tree_node(list[0])
#     if len(list)==0:
#         return None
#     low,high = 0,len(list)-1
#     mid = (low+high)//2
#     node = tree_node(list[mid])
#     node.left = create_bst(list[:mid])
#     node.right = create_bst(list[mid+1:])
#     return node
# root = create_bst(random_list)
# print(root)