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

print("--------------Decimal to Binary--------------")
decimal = int(input("Please input the number to convert to binary:"))
print("Using the building fuunction to convert {} to binary is {}".format(decimal,bin(decimal)))
print("Iteration:")
def decimal_to_binary_iteration(i):
    s = ""
    while i!=0:
        t = i%2
        s =str(t)+s
        i = i//2
    return s
print("Conert decimal {} to binary is {}".format(decimal,decimal_to_binary_iteration(decimal)))


print("Recursive:")
def decimal_to_binary_recursive(i,result=None):
    if result is None:
        result = ""
    if i==0:
        return result
    result = str(i%2)+result
    return decimal_to_binary_recursive(i//2,result)
print("Conert decimal {} to binary is {}".format(decimal,decimal_to_binary_recursive(decimal)))

