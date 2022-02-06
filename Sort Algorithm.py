import random
def print_a_list(list):
    for item in list:
        print(item,end = " ")
    print()
list1 = list(range(10))
list2 = list(range(10))
random.shuffle(list1)
list_for_test = list1
print("Before Bubble Sort")
print_a_list(list_for_test)
def Bubble_Sort(list):
    for j in range(0,len(list)):
        for i in range(1,len(list)):
            if list[i]<list[i-1]:
                list[i],list[i-1]=list[i-1],list[i]
    return list
print("After Bubble Sort:")
print_a_list(Bubble_Sort(list_for_test))

def Insertion_Sort(list):
    for i in range(1,len(list)):
        key = list[i]
        j = i-1
        while j>=0 and key<list[j]:
            list[j+1] = list[j]
            j=j-1
        list[j+1] = key
    return list
# list_for_insertion_sort = list(range(10))
random.shuffle(list2)
print("Before Insertion Sort")
print_a_list(list2)
print("After Insertion Sort")
print_a_list(Insertion_Sort(list2))