import random
def print_a_list(list):
    for item in list:
        print(item,end = " ")
    print()
list1 = list(range(20))
list2 = list(range(20))
list3 = list(range(20))
list4 = list(range(20))
random.shuffle(list1)
list_for_test = list1
# Selection Sort: Find the index of the smallest item for the rest of the list and exchange with the current item
# list has two sections, sorted and unsorted.find the smallest item and keep its index, exchange it with the current item
print("Selection Sort")
def selection_sort(list):
    for i in range(len(list)-1):
        min_index=i
        min_value = list[i]
        for j in range(i+1,len(list)):
            if list[j]<min_value:
                min_value=list[j]
                min_index = j
        if min_index!=i:
            list[i],list[min_index]=list[min_index],list[i]
    return list
print("Before Selection Sort:")
print_a_list(list_for_test)
print("After Selection Sort:")
print_a_list(selection_sort(list_for_test))
print("Before Bubble Sort")
random.shuffle(list1)
list_for_test=list1
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
        #move the element of list[0...i-1],
        #that are greater than key, to one position
        #ahead of their current position
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
random.shuffle(list3)
print("Before Merger Sort")
print_a_list(list3)
print("After Merge Sort")
def Merge_Sort(list):
    if len(list)<=1:
        return list
    mid = len(list)//2
    left = list[:mid]
    right = list[mid:]
    left_sorted = Merge_Sort(left)
    right_sorted = Merge_Sort(right)
    sorted_list= Merge(left_sorted,right_sorted)
    return sorted_list
def Merge(l_list,r_list):
    merge = []
    i=0
    j=0
    while i<len(l_list) and j<len(r_list):
        if l_list[i]<=r_list[j]:
            merge.append(l_list[i])
            i=i+1
        else:
            merge.append(r_list[j])
            j=j+1

    l_list_left = l_list[i:]
    r_list_left = r_list[j:]

    return merge+l_list_left+r_list_left
print_a_list(Merge_Sort(list3))
print("Start Quick Sort")
random.shuffle(list4)
print_a_list(list4)

def QuickSort(list,low = 0, high = None):
    if high is None:
        high = len(list)-1
    if len(list)==1:
        return list
    if low <high:
        # pi is the partitioning index, list[pi] is now at the right place
        pi = Partition(list,low,high)
        # separately sort the elements before and after the partition
        QuickSort(list,low,pi-1)
        QuickSort(list,pi+1,high)
    return list
def Partition(list,low,high):
    #index of smaller element
    i = (low-1)
    # set the right element as the pivot element
    pivot = list[high]
    for j in range(low,high):
        # if the current element is smaller or equal to the pivot element
        if list[j]<=pivot:
        # increament the smaller index
            i= i+1
        # since the smaller index +1 (i = i+1), swap the elements with the
        # current smaller element list[j] and the first larger
        # element list[i]
            list[i],list[j] = list[j],list[i]
    list[i+1],list[high] = list[high],list[i+1]
    return (i+1)
print("After the Quick Sort")
print_a_list(QuickSort(list4))