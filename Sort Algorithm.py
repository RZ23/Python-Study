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
# print("Heap Sort")
random.shuffle(list1)
print("Before Heap List (Recurssive)")
print_a_list(list1)
# create the max heap
def heapfiy_Recurssive(list,n,i):
    largest = i
    left = 2*i+1
    right = 2*i+2
    if left<n and list[i]<list[left]:
        largest = left
    if right<n and list[largest]<list[right]:
        largest = right
    if largest!=i:
        list[i],list[largest] = list[largest],list[i]
        heapfiy_Recurssive(list,n,largest)
def heap_sort_recurssive(list):
    n = len(list)
    for i in range(n//2-1,-1,-1):
        heapfiy_Recurssive(list,n,i)
    for i in range(n-1,0,-1):
        list[i],list[0]=list[0],list[i]
        heapfiy_Recurssive(list,i,0)
    return list
print("After Heap Sort:")
list1 = heap_sort_recurssive(list1)
print_a_list(list1)
print("Before Heap Sort (Iteration)")
random.shuffle(list4)
print_a_list(list4)
# create max heap
def sifdown(lst,i,upper):
    while (True):
        left,right = i*2+1,i*2+2
        # both in the range
        if max(left,right)<upper:
            if lst[i]>max(lst[left],lst[right]):
                break
            elif lst[left]<lst[right]:
                lst[right],lst[i] = lst[i],lst[right]
                i = right
            else:
                lst[left],lst[i]=lst[i],lst[left]
                i=left
        elif left <upper:
            if lst[left]>lst[i]:
                lst[left],lst[i]=lst[i],lst[left]
                i=left
            else:
                break
        elif right<upper:
            if lst[right]<lst[i]:
                lst[right],lst[i]=lst[i],lst[right]
                i = right
            else: break
        else:
            break
def heap_sort(lst):
    for j in range((len(lst)-2)//2,-1,-1):
        # create the heap
        sifdown(lst,j,len(lst))
    for end in range(len(lst)-1,0,-1):
        lst[0],lst[end] = lst[end],lst[0]
        sifdown(lst,0,end)
    return lst
list4 = heap_sort(list4)
print("After Heap Sort")
print_a_list(heap_sort(list4))
# For the Heap, Left Child = 2*i+1
#               Right Child = 2*i+2
#               Parent = (i-1)/2
print("Create the Min Heap Class <the min number is on the top>")
class MinHeap():
    def __init__ (self,lst):
        # def __init__(self, arc=None):
        # self.heap = lst.copy()
        # self.heap = []
        # if type(arc) is list:
        #     self.heap = arc.copy()
    # for i in range(len(self.heap))[::-1]:
    #     self._siftdown(i)
        self.heap = lst
        # use len(list-2)//2 to get the first parent node
        # right child is 2*i+2, so the parent node is (i-2)/2
        for i in range((len(self.heap)-2)//2)[::-1]:
            self._siftdown(i)

    def _siftup(self,i):
        parent = (i-1)//2
        while i!=0 and self.heap[i]<self.heap[parent]:
            self.heap[i],self.heap[parent] = self.heap[parent],self.heap[i]
            i=parent
            parent = (i-1)//2
    def _siftdown(self,i):
        left = 2*i+1
        right = 2*i+2
        while (left<len(self.heap) and self.heap[i]>self.heap[left]) or (right<len(self.heap) and self.heap[i]>self.heap[right]):
            smallest = left if (right>=len(self.heap) or self.heap[left]<self.heap[right]) else right
            self.heap[i],self.heap[smallest]=self.heap[smallest],self.heap[i]
            i =smallest
            left = 2*i+1
            right = 2*i+2
    def insert(self,element):
        self.heap.append(element)
        self._sifftup(len(self.heap)-1)
    def get_min(self):
        return self.heap[0] if len(self.heap)>0 else None
    def extract_min(self):
        if len(self.heap)==0:
            return None
        minval = self.heap[0]
        self.heap[0],self.heap[-1] = self.heap[-1],self.heap[0]
        self.heap.pop()
        self._siftdown(0)
        return minval
    def update_by_index(self,i,new):
        old = self.heap[i]
        self.heap[i]=new
        if new>old:
            self._siftup(i)
        else:
            self._siftdown(i)
    def update(self,old,new):
        if old in self.heap:
            self.update_by_index(self.index(old),new)
def heap_sort_with_heap_class(lst):
    heap = MinHeap(lst)
    return [heap.extract_min() for i in range(len(heap.heap))]
random.shuffle(list4)
list4 = list4
print_a_list(list4)
print_a_list(heap_sort_with_heap_class(list4))


