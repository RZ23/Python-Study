from collections import deque
from collections import defaultdict
# add value to dict
def add_to_dict(dict,key,value):
    if key in dict:

        dict[key].append(value)
    else:
        dict[key]=list(value)
dict = {}
key1 = "A"
key2 = "B"

add_to_dict(dict,key1,"a")
print(dict)
add_to_dict(dict,key2,"b")
print(dict)
add_to_dict(dict,key2,"2a")
print(dict)

# Linked list
class Node():
    def __init__(self,value):
        self. value = value
        self.next = None
a= Node("a")
b= Node("b")
c=Node("c")
d= Node("d")
e = Node("e")
a.next = b
b.next = c
c.next = d
d.next = e
print("a->b->c->d->e")
# Travel Linked List
print("-------------Travel the Linked List -------------------")
print("Iteration:")
def travelLinked(Node):
    current = Node
    list = []
    while current != None:
        list.append(current.value)
        print(current.value,end = " ")
        current = current.next
    return list
travelLinked(a)
print()
print("Recursive:")
def travelLinkedRecurssive(Node):
    list = []
    if Node == None:
        return
    travelLinkedRecurssive_Sub(Node,list)
    for item in list:
        print(item,end=" ")
def travelLinkedRecurssive_Sub(Node,list):
    if Node is None:
        return
    list.append(Node.value)
    travelLinkedRecurssive_Sub(Node.next,list)

# for item in travelLinkedRecurssive(a):
#     print(item,end = ",")
travelLinkedRecurssive(a)
print()
print("-------------Calculate the size of Linked List -------------------- ")
# Calculate the size of Linked List
print("Iteration:")
def Size_Of_LinkedList_Iteration(Node):
    count = 0
    while Node is not None:
        count = count+1
        Node =Node.next
    return count
print("the Size  of Linked list is "+str(Size_Of_LinkedList_Iteration(a)))
print("Recursive:")
def Size_Of_LinkedList_Recurssive(Node):
    if Node is None:
        return 0
    return 1+Size_Of_LinkedList_Recurssive(Node.next)
print("the Size  of Linked list is "+str(Size_Of_LinkedList_Recurssive(a)))
print("-------------Find the Item in Linked List ---------------------")
# Find the Item in Linked List
print("Iteration:")
def Find_Item_Iteration(Node,target):
    if Node.value == target:
        return True
    else:
        while Node is not None:
            if Node.value ==target:
                return True
            Node = Node.next
    return False
target_list=["d","j"]
for item in target_list:
    if Find_Item_Iteration(a,item):
        print("The {} is in the Linked List".format(item))
    else:
        print("the {} is not in the Linked List".format(item))

print("Recursive:")
def Find_Item_recursive(Node,target):
    if Node is None:
        return False
    if Node.value == target:
        return True
    else:
        return Find_Item_recursive(Node.next,target)

for item in target_list:
    if Find_Item_recursive(a,item):
        print("The {} is in the Linked List".format(item))
    else:
        print("the {} is not in the Linked List".format(item))
print("2->8->3->7")
Node_x=Node(2)
Node_y=Node(8)
Node_i=Node(3)
Node_j=Node(7)
Node_x.next =Node_y
Node_y.next=Node_i
Node_i.next = Node_j
print("-------------Calculate the Sum of Linked List ---------------------")
print("Recursive:")
def cauculate_Linked_List_Sum_recursive(Node):
    if Node is None:
        return 0
    else:
        return Node.value+cauculate_Linked_List_Sum_recursive(Node.next)

print("The sum of Linked List is "+str(cauculate_Linked_List_Sum_recursive(Node_x)))
print("Iteration:")
def cauculate_Linked_List_Sum_Iteration(Node):
    sum =0
    while Node is not None:
        sum = sum + Node.value
        Node = Node.next
    return sum
print("The sum of Linked List is "+str(cauculate_Linked_List_Sum_Iteration(Node_x)))
print("---------------Return the value of Node with Given Index-------------------")
print("a->b->c->d->e")
print("Iteration:")
def Find_value_by_Index_Iteration(Node, Index):
    if Node is None:
        return False
    current = Node
    while Index>0:
        if current.next is None:
            return False
        current = current.next
        Index = Index-1
    return current.value
item_list = [x for x in range(7)]
for item in item_list:
    if Find_value_by_Index_Iteration(a,item) is False:
        print("The Index {} is out of the range".format(item))
    else:
        print("The {} item in the Linked List is {}".format(item,Find_value_by_Index_Iteration(a,item)))
print("Recursive:")
def Find_value_by_Index_Recursive(Node, Index):
    if Node is None:
        return False
    if Index==0:
        return Node.value
    return Find_value_by_Index_Recursive(Node.next,Index-1)
for item in item_list:
    if Find_value_by_Index_Recursive(a,item) is False:
        print("The Index {} is out of the range".format(item))
    else:
        print("The {} item in the Linked List is {}".format(item,Find_value_by_Index_Recursive(a,item)))

print("---------------Reverse the Linked List-------------------")
print("Iteration:")
def Reversed_Iteration(Node):
    current = Node
    prev = None
    while current is not None:
        next = current.next
        current.next = prev
        prev = current
        current = next
    return prev

travelLinked(Reversed_Iteration(a))
print()
print("Recursive:")
def Reversed_Recursive(Node,prev = None):
    if Node is None:
        return prev
    next_node = Node.next
    Node.next = prev
    return Reversed_Recursive(next_node,Node)
travelLinked(Reversed_Recursive(e))

# Zipper List
print()
print("---------------Zipper List-----------------")
print("a->b->c->d->e, 2->8->3->7, result: a->2->b->8->c->3->d->7->e")
print("Iteration:")
def ziplist_Iteration(head1,head2):
    tail = head1
    current1 = head1.next
    current2 = head2
    count = 0
    while (current1 is not None) and (current2 is not None):
        if count%2==0:
         # take current2 to current1
            tail.next = current2
            current2 = current2.next
        else:
            tail.next = current1
            current1 =current1.next
        tail=tail.next
        count = count+1
    if current1 is not None:
        tail.next = current1
    else:
        tail.next = current2
    return head1
travelLinked(a)
print()
travelLinked(Node_x)
print()
travelLinked(ziplist_Iteration(a,Node_x))
print()
print("Recursive:")
Node_x=Node(2)
Node_y=Node(8)
Node_i=Node(3)
Node_j=Node(7)
Node_x.next =Node_y
Node_y.next=Node_i
Node_i.next = Node_j
a= Node("a")
b= Node("b")
c=Node("c")
d= Node("d")
e = Node("e")
a.next = b
b.next = c
c.next = d
d.next = e
def ziplist_Recursive(head1,head2):
    if (head1 is None) and (head2 is None):
        return None
    if head1 is None:
        return head2
    if head2 is None:
        return head1
    next1 = head1.next
    next2 = head2.next
    head1.next = head2
    head2.next = ziplist_Recursive(next1,next2)
    return head1
travelLinked(a)
print()
travelLinked(Node_x)
print()
travelLinked(ziplist_Recursive(a,Node_x))
print()
