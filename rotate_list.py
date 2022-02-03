test0 = {
    'input': {
        'nums': [19, 25, 29, 3, 5, 6, 7, 9, 11, 14]
    },
    'output': 3
}
# A list of size 8 rotated 5 times.
test1 = {
    'input': {
        'nums': [4, 5, 6, 7, 8, 1, 2, 3]
    },
    'output': 5
}
# A list that wasn't rotated at all.
test2 = {
    'input': {
        'nums': [1,2,3,4,5,6,7,8]
    },
    'output': 0
}
# A list that was rotated just once.
test3 = {
    'input': {
        'nums': [7,1,2,3,4,5,6]
    },
    'output': 1
}
# A list that was rotated n-1 times, where n is the size of the list.
test4 = {
    'input': {
        'nums': [2,3,4,5,6,7,8,1]
    },
    'output': 7
}
# An empty list.
test6 = {
    'input': {
        'nums': []
    },
    'output': 0
}
test7 = {
    'input': {
        'nums': [23]
    },
    'output': 0
}
tests = [test0, test1, test2, test3, test4, test6, test7]

def linear_search(cards):
    while len(cards)>0:
        min_item = cards[0]
        target_index = 0
        i=1
        while i <len(cards):
            if cards[i]< min_item:
                return i
            i = i+1
        return 0
    return 0

def Binary_search(cards):
    low = 0
    high = len(cards)-1
    while low <= high:
        mid = (low + high) // 2
        mid_number = cards[mid]
        # print("low:",low," high:",high," mid",mid," mid_number",mid_number)
        precessor= cards[mid-1]
        if mid_number < precessor and mid>0:
            return mid
        elif mid_number > precessor and mid_number < cards[high]:
                high = mid-1
        elif mid_number > precessor and mid_number > cards[high]:
                low = mid+1
        elif low == high:
            return high
    return 0
# print(linear_search(test1["input"]["nums"]))
# print(Binary_search(test0["input"]["nums"]))
print("Start Linear Search for the Rotated Function")

for item in tests:
    print(item["input"]["nums"])
    print("Rotated "+str(linear_search(item["input"]["nums"]))+" times and the reference is "+str(item["output"]),end = " ")
    if linear_search(item["input"]["nums"])==item["output"]:
        Status ="Pass"
    else:
        Status == "Fail"
    print("Status:"+Status)
print("The Linear Search for the Rotated Function is finished")

print()

print("Start Binary Search for the Rotated Function")

for item in tests:
    print(item["input"]["nums"])
    print("Rotated "+str(Binary_search(item["input"]["nums"]))+" times and the reference is "+str(item["output"]),end = " ")
    if Binary_search(item["input"]["nums"])==item["output"]:
        Status ="Pass"
    else:
        Status == "Fail"
    print("Status:"+Status)
print("The Binary Search for the Rotated Function is finished")