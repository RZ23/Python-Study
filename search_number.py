import time
tests = []
test = {
    'input': {
        'cards': [13, 11, 10, 7, 4, 3, 1, 0],
        'query': 7
    },
    'output': 3
}
tests.append(test)
# query occurs in the middle
tests.append({
    'input': {
        'cards': [13, 11, 10, 7, 4, 3, 1, 0],
        'query': 1
    },
    'output': 6
})
# query is the first element
tests.append({
    'input': {
        'cards': [4, 2, 1, -1],
        'query': 4
    },
    'output': 0
})
# query is the last element
tests.append({
    'input': {
        'cards': [3, -1, -9, -127],
        'query': -127
    },
    'output': 3
})
# cards contains just one element, query
tests.append({
    'input': {
        'cards': [6],
        'query': 6
    },
    'output': 0
})
# cards does not contain query
tests.append({
    'input': {
        'cards': [9, 7, 5, 2, -9],
        'query': 4
    },
    'output': -1
})
# cards is empty
tests.append({
    'input': {
        'cards': [],
        'query': 7
    },
    'output': -1
})
tests.append({
    'input': {
        'cards': [8, 8, 6, 6, 6, 6, 6, 3, 2, 2, 2, 0, 0, 0],
        'query': 3
    },
    'output': 7
})
# query occurs multiple times
tests.append({
    'input': {
        'cards': [8, 8, 6, 6, 6, 6, 6, 6, 3, 2, 2, 2, 0, 0, 0],
        'query': 6
    },
    'output': 2
})
# for item in tests:
#     print(item)
def linear_Search_located_number(cards,target_number):
    index = 0
    if len(cards) == 0:
        return -1
    for item in cards:
        if item == target_number:
            return index
        index = index+1
    if index == len(cards):
        return -1
# print(located_number(tests[5]['input']['cards'],tests[5]['input']['query']))
start_time =time.time()
for test in tests:
        print(test['input']['cards'])
        calc_out=linear_Search_located_number(test['input']['cards'],test['input']['query'])
        if calc_out==-1:
            print("Not found "+str(test['input']['query']))
        elif calc_out == test['output']:
            print("found "+str(test['input']['query'])+", index is "+str(calc_out))
end_time = time.time()
print("runnint time is "+str((end_time-start_time))+"ms")

