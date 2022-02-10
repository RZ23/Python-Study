def lcs_recursive(sq1,sq2,idx1 =0,idx2 = 0):
    if idx1 == len(sq1) or idx2==len(sq2):
        return 0
    if sq1[idx1]==sq2[idx2]:
        return 1+lcs_recursive(sq1,sq2,idx1+1,idx2+1)
    else:
        return max(lcs_recursive(sq1, sq2, idx1+1, idx2),lcs_recursive(sq1, sq2,idx1, idx2+1))

def lcs_mem(sq1,sq2):
    mem = {}
    def recursive(idx1,idx2):
        key = idx1,idx2
        if key in mem:
            return mem[key]
        if idx1==len(sq1) or idx2==len(sq2):
            mem[key] = 0
        elif sq1[idx1]==sq2[idx2]:
            mem[key]=1+recursive(idx1+1,idx2+1)
        else:
            mem[key] = max(recursive(idx1,idx2+1),recursive(idx1+1,idx2))
        return mem[key]
    return recursive(0,0)

list1 ='hello'
list2 = 'llo'
print(lcs_recursive(list1,list2))
print(lcs_mem(list1,list2))

def dynamic_program(seq1,seq2):
    n1,n2 = len(seq1),len(seq2)
    result=[[0 for i in range(n2+1)] for j in range(n1+1)]
    for idx1 in range(n1):
        for idx2 in range(n2):
            if seq1[idx1]==seq2[idx2]:
                result[idx1+1][idx2+1] = 1+result[idx1][idx2]
            else:
                result[idx1+1][idx2+1] = max(result[idx1][idx2+1],result[idx1+1][idx2])
    return result[n1][n2]
print(dynamic_program(list1,list2))

def max_profit(capacity, weight,profits,idx=0):
    if idx==len(weight):
        return 0
    if weight[idx]>capacity:
        return max_profit(capacity,weight,profits, idx+1)
    else:
        return max(max_profit(capacity,weight,profits,idx+1),profits[idx]+max_profit(capacity-weight[idx],weight,profits,idx+1))

print(max_profit(165,[23,31,29,44,53,38,63,85,89,82],[92,57,49,68,60,43,67,84,87,72]))

def knapsack_mem(capacity, weights,profits):

    memo = {}

    def recurse(idx,remaining):
        key = (idx, remaining)
        if key in memo:
            return memo[key]
        elif idx==len(weights):
            memo[key]=0
        elif weights[idx]>remaining:
            memo[key] = recurse(idx+1,remaining)
        else:
            memo[key] = max(recurse(idx+1,remaining),
                            profits[idx]+recurse(idx+1,remaining-weights[idx]))
        return memo[key]
    return recurse(0,capacity)
print(knapsack_mem(165,[23,31,29,44,53,38,63,85,89,82],[92,57,49,68,60,43,67,84,87,72]))