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