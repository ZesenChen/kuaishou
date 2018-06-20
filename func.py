import numpy as np

def conseq(arr):
    if arr[0] == 0:
        count = 0
        tmp = 0
    else:
        count = 1
        tmp = 1
    for i in range(1,arr.shape[0]):
        if arr[i] == 0:
            tmp = 0
        else:
            tmp += 1
            count = max(count,tmp)
    return count
