import numpy as np
import cv2

def conv(x, k, borderType=cv2.BORDER_CONSTANT):
    """
    Convolution on x with kernal k
    Parameters:
    x - Matrix,
    k - Kernal,
    borderType - border type,
    value - the constant, if border constant is selected
    """
    return cv2.filter2D(x, -1, np.flip(k, -1), borderType=borderType)

def me4(A):
    """
    Measure of effectivness using the 4 neighborhood
    """
    W = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.float64)
    
    return me(A, W)

def me8(A):
    """
    Measure of effectivness using the 8 neighborhood
    """
    W = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    return me(A, W)

def me(A, W):
    S = A*conv(A, W, borderType=cv2.BORDER_ISOLATED)
    D = np.sum(S*S)
    
    return D

if __name__ == "__main__":
    data = np.loadtxt('./Data/small_random.tsv', delimiter='\t')
    print(me4(data))

if __name__ == "__main__":
    data = np.loadtxt('./Data/small_random.tsv', delimiter='\t')
    print(me4(data))