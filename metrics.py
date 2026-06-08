import numpy as np
import cv2

N4 = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.float64)

N8 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    

def conv(x, k, borderType=cv2.BORDER_ISOLATED):
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
    return _me(A, N4)

def me8(A):
    """
    Measure of effectivness using the 8 neighborhood
    """
    return _me(A, N8)

def _me(A, W):
    S = A*conv(A, W, borderType=cv2.BORDER_ISOLATED)
    D = np.sum(S)
    
    return D

def moore_stress4(A):
    T = np.ones_like(A) * 4
    (a,b) = T.shape
    (a,b) = (a-1, b-1)
    
    T[:, 0] = 3
    T[:, b] = 3
    T[0, :] = 3
    T[a, :] = 3
    
    T[0,0] = 2
    T[0,b] = 2
    T[a,0] = 2
    T[a,b] = 2

    print(T)

    return(_moore_stress(A, N4, T))

def moore_stress8(A):
    T = np.ones_like(A) * 8
    (a,b) = T.shape
    (a,b) = (a-1, b-1)
    
    T[:, 0] = 5
    T[:, b] = 5
    T[0, :] = 5
    T[a, :] = 5
    
    T[0,0] = 3
    T[0,b] = 3
    T[a,0] = 3
    T[a,b] = 3

    print(T)

    return(_moore_stress(A, N8, T))

def _moore_stress(A, k, T):
    A2 = A * A
    D = T * A2 - 2 * A * conv(A, k) + conv(A2, k)
    return np.sum(D)

if __name__ == "__main__":
    data = np.loadtxt('./Data/small_random.tsv', delimiter='\t')
    print(moore_stress8(data))
