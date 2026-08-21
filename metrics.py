import numpy as np
import cv2
import numpy as np

from enum import Enum

class Metric(Enum):
    NS = 0
    MS = 1
    ME4 = 2
    ME8 = 3
    MI4 = 4
    MI8 = 5
    WV2 = 6
    WV3 = 7
    L2 = 8

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
    return(_moore_stress(A, N4, get_num_of_4_neighbors(A)))

def get_num_of_4_neighbors(A):
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

    return T

def get_num_of_8_neighbors(A):
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

    return T

def moore_stress8(A):
    return(_moore_stress(A, N8, get_num_of_8_neighbors(A)))

def _moore_stress(A, k, T):
    A2 = A * A
    D = T * A2 - 2 * A * conv(A, k) + conv(A2, k)
    return np.sum(D)

def MoransI8(A):
    (a,b) = A.shape
    t = (a-1)*b + a*(b-1) + 2*(a-1)*(b-1)
    return MoransI(A, N8, get_num_of_8_neighbors(A), t)

def MoransI4(A):
    (a,b) = A.shape
    t = (a-1)*b + a*(b-1)
    return MoransI(A, N4, get_num_of_4_neighbors(A), t)


def MoransI(A, k, T, t):
    x_ = np.mean(A)
    A = A - x_
    Ak = conv(A, k)
    (a, b) = A.shape
    r = a * b
    M = A*Ak

    I = (r/t) * ((np.sum(M)) / np.sum((A)**2))
    return I

def full_eval(A):
    return {
        "ME4" : me4(A),
        "ME8" : me8(A),
        "MS4" : moore_stress4(A),
        "MS8" : moore_stress8(A),
        "MI4" : MoransI4(A),
        "MI8" : MoransI8(A),
        "WV2" : wv2(A),
        "WV3" : wv3(A)
    }

def wv2(A):
    return wv(A, 2)

def wv3(A):
    return wv(A, 3)

def wv(A, n):
    # 1. Create a sample 4x4 array
    win_shape = (n, n)
    windows = np.lib.stride_tricks.sliding_window_view(A, window_shape=win_shape)
    variances = np.var(windows, axis=(-2, -1))
    r = np.sum(variances)
    return r

if __name__ == "__main__":

    # Test with a simple pattern
    A = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ], dtype=np.float64)
    C = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ], dtype=np.float64)
    B = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ], dtype=np.float64)
    D = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    E = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    F = np.array([ 
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float64)

    G = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=np.float64)

    print(wv2(None))