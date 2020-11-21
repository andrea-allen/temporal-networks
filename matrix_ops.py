import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import itertools
import scipy
from matplotlib import rc
import timeit
import time
from skimage import measure
from scipy.linalg import expm, sinm, cosm

def test_ops(T1, T2, beta, t):
    # A = np.random.rand(10,10)
    # B = np.random.rand(10,10)
    # e_temp = matrix_exp(B).dot(matrix_exp(A))
    # e_agg = matrix_exp(B+A)
    # e_temp_data = matrix_exp(T2).dot(matrix_exp(T1))
    # e_agg_data = matrix_exp((T2+T1)/2)
    return theoretical_error(T1, T2, beta, t)
    P_0 = np.zeros(len(T2[0]))
    for i in range(len(T2[0])):
        P_0[i] = 1 / len(T2[0])
    beta_deltat = beta*t
    e_t = matrix_exp(beta_deltat * T2).dot(matrix_exp(beta_deltat * T1))
    e_a = matrix_exp(beta_deltat * (T1 + T2))
    p_t = e_t.dot(P_0)
    p_a = e_a.dot(P_0)
    diff = np.sum(p_t) - np.sum(p_a)
    # max_diff = max(diff)
    # print(max_diff)
    return diff/(len(T2[0]))

def theoretical_error(T1, T2, beta, t):
    print(beta*t)
    N = len(T2[0])
    P_0 = np.zeros(N)
    for i in range(N):
        P_0[i] = 1 / N
    AAB = beta*t * (T1.dot(T1.dot(T2)))
    ABB = beta*t * (T1.dot(T2.dot(T2)))
    ABA = beta*t * (T1.dot(T2.dot(T1)))
    BAB = beta*t * (T2.dot(T1.dot(T2)))
    E = np.sum((AAB + ABB + ABA + BAB).dot(P_0))
    return E

def matrix_exp(A):
    return expm(A)