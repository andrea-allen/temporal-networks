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

from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def less_than(x, y) -> object:
    if x > y: return 0;
    return 1

@vectorize([float64(float64)])
def flatten_weights(x):
    if x > 0: return 1;
    return 0

def SIR_step(current_infected_nodes, adj_matx, beta):
    N = len(current_infected_nodes)
    next_infected_nodes = current_infected_nodes
    for i in np.where(current_infected_nodes == 1)[0]:
        R = less_than(np.random.rand(N), beta)
        infect_possible_vec = np.multiply(adj_matx[i], R)
        for j in np.where(infect_possible_vec == 1)[0]:
            if current_infected_nodes[j] == 0:
                next_infected_nodes[j] = infect_possible_vec[j]
    next_infected_nodes = flatten_weights(next_infected_nodes)
    I = np.sum(next_infected_nodes)
    return I, next_infected_nodes

def run_sample_sim_temporal(A,B, beta, gamma, I, delta_t):
    time_series = np.arange(2*delta_t)
    N = len(B)
    current_infected_nodes = np.zeros(N)
    patient_zero = random.randint(0, N-1)
    current_infected_nodes[patient_zero]=1
    # start_time = time.time()
    for t in range(delta_t):
        I, current_infected_nodes = SIR_step(current_infected_nodes, A, beta)
        time_series[t]=I
    for t in range(delta_t, 2*delta_t):
        I, current_infected_nodes = SIR_step(current_infected_nodes, B, beta)
        time_series[t]=I
    # print("--- %s temporal inner seconds ---" % (time.time() - start_time))
    return time_series/N

def run_sample_sim_aggregate(A,B, beta, gamma, I, delta_t):
    A = flatten_weights(A+B)
    time_series = np.arange(2*delta_t)
    N = len(A)
    current_infected_nodes = np.zeros(N)
    patient_zero = random.randint(0, N-1)
    current_infected_nodes[patient_zero]=1
    for t in range(2*delta_t):
        I, current_infected_nodes = SIR_step(current_infected_nodes, A, beta)
        time_series[t]=I
    return time_series/N

def compare(N, beta, gamma):
    A = np.random.random_integers(0, 1, (N,N))
    B = np.random.random_integers(0, 1, (N,N))
    GA = nx.generators.erdos_renyi_graph(N, .05)
    GA = nx.generators.connected_watts_strogatz_graph(1000, 3, .02)
    GB = nx.generators.erdos_renyi_graph(len(GA.nodes()), .05)
    GB = nx.generators.connected_watts_strogatz_graph(1000, 3, .02)
    A = np.array(nx.adjacency_matrix(GA).todense())
    B = np.array(nx.adjacency_matrix(GB).todense())
    T = 100
    start_time = time.time()
    I_agg = run_sample_sim_aggregate(A,B, beta, gamma, 1, delta_t=T)
    print("--- %s agg seconds ---" % (time.time() - start_time))
    start_time = time.time()
    I_temp = run_sample_sim_temporal(A,B, beta, gamma, 1, delta_t=T)
    print("--- %s temporal seconds ---" % (time.time() - start_time))
    plt.plot(np.arange(2*T), I_agg, label="aggregate")
    plt.plot(np.arange(2*T), I_temp, label="temporal")
    plt.legend(loc='lower right')
    plt.show()
    return 0

def run_given_layers(layer1, layer2, beta):
    A = np.array(nx.adjacency_matrix(layer1).todense())
    B = np.array(nx.adjacency_matrix(layer2).todense())
    T = 100
    gamma=0
    start_time = time.time()
    I_agg = run_sample_sim_aggregate(A, B, beta, gamma, 1, delta_t=T)
    # print("--- %s agg seconds ---" % (time.time() - start_time))
    start_time = time.time()
    I_temp = run_sample_sim_temporal(A, B, beta, gamma, 1, delta_t=T)
    # print("--- %s temporal seconds ---" % (time.time() - start_time))
    return I_agg, I_temp
    # plt.plot(np.arange(2 * T), I_agg, label="aggregate")
    # plt.plot(np.arange(2 * T), I_temp, label="temporal")
    # plt.legend(loc='lower right')
    # plt.show()

def run_multiple_sims(layer1, layer2, beta, times, T):
    results_agg = np.zeros((times, 2*T))
    results_temp = np.zeros((times, 2*T))
    for t in range(times):
        results_agg[t], results_temp[t] = run_given_layers(layer1, layer2, beta)
    results_agg = np.mean(results_agg, axis=0)
    results_temp = np.mean(results_temp, axis=0)
    plt.plot(np.arange(2 * T), results_agg, label="aggregate")
    plt.plot(np.arange(2 * T), results_temp, label="temporal")
    plt.legend(loc='lower right')
    plt.show()
    error = results_agg-results_temp
    plt.plot(np.arange(2*T), error, label='aggregation error')
    plt.legend(loc='lower right')
    plt.show()
    return error

def identity_experiment():
    GA = nx.generators.connected_watts_strogatz_graph(1000, 3, .02)
    A = np.array(nx.adjacency_matrix(GA).todense())
    error = run_multiple_sims(GA, GA, .2, 50, 100)
    plt.plot(np.arange(2*100), error, label='aggregation error')
    plt.legend(loc='lower right')
    plt.show()
