import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import itertools
import scipy
from matplotlib import rc
import timeit
import math
import time
from skimage import measure
import matrix_ops
import event_driven_sims

from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def less_than(x, y) -> object:
    if x > y: return 0;
    return 1

@vectorize([float64(float64)])
def flatten_weights(x):
    if x > 0: return 1;
    return 0

def SI_step(current_infected_nodes, adj_matx, beta):
    #for each infectious node, loop through its neighbors and transmit with p=beta
    N = len(current_infected_nodes) #this is misleading, N is the length of the whole network
    next_infected_nodes = current_infected_nodes
    for i in np.where(current_infected_nodes == 1)[0]:
        # make a vector of coin flips
        R = less_than(np.random.rand(N), beta) #represents yes/no node is going to get infected next
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
        I, current_infected_nodes = SI_step(current_infected_nodes, A, beta)
        time_series[t]=I
    for t in range(delta_t, 2*delta_t):
        I, current_infected_nodes = SI_step(current_infected_nodes, B, beta)
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
        I, current_infected_nodes = SI_step(current_infected_nodes, A, beta)
        time_series[t]=I
    return time_series/N

def run_aggregate_event_driven(G1, G2, beta, gamma, I, delta_t):
    # AB = flatten_weights(A + B)
    G = nx.Graph()
    G.add_nodes_from(G1.nodes())
    G.add_nodes_from(G2.nodes())
    G.add_edges_from(nx.edges(G1))
    G.add_edges_from(nx.edges(G2))
    pos = nx.spring_layout(G)
    gamma=0
    N=len(G.nodes())
    Lambda = np.zeros((N, N))
    Gamma = np.zeros(N)
    for i in range(N):
        Gamma[i] = 0
        for j in range(N):
            Lambda[i][j] = beta
    sim = event_driven_sims.Simulation(delta_t, G, Lambda, Gamma, pos)
    sim.run_sim()
    print(len(sim.has_been_infected_labels))
    # plt.plot(sim.sim_time_vec, sim.percent_infected, label='aggregate')
    # plt.legend(loc='upper left')
    # plt.show()
    return sim.percent_infected, sim.sim_time_vec

def run_temporal_event_driven(G1, G2, beta, gamma, I, delta_t):
    # AB = flatten_weights(A + B)
    # G = nx.Graph()
    # G.add_nodes_from(G1.nodes())
    # G.add_nodes_from(G2.nodes())
    # G.add_edges_from(nx.edges(G1))
    # G.add_edges_from(nx.edges(G2))
    pos = nx.spring_layout(G2)
    gamma=0
    N=len(G2.nodes())
    Lambda = np.zeros((N, N))
    Gamma = np.zeros(N)
    for i in range(N):
        Gamma[i] = 0
        for j in range(N):
            Lambda[i][j] = beta
    sim = event_driven_sims.Simulation(delta_t, G1, Lambda, Gamma, pos)
    sim.set_G2(G2)
    sim.run_sim_temporal()
    print(len(sim.has_been_infected_labels))
    # plt.plot(sim.sim_time_vec, sim.percent_infected, label='temporal')
    # plt.legend(loc='upper left')
    # plt.show()
    return sim.percent_infected, sim.sim_time_vec

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

def run_given_layers(layer1, layer2, beta, T):
    A = np.array(nx.adjacency_matrix(layer1).todense())
    B = np.array(nx.adjacency_matrix(layer2).todense())
    gamma = 0
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
        results_agg[t], results_temp[t] = run_given_layers(layer1, layer2, beta, T)
    results_agg = np.mean(results_agg, axis=0)
    results_temp = np.mean(results_temp, axis=0)
    plt.plot(np.arange(2 * T), results_agg, label="aggregate")
    plt.plot(np.arange(2 * T), results_temp, label="temporal")
    plt.legend(loc='lower right')
    plt.show()
    error = results_agg-results_temp
    # plt.plot(np.arange(2*T), abs(error), label='aggregation error')
    A = np.array(nx.adjacency_matrix(layer1).todense())
    B = np.array(nx.adjacency_matrix(layer2).todense())
    theo_error_range = np.array([1])
    error_theroetical = []
    E = matrix_ops.test_ops(A, B, beta, 1/7)
    print(E)
    print(t)
    error_theroetical.append(abs(E))
    # plt.plot(theo_error_range, error_theroetical, label='theoretical error')
    # plt.ylim(0, 1)
    # plt.legend(loc='lower right')
    # plt.show()
    return abs(error), error_theroetical[-1]

def real_time_map(times, results, T):
    real_time = np.zeros(T)
    current_floor = 0
    for t in range(len(times)):
        time = times[t]
        time_floor = int(math.floor(time))
        if time_floor > current_floor+1:
            for i in range(current_floor+1, current_floor+time_floor-current_floor):
                try:
                    real_time[i] = real_time[i-1]
                except IndexError:
                    continue
        try:
            real_time[time_floor] = results[t]
        except IndexError:
            print('time floor, t')
            print(time_floor, t)
            continue
        current_floor = time_floor
    if T > len(results):
        for t in range(len(results)+1, T):
            real_time[t] = real_time[t-1]
    return real_time


def identity_experiment():
    GA = nx.generators.connected_watts_strogatz_graph(100, 3, .02)
    A = np.array(nx.adjacency_matrix(GA).todense())
    granularity = 30
    max_errors = np.zeros(granularity)
    final_errors = np.zeros(granularity)
    theo_error = np.zeros(granularity)
    gran = (2000) / granularity
    for i in range(granularity):
        print('Layer ', i, ' out of ', granularity)
        # l1, l2 = produce_layers(df, min_t + i * gran, gran)
        g1 = GA
        g2 = GA
        start_time = time.time()
        try:
            error, theo = run_multiple_sims(g1, g2, .04, 10, 30)
            max_errors[i] = max(error)
            final_errors[i] = error[-1]
            theo_error[i] = theo
        except nx.exception.NetworkXError:
            max_errors[i] = -1
        print("--- %s sim seconds ---" % (time.time() - start_time))
    error, theo = run_multiple_sims(GA, GA, .2, 50, 30) #for range of beta, still agrees
    plt.plot(np.arange(30), error, label='aggregation error')
    # plt.plot([1, 50, 100, 150, 195, 199], error_theroetical, label='theoretical error')
    # plt.ylim(0,1)
    plt.legend(loc='lower right')
    plt.show()

    plt.scatter(max_errors, theo_error, label='max error vs theo error')
    plt.scatter(final_errors, theo_error, label='final error vs theo error')
    # plt.plot(np.arange(min_t, max_t, gran), theo_error, label='theo error')
    plt.legend(loc='upper left')
    plt.show()

def experiment():
    GA = nx.generators.connected_watts_strogatz_graph(100, 3, .02)
    GB = nx.generators.erdos_renyi_graph(100, .05)
    A = np.array(nx.adjacency_matrix(GA).todense())
    B = np.array(nx.adjacency_matrix(GB).todense())
    error, theo = run_multiple_sims(GA, GB, .2, 50, 100)  # for range of beta, still agrees
    plt.plot(np.arange(2 * 100), error, label='aggregation error')
    error_theroetical = []
    for t in [1, 50, 100, 150, 195, 199]:
        if t in [1, 50, 100, 150, 195, 199]:
            print(t)
            E = matrix_ops.test_ops(A, B, .2, 1/14)
            error_theroetical.append(abs(E))
    plt.plot([1, 50, 100, 150, 195, 199], error_theroetical, label='theoretical error')
    plt.ylim(0,1)
    plt.legend(loc='lower right')
    plt.show()
