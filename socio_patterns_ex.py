import sample_sir_sims
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import time
import pandas as pd
import matrix_ops
import math
from scipy import stats

def run_one_layer():
    layer1, layer2 = parse_data('../tij_SFHH.dat_')
    sample_sir_sims.run_multiple_sims(layer1, layer2, .04, 100, 100)
    matrix_ops.test_ops(layer1, layer2)

def error_over_time(granularity, filename):
    df = pd.read_csv(filename, delimiter=' ', header=None)
    df.columns = ['t', 'i', 'j']
    min_t = min(df['t'])
    max_t = max(df['t'])
    max_errors = np.zeros(granularity)
    final_errors = np.zeros(granularity)
    theo_error = np.zeros(granularity)
    gran = (max_t-min_t)/granularity
    for i in range(granularity):
        print('Layer ', i, ' out of ', granularity)
        l1, l2 = produce_layers(df, min_t+i*gran, gran)
        g1 = graphOf(l1, l2)
        g2 = graphOf(l2, l1)
        start_time = time.time()
        try:
            error, theo = sample_sir_sims.run_multiple_sims(g1, g2, .003, 50, 20)
            max_errors[i] = max(error)
            final_errors[i] = error[-1]
            theo_error[i] = theo
        except nx.exception.NetworkXError:
            max_errors[i] = -1
        print("--- %s sim seconds ---" % (time.time() - start_time))
    # plt.scatter(max_errors, theo_error, label='max error vs theo error')
    plt.scatter(final_errors, theo_error, label='final error vs theo error')
    slope, intcpt, r_val, p_val, std_err  = stats.linregress(final_errors, theo_error)
    x_range = final_errors
    y_vals = intcpt + slope*x_range
    plt.plot(x_range, y_vals)
    # plt.plot(np.arange(min_t, max_t, gran), theo_error, label='theo error')
    plt.legend(loc='upper left')
    plt.show()

def error_over_time_hospital(granularity, filename):
    df = pd.read_csv(filename, delimiter='	', header=None)
    df.columns = ['t', 'i', 'j', 'dept', 'dept2']
    min_t = min(df['t'])
    max_t = max(df['t'])
    max_errors = np.zeros(granularity)
    final_errors = np.zeros(granularity)
    theo_error = np.zeros(granularity)
    gran = (max_t-min_t)/granularity
    for i in range(granularity):
        print('Layer ', i, ' out of ', granularity)
        l1, l2 = produce_layers(df, min_t+i*gran, gran)
        g1 = graphOf(l1, l2)
        g2 = graphOf(l2, l1)
        start_time = time.time()
        try:
            error, theo = sample_sir_sims.run_multiple_sims(g1, g2, .005, 5, 20)
            max_errors[i] = max(error)
            final_errors[i] = error[-1]
            theo_error[i] = theo
        except nx.exception.NetworkXError:
            max_errors[i] = -1
        print("--- %s sim seconds ---" % (time.time() - start_time))
    # plt.scatter(max_errors, theo_error, label='max error vs theo error')
    plt.scatter(final_errors, theo_error, label='final error vs theo error')
    slope, intcpt, r_val, p_val, std_err  = stats.linregress(final_errors, theo_error)
    x_range = final_errors
    y_vals = intcpt + slope*x_range
    plt.plot(x_range, y_vals)
    # plt.plot(np.arange(min_t, max_t, gran), theo_error, label='theo error')
    plt.legend(loc='upper left')
    plt.show()

def random_graph_error(granularity, filename):
    # df = pd.read_csv(filename, delimiter='	', header=None)
    # df.columns = ['t', 'i', 'j', 'dept', 'dept2']
    # min_t = min(df['t'])
    # max_t = max(df['t'])
    max_errors = np.zeros(granularity)
    final_errors = np.zeros(granularity)
    theo_error = np.zeros(granularity)
    gran = 2000/granularity
    for i in range(granularity):
        print('Layer ', i, ' out of ', granularity)
        # l1, l2 = produce_layers(df, min_t+i*gran, gran)
        N = 100
        p = random.random()
        g2 = nx.generators.erdos_renyi_graph(N, p)
        g1 = nx.generators.erdos_renyi_graph(N, 0.01)
        # g1 = graphOf(l1, l2)
        # g2 = graphOf(l2, l1)
        start_time = time.time()
        try:
            T = 20
            error, theo = sample_sir_sims.run_multiple_sims(g1, g2, .005, 50, T)
            max_errors[i] = max(error)
            final_errors[i] = error[T-1]
            theo_error[i] = theo
        except nx.exception.NetworkXError:
            max_errors[i] = -1
        print("--- %s sim seconds ---" % (time.time() - start_time))
    # plt.scatter(max_errors, theo_error, label='max error vs theo error')
    plt.scatter(final_errors, theo_error, label='final error vs theo error')
    slope, intcpt, r_val, p_val, std_err = stats.linregress(final_errors, theo_error)
    x_range = final_errors
    y_vals = intcpt + slope*x_range
    plt.plot(x_range, y_vals, label='std error= '+str(np.round(std_err,2)))
    plt.legend(loc='upper left')
    # plt.plot(np.arange(min_t, max_t, gran), theo_error, label='theo error')
    # plt.show()

def other_random_graph_error(granularity, filename):
    # df = pd.read_csv(filename, delimiter='	', header=None)
    # df.columns = ['t', 'i', 'j', 'dept', 'dept2']
    # min_t = min(df['t'])
    # max_t = max(df['t'])
    max_errors = np.zeros(granularity)
    final_errors = np.zeros(granularity)
    theo_error = np.zeros(granularity)
    gran = 2000/granularity
    for i in range(granularity):
        print('Layer ', i, ' out of ', granularity)
        # l1, l2 = produce_layers(df, min_t+i*gran, gran)
        N = 500
        g2 = nx.generators.connected_watts_strogatz_graph(N, 5, 0.05) #go home
        g1 = generate_graph(N, [0.0, 0.10, .20, .20, .20, .20, .05, .05]) #work
        if (len(g1.nodes())!=N):
            num_nodes = len(g1.nodes())
            if num_nodes>N:
                for n in range(N, num_nodes):
                    g1.remove_node(n)
            if num_nodes<N:
                for n in range(num_nodes, N):
                    g1.add_node(n)
        # g1 = graphOf(l1, l2)
        # g2 = graphOf(l2, l1)
        start_time = time.time()
        try:
            T = 40
            error, theo = sample_sir_sims.run_multiple_sims(g1, g2, .3, 5, T)
            # max_errors[i] = max(error)
            final_errors[i] = error[T-5]
            theo_error[i] = theo
        except nx.exception.NetworkXError:
            max_errors[i] = -1
        print("--- %s sim seconds ---" % (time.time() - start_time))
    # plt.scatter(max_errors, theo_error, label='max error vs theo error')
    plt.scatter(final_errors, theo_error, label='final error vs theo error')
    plt.legend(loc='upper left')
    # plt.plot(np.arange(min_t, max_t, gran), theo_error, label='theo error')
    # plt.show()

def parse_data(filename, t_start, increment):
    df = pd.read_csv(filename, delimiter=' ', header=None)
    df.columns = ['t', 'i', 'j']
    layer1, layer2 = produce_layers(df, t_start, increment)
    return graphOf(layer1, layer2), graphOf(layer2, layer1)

def produce_layers(df, t_start, increment):
    layer_1 = df[['i', 'j']].where((df['t'] >= t_start) & (df['t'] < t_start+increment))
    layer_2 = df[['i', 'j']].where((df['t'] >= t_start+increment) & (df['t'] < t_start+2*increment))
    return layer_1, layer_2

def graphOf(layer, second_layer):
    graph = nx.from_pandas_edgelist(layer.dropna(), 'i', 'j')
    second_layer_graph = nx.from_pandas_edgelist(second_layer.dropna(), 'i', 'j')
    graph.add_nodes_from(nx.nodes(second_layer_graph))
    return graph

def aggregate(df, t_a, t_b):
    t_a_layer = df[['i', 'j']].where(df['t'] == t_a)
    t_b_layer = df[['i', 'j']].where(df['t'] == t_b)
    aggregate = graphOf(pd.concat([t_a_layer, t_b_layer]))
    return aggregate

def generate_graph(N, deg_dist):
    number_of_nodes = N*np.array(deg_dist)
    degree_sequence = []
    for i in range(int(math.floor(len(number_of_nodes)))):
        number_with_that_degree = number_of_nodes[i]
        for k in range(int(math.floor(number_with_that_degree))):
            degree_sequence.append(i)
    # z = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    graphical = nx.is_graphical(degree_sequence)
    print('Degree sequence is graphical: ', graphical)
    if not graphical:
        print('Adding node of degree 1')
        degree_sequence.append(1)
    print("Configuration model")
    G = nx.configuration_model(degree_sequence)
    # Remove self-loops and parallel edges
    try:
        G.remove_edges_from(nx.selfloop_edges(G))
    except RuntimeError:
        print('No self loops to remove')
    pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos=pos, with_labels=True)
    # nx.draw_networkx_labels(G, pos=pos, with_labels=True)
    # nx.draw_networkx_edges(G, pos=pos)
    # plt.show()
    # Check:
    inferred_degree_dist = np.array(nx.degree_histogram(G))/N
    print('Inferred equals given degree distribution: ', inferred_degree_dist == deg_dist)
    return G