import sample_sir_sims
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import time
import pandas as pd

def run_one_layer():
    layer1, layer2 = parse_data('../tij_SFHH.dat_')
    sample_sir_sims.run_multiple_sims(layer1, layer2, .04, 100, 100)

def error_over_time(granularity, filename):
    df = pd.read_csv(filename, delimiter=' ', header=None)
    df.columns = ['t', 'i', 'j']
    min_t = min(df['t'])
    max_t = max(df['t'])
    max_errors = np.zeros(granularity)
    gran = (max_t-min_t)/granularity
    for i in range(granularity):
        print('Layer ', i, ' out of ', granularity)
        l1, l2 = produce_layers(df, min_t+i*gran, gran)
        g1 = graphOf(l1, l2)
        g2 = graphOf(l2, l1)
        start_time = time.time()
        try:
            error = sample_sir_sims.run_multiple_sims(g1, g2, .04, 10, 100)
            max_errors[i] = max(error)
        except nx.exception.NetworkXError:
            max_errors[i] = -1
        print("--- %s sim seconds ---" % (time.time() - start_time))
    plt.plot(np.arange(min_t, max_t, gran), max_errors)
    plt.show()

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
