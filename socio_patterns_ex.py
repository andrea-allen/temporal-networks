import sample_sir_sims
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import time
import pandas as pd

def import_data():
    start_time = time.time()
    df = parse_data('../tij_SFHH.dat_')
    print("--- %s agg seconds ---" % (time.time() - start_time))
    # df = pd.read_csv('../tij_SFHH.dat_')
    return 0

def run_one_layer():
    layer1, layer2 = parse_data('../tij_SFHH.dat_')
    sample_sir_sims.run_multiple_sims(layer1, layer2, .04, 100, 100)


def parse_data(filename):
    data = pd.read_csv(filename)
    df = pd.read_csv(filename, delimiter=' ', header=None)
    df.columns = ['t', 'i', 'j']
    min_t = min(df['t'])
    max_t = max(df['t'])
    some_layer = 35720
    layer1, layer2 = produce_layers(df, 33400, 600)
    # aggregate_graph = graphOf(pd.concat([t_101_layer, t_103_layer]))
    return graphOf(layer1, layer2), graphOf(layer2, layer1)

def produce_layers(df, t_start, increment):
    layer_1 = df[['i', 'j']].where((df['t'] >= t_start) & (df['t'] < t_start+increment))
    layer_2 = df[['i', 'j']].where((df['t'] >= t_start+increment) & (df['t'] < t_start+2*increment))
    return layer_1, layer_2

def graphOf(layer, second_layer):
    graph = nx.from_pandas_edgelist(layer.dropna(), 'i', 'j')
    second_layer_graph = nx.from_pandas_edgelist(second_layer.dropna(), 'i', 'j')
    graph.add_nodes_from(nx.nodes(second_layer_graph))
    mat1 = nx.adjacency_matrix(graph).todense()
    return graph

def aggregate(df, t_a, t_b):
    t_a_layer = df[['i', 'j']].where(df['t'] == t_a)
    t_b_layer = df[['i', 'j']].where(df['t'] == t_b)
    aggregate = graphOf(pd.concat([t_a_layer, t_b_layer]))
    return aggregate
