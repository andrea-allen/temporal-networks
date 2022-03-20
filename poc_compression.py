import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import deterministic
import random


### Goal here is to start with a series of 10-12 networks
### then compress them pairwise by computing E for successive pairs
### for an appropriate t and beta, then running
### a determinisitc disease process (using the ODE method, integration approximation)
### on the full set of networks and the compressed set and assessing that no integrity is lost.
### then compare that against a random aggregation (and full aggregation) of all
### the networks, and try to overall compare "networks aggregated by the E method"
### vs fully aggregated against the baseline "true" simulation.

### Things to do:
# Create a deterministic integration method to run as many networks and at each start time desired
# ^^ DONE, YAY (Saturday)
# Find networks that might work for this example
## For now, pause on finding the "perfect" error approximation E: I have a good one, that can work
# in the interim, for proof of concept. Then when proof of concept is in place, can go back
# and iterate on the E approximation to make it better and cleaner. Go back and forth between the processes.
## Function for creating networks:
def new_matexp_approximation(X, Y, t, beta):
    A = beta * t / 2 * X
    B = beta * t / 2 * Y
    N = len(A)
    I = np.identity(N)
    P0 = np.full(N, 1 / N)
    # error = (-1*I) - (A @ B) - (1/12)*(B@B@A) - (1/3)*(B@A@B) - (1/12)*(A@B@B) - (1/3)*(A@B@A) - (1/12)*(A@A@B) \
    #         + (1/4)*(B@A@A) + (1/2)*(A@A) + (1/2)*(B@B) + (1/6)*(A@A@A) + (1/6)*(B@B@B)
    error = compute_Z(X, Y, t, beta) - (
                A + B)  # - (((A+B)@(A+B))/2 + ((A+B)@(A+B)@(A+B))/6) + ((B+A + (1/2)*(B@A - A@B))@(B+A + (1/2)*(B@A - A@B)))/2 - (((1/2)*(B@A - A@B))@((1/2)*(B@A - A@B)))/2
    # error = scipy.linalg.expm(compute_Z(A, B, t, beta)).dot(P0) - scipy.linalg.expm(compute_Z(A, B, t, beta)).dot(P0)
    total_infect_diff = np.sum(np.abs(error) @ P0)
    return total_infect_diff


def compute_Z(X, Y, t, beta):
    ## Compute Z to third order matrix products
    A = beta * t / 2 * X
    B = beta * t / 2 * Y
    Z = B + A + (1 / 2) * (B @ A - A @ B) + (1 / 12) * (B @ B @ A + A @ A @ B + A @ B @ B + B @ A @ A) - (1 / 6) * (
                B @ A @ B + A @ B @ A)
    return Z


def erdos_renyi_graph(N, p):
    G1 = nx.generators.erdos_renyi_graph(n=N, p=p)
    G1.add_nodes_from(np.arange(N))
    adj_1 = np.array(nx.adjacency_matrix(G1).todense())
    # nx.draw(G1)
    # plt.show()
    return G1, adj_1


def configuration_model_graph(N):
    degree_distribution = [0, 20 / 100, 65 / 100, 0 / 100, 0, 0, 0, 10 / 100, 0, 0, 0, 0, 5 / 100]
    got_config_model = False
    while not got_config_model:
        try:
            config_model = nx.generators.configuration_model(
                np.random.choice(np.arange(len(degree_distribution)), p=degree_distribution, size=N))
            got_config_model = True
        except:
            got_config_model = False
    config_model.add_nodes_from(list(np.arange(N)))
    config_adj = np.array(nx.adjacency_matrix(config_model).todense())
    return config_model, config_adj


def sbm(N, groups, probs):
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    node_groups = {g: [] for g in range(groups)}
    nodes_per_group = int(N / groups)
    blocks = []
    for g in range(groups):
        for n in range(g * nodes_per_group, (g + 1) * nodes_per_group):
            node_groups[g].append(n)
        # H = nx.Graph()
        # H.add_nodes_from(node_groups[g])
        H = nx.generators.erdos_renyi_graph(nodes_per_group, probs[g])
        H_nodes = list(H.nodes())
        label_map = {H_nodes[i]: node_groups[g][i] for i in range(nodes_per_group)}
        H = nx.relabel_nodes(H, label_map)
        blocks.append(H)
    for block in blocks:
        # G.add_nodes_from(block.nodes())
        G.add_edges_from(block.edges())
    for g in range(groups):
        for j in range(g, groups):
            try:
                p = probs[(g, j)]
                for n in node_groups[g]:
                    for m in node_groups[j]:
                        flip_coin = random.random()
                        if flip_coin < p:
                            G.add_edge(n, m)
            except KeyError:
                print(g, j)
                pass
    adj = np.array(nx.adjacency_matrix(G).todense())
    return G, adj


def cycle_graph(N):
    G3 = nx.generators.cycle_graph(n=N)
    G3.add_nodes_from(np.arange(N))
    Cycle_adj = np.array(nx.adjacency_matrix(G3).todense())
    return G3, Cycle_adj


def barbell_graph(N):
    G = nx.generators.barbell_graph(int(N / 2), 0)
    adj = np.array(nx.adjacency_matrix(G).todense())
    return G, adj

#
# ### Creating the networks:
# N = 100
# G1, A1 = configuration_model_graph(N)
# G2, A2 = barbell_graph(N)
# G3, A3 = configuration_model_graph(N)
# G4, A4 = barbell_graph(N)
# G5, A5 = configuration_model_graph(N)
# G6, A6 = erdos_renyi_graph(N, .1)
# G7, A7 = configuration_model_graph(N)
# G8, A8 = erdos_renyi_graph(N, .05)
# G9, A9 = cycle_graph(N)
#
# fig, ax = plt.subplots(3, 3)
# nx.draw(G1, ax=ax[0, 0], node_size=2, width=0.5, node_color='red')
# nx.draw(G2, ax=ax[0, 1], node_size=2, width=0.5, node_color='orange')
# nx.draw(G3, ax=ax[0, 2], node_size=2, width=0.5, node_color='purple')
# nx.draw(G4, ax=ax[1, 0], node_size=2, width=0.5, node_color='green')
# nx.draw(G5, ax=ax[1, 1], node_size=2, width=0.5, node_color='blue')
# nx.draw(G6, ax=ax[1, 2], node_size=2, width=0.5, node_color='gold')
# nx.draw(G7, ax=ax[2, 0], node_size=2, width=0.5, node_color='pink')
# nx.draw(G8, ax=ax[2, 1], node_size=2, width=0.5, node_color='cyan')
# nx.draw(G9, ax=ax[2, 2], node_size=2, width=0.5, node_color='forestgreen')
# plt.show()
# # _, A6 = erdos_renyi_graph(N, .1)

# ### BASE CASE: ALL SAME NETWORK
# _, A1 = configuration_model_graph(N)
# _, A2 =configuration_model_graph(N)
# # _, A2 =configuration_model_graph(N)
# _, A3 =configuration_model_graph(N)
# _, A4 =configuration_model_graph(N)
# # _, A4 =configuration_model_graph(N)
# _, A5 =configuration_model_graph(N)
# _, A6 =configuration_model_graph(N)
# _, A7 =configuration_model_graph(N)
# _, A8 =configuration_model_graph(N)
# _, A9 =configuration_model_graph(N)

#
# ### Running them in full temporal mode:
# # t_intervals = int(np.linspace(10, 61, 6))
# t_interval = 10
# beta = .005
# model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=6*t_interval,
#                         networks={t_interval: A1, 2*t_interval: A2, 3*t_interval: A3,
#                                   4*t_interval: A4, 5*t_interval: A5, 6*t_interval: A6, 7*t_interval: A7,
#                                   8*t_interval: A8, 9*t_interval: A9})
# solution_t_temporal, solution_p = model.solve_model()
# temporal_timeseries = np.sum(solution_p, axis=0)
# plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
# model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=6*t_interval,
#                         networks={9*t_interval: (A1+A2+A3+A4+A5+A6+A7+A8+A9)/9})
# solution_t_agg, solution_p = model.solve_model()
# aggregate_timeseries = np.sum(solution_p, axis=0)
# plt.plot(solution_t_agg, aggregate_timeseries, label='fully aggregated')
# plt.legend()
# plt.show()
# # So, as we can see from this plot, the number of infected individuals agrees at the end, but the
# # timeseries themselves look drastically different. We eventually want to do a good job of modeling
# # the correct timeseries, by doing a sort of step-wise integration where we can compress SOME but not
# # all of the layers by taking their end errors (not caring about the time series for tiny tau)
# # so they build up a good fitting time series that looks like this one
#
# # Example by hand: from looking at the networks, this version might make the most sense for
# # compression (then we'll do it via an error approximation):
# # model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=6*t_interval,
# #                         networks={t_interval: A1, 2*t_interval: A2, 6*t_interval: (A3+A4+A5+A6)/4})
# # solution_t, solution_p = model.solve_model()
# # partial_aggregate = np.sum(solution_p, axis=0)
# plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal')
# plt.plot(solution_t_agg, aggregate_timeseries, label='fully aggregated')
# # plt.plot(solution_t, partial_aggregate, label='partial aggregated')
# plt.legend()
# plt.show()

# ### Computing pairwise compression errors:
# pair_errors = {('A1', 'A2'): 0, ('A2', 'A3'): 0, ('A3', 'A4'): 0, ('A4', 'A5'): 0, ('A5', 'A6'): 0}
# pair_ids = {'A1':A1, 'A2':A2, 'A3':A3, 'A4':A4, 'A5':A5, 'A6':A6}
# pair_order = {'A1': 1*t_interval, 'A2': t_interval*2, 'A3': t_interval*3, 'A4': t_interval*4, 'A5': t_interval*5, 'A6': t_interval*6} # find a better way to store this info
# pair_order_flip = {1*t_interval: A1, t_interval*2: A2, t_interval*3: A3, t_interval*4: A4, t_interval*5: A5, t_interval*6: A6} # find a better way to store this info
# for pair in list(pair_errors.keys()):
#     pair_errors[pair] = new_matexp_approximation(pair_ids[pair[0]], pair_ids[pair[1]], t_interval, beta)
#
# sorted_pair_error = {k: v for k, v in sorted(pair_errors.items(), key=lambda item: item[1])}
# print(sorted_pair_error)
#
# #### TODO: better way to automatically compute errors and compress along the way
# ## the original time-to-network map
# end_time_network_map = {t_interval: A1, 2*t_interval: A2, 3*t_interval: A3,
#                                   4*t_interval: A4, 5*t_interval: A5, 6*t_interval: A6, 7*t_interval: A7,
#                                   8*t_interval: A8, 9*t_interval: A9}
### setting up a new one that can be modified
# new_end_time_network_map = {}
# # for k, v in end_time_network_map.items():
# #     new_end_time_network_map[k] = v
# ### compressing pairs one by one
# errors_by_start_time = {}
# ## now a new dict where the value for each time stamp is the error between that matrix and the next one
# for timestamp, matrix in end_time_network_map.items():
#     try:
#         errors_by_start_time[timestamp] = new_matexp_approximation(end_time_network_map[timestamp], end_time_network_map[timestamp + t_interval], t_interval, beta)
#     except KeyError:
#         continue
# sorted_pair_error = {k: v for k, v in sorted(errors_by_start_time.items(), key=lambda item: item[1])}
# compression_amt = 6
# times_to_compress = list(sorted_pair_error.keys())[:compression_amt]
# # now times_to_compress are the timestamps to compress that with the next timestamp
# for t in times_to_compress:
#     matrix_one = end_time_network_map[t]
#     matrix_two = end_time_network_map[t+t_interval]
#     compressed_matrix = (matrix_one+matrix_two)/2
#     new_end_time_network_map[t+t_interval] = compressed_matrix # TODO need to make it so that not just all pairs get smooshed, but multiple consecutive networks in a row get smooshed
# for t in list(end_time_network_map.keys()):
#     if t not in new_end_time_network_map.keys():
#         new_end_time_network_map[t] = end_time_network_map[t]
#
# ####

### Picking the best compression:
# def compress_3(matrices):
#     first_matrix = pair[0]
#     second_matrix = pair[0]
#     third_matrix = pair[0]
#     t_first = pair_order[first_matrix]
#     highest_pair = first_matrix
#     t_second = pair_order[second_matrix]
#     if t_second > t_first:
#         highest_pair = second_matrix
#     t_third = pair_order[third_matrix]
#     if t_third > t_second and t_third > t_first:
#         highest_pair = third_matrix
#     highest_pair_matrix = pair_ids[highest_pair]
#     # TODO finish this tomorrow

# pick the lowest 2 errors to compress:
# compression_amt = 2
# pairs_to_compress = list(sorted_pair_error.keys())[:compression_amt]
# # finish some smooth way to then automate setting up a model with the top two smallest errors compressed.
# new_network_set = {}
# for k, v in pair_order_flip.items():
#     new_network_set[k] = v
# for pair in pairs_to_compress:
#     lower_pair = pair[0]
#     lower_pair_matrix = pair_ids[lower_pair]
#     higher_pair = pair[1]
#     higher_pair_matrix = pair_ids[higher_pair]
#     t_higher_pair = pair_order[higher_pair]
#     new_network_set[t_higher_pair] = (lower_pair_matrix+higher_pair_matrix) / 2
#     new_network_set.pop(pair_order[lower_pair])
# # TODO: need to handle when 2 pairs contain one of the same index


# # For now, just doing it by hand because it's complicated.
# # pairs to compress are:
# #('A5', 'A6'): 0.009870406874999994, ('A3', 'A4'): 0.06965407687499998
# model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=9*t_interval,
#                         networks=new_end_time_network_map)
# solution_t_compressed, solution_p = model.solve_model()
# compressed_aggregate = np.sum(solution_p, axis=0)
# plt.plot(solution_t_temporal, temporal_timeseries, label='fully temporal', color='m', lw=4)
# plt.plot(solution_t_agg, aggregate_timeseries, label='fully aggregated', color='y', lw=4)
# # plt.plot(solution_t, partial_aggregate, label='partial aggregated')
# plt.plot(solution_t_compressed, compressed_aggregate, label='compression algorithm', color='c', lw=2.5)
# ## vertical lines to show compression
# plt.vlines(end_time_network_map.keys(), ymin=0, ymax=100, ls=':', color='m', lw=1, alpha=0.5)
# plt.vlines(list(set(end_time_network_map.keys()).difference(set(times_to_compress))), ymin=0, ymax=100, ls='--', color='c', lw=1.5)
# plt.xlabel('Time')
# plt.ylabel('Number nodes infected')
# plt.xticks(list(end_time_network_map.keys()))
# plt.legend()
# plt.show()


### Running them in compressed mode:

### Running them in full aggregate mode:

# # Next idea for another figure:
# X-axis: Number of pairs compressed (or number of compression steps)
# y-axis: Time-series error for every point t (like an integral)
# compare random vs pairwise vs full aggregate or something
# need to make a random algo for compression / better pairwise one

# # pseudocode for TDD version:
# set up a set of networks where I know the outcome
# doesn't matter what the error specifically is (because they are random networks) but I know which pairs should
# be compressed
# Make 2 assertions: this pair IS compressed/ranked
# this pair is not compressed
# then, in a DIFFERENT test, measure the error, and use a static (non-random) network here.
