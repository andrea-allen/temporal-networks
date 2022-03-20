import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import datetime
import random
import numpy as np
import time
import matrix_ops
import math
import seaborn as sns
from scipy import stats
from epintervene.simobjects import AbsoluteTimeNetworkSwitchSim
from epintervene.simobjects import Simulation
from epintervene.simobjects import network as netbd
# beta delta t times leading eigenvalue (R0=beta*eigen_val = spreading rate X t gives a number)
# use relative value: half of characteristic value? (Then use relative error as percentage of this leading spreading number)
# Plot as percentage wrong vs analytical / characteristic

## Goal: want to have infrastructure to feed some piece of code two layers of a network (as a graph and/or adjacency matrix)
## Also be able to feed it given end timesteps, rate beta, and appropriate switch time
## have some function that runs a first ensemble of epidemic simulations on the pair of networks and gives back the epidemic
## peak, in time steps, so that then you can specify how long to run the simulation based on that (and it can be changed automatically)
## plot error over range
## randomize edges of network from sociopatterns?
## work on this tomorrow
## do a clear base case thing

# full one first, empty one second
def base_case_experiment():
    #Gives virtually no error numerically, since the curves "meet" after t=40, with beta cut in half
    beta = .05
    t = 40
    gamma = .0001
    graph1 = nx.generators.erdos_renyi_graph(n=100, p=5/100)
    graph2 = nx.generators.erdos_renyi_graph(n=100, p=0)
    ensemble_temporal(graph1, graph2, t/2, t, beta, gamma, 50, show=True)

#completely disjoint networks
def base_case_experiment2(t=5, beta=.05,show_graphs=False, show_networks=False):
    #Gives virtually no error numerically, since the curves "meet" after t=40, with beta cut in half
    # beta = .05
    gamma = .0001
    whoole_graph = nx.generators.erdos_renyi_graph(n=100, p=4/100)
    graph1 = nx.subgraph(whoole_graph, list(np.arange(50)))
    graph2 = nx.subgraph(whoole_graph, list(np.arange(50,100)))
    graph1 = graph1.copy()
    graph1.add_nodes_from(graph2.nodes())
    graph2 = graph2.copy()
    graph2.add_nodes_from(graph1.nodes())
    recon_graph = nx.Graph()
    recon_graph.add_nodes_from(graph2.nodes())
    recon_graph.add_edges_from(graph1.edges())
    recon_graph.add_edges_from(graph2.edges())

    if show_networks:
        pos = nx.spring_layout(whoole_graph)
        plt.figure('1')
        nx.draw(graph1, pos, with_labels=True)
        plt.figure('2')
        nx.draw(graph2, pos, with_labels=True)
        plt.figure('3')
        nx.draw(recon_graph, pos, with_labels=True)
        plt.show()
    _, final_error, theo_error, _, _, _, _, _, _, _, _, _ = ensemble_temporal(graph1, graph2, t/2, t, beta, gamma, 50, show=show_graphs)
    return final_error, theo_error

def ensemble_base_case_2(show_now=False):
    # This plot will show a relationship between numerical and predicted error over a range of tau, fixed beta
# and a fixed network, but where the network error should be non-negligible
    numerical_results = []
    theo_results = []
    tau_val = []
    random_betas = np.arange(.1, .95, .1)
    random_t = np.arange(.05, 15, .5)
    total_num = len(random_betas)*len(random_t)
    current_num = 0
    for b in list(random_betas)[:20]:
        for t in list(random_t)[:20]:
            print(f'beta: {b}, t: {t}, tau: {b*t}')
            current_num+=1
            if (b*t < 1.82) and (b*t > .001):
                print(f'RUNNING {current_num} out of {total_num}')
                final_error, theo_error = base_case_experiment2(t=t, beta=b, show_graphs=False, show_networks=False)
                numerical_results.append(final_error)
                theo_results.append(theo_error)
                tau_val.append(b*t)
    plt.scatter(theo_results, numerical_results, label=tau_val)
    for i in range(len(numerical_results)):
        plt.text(theo_results[i], numerical_results[i], np.round(tau_val[i], 2))

    plt.legend('upper left')
    if show_now:
        plt.show()

def experiment_er(t=5, beta=.05, show_graphs=False):
    # beta = .05
    gamma = .0001
    graph1 = nx.generators.erdos_renyi_graph(n=100, p=1.5/100)
    graph2 = nx.generators.erdos_renyi_graph(n=100, p=10/100)

    if show_graphs:
        plt.figure('1')
        nx.draw(graph1)
        plt.figure('2')
        nx.draw_spring(graph2)
        plt.show()
    _, final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = ensemble_temporal(graph1, graph2, t / 2, t, beta, gamma, 50, show=show_graphs)
    return final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB

def experiment_WS(t=5, beta=0.5, show_graphs=False, show_networks=True):
    gamma = .0001
    graph1 = nx.generators.watts_strogatz_graph(n=100, k=4, p=7/100) #1008
    # graph1 = nx.generators.watts_strogatz_graph(n=100, k=2, p=5/100)
    graph2 = nx.generators.erdos_renyi_graph(n=100, p=1.5/100) #for ER 1001
    # graph2 = nx.generators.erdos_renyi_graph(n=100, p=10/100) #for ER 1002

    if show_networks:
        plt.figure('1')
        nx.draw(graph1)
        plt.figure('2')
        nx.draw_spring(graph2)
        plt.show()
    _, final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = ensemble_temporal(graph1, graph2, t / 2, t, beta, gamma, 50, show=show_graphs)
    return final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB

def experiment_SBM(t=5, beta=0.5, show_graphs=False, show_networks=True, prob_matrix_1=None, prob_matrix_2=None):
    gamma = .0001
    graph1 = temporal_stochastic_block_model(90, prob_matrix_1)
    graph2 = temporal_stochastic_block_model(90, prob_matrix_2)

    if show_networks:
        plt.figure('1')
        nx.draw(graph1)
        plt.figure('2')
        nx.draw_spring(graph2)
        plt.show()
    _, final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = ensemble_temporal(graph1, graph2, t / 2, t, beta, gamma, 50, show=show_graphs)
    return final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB

def experiment_SBM_ER(t=5, beta=0.5, show_graphs=False, show_networks=True, prob_matrix_1=None):
    #TODO start here
    gamma = 1
    graph1 = temporal_stochastic_block_model(99, prob_matrix_1)
    graph2 = nx.generators.erdos_renyi_graph(n=100, p=1.5/100) #for ER 1001

    if show_networks:
        plt.figure('1')
        nx.draw(graph1)
        plt.figure('2')
        nx.draw_spring(graph2)
        plt.show()
    _, final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = ensemble_temporal(graph1, graph2, t / 2, t, beta, gamma, 50, show=show_graphs)
    return final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB

### KEEP THIS CODE
def ensemble_experiment_er(show_now=False, random_betas=None, random_t=None):
    # This plot will show a relationship between numerical and predicted error over a range of tau, fixed beta
# and a fixed network, but where the network error should be non-negligible
    numerical_results = []
    theo_results = []
    tau_val = []
    if random_betas is None:
        random_betas = np.arange(.1, .95, .1)
    if random_t is None:
        random_t = np.arange(.05, 15, .2)
    total_num = len(random_betas)*len(random_t)
    current_num = 0
    row_results = []
    for b in list(random_betas):
        for t in list(random_t):
            print(f'beta: {b}, t: {t}, tau: {b*t}')
            current_num+=1
            if (b*t < 1.82) and (b*t > .001):
                print(f'RUNNING {current_num} out of {total_num}')
                final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_er(t=t, beta=b, show_graphs=False) #need to capture all the elements we wanted to record here
                numerical_results.append(final_error)
                theo_results.append(theo_error)
                tau_val.append(b*t)
                row_results.append([b, t, eigenvalue_1, eigenvalue_2, eigenvalue3, theo_error, count_temp, count_agg, AAB, ABB, ABA, BAB, 50]) #50 num sims
    if show_now:
        plt.scatter(theo_results, numerical_results, label=tau_val)
        for i in range(len(numerical_results)):
            plt.text(theo_results[i], numerical_results[i], np.round(tau_val[i], 2))

    plt.legend('upper left')
    if show_now:
        plt.show()
    return row_results


def ensemble_experiment_WS(show_now=False, random_betas=None, random_t=None):
    # This plot will show a relationship between numerical and predicted error over a range of tau, fixed beta
# and a fixed network, but where the network error should be non-negligible
    numerical_results = []
    theo_results = []
    tau_val = []
    if random_betas is None:
        random_betas = np.arange(.1, .95, .1)
    if random_t is None:
        random_t = np.arange(.05, 15, .5)
    total_num = len(random_betas)*len(random_t)
    current_num = 0
    row_results = []
    for b in list(random_betas):
        for t in list(random_t):
            print(f'beta: {b}, t: {t}, tau: {b*t}')
            current_num+=1
            if (b*t < 1.82) and (b*t > .001):
                print(f'RUNNING {current_num} out of {total_num}')
                if b*t > 1.5: #this is to show that even tau of 1.5 is "early" spread... the epidemics haven't reached their peaks (shown in the figure screenshot) need a good way to determine the limit of "early" based on beta, tau, and the network, maybe the leading eigenvalue
                    final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_WS(t=t, beta=b, show_graphs=False, show_networks=False)
                else:
                    final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_WS(t=t, beta=b, show_graphs=False, show_networks=False)
                numerical_results.append(final_error)
                theo_results.append(theo_error)
                tau_val.append(b*t)
                row_results.append(
                    [b, t, eigenvalue_1, eigenvalue_2, eigenvalue3, theo_error, count_temp, count_agg, AAB, ABB, ABA,
                     BAB, 50])  # 50 num sims
    if show_now:
        plt.scatter(theo_results, numerical_results, label=tau_val)
    for i in range(len(numerical_results)):
        plt.text(theo_results[i], numerical_results[i], np.round(tau_val[i], 2))

    if show_now:
        plt.legend('upper left')
        plt.show()
    return row_results


def ensemble_experiment_SBM(show_now=False, random_betas=None, random_t=None, prob_matrix_1=None, prob_matrix_2=None):
    # This plot will show a relationship between numerical and predicted error over a range of tau, fixed beta
# and a fixed network, but where the network error should be non-negligible
    numerical_results = []
    theo_results = []
    tau_val = []
    if random_betas is None:
        random_betas = np.arange(.1, .95, .1)
    if random_t is None:
        random_t = np.arange(.05, 15, .5)
    total_num = len(random_betas)*len(random_t)
    current_num = 0
    row_results = []
    for b in list(random_betas):
        for t in list(random_t):
            print(f'beta: {b}, t: {t}, tau: {b*t}')
            current_num+=1
            if (b*t < 5) and (b*t > .001):
                print(f'RUNNING {current_num} out of {total_num}')
                if b*t > 1.5: #SBM was less good fit, but because it was "beyond peak"... need to investigate
                    final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_SBM(t=t, beta=b, show_graphs=False, show_networks=False, prob_matrix_1=prob_matrix_1, prob_matrix_2=prob_matrix_2)
                else:
                    final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_SBM(t=t, beta=b, show_graphs=False, show_networks=False, prob_matrix_1=prob_matrix_1, prob_matrix_2=prob_matrix_2)
                numerical_results.append(final_error)
                theo_results.append(theo_error)
                tau_val.append(b*t)
                row_results.append(
                    [b, t, eigenvalue_1, eigenvalue_2, eigenvalue3, theo_error, count_temp, count_agg, AAB, ABB, ABA,
                     BAB, 50])  # 50 num sims
    if show_now:
        plt.scatter(theo_results, numerical_results, label=tau_val)
        plt.legend('upper left')
    for i in range(len(numerical_results)):
        plt.text(theo_results[i], numerical_results[i], np.round(tau_val[i], 2))

    if show_now:
        plt.show()
    return row_results

def ensemble_experiment_SBM_ER(show_now=False, random_betas=None, random_t=None, prob_matrix_1=None, prob_matrix_2=None):
    # This plot will show a relationship between numerical and predicted error over a range of tau, fixed beta
# and a fixed network, but where the network error should be non-negligible
    numerical_results = []
    theo_results = []
    tau_val = []
    if random_betas is None:
        random_betas = np.arange(.1, .95, .1)
    if random_t is None:
        random_t = np.arange(.05, 15, .5)
    total_num = len(random_betas)*len(random_t)
    current_num = 0
    row_results = []
    for b in list(random_betas):
        for t in list(random_t):
            print(f'beta: {b}, t: {t}, tau: {b*t}')
            current_num+=1
            if (b*t < 2) and (b*t > .001):
                print(f'RUNNING {current_num} out of {total_num}')
                if b*t > 1.0: #SBM was less good fit, but because it was "beyond peak"... need to investigate
                    final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_SBM_ER(t=t, beta=b, show_graphs=False, show_networks=False, prob_matrix_1=prob_matrix_1)
                else:
                    final_error, theo_error, count_agg, count_temp, eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB = experiment_SBM_ER(t=t, beta=b, show_graphs=False, show_networks=False, prob_matrix_1=prob_matrix_1)
                numerical_results.append(final_error)
                theo_results.append(theo_error)
                tau_val.append(b*t)
                row_results.append(
                    [b, t, eigenvalue_1, eigenvalue_2, eigenvalue3, theo_error, count_temp, count_agg, AAB, ABB, ABA,
                     BAB, 50])  # 50 num sims
    if show_now:
        plt.scatter(theo_results, numerical_results, label=tau_val)
        plt.legend('upper left')
    for i in range(len(numerical_results)):
        plt.text(theo_results[i], numerical_results[i], np.round(tau_val[i], 2))

    if show_now:
        plt.show()
    return row_results


def computeEdges(G):
    return len(G.edges())

def computeClustering(G):
    clustering = nx.clustering(G)
    cluster_sum = np.sum(list(clustering.values()))
    avg_clustering = cluster_sum / len(list(clustering.values()))
    return avg_clustering

def avgDegrees(G):
    degree_hist = nx.degree_histogram(G)
    degree_dist = degree_hist / np.sum(degree_hist)
    avg_degree_k = np.sum([k*degree_dist[k] for k in range(len(degree_dist))])
    excess_distribution = np.zeros(len(degree_dist))
    for i in range(len(degree_dist)-1):
        excess_distribution[i] = (i+1) * degree_dist[i+1]
    excess_distribution = excess_distribution / avg_degree_k
    avg_excess_degree_q = np.sum([q * excess_distribution[q] for q in range(len(excess_distribution))])
    return avg_degree_k, avg_excess_degree_q

def network_statistics(keyword, N, p, WS_k=None):
    # Create 5-10 networks, compute their scores, average them and return the values
    edges = 0
    clustering = 0
    k = 0
    q = 0
    for i in range(10):
        if keyword.upper() == 'ER1':
            G = nx.generators.erdos_renyi_graph(N, p=p)
        elif keyword.upper() == 'ER2':
            G = nx.generators.erdos_renyi_graph(N, p=p)
        elif keyword.upper() == 'WS1':
            G = nx.generators.watts_strogatz_graph(n=N, k=WS_k, p=p)
        elif keyword.upper() == 'WS2':
            G = nx.generators.watts_strogatz_graph(n=N, k=WS_k, p=p)
        elif keyword.upper() == 'SBM1':
            prob_matrix_1 = np.array([[.01, .02, .03],
                                      [.02, .1, .02],
                                      [.03, .02, .4]])
            G = temporal_stochastic_block_model(90, prob_matrix_1)
        elif keyword.upper() == 'SBM2':
            prob_matrix_2 = np.array([[.1, .2, .3],
                                      [.2, .1, .2],
                                      [.3, .2, .4]])
            G = temporal_stochastic_block_model(90, prob_matrix_2)
        elif keyword.upper() == 'SBM3':
            prob_matrix_2 = np.array([[.02, .25, .25], # low in-groups, high out groups
                                      [.25, .02, .25],
                                      [.25, .25, .02]])
            G = temporal_stochastic_block_model(99, prob_matrix_2)
        elif keyword.upper() == 'SBM4':
            prob_matrix_2 = np.array([[.25, .02, .02], # high in-groups, low out groups
                                      [.02, .25, .02],
                                      [.02, .02, .25]])
            G = temporal_stochastic_block_model(99, prob_matrix_2)
            nx.draw_spring(G)
            plt.savefig('sbm_example_fig.png')
            plt.show()
        elif keyword.upper() == 'SBM5':
            prob_matrix_2 = np.array([[.1, .02, .02], # higher in-groups, low out groups
                                      [.02, .1, .02],
                                      [.02, .02, .1]])
            G = temporal_stochastic_block_model(99, prob_matrix_2)

        edges += computeEdges(G)
        clustering += computeClustering(G)
        k_now, q_now = avgDegrees(G)
        k += k_now
        q += q_now
    edges = edges/10
    clustering = clustering/10
    k = k/10
    q = q/10

    return edges, clustering, k, q






def ensemble_temporal(graph1, graph2, intervention_time, max_time, beta, gamma, num_runs, show=False, p_zeros=1):
    # Do something to catch/ estimate the average peak of an epidemic (the timescale thing) captured by eigenvalue?
    nb = netbd.NetworkBuilder
    # graph1.add_edges_from(list(graph2.edges()))
    adjlist1 = nb.create_adjacency_list(graph1, multi=False)
    adjlist2 = nb.create_adjacency_list(graph2, multi=False)

    G_combined = nx.MultiGraph() # TODO change this to a multigraph once changes are in place
    G_combined.add_edges_from(graph1.edges())
    G_combined.add_edges_from(graph2.edges())

    A_combo = np.array(nx.adjacency_matrix(G_combined, nodelist=np.arange(0,len(G_combined.nodes()))).todense())
    np.fill_diagonal(A_combo, 0)
    eigenvalue3 = np.linalg.eigvals(A_combo)[0]

    #### degree stuff
    # degree_hist = nx.degree_histogram(G_combined)
    # degree_dist = degree_hist / np.sum(degree_hist)
    # avg_degree_k = np.sum([k*degree_dist[k] for k in range(len(degree_dist))])
    # excess_distribution = np.zeros(len(degree_dist))
    # for i in range(len(degree_dist)-1):
    #     excess_distribution[i] = (i+1) * degree_dist[i+1]
    # excess_distribution = excess_distribution / avg_degree_k
    # avg_excess_degree_q = np.sum([q * excess_distribution[q] for q in range(len(excess_distribution))])
    ####

    adjlist_combined = nb.create_adjacency_list(G_combined, multi=True)

    inft_reg_results = np.zeros(1000)
    rec_reg_results = np.zeros(1000)
    inft_results = np.zeros(1000)
    rec_results = np.zeros(1000)
    total_count_agg = 0
    total_count_temp = 0
    # TODO idea from 10/1 here would be to add recovery number to the error number because its total number infected
    for i in range(num_runs):
        print(i)
        sim_regular = Simulation(N=len(adjlist_combined), adj_list=adjlist_combined)
        sim_regular.set_uniform_beta(beta/2)  # Split beta for regular simulation
        sim_regular.set_uniform_gamma(gamma)
        if p_zeros > 1:
            p_zeros_list = [None for i in range(p_zeros)]
        elif p_zeros == 1:
            p_zeros_list = None
        sim_regular.run_sim(with_memberships=False, wait_for_recovery=True, uniform_rate=True, p_zero=p_zeros_list)
        ts, infct, rec = sim_regular.tabulate_continuous_time(time_buckets=1000, custom_range=True,
                                                              custom_t_lim=max_time)


        sim = AbsoluteTimeNetworkSwitchSim(N=len(adjlist1), adjlist=adjlist1)
        sim.configure_intervention(intervention_time=intervention_time, new_adjlist=adjlist2)
        sim.set_uniform_beta(beta)
        sim.set_uniform_gamma(gamma)
        sim.run_sim(with_memberships=False, wait_for_recovery=True, uniform_rate=True, p_zero=p_zeros_list)
        ts_switch, infct_switch, rec_switch = sim.tabulate_continuous_time(time_buckets=1000, custom_range=True, custom_t_lim=max_time)

        # TODO need to figure out how to address this because when the number is too big, then early tau value results
        # aren't counted at all, but too low of a number causes tau vals of higher to have their average brought down
        # if np.sum(infct_switch)/len(adjlist_combined) > 6:
        #     total_count += 1 ## what the hell was this????
        ## correlation is now working, but maybe there's a TIME issue? Is the continuous time
        # being allowed to run the same time as the deterministic solution? The graphs are well connected
        # so not sure what's wrong 
        if infct_switch[-1]-infct_switch[0] > 0:
            total_count_temp += 1
            inft_results += infct_switch
            rec_results += rec_switch
            print(f'TOTAL TEMP {infct_switch[-1] - infct_switch[0]} INFECTED')
        else:
            print(f'ONLY TEMP {infct_switch[-1]-infct_switch[0]} INFECTED')
        if infct[-1] - infct[0] > 0:
            inft_reg_results += infct
            rec_reg_results += rec
            total_count_agg += 1
            print(f'TOTAL AGG {infct[-1] - infct[0]} INFECTED')
        else:
            print(f'ONLY AGG {infct[-1]-infct[0]} INFECTED')

    inft_reg_results = inft_reg_results / total_count_agg
    rec_reg_results = rec_reg_results / total_count_agg
    inft_results = inft_results / total_count_temp
    rec_results = rec_results / total_count_temp


    diff_sum = np.sum(np.abs(inft_reg_results - inft_results))
    diff_sum = np.sum(np.abs(inft_reg_results+rec_reg_results - (inft_results + rec_results)))          # Adding recovery to the numbers
    final_diff = np.sum(np.abs((inft_reg_results[-1] + rec_reg_results[-1]) - (inft_results[-1]+ rec_results[-1]))) # adding recovery to the numbers
    final_diff = np.abs((inft_reg_results[-1]+rec_reg_results[-1]) - (inft_results[-1] + rec_results[-1]))
    A = np.array(nx.adjacency_matrix(graph1, nodelist=np.arange(0,len(graph1.nodes()))).todense())
    np.fill_diagonal(A, 0)
    # Need to sort the nodes first to avoid adjacency matrix mis-labeling
    A_switch = np.array(nx.adjacency_matrix(graph2, nodelist=np.arange(0,len(graph1.nodes()))).todense())
    np.fill_diagonal(A_switch, 0)
    # TODO: eigenvalues?
    eigenvalue_1 = np.linalg.eigvals(A)[0]
    eigenvalue_2 = np.linalg.eigvals(A_switch)[0]
    theo_error, AAB, ABB, ABA, BAB = matrix_ops.theoretical_error(A, A_switch, beta=beta, t=intervention_time)
    print(f'Numerical error after {max_time}: {final_diff}')
    print(f'Theoretical error after {max_time}: {theo_error}')
    if show:
        plt.figure(2)
        plt.plot(ts, inft_reg_results, label='Regular Infected')
        plt.plot(ts, rec_reg_results, label='Regular Recovered')
        # plt.legend(loc='uppwer right')
        #
        plt.plot(ts, inft_results, label='Net Switch Infected')
        plt.plot(ts, rec_results, label='Net Switch Recovered')
        plt.vlines(intervention_time, ymin=0, ymax=np.max(rec_reg_results), ls='--', color='red',
                   label='Network switch time')
        # plt.scatter(intervention_time, (beta*(intervention_time**2)*avg_excess_degree_q/2), color='orange', s=15)
        # plt.text(intervention_time, (beta*(intervention_time**2)*avg_excess_degree_q/2), f'{avg_excess_degree_q}')
        plt.scatter(intervention_time, np.abs(beta*(intervention_time)*eigenvalue3), color='yellow', s=15)
        plt.text(intervention_time, np.abs(beta*(intervention_time)*eigenvalue3), f'{eigenvalue3}')

        plt.legend(loc='upper right')
        # plt.show()

        plt.figure('difference')
        plt.plot(ts, np.abs(inft_reg_results - inft_results) / (max(np.abs(inft_reg_results - inft_results))),
                 label='Difference in Infections')
        plt.vlines(intervention_time, ymin=0, ymax=1, ls='--', color='red', label='Network switch time')
    if show:
        plt.show()

    return diff_sum, final_diff, theo_error, inft_reg_results[-1]+rec_reg_results[-1], inft_results[-1]+rec_results[-1], eigenvalue_1, eigenvalue_2, eigenvalue3, AAB, ABB, ABA, BAB


    #
   # layer1, layer2 = parse_data('./tij_InVS.dat', 30400, 60)
def socio_patterns1():
    layer1, layer2 = parse_data('./tij_InVS.dat', 30400, 100)
    ensemble_temporal(layer1, layer2, intervention_time=60, max_time=100, beta=.99, gamma=.01, num_runs=100, show=True)

def parse_data(filename, t_start, increment):
    if 'Hospital' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j', 'i_type', 'j_type']
        df = df[['t', 'i', 'j']]
    elif 'listcontacts' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j']
    else:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j']
    print(f'There are {len(set(df["i"]).union(set(df["j"])))} nodes')
    layer1, layer2 = produce_layers(df, t_start, increment)
    return graphOf(layer1, layer2), graphOf(layer2, layer1)

def parse_data_hospital(filename, t_start, increment):
    df = pd.read_csv(filename, delimiter='\t', header=None)
    df.columns = ['t', 'i', 'j', 'i_type', 'j_type']
    df = df[['t', 'i', 'j']]
    print(f'There are {len(set(df["i"]).union(set(df["j"])))} nodes')
    layer1, layer2 = produce_layers(df, t_start, increment)
    return graphOf(layer1, layer2), graphOf(layer2, layer1)

def dataset_statistics(filename):
    """
    Compute some statistics on the given temporal network
    :param filename:
    :return:
    """
    if 'Hospital' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j', 'type_i', 'type_j']
    elif 'listcontacts' in filename:
        df = pd.read_csv(filename, delimiter='\t', header=None)
        df.columns = ['t', 'i', 'j']
    else:
        df = pd.read_csv(filename, delimiter=' ', header=None)
        df.columns = ['t', 'i', 'j']
    info = {}
    unique_timestamps = set(df['t'])
    info['num_timestamps'] = len(unique_timestamps)
    info['max_timestamp'] = max(unique_timestamps)
    info['min_timestamp'] = min(unique_timestamps)
    ## WANT: Distribution of contacts per timestamp
    ## Distribution of frequency of the same contact
    histo = plt.hist(df.groupby(['i', 'j']).size(), bins='auto')
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(8,4)
    ax[0].hist(df.groupby(['t']).size(), bins='auto')  # this is how many contacts per timestep
    ax[0].semilogy()
    ax[0].set_xlabel('Contacts (edges)\n per timestamp')
    ax[0].set_ylabel('Distribution (log)')
    ax[1].scatter(np.log10(np.arange(len(histo[0]))), np.log10(histo[0]), s=8, color='blue')
    ax[1].set_xlabel('Frequency of same\n contact (log scale)')
    ax[1].set_ylabel('Distribution (log)')
    static_dd = df.groupby('i')['j'].nunique()
    ax[2].hist(static_dd, color='green', alpha=0.6, density='true', label=f'<k>={np.round(np.mean(static_dd), 1)}')
    # check: set(df.where(df['i']==122).dropna()['j'])
    ax[2].set_xlabel('Degree in static network')
    ax[2].set_ylabel('Distribution')
    ax[2].legend()
    durations = df['t'].diff(1).dropna() # really weird giant duration gap?
    ax[3].scatter(np.arange(len(durations)), durations + 1, s=8, color='green') # +1 to make log scale work, still effectively 0
    ax[3].semilogy()
    ax[3].set_xlabel('Time steps \nin dataset')
    ax[3].set_ylabel('Time between consecutive timestamps')
    plt.tight_layout(0.1)
    plt.show()

    return info



def produce_layers(df, t_start, increment):
    layer_1 = df[['i', 'j']].where((df['t'] >= t_start) & (df['t'] < t_start+increment))
    layer_2 = df[['i', 'j']].where((df['t'] >= t_start+increment) & (df['t'] < t_start+2*increment))
    return layer_1, layer_2

def graphOf(layer, second_layer):
    """
    Makes a networkx graph of nodes AND EDGES from layer1, (first arugment)
    Adds nodes from second_layer to preserve same dimensions
    :param layer:
    :param second_layer:
    :return:
    """
    layer = layer.dropna()
    layer['i'] = layer['i'].astype(int)
    layer['j'] = layer['j'].astype(int)
    second_layer = second_layer.dropna()
    second_layer['i'] = second_layer['i'].astype(int)
    second_layer['j'] = second_layer['j'].astype(int)
    graph = nx.from_pandas_edgelist(layer.dropna(), 'i', 'j')
    second_layer_graph = nx.from_pandas_edgelist(second_layer.dropna(), 'i', 'j')
    graph.add_nodes_from(nx.nodes(second_layer_graph))
    # nx.draw(graph)
    # plt.show()
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

def temporal_null_hyp():
    intervention_time = 1.5
    max_time = 3
    network_size = 99
    beta_val = .5
    gamma_val = .01
    nb = netbd.NetworkBuilder
    G = nx.generators.erdos_renyi_graph(network_size, p=6/network_size)
    G_switch = nx.generators.erdos_renyi_graph(network_size, p=0)
    adjlist = nb.create_adjacency_list(G)
    adjlist_switch = nb.create_adjacency_list(G_switch)

    num_experiments = 50
    diff_sum = np.zeros(num_experiments)
    final_diff = np.zeros(num_experiments)
    theo_error = np.zeros(num_experiments)
    for expermnt in range(25, num_experiments-20):
        print(f'EXPERIMENT {expermnt}')
        # G = nx.generators.erdos_renyi_graph(network_size, p=(expermnt/10) / network_size)
        G = nx.generators.watts_strogatz_graph(network_size, k=expermnt%10+1, p=.15)
        # G_switch = nx.generators.erdos_renyi_graph(network_size, p=0 / network_size)
        G_switch = nx.generators.watts_strogatz_graph(network_size, k=expermnt%10+1, p=.04)
        adjlist = nb.create_adjacency_list(G)
        N = len(adjlist)
        if N == 0:
            continue
        adjlist_switch = nb.create_adjacency_list(G_switch)
        N_switch = len(adjlist_switch)
        if N_switch == 0:
            continue

        G_combined = nx.Graph()
        G_combined.add_edges_from(G.edges())
        G_combined.add_edges_from(G_switch.edges())

        adjlist_combined = nb.create_adjacency_list(G_combined)


        inft_reg_results = np.zeros(1000)
        rec_reg_results = np.zeros(1000)
        inft_results = np.zeros(1000)
        rec_results = np.zeros(1000)

        num_sims = 500
        for i in range(num_sims):
            #TODO something isn't making sense why the rate is lowering for the switch version
            #need to do some more null hypothesis testing
            print(i)
            sim_regular = Simulation(N=len(adjlist_combined), adj_list=adjlist_combined)
            sim_regular.set_uniform_beta(beta_val) #Split beta for regular simulation
            sim_regular.set_uniform_gamma(gamma_val)
            #TODO fix issue with graph with empty adjacency list
            sim_regular.run_sim(with_memberships=False, wait_for_recovery=True, uniform_rate=True)
            ts, infct, rec = sim_regular.tabulate_continuous_time(time_buckets=1000, custom_range=True, custom_t_lim=max_time)
            inft_reg_results += infct
            rec_reg_results += rec

            sim = AbsoluteTimeNetworkSwitchSim(N=len(adjlist),  adjlist=adjlist)
            sim.configure_intervention(intervention_time=intervention_time, new_adjlist=adjlist_switch)
            sim.set_uniform_beta(beta_val)
            sim.set_uniform_gamma(gamma_val)
            sim.run_sim(with_memberships=False, wait_for_recovery=True, uniform_rate=True)
            ts, infct, rec = sim.tabulate_continuous_time(time_buckets=1000, custom_range=True, custom_t_lim=max_time)
            inft_results += infct
            rec_results += rec

        inft_reg_results = inft_reg_results/num_sims
        rec_reg_results = rec_reg_results/num_sims
        inft_results = inft_results/num_sims
        rec_results = rec_results/num_sims

        plt.figure(2)
        plt.plot(ts, inft_reg_results, label='Regular Infected')
        plt.plot(ts, rec_reg_results, label='Regular Recovered')
        # plt.legend(loc='uppwer right')
        #
        plt.plot(ts, inft_results, label='Net Switch Infected')
        plt.plot(ts, rec_results, label='Net Switch Recovered')
        plt.vlines(intervention_time, ymin=0, ymax=np.max(rec_reg_results), ls='--', color='red', label='Network switch time')

        plt.legend(loc='upper right')
        # plt.show()

        plt.figure('difference')
        plt.plot(ts, np.abs(inft_reg_results-inft_results)/(max(np.abs(inft_reg_results-inft_results))), label='Difference in Infections')
        plt.vlines(intervention_time, ymin=0, ymax=1, ls='--', color='red', label='Network switch time')
        # plt.show()
        diff_sum[expermnt] = np.sum(np.abs(inft_reg_results-inft_results))
        final_diff[expermnt] = np.sum(np.abs(inft_reg_results[-1]-inft_results[-1]))
        A = np.array(nx.adjacency_matrix(G).todense())
        np.fill_diagonal(A, 1)
        A_switch = np.array(nx.adjacency_matrix(G_switch).todense())
        np.fill_diagonal(A_switch, 1)
        theo_error[expermnt] = matrix_ops.theoretical_error(A, A_switch, beta=beta_val, t=intervention_time)
    plt.figure('RESULTS')

    # np.savetxt('./data/temporal/sbm999_t1_2_b5_theo.txt', theo_error, delimiter=',')
    # np.savetxt('./data/temporal/sbm999_t1_2_b5_emp.txt', diff_sum, delimiter=',')
    plt.scatter(theo_error, diff_sum, color='blue')
    plt.xlabel('Theoretical error')
    plt.ylabel('Sum difference in simulation')
    # plt.semilogy()
    plt.legend(loc='upper left')

    plt.figure('final difference')
    plt.scatter(theo_error, final_diff, color='orange')
    plt.xlabel('Theoretical error')
    plt.ylabel('Sum difference in simulation')
    # plt.semilogy()
    plt.legend(loc='upper left')
    plt.show()


def temporal_stochastic_block_model(N, prob_matrix):
    graph1 = nx.Graph()
    graph1.add_nodes_from(np.arange(N))
    num_groups = len(prob_matrix)
    for i in range(num_groups):
        for j in range(i, num_groups):
            density = prob_matrix[i][j]
            for node in range(int(i * (N/num_groups)), int(i * (N/num_groups) + int(N/num_groups))):
                for neighbor in range(int(j * (N/num_groups)), int(j * (N/num_groups) + int(N/num_groups))):
                    flip_coin = random.uniform(0, 1)
                    if flip_coin < density:
                        graph1.add_edge(node, neighbor)

    adj_mat = np.array(nx.adjacency_matrix(graph1, nodelist=np.arange(N), ).todense())
    # nx.draw(graph1, with_labels=True) # check if adjmat is symmetric
    np.fill_diagonal(adj_mat, 0)
    graph1.remove_edges_from(nx.selfloop_edges(graph1))
    # plt.show()
    return graph1



### Linear regression code for reference
    # slope, intcpt, r_val, p_val, std_err  = stats.linregress(final_errors, theo_error)
    # x_range = final_errors
    # y_vals = intcpt + slope*x_range

# Database for results design:
# File 1: NetworkTable
# Data structure: numpy array (pandas)
# PK: NetID (start at 1001, 1002, 1003... etc)
# Columns: NetId, N, AvgEdges, Eigen1, C, k, q, sAAB, sABB, sABA, sBAB


# File 2: ErrorSimsTable
# Data structure: pandas array
# PK: EnsembleId (start at 10001, 10002, etc.)
# Columns: EnsembleId, NetId, beta, dt, epsilon, simError, numSims

def make_network_table():
    network_table = pd.DataFrame(columns=['NetId', 'KeyWord', 'N', 'AvgEdges', 'C', 'k', 'q', 'InsertDateTime'])
    network_table.to_csv('./data/september/NetworkTableV1.csv', index=False)

def make_data_table():
    # Note for future analysis: When agreement starts to get bad, can pick the tau parameters and re-run some sims to see the arch
    ensemble_table = pd.DataFrame(columns=['EnsembleId', 'NetId_1', 'NetId_2', 'beta', 'dt', 'Eigen_1', 'Eigen_2', 'Eigen_3', 'epsilon', 'tempCount', 'aggCount', 'sAAB', 'sABB', 'sABA', 'sBAB', 'numSims', 'InsertDateTime'])
    ensemble_table.to_csv('./data/september/EnsembleTableV2.csv', index=False)

def load_network_table(fpath=None):
    if fpath is not None:
        network_table = pd.read_csv(fpath, delimiter=',')
    else:
        network_table = pd.read_csv('./data/NetworkTable.csv', delimiter=',')
    return network_table

def load_ensemble_table(fpath=None):
    if fpath is not None:
        ensemble_table = pd.read_csv(fpath, delimiter=',')
    else:
        ensemble_table = pd.read_csv('./data/EnsembleTable.csv', delimiter=',')
    return ensemble_table

def save_network_table(network_table, fpath=None):
    if fpath is None:
        today = datetime.datetime.today()
        today_string = today.strftime("%m-%d-%Y-%H:%M:%S")
        network_table.to_csv(f'./data/NetworkTable{today_string}.csv', index=False)
    else:
        network_table.to_csv(fpath, index=False)

def save_ensemble_table(ensemble_table, fpath=None):
    if fpath is None:
        today = datetime.datetime.today()
        today_string = today.strftime("%m-%d-%Y %H:%M:%S")
        ensemble_table.to_csv(f'./data/EnsembleTable{today_string}.csv', index=False)
    else:
        ensemble_table.to_csv(fpath, index=False)

def do_stuff(network_table):
    last_row = len(network_table)
    today = datetime.datetime.today()
    today_string = today.strftime("%m-%d-%Y %H:%M:%S")
    network_table.loc[last_row] = [1002, 'ER',200, 42, 2.3, 4.2, 6.5, 10, 8, 17, 11, 2, today_string]
    # network_table.append(new_results, ignore_index=True)
    return network_table