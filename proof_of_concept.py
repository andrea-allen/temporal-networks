import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import deterministic
import poc_compression
import network_objects
import seaborn as sns
import socio_patterns_ex
from epintervene.simobjects import network as netbd

def random_temporal_network_mix(num_layers, t_interval, beta):
    ### Creating the networks:
    N = 100
    G1, A1 = poc_compression.configuration_model_graph(N)
    G2, A2 = poc_compression.barbell_graph(N)
    G3, A3 = poc_compression.configuration_model_graph(N)
    G4, A4 = poc_compression.erdos_renyi_graph(N, .3)
    G5, A5 = poc_compression.configuration_model_graph(N)
    G6, A6 = poc_compression.erdos_renyi_graph(N, .1)
    G7, A7 = poc_compression.configuration_model_graph(N)
    G8, A8 = poc_compression.erdos_renyi_graph(N, .15)
    G9, A9 = poc_compression.cycle_graph(N)
    G10, A10 = poc_compression.configuration_model_graph(N)
    options = [A1, A2, A2, A2, A2, A3, A4, A5, A6, A7, A8, A9, A10]
    layers = []
    current_start_t = 0
    for l in range(num_layers):
        rand_choice = np.random.randint(0,10)
        layers.append(network_objects.Layer(current_start_t, t_interval+current_start_t, beta, options[rand_choice]))
        current_start_t += t_interval
    # temporal_network = network_objects.TemporalNetwork([network_objects.Layer(0, 1 * t_interval, beta, A1),
    #                                                     network_objects.Layer(10, 2 * t_interval, beta, A2),
    #                                                     network_objects.Layer(20, 3 * t_interval, beta, A3),
    #                                                     network_objects.Layer(30, 4 * t_interval, beta, A4),
    #                                                     network_objects.Layer(40, 5 * t_interval, beta, A5),
    #                                                     network_objects.Layer(50, 6 * t_interval, beta, A6),
    #                                                     network_objects.Layer(60, 7 * t_interval, beta, A7),
    #                                                     network_objects.Layer(70, 8 * t_interval, beta, A8),
    #                                                     network_objects.Layer(80, 9 * t_interval, beta, A9),
    #                                                     network_objects.Layer(90, 10 * t_interval, beta, A10)
    #                                                     ])
    return network_objects.TemporalNetwork(layers)

## TO THINK ABOUT FOR TOMORROW: HOW BEST TO HANDLE THE TIME SERIES DIFFERENTIAL
# do something with binning:
def digitize_me(time_vector, infected, number_layers, t_interval):
    digitized = np.digitize(np.array(time_vector), np.linspace(0,number_layers*t_interval,50))
    bin_means = [np.array(time_vector)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    bins_infected = [np.array(infected)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    return bin_means, bins_infected

def run_random(temporal_network, t_interval, beta, number_layers, levels, iters):
    N = len(temporal_network.layers[0].A)
    random_compressed = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters,
                                                            optimal=False)
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N),
                                          end_time=number_layers * t_interval,
                                          networks=random_compressed.get_time_network_map())
    solution_t_random, solution_p = model.solve_model()
    random_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_random, random_solution, number_layers, t_interval)
    return d[0], d[1], random_compressed

def run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters):
    N = len(temporal_network.layers[0].A)
    optimal_network = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters)
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N),
                                          end_time=number_layers * t_interval,
                                          networks=optimal_network.get_time_network_map())
    solution_t_compressed, solution_p = model.solve_model()
    compressed_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_compressed, compressed_solution, number_layers, t_interval)
    return d[0], d[1], optimal_network

def run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters):
    N = len(temporal_network.layers[0].A)
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=number_layers*t_interval,
                            networks=temporal_network.get_time_network_map())
    solution_t_temporal, solution_p = model.solve_model()
    temporal_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_temporal, temporal_solution, number_layers, t_interval)
    return d[0], d[1], temporal_network


def one_round(temporal_network, t_interval, beta, number_layers, levels, iters, plot=False):
    temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters)
    rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
    opt_t, opt_inf, opt_net = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters)
    print(f"opt net layers {opt_net.length}")
    print(f"temp net layers {temp_net.length}")
    print(f"rand net layers {rand_net.length}")
    total_optimal_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(opt_inf))), 2)
    total_random_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(rand_inf))), 2)

    ##########
    if plot:
        colors = sns.color_palette("hls", 8)
        fig, axs = plt.subplots(2, 1, sharex=True)
        ax = axs[0]
        ax.plot(temp_t, temp_inf, label='Temporal', color=colors[1], lw=2, alpha=0.9)
        ax.plot(opt_t, opt_inf, label='Algorithmic', color=colors[0], lw=2, alpha=0.6, ls='--')
        ax.plot(rand_t, rand_inf, label='Random', color=colors[6], lw=2, alpha=0.6, ls='-.')
        ## vertical lines to show compression
        max_infected_buffer = max(max(temp_inf), max(opt_inf), max(rand_inf)) + 2
        ax.vlines(temporal_network.get_time_network_map().keys(), ymin=0, ymax=max_infected_buffer/3, ls=':', color=colors[1], lw=1, alpha=0.95)
        ax.vlines(opt_net.get_time_network_map().keys(), ymin=2*max_infected_buffer/3, ymax=max_infected_buffer, ls='-', color=colors[0], lw=1, alpha=0.95)
        ax.vlines(rand_net.get_time_network_map().keys(), ymin=max_infected_buffer/3, ymax=2*max_infected_buffer/3, ls='--', color=colors[6], lw=1, alpha=0.95)
        # ax.xlabel('Time')
        ax.set_ylabel('Infected nodes')
        # ax.set_xticks(list(temporal_network.get_time_network_map().keys())[::4])
        axs[0].legend(loc='lower right')
        # plt.show()
        ax = axs[1]
        # plt.figure('error')
        # for the plot, have it be normalized error?
        # ax.plot(opt_t, (-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf), label=f'Optimal: {total_optimal_error}', color=colors[0], ls='--')
        ax.plot(opt_t, (-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf), label=f'Algorithmic', color=colors[0], ls='--')
        # ax.plot(rand_t, (-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf), label=f'Random: {total_random_error}', color=colors[6], ls='-.')
        ax.plot(rand_t, (-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf), label=f'Random', color=colors[6], ls='-.')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized error')
        axs[1].legend()
        # plt.show()
        plt.tight_layout()
        # fig.set_size_inches(5,5)
    return total_optimal_error, total_random_error

def run_multiple(temporal_network, t_interval, beta, number_layers, levels, iters, rand_only=False, temp_inf=None):
    print(iters)
    if rand_only:
        rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
        print(f"rand net layers {rand_net.length}")
        total_random_error = round(np.sum(np.abs((-np.array(temp_inf) + np.array(rand_inf))/np.array(temp_inf))), 2)
        return total_random_error
    else:
        temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters)
        rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
        opt_t, opt_inf, opt_net = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters)
        total_optimal_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf))), 2)
        total_random_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf))), 2)
        print(f"opt net layers {opt_net.length}")
        print(f"temp net layers {temp_net.length}")
        print(f"rand net layers {rand_net.length}")
        return total_optimal_error, total_random_error, temp_inf

def experiment(beta):
    colors = sns.color_palette("hls", 8)
    this_t_interval = 4
    this_beta = beta
    this_number_layers = 20

    # temporal_network = random_temporal_network_mix(number_layers, t_interval, beta)
    # print(temporal_network.length)
    # one_round(temporal_network, t_interval, beta, number_layers, levels=1, iters=temporal_network.length-1, plot=True)
    # plt.show()

    # for lev in range(5):
    this_temporal_network = random_temporal_network_mix(this_number_layers, this_t_interval, this_beta)
    iter_range = this_temporal_network.length
    ensemble_length = 10
    random_errors = np.zeros((ensemble_length, iter_range))
    optimal_errors = np.zeros((ensemble_length, iter_range))
    # one_round(this_temporal_network, this_t_interval, this_beta, this_number_layers, levels=1, iters=6, plot=True)

    # plt.show()
    lev = 1
    for i in range(iter_range):
        print(lev, i)
        _total_opt, total_rand, _temp = run_multiple(this_temporal_network, this_t_interval, this_beta, this_number_layers, lev, iters=i)
        random_errors[0][i] = total_rand
        optimal_errors[0][i] = _total_opt
        for t in range(1, ensemble_length):
            # TODO make this so it doesn't run temporal and optimal a hundred times
            total_rand = run_multiple(this_temporal_network, this_t_interval, this_beta, this_number_layers, levels=lev, iters=i, rand_only=True, temp_inf=_temp)
            random_errors[t][i] = total_rand
            optimal_errors[t][i] = _total_opt

    plt.plot(np.mean(random_errors, axis=0), label='Mean random', color=colors[6])
    std_random = np.sqrt(np.var(random_errors, axis=0))
    above_r = np.mean(random_errors, axis=0) + std_random
    below_r = np.mean(random_errors, axis=0) - std_random
    plt.fill_between(np.arange(iter_range), below_r, above_r, color=colors[6], alpha=0.4)
    plt.plot(np.mean(optimal_errors, axis=0), color=colors[0], label='Algorithmic')
    std_opt = np.sqrt(np.var(optimal_errors, axis=0))
    above_o = np.mean(optimal_errors, axis=0) + std_opt
    below_o = np.mean(optimal_errors, axis=0) - std_opt
    plt.fill_between(np.arange(iter_range), below_o, above_o, color=colors[0], alpha=0.4)
    # plt.xticks(np.linspace(0, iter_range+1, 10))
    # plt.xticks(list(int(np.arange(0, iter_range+1))))
    plt.xlabel('Iterations')
    plt.ylabel('Total normalized error')
    plt.xticks([0, 5, 10, 15, 19])
    plt.legend(loc='upper left')
    # plt.show()

    print(random_errors)

def data_experiment(interval, beta, number_layers):
    graphs = []
    start_times = []
    end_times = []
    start = 28820
    min_real_t = 28820
    for i in range(number_layers):
        data = socio_patterns_ex.parse_data('./tij_InVS.dat', start, interval)
        graphs.append(data[0])
        start_times.append(start - min_real_t)
        end_times.append(start+interval - min_real_t)
        graphs.append(data[1])
        start_times.append(start+interval - min_real_t)
        end_times.append(start+2*interval - min_real_t)
        start += 2*interval
    print(len(graphs))
    all_nodes = list(graphs[0].nodes())
    for graph in graphs:
        all_nodes.extend(list(graph.nodes()))
    all_nodes = set(all_nodes)
    new_labels = {sorted(all_nodes)[i]:i for i in range(len(all_nodes))}
    relabeled_graphs = []
    for graph in graphs:
        relabeled_graphs.append(nx.relabel_nodes(graph, new_labels))
    adj_m = []
    for graph in relabeled_graphs:
        matrix = np.zeros((len(all_nodes), len(all_nodes)))
        for edge in graph.edges():
            matrix[edge[0], edge[1]] = 1
            matrix[edge[1], edge[0]] = 1
        check = np.sum(matrix - matrix.T)
        if check != 0:
            print(f'failed check on graph with sum {check}')
        adj_m.append(matrix)
    final_layers = []
    for i, A in enumerate(adj_m):
        final_layers.append(network_objects.Layer(start_time=start_times[i], end_time=end_times[i], beta=beta, A=A))
    # graph1_nodes = data[0].nodes()
    # graph2_nodes = data[1].nodes()
    # Need a good method of extracting the adjacency matrix of each layer without messing up the node ordering
    # Maybe do that via an adjacency matrix? If it gets handed a networkx graph, do the node ordering itself?
    return final_layers


# total_time = 1016440 - 28820
# num_layers = 200
# layers = data_experiment(interval=int(total_time/num_layers), beta=.000005, number_layers=int(num_layers/2))
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), .000005, len(layers), levels=1, iters=100, plot=True)
# plt.show()

# a_temporal_network = random_temporal_network_mix(20, 5, .0022)

# one_round(a_temporal_network, 5, .0022, 20, levels=1, iters=10, plot=True)
# plt.savefig("./examples/nerccsfig_1.png")
# plt.savefig("./examples/nerccsfig_1.svg", fmt='svg')
# plt.show()
# plt.figure('.00001 beta')
# experiment(beta=.00001)
# # plt.figure('.006 beta')
# # experiment(beta=.006)
# plt.show()

plt.plot(np.arange(10), np.arange(10), ls='--', color='k')
plt.text(1, 6, 'placeholder')
plt.show()



