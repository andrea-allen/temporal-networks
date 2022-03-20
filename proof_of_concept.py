import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import deterministic
import poc_compression
import network_objects
import seaborn as sns
import socio_patterns_ex
from epintervene.simobjects import network as netbd
import time

def random_temporal_network_mix(num_layers, t_interval, beta, return_ordered=False):
    ### Creating the networks:
    N = 200
    A0 = np.zeros((N, N))
    A0[50,21] = 1
    A0[21,50] = 1
    A0[27,42] = 1
    A0[42,27] = 1
    A0B = np.zeros((N, N))
    A0B[57,49] = 1
    A0B[49,57] = 1
    A0B[13,4] = 1
    A0B[4,13] = 1
    G1, A1 = poc_compression.configuration_model_graph(N)
    G2, A2 = poc_compression.barbell_graph(N)
    G3, A3 = poc_compression.configuration_model_graph(N)
    G4, A4 = poc_compression.erdos_renyi_graph(N, .03)
    G5, A5 = poc_compression.configuration_model_graph(N)
    G6, A6 = poc_compression.erdos_renyi_graph(N, .01)
    G7, A7 = poc_compression.configuration_model_graph(N)
    G8, A8 = poc_compression.erdos_renyi_graph(N, .15)
    G9, A9 = poc_compression.cycle_graph(N)
    G10, A10 = poc_compression.configuration_model_graph(N)
    options = [A1,A2,A2,A2, A3, A4, A2, A5, A6, A2, A2, A7, A8, A9, A10, A0, A0B, A0, A0B, A0, A0B, A0B]
    # Pairs in order:
    ordered_pairs = [A1, A1, A2, A2, A3, A3, A4, A4, A5, A5, A6, A6, A7, A7, A8, A8, A9, A9, A10, A10, A1, A1, A2, A2,
                     A3, A3, A4, A4, A5, A5, A6, A6, A7, A7, A8, A8, A9, A9, A10, A10, A1, A1, A2, A2, A3, A3, A4, A4, A5, A5,]
                     # A1, A1,A1, A1,A1, A1,A1, A1,A1, A1,]
    # Pairs but also with 8 zeros
    ordered_pairs = [A1, A0, A2, A0, A3, A3, A0, A0, A0, A0, A0, A0, A0, A0, A8, A8, A9, A9, A10, A10, A1, A1, A2, A2,
                     A3, A3, A4, A4, A5, A0, A0, A0, A0, A0, A0, A8, A9, A9, A10, A10, A1, A1, A2, A2, A3, A3, A4, A4, A5, A5,]
    ordered_pairs = [A2, A2, A0, A0B, A0,A0B, A0B, A0, A0, A0,A0, A0, A0, A0, A2,A2, A3, A4, A0, A0B,A0, A0B, A2, A5, A6,
                     A2, A0, A2, A4, A5,A3, A3, A4, A4, A0, A0, A0,A0, A4, A5,A3, A3, A4, A4, A5,A0, A0, A4, A4, A5,]
    # ordered_pairs = [A3,A3,A3,A3,A3,A2,A2,A2,A2,A2,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A2,A2,A2,A2,A3,A3,A3,A3,A3,A3,
    #                  A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A6,A6,A6,A6,A6,A6,A6,A6,A6,A6]
    # ordered_pairs = [A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A3,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,A2,
    #                  A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1,A1, A1]
    # ordered_pairs = [A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A3, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8, A8]
    if return_ordered:
        layers = []
        current_start_t = 0
        for l in range(len(ordered_pairs)):
            layers.append(
                network_objects.Layer(current_start_t, t_interval + current_start_t, beta, ordered_pairs[l]))
            current_start_t += t_interval
        return network_objects.TemporalNetwork(layers)
    layers = []
    current_start_t = 0
    for l in range(num_layers):
        rand_choice = np.random.randint(0,len(options))
        # rand_choice = l % len(options)
        layers.append(network_objects.Layer(current_start_t, t_interval+current_start_t, beta, options[rand_choice]))
        current_start_t += t_interval
    return network_objects.TemporalNetwork(layers)

## TO THINK ABOUT FOR TOMORROW: HOW BEST TO HANDLE THE TIME SERIES DIFFERENTIAL
# do something with binning:
def digitize_me(time_vector, infected, number_layers, t_interval):
    digitized = np.digitize(np.array(time_vector), np.linspace(0,number_layers*t_interval,50))
    bin_means = [np.array(time_vector)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    bins_infected = [np.array(infected)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    # interpolate the nans with backfill:
    for i, d in enumerate(bin_means):
        if np.isnan(d):
            bin_means[i] = bin_means[i-1]
    for i, d in enumerate(bins_infected):
        if np.isnan(d):
            bins_infected[i] = bins_infected[i-1]
    return bin_means, bins_infected

def run_random(temporal_network, t_interval, beta, number_layers, levels, iters):
    N = len(temporal_network.layers[0].A)
    random_compressed = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters,
                                                            how='random')
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N),
                                          end_time=number_layers * t_interval,
                                          networks=random_compressed.get_time_network_map())
    solution_t_random, solution_p = model.solve_model()
    random_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_random, random_solution, number_layers, t_interval)
    return d[0], d[1], random_compressed

def run_even(temporal_network, t_interval, beta, number_layers, levels, iters):
    N = len(temporal_network.layers[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.layers[0].dd_normalized
    begin_compress = time.time()
    even_compressed = network_objects.Compressor.compress(temporal_network, iterations=iters,
                                                            how='even')
    print(f"compressed evenly in {time.time()-begin_compress} seconds.")
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=y_init,
                                          end_time=number_layers * t_interval,
                                          networks=even_compressed.get_time_network_map())
    solution_t_even, solution_p = model.solve_model()
    even_solution = np.sum(solution_p, axis=0)
    start_time = time.time()
    d = digitize_me(solution_t_even, even_solution, number_layers, t_interval)
    return d[0], d[1], even_compressed

def run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters, error_type):
    N = len(temporal_network.layers[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.layers[0].dd_normalized
    begin_compress = time.time()
    optimal_network, total_chosen_error = network_objects.Compressor.compress(temporal_network, iterations=iters, how='optimal', error_type=error_type)
    # optimal_network = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters, how='optimal', error_type='terminal')
    print(f"compressed optimally in {time.time()-begin_compress} seconds.")
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=y_init,
                                          end_time=number_layers * t_interval,
                                          networks=optimal_network.get_time_network_map())
    solution_t_compressed, solution_p = model.solve_model()
    compressed_solution = np.sum(solution_p, axis=0)
    # there are nans in the digitized vector, need to fix:
    start_time = time.time()
    d = digitize_me(solution_t_compressed, compressed_solution, number_layers, t_interval)
    return d[0], d[1], optimal_network, total_chosen_error

def run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters):
    N = len(temporal_network.layers[0].A)
    y_init = np.full(N, 1 / N)
    # y_init = temporal_network.layers[0].dd_normalized
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=y_init, end_time=number_layers*t_interval,
                            networks=temporal_network.get_time_network_map())
    # plt.plot([np.sum(model.networks[key]) for key, val in model.networks.items()]) # 2x contacts per snapshot, plotted
    # plt.show()
    solution_t_temporal, solution_p = model.solve_model()
    temporal_solution = np.sum(solution_p, axis=0)
    start_time = time.time()
    d = digitize_me(solution_t_temporal, temporal_solution, number_layers, t_interval)
    return d[0], d[1], temporal_network


def one_round(temporal_network, t_interval, beta, number_layers, levels, iters, plot=False):
    total_time = 0
    temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters)
    # rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
    even_t, even_inf, even_net = run_even(temporal_network, t_interval, beta, number_layers, levels, iters)
    # opt_t_h, opt_inf_h, opt_net_h = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters, 'halftime')
    # opt_t, opt_inf, opt_net = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters, 'terminal')
    opt_t_c, opt_inf_c, opt_net_c, total_chosen_error = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters, 'combined')
    # print(f"opt net layers {opt_net.length}")
    print(f"temp net layers {temp_net.length}")
    # print(f"rand net layers {rand_net.length}")
    print(f"TOTAL TIME FOR DIGITIZE {total_time}")
    # total_optimal_error_nm = round(integrate_error_ts(temp_t, temp_inf, opt_t, opt_inf), 3)
    # total_optimal_h_error_nm = round(integrate_error_ts(temp_t, temp_inf, opt_t_h, opt_inf_h), 3)
    total_optimal_c_error_nm = round(integrate_error_ts(temp_t, temp_inf, opt_t_c, opt_inf_c), 3)
    # total_optimal_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(opt_inf))), 3)
    # total_random_error_nm = round(np.sum(np.abs(-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf)), 3)
    # total_random_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(rand_inf))), 3)
    total_even_error_nm = round(integrate_error_ts(temp_t, temp_inf, even_t, even_inf), 3)
    total_even_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(even_inf))), 3)

    ##########
    print("STARTING PLOTTING")
    if plot:
        colors = sns.color_palette("hls", 8)
        colors = sns.color_palette()
        colors = sns.color_palette("Paired")
        # colors = ["#C54CC5", "#00A4D4", "#FFC626"]
        # colors = ["crimson", "#00A4D4", "#FFC626", 'blue']
        type_colors = {'temp': colors[1], 'even': colors[3], 'algo': colors[0]}
        type_colors = {'temp': 'grey', 'even': colors[7], 'algo': 'c'}
        type_colors = {'temp': 'grey', 'even': "#FFC626", 'algo': "#00A4D4"}
        fig, axs = plt.subplots(2, 1, sharex=True)
        ax = axs[0]
        ax.plot(temp_t, temp_inf, label='Temporal', color=type_colors['temp'], lw=2.5, alpha=1.0)
        # ax.plot(opt_t, opt_inf, label='Algorithmic-terminal', color=colors[1], lw=2, alpha=0.6, ls='--')
        # ax.plot(opt_t_h, opt_inf_h, label='Algorithmic-halftime', color=colors[3], lw=2, alpha=0.6, ls='--')
        ax.plot(opt_t_c, opt_inf_c, label='Algorithmic', color=type_colors['algo'], lw=2, alpha=1.0, ls='--')
        # ax.plot(rand_t, rand_inf, label='Random', color=colors[6], lw=2, alpha=0.6, ls='-.')
        ax.plot(even_t, even_inf, label='Even', color=type_colors['even'], lw=2, alpha=1.0, ls='-.')
        ## vertical lines to show compression
        max_infected_buffer = max(max(temp_inf), max(opt_inf_c)) + 2
        ax.vlines(temporal_network.get_time_network_map().keys(), ymin=0, ymax=max_infected_buffer / 3, ls='-',
                  color=type_colors['temp'], lw=0.5, alpha=1.0)
        # ax.vlines(opt_net.get_time_network_map().keys(), ymin=2*max_infected_buffer/3, ymax=max_infected_buffer, ls='-', color=colors[1], lw=2, alpha=1.0)
        # ax.vlines(opt_net_h.get_time_network_map().keys(), ymin=2*max_infected_buffer/3, ymax=max_infected_buffer, ls='-', color=colors[3], lw=1, alpha=0.95)
        ax.vlines(opt_net_c.get_time_network_map().keys(), ymin=2 * max_infected_buffer / 3, ymax=max_infected_buffer,
                  ls='--', color=type_colors['algo'], lw=1, alpha=0.95)
        # ax.vlines(rand_net.get_time_network_map().keys(), ymin=max_infected_buffer/3, ymax=2*max_infected_buffer/3, ls='--', color=colors[6], lw=1, alpha=0.95)
        ax.vlines(even_net.get_time_network_map().keys(), ymin=max_infected_buffer / 3,
                  ymax=2 * max_infected_buffer / 3, ls='-.', color=type_colors['even'], lw=1, alpha=0.95)
        # ax.xlabel('Time')
        ax.set_ylabel('Infected nodes')
        # ax.set_xticks(list(temporal_network.get_time_network_map().keys())[::4])
        axs[0].legend(loc='lower right', frameon=False)
        # plt.show()
        ax = axs[1]
        # plt.figure('error')
        # for the plot, have it be normalized error?
        # TODO 3/3: THIS ONE IS STILL NORMALIZING
        # ax.plot(opt_t_c, (-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf), label=f'Optimal terminal: {total_optimal_error_nm}', color=colors[1], ls='--')
        # ax.plot(opt_t_c, (-np.array(temp_inf)+np.array(opt_inf_h))/np.array(temp_inf), label=f'Optimal halftime: {total_optimal_h_error_nm}', color=colors[3], ls='--')
        ax.plot(opt_t_c, (-np.array(temp_inf) + np.array(opt_inf_c)) / np.array(temp_inf),
                label=f'Algorithmic: {total_optimal_c_error_nm}\n TCE: {total_chosen_error}',
                color=type_colors['algo'], ls='--', lw=2, alpha=1.0)
        ax.fill_between(opt_t_c, (-np.array(temp_inf) + np.array(opt_inf_c)) / np.array(temp_inf),
                        color=type_colors['algo'], alpha=0.5)
        # ax.plot(opt_t, (-np.array(temp_inf)+np.array(opt_inf)), label=f'Algorithmic {total_optimal_error}', color=type_colors['even', ls='--')
        # ax.plot(rand_t, (-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf), label=f'Random: {total_random_error_nm}', color=colors[6], ls='-.')
        # ax.plot(rand_t, (-np.array(temp_inf)+np.array(rand_inf)), label=f'Random {total_random_error}', color=colors[6], ls='-.')
        ax.plot(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf),
                label=f'Even {total_even_error_nm}',
                color=type_colors['even'], ls='-.', lw=2)
        ax.fill_between(even_t, (-np.array(temp_inf) + np.array(even_inf)) / np.array(temp_inf), color=type_colors['even'], alpha=0.5)
        # TODO do actual integral
        # ax.plot(even_t, (-np.array(temp_inf)+np.array(even_inf)), label=f'Even {total_even_error}', color=colors[3], ls='-.')
        ax.plot(opt_t_c, np.zeros(len(opt_t_c)), color='k', ls='-', alpha=1.0)
        ax.set_xlabel('Time')
        # ax.set_ylabel('Normalized error')
        ax.set_ylabel('Normalized error')
        axs[1].legend(frameon=False)
        # plt.show()
        plt.tight_layout()
        # fig.set_size_inches(5,5)
        print("ENDING PLOT")
    return total_optimal_c_error_nm

def run_multiple(temporal_network, t_interval, beta, number_layers, levels, iters, rand_only=False, temp_inf=None):
    print(iters)
    if rand_only:
        rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
        print(f"rand net layers {rand_net.length}")
        # total_random_error = round(np.sum(np.abs((-np.array(temp_inf) + np.array(rand_inf))/np.array(temp_inf))), 2)
        total_random_error = round(np.sum(np.abs((-np.array(temp_inf) + np.array(rand_inf)))), 2)
        return total_random_error
    else:
        temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters)
        rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
        even_t, even_inf, even_net = run_even(temporal_network, t_interval, beta, number_layers, levels, iters)
        opt_t, opt_inf, opt_net = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters)
        # total_optimal_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf))), 2)
        total_optimal_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(opt_inf)))), 2)
        # total_random_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(rand_inf))/np.array(temp_inf))), 2)
        total_random_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(rand_inf)))), 2)
        # total_even_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(even_inf)))/np.array(temp_inf)), 2)
        total_even_error = round(np.sum(np.abs((-np.array(temp_inf)+np.array(even_inf)))), 2)
        print(f"opt net layers {opt_net.length}")
        print(f"temp net layers {temp_net.length}")
        print(f"rand net layers {rand_net.length}")
        return total_optimal_error, total_random_error, total_even_error, temp_inf

def experiment(beta):
    colors = sns.color_palette("hls", 8)
    this_t_interval = 5
    this_beta = beta
    this_number_layers = 30

    # temporal_network = random_temporal_network_mix(number_layers, t_interval, beta)
    # print(temporal_network.length)
    # one_round(temporal_network, t_interval, beta, number_layers, levels=1, iters=temporal_network.length-1, plot=True)
    # plt.show()

    # for lev in range(5):
    this_temporal_network = random_temporal_network_mix(this_number_layers, this_t_interval, this_beta)
    iter_range = np.arange(2, this_number_layers-1)
    iter_range = np.arange(2, 10)
    ensemble_length = 1
    random_errors = np.zeros((ensemble_length, len(iter_range)))
    optimal_errors = np.zeros((ensemble_length, len(iter_range)))
    even_errors = np.zeros((ensemble_length, len(iter_range)))
    one_round(this_temporal_network, this_t_interval, this_beta, this_number_layers, levels=1, iters=15, plot=True)

    plt.show()
    lev = 3
    for i in range(len(iter_range)):
        print(lev, i)
        _total_opt, total_rand, _total_even, _temp = run_multiple(this_temporal_network, this_t_interval, this_beta, this_number_layers, lev, iters=int(iter_range[i]))
        random_errors[0][i] = total_rand
        optimal_errors[0][i] = _total_opt
        even_errors[0][i] = _total_even
        for t in range(1, ensemble_length):
            # TODO make this so it doesn't run temporal and optimal a hundred times
            # total_rand = run_multiple(this_temporal_network, this_t_interval, this_beta, this_number_layers, levels=lev, iters=int(iter_range[i]), rand_only=True, temp_inf=_temp)
            random_errors[t][i] = total_rand
            optimal_errors[t][i] = _total_opt
            even_errors[t][i] = _total_even

    plt.plot(np.mean(random_errors, axis=0), label='Mean random', color=colors[6])
    std_random = np.sqrt(np.var(random_errors, axis=0))
    above_r = np.mean(random_errors, axis=0) + std_random
    below_r = np.mean(random_errors, axis=0) - std_random
    # plt.fill_between(np.arange(iter_range), below_r, above_r, color=colors[6], alpha=0.4)
    plt.plot(np.mean(optimal_errors, axis=0), color=colors[0], label='Algorithmic')
    std_opt = np.sqrt(np.var(optimal_errors, axis=0))
    above_o = np.mean(optimal_errors, axis=0) + std_opt
    below_o = np.mean(optimal_errors, axis=0) - std_opt
    # plt.fill_between(np.arange(iter_range), below_o, above_o, color=colors[0], alpha=0.4)
    plt.plot(np.mean(even_errors, axis=0), color=colors[3], label='Even')
    # plt.xticks(np.linspace(0, iter_range+1, 10))
    # plt.xticks(list(int(np.arange(0, iter_range+1))))
    plt.xlabel('Iterations')
    plt.ylabel('Total normalized error')
    # plt.xticks([0, 5, 10, 15, 19])
    plt.legend(loc='upper left')
    # plt.show()

    print(random_errors)

def data_experiment(filename, interval, beta, number_layers, start):
    print('tau:')
    print(interval*beta)
    graphs = []
    start_times = []
    end_times = []
    # start = 28820
    # start = 0
    min_real_t = start
    # min_real_t = 0
    for i in range(number_layers):
        data = socio_patterns_ex.parse_data(filename, start, interval)
        # data = socio_patterns_ex.parse_data_hospital(filename, start, interval)
        graphs.append(data[0])
        print(f'number of nodes {len(data[0].nodes())}')
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
# TAU OF .025

def plot_ts_and_bars(ax, temp_t, temp_inf, opt_t, opt_inf, even_t, even_inf, temporal_network, opt_net, even_net):
    colors = sns.color_palette("hls", 8)
    ax.plot(temp_t, temp_inf, label='Temporal', color=colors[1], lw=2, alpha=0.9)
    ax.plot(opt_t, opt_inf, label='Algorithmic', color=colors[0], lw=2, alpha=0.6, ls='--')
    # ax.plot(rand_t, rand_inf, label='Random', color=colors[6], lw=2, alpha=0.6, ls='-.')
    ax.plot(even_t, even_inf, label='Even', color=colors[3], lw=2, alpha=0.6, ls='-.')
    ## vertical lines to show compression
    max_infected_buffer = max(max(temp_inf), max(opt_inf)) + 2
    ax.vlines(temporal_network.get_time_network_map().keys(), ymin=0, ymax=max_infected_buffer / 3, ls=':',
              color=colors[1], lw=1, alpha=0.95)
    ax.vlines(opt_net.get_time_network_map().keys(), ymin=2 * max_infected_buffer / 3, ymax=max_infected_buffer, ls='-',
              color=colors[0], lw=1, alpha=0.95)
    # ax.vlines(rand_net.get_time_network_map().keys(), ymin=max_infected_buffer/3, ymax=2*max_infected_buffer/3, ls='--', color=colors[6], lw=1, alpha=0.95)
    ax.vlines(even_net.get_time_network_map().keys(), ymin=max_infected_buffer / 3, ymax=2 * max_infected_buffer / 3,
              ls='--', color=colors[3], lw=1, alpha=0.95)
def integrate_error_ts(temporal_ts, temporal_inf, other_ts, other_inf):
    other_inf = np.array(other_inf)
    temporal_inf = np.array(temporal_inf)
    if other_inf[0] == 0 or temporal_inf[0]==0:
        print('correcting')
        other_inf = np.array(other_inf) + .0000000001
        temporal_inf = np.array(temporal_inf) + .0000000001
    absolute_diff_normed = np.abs(other_inf[1:] - temporal_inf[1:]) / temporal_inf[1:]
    time_delta = np.diff(np.array(temporal_ts))
    integrand = np.array([time_delta[i]*absolute_diff_normed[i] for i in range(len(time_delta))])
    total_error_integrand = np.sum(integrand)
    return total_error_integrand

def error_by_compression_faster(layers, beta, iter_range, plot=True):
    t_interval = layers[10].duration
    temp_net = network_objects.TemporalNetwork(layers)
    # num_layers = len(layers)
    levels = 1
    if iter_range is None:
        gap = 5
        iter_range = np.arange(0, temp_net.length, gap)
    even_errors = np.zeros(len(iter_range))
    even_errors_norm = np.zeros(len(iter_range))
    optimal_errors = np.zeros(len(iter_range))
    optimal_errors_norm = np.zeros(len(iter_range))
    tce_all = np.zeros(len(iter_range))
    temp_t, temp_inf, temp_net = run_temporal(temp_net, t_interval, beta, temp_net.length, None, None)

    current_optimal_temp_net = temp_net
    current_iters_for_optim = 0
    if plot:
        fig2, axs = plt.subplots(4,10)
        all_axs = []
        for a in range(len(axs)):
            all_axs.extend(list(axs[a]))
    for i, r in enumerate(iter_range):
        c = int(r)
        even_t, even_inf, even_net = run_even(temp_net, t_interval, beta, temp_net.length, levels, c)
        opt_t, opt_inf, opt_net, tce = run_optimal(current_optimal_temp_net, t_interval, beta,
                                              temp_net.length, levels, c-current_iters_for_optim, 'combined')
        current_iters_for_optim = c
        current_optimal_temp_net = opt_net
        # total_optimal_error_nm = round(np.sum(np.abs((-np.array(temp_inf)+np.array(opt_inf))/np.array(temp_inf))), 2)
        total_optimal_error_nm = integrate_error_ts(temp_t, temp_inf, opt_t, opt_inf)
        total_optimal_error = np.sum(np.abs((-np.array(temp_inf)+np.array(opt_inf))))
        print(f"***, {i}")
        print(total_optimal_error)
        # total_even_error_nm = np.sum(np.abs((-np.array(temp_inf)+np.array(even_inf)))/np.array(temp_inf))
        total_even_error_nm = integrate_error_ts(temp_t, temp_inf, even_t, even_inf)
        total_even_error = np.sum(np.abs((-np.array(temp_inf)+np.array(even_inf))))
        print(total_even_error)

        optimal_errors[i] = total_optimal_error
        optimal_errors_norm[i] = total_optimal_error_nm
        even_errors[i] = total_even_error
        even_errors_norm[i] = total_even_error_nm
        tce_all[i] = tce
        if plot:
            plot_ts_and_bars(all_axs[i], temp_t, temp_inf, opt_t, opt_inf, even_t, even_inf, temp_net, opt_net, even_net)

    if plot:
        fig, ax = plt.subplots(2, 1)
        colors = sns.color_palette("hls", 8)
        # ax[0].scatter(iter_range, optimal_errors, color=colors[0], label='Algorithmic')
        # ax[0].plot(iter_range, optimal_errors, color=colors[0], label='Algorithmic')
        # ax[0].scatter(iter_range, even_errors, color=colors[3], label='Even')
        # ax[0].plot(iter_range, even_errors, color=colors[3], label='Even')
        ax[1].scatter(iter_range, optimal_errors_norm, color=colors[0], label='Algorithmic')
        ax[1].plot(iter_range, optimal_errors_norm, color=colors[0], label='Algorithmic')
        ax[1].scatter(iter_range, even_errors_norm, color=colors[3], label='Even')
        ax[1].plot(iter_range, even_errors_norm, color=colors[3], label='Even')
        ax[0].scatter(iter_range, tce_all, color=colors[3], label='TCE')
        ax[0].plot(iter_range, tce_all, color=colors[3], label='TCE')
        # plt.xticks(np.linspace(0, iter_range+1, 10))
        # plt.xticks(list(int(np.arange(0, iter_range+1))))
        ax[1].set_xlabel('Iterations')
        ax[0].set_ylabel('Total chosen error')
        ax[1].set_ylabel('Total normalized error')
        # plt.xticks([0, 5, 10, 15, 19])
        plt.legend(loc='upper left')
    return optimal_errors_norm - even_errors_norm



def heatmap_results(matrix, betas, compressions):
    print(matrix)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    ax = sns.heatmap(matrix, cmap=cmap, center=0)
    ax.set_xlabel('Number of compressions')
    ax.set_xticklabels(compressions)
    ax.set_yticklabels(betas, rotation=45)
    ax.set_ylabel('Beta')
    ax.set_title('Ratio of Even/Algorithmic Total Error')
    plt.show()

def error_by_compression(layers, beta, iter_range=None, lev_range=None):
    # TODO write a whole new code block that iteratively saves the optimal network and just does one more compression
    # beta = .000005
    t_interval = layers[10].duration
    temp_net = network_objects.TemporalNetwork(layers)
    if iter_range is None:
        gap = 5
        iter_range = np.arange(0, temp_net.length, gap)
    ensemble_length = 1
    random_errors = np.zeros((ensemble_length, len(iter_range)))
    even_errors = np.zeros((ensemble_length, len(iter_range)))
    optimal_errors = np.zeros((ensemble_length, len(iter_range)))
    # one_round(temp_net, t_interval, beta, len(layers), levels=1, iters=30, plot=True)

    # plt.show()
    if lev_range is not None:
        lev = lev_range
    else:
        lev = 1
    current_optimal_temp_net = temp_net
    for i, r in enumerate(iter_range):
        print(lev, i)
        c = int(r)
        _total_opt, total_rand, _total_even, _temp = run_multiple(temp_net, t_interval, beta,
                                                     len(temp_net.layers), lev, iters=c)
        random_errors[0][i] = total_rand
        optimal_errors[0][i] = _total_opt
        even_errors[0][i] = _total_even
        if np.isnan(_total_opt):
            print(_total_opt)
        for t in range(1, ensemble_length):
            # TODO make this so it doesn't run temporal and optimal a hundred times
            # total_rand = run_multiple(temp_net, t_interval, beta, len(layers), levels=lev,
            #                           iters=c, rand_only=True, temp_inf=_temp)
            # random_errors[t][i] = total_rand
            random_errors[t][i] = total_rand
            optimal_errors[t][i] = _total_opt
            even_errors[t][i] = _total_even

    colors = sns.color_palette("hls", 8)
    plt.plot(iter_range, np.mean(random_errors, axis=0), label='Mean random', color=colors[6])
    std_random = np.sqrt(np.var(random_errors, axis=0))
    above_r = np.mean(random_errors, axis=0) + std_random
    below_r = np.mean(random_errors, axis=0) - std_random
    plt.fill_between(iter_range, below_r, above_r, color=colors[6], alpha=0.4)
    plt.plot(iter_range, np.mean(optimal_errors, axis=0), color=colors[0], label='Algorithmic')
    std_opt = np.sqrt(np.var(optimal_errors, axis=0))
    above_o = np.mean(optimal_errors, axis=0) + std_opt
    below_o = np.mean(optimal_errors, axis=0) - std_opt
    plt.fill_between(iter_range, below_o, above_o, color=colors[0], alpha=0.4)
    plt.plot(iter_range, np.mean(even_errors, axis=0), color=colors[3], label='Even')
    # plt.xticks(np.linspace(0, iter_range+1, 10))
    # plt.xticks(list(int(np.arange(0, iter_range+1))))
    plt.xlabel('Iterations')
    plt.ylabel('Total normalized error')
    # plt.xticks([0, 5, 10, 15, 19])
    plt.legend(loc='upper left')
    # plt.show()
    data_to_save = np.array([np.mean(optimal_errors), np.mean(even_errors)])
    # np.savetxt('./hospital_compression_error_400_to_500.txt', data_to_save, delimiter=',')

    print(random_errors)

integrate_error_ts([0,.5,1, 1.5,2,2.5,3,3.5,4], np.array([0, 1,2,3,4,5,6,7,8]), [0,.5,1, 1.5,2,2.5,3,3.5,4], np.array([0, 2,4,6,8,10,12,14,16]))
integrate_error_ts([0,.5,1, 1.5,2,2.5,3,3.5,4], np.array([0, 1,2,3,4,5,6,7,8]), [0,.5,1, 1.5,2,2.5,3,3.5,4], np.array([0, 2,5,5,8,10,11,12,15]))
# heatmap_results()
# UNCOMMENT FOR DATASET STATISTICS FOR CONFERENCE DATA
# data_info = socio_patterns_ex.dataset_statistics('./tij_InVS.dat')
# data_info = socio_patterns_ex.dataset_statistics('./detailed_list_of_contacts_Hospital.dat_')
# data_info = socio_patterns_ex.dataset_statistics('./listcontacts_2009_05_06.txt')
total_time = 1016440 - 28820
total_time = 347640 - 140
# total_time = 1241632099 - 1241604119
num_layers = 400
beta = .00002
# beta = .00001
# beta = .00002
# beta = .0001
# beta = .01
# layers = data_experiment(filename='./tij_InVS.dat', interval=int(total_time/num_layers), beta=beta, number_layers=int(num_layers/2), start=28820)
# layers = data_experiment(filename='./detailed_list_of_contacts_Hospital.dat_', interval=int(total_time/num_layers), beta=beta, number_layers=int(num_layers/2), start=140)
# layers = data_experiment(filename='./listcontacts_2009_05_06.txt', interval=int(total_time/num_layers), beta=beta, number_layers=int(num_layers/2), start=1241604119)
# my_start = time.time()

# Conference set
# iter_range = np.arange(170, 200, 2)
# betas = np.array([.000005, .000007, .00001, .00002, .00003, .00004]) # hospital
# # # betas = np.array([.00005, .00007, .00009, .0001, .0002, .0003])
# results_matrix = np.zeros((len(betas), len(iter_range)))
# for b in range(len(betas)):
#     layers = data_experiment(filename='./tij_InVS.dat',
#                              interval=int(total_time / num_layers), beta=betas[b], number_layers=int(num_layers / 2),
#                              start=28820)
#     temp_net = network_objects.TemporalNetwork(layers)
#     error_difference = error_by_compression_faster(temp_net.layers, betas[b], iter_range, plot=True)
#     plt.show()
#     results_matrix[b] = error_difference
# heatmap_results(results_matrix, betas, iter_range)
# plt.show()

#hospital
iter_range = np.arange(175, 200, 1)
iter_range = np.arange(575, 600, 1)
betas = np.array([.000005, .000007, .00001, .00002, .00003, .00004]) # hospital
betas = np.array([.00002, .000007, .00001, .00002, .00003, .00004]) # hospital
# # betas = np.array([.00005, .00007, .00009, .0001, .0002, .0003])
# results_matrix = np.zeros((len(betas), len(iter_range)))
# for b in range(len(betas)):
#     layers = data_experiment(filename='./detailed_list_of_contacts_Hospital.dat_',
#                              interval=int(total_time / num_layers), beta=betas[b], number_layers=int(num_layers / 2),
#                              start=140)
#     temp_net = network_objects.TemporalNetwork(layers)
#     error_difference = error_by_compression_faster(temp_net.layers, betas[b], iter_range, plot=True)
#     plt.show()
#     results_matrix[b] = error_difference
# heatmap_results(results_matrix, betas, iter_range)
# plt.show()

# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=390, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=380, plot=True)
# plt.show()
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=185, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=180, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=390, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=42, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=45, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=90, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=785, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=790, plot=True)
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=792, plot=True)
# print(f"total seconds to run one round is {time.time()-my_start}")
plt.show()
iter_range = np.arange(170, 200, 1)
# iter_range = np.arange(100, 167, 3)
# error_by_compression(layers, beta, iter_range)
# error_by_compression_faster(layers, beta, iter_range)
# plt.show()
# num_layers = 500
# FIXED! It was an issue with the solver having fewer timesteps than 2 and so never getting "off the ground"
num_layers = 500
beta = .0003
# layers = data_experiment(filename='./tij_InVS.dat', interval=int(total_time/num_layers), beta=beta, number_layers=int(num_layers/2))
# one_round(network_objects.TemporalNetwork(layers), int(total_time/num_layers), beta, len(layers), levels=1, iters=480, plot=True)

# plt.show()
a_temporal_network = random_temporal_network_mix(50, 5, .001, return_ordered=True)
one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .001, a_temporal_network.length, levels=1, iters=44, plot=True)
one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .001, a_temporal_network.length, levels=1, iters=30, plot=True)
plt.show()
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0006, a_temporal_network.length, levels=1, iters=25, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0006, a_temporal_network.length, levels=1, iters=27, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0006, a_temporal_network.length, levels=1, iters=42, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0006, a_temporal_network.length, levels=1, iters=44, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0006, a_temporal_network.length, levels=1, iters=46, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0006, a_temporal_network.length, levels=1, iters=47, plot=True)
# plt.show()
iter_range = np.arange(30,50,1)
iter_range = np.arange(30,50,2)
iter_range = np.arange(1,50,2)
# error_by_compression_faster(a_temporal_network.layers, .0006, iter_range, plot=True)
# plt.show()
betas = np.array([.0001, .0002, .0003, .0004, .0005, .0006, .0007, .0008, .0009, .001])
# betas = np.array([.0001, .0003, .0005, .0007, .0009, .001])
results_matrix = np.zeros((len(betas), len(iter_range)))
for b in range(len(betas)):
    a_temporal_network = random_temporal_network_mix(50, 5, betas[b], return_ordered=True)
    error_ratio = error_by_compression_faster(a_temporal_network.layers, betas[b], iter_range, plot=True)
    plt.show()
    results_matrix[b] = error_ratio
heatmap_results(results_matrix, betas, iter_range)
plt.show()
iter_range = np.arange(50,100,2)
a_temporal_network = random_temporal_network_mix(200, 5, .0002)
one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0002, a_temporal_network.length, levels=1, iters=190, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0002, a_temporal_network.length, levels=1, iters=80, plot=True)
# one_round(network_objects.TemporalNetwork(a_temporal_network.layers), 5, .0002, a_temporal_network.length, levels=1, iters=90, plot=True)
plt.show()
iter_range = np.arange(4,20,1)
error_by_compression_faster(a_temporal_network.layers, .002, iter_range)
plt.show()

one_round(a_temporal_network, 5, .0003, 50, levels=3, iters=15, plot=True)
# plt.savefig("./examples/nerccsfig_1.png")
# plt.savefig("./examples/nerccsfig_1.svg", fmt='svg')
plt.show()
# plt.figure('.00001 beta')
# experiment(beta=.00001)
plt.figure('.0006 beta')
# experiment(beta=.0006)
# plt.show()

plt.plot(np.arange(10), np.arange(10), ls='--', color='k')
plt.text(1, 6, 'placeholder')
plt.show()

## do it again with the hospital dataset? or something. Or a big pair of SBMs



