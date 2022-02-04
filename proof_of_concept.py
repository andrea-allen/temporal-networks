import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import deterministic
import poc_compression
import network_objects

### Creating the networks:
N = 50
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

def random_temporal_network_mix(num_layers, t_interval, beta):
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
def digitize_me(time_vector, infected):
    digitized = np.digitize(np.array(time_vector), np.linspace(0,number_layers*t_interval,50))
    bin_means = [np.array(time_vector)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    bins_infected = [np.array(infected)[digitized == i].mean() for i in range(1, len(np.linspace(0,number_layers*t_interval,50)))]
    return bin_means, bins_infected

def run_random(temporal_network, t_interval, beta, number_layers, levels, iters):
    random_compressed = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters,
                                                            optimal=False)
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N),
                                          end_time=number_layers * t_interval,
                                          networks=random_compressed.get_time_network_map())
    solution_t_random, solution_p = model.solve_model()
    random_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_random, random_solution)
    return d[0], d[1], random_compressed

def run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters):
    optimal_network = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters)
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N),
                                          end_time=number_layers * t_interval,
                                          networks=optimal_network.get_time_network_map())
    solution_t_compressed, solution_p = model.solve_model()
    compressed_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_compressed, compressed_solution)
    return d[0], d[1], optimal_network

def run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters):
    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=number_layers*t_interval,
                            networks=temporal_network.get_time_network_map())
    solution_t_temporal, solution_p = model.solve_model()
    temporal_solution = np.sum(solution_p, axis=0)
    d = digitize_me(solution_t_temporal, temporal_solution)
    return d[0], d[1], temporal_network


def one_round(temporal_network, t_interval, beta, number_layers, levels, iters, plot=False):
    temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters)
    rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
    opt_t, opt_inf, opt_net = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters)

    ##########
    if plot:
        plt.figure('results')
        plt.subplot(211)
        plt.plot(temp_t, temp_inf, label='fully temporal', color='c', lw=3)
    # d = digitize_me(temp_t, temp_inf)
    # plt.plot(d[0], d[1], label='fully temporal', color='c', lw=3)
    if plot:
        plt.plot(opt_t, opt_inf, label='optimal compressed temporal', color='m', lw=2, alpha=0.6)
    # d2 = digitize_me(opt_t, opt_inf)
    # plt.plot(d2[0], d2[1], label='optimal compressed temporal', color='m', lw=2, alpha=0.6)
    if plot:
        plt.plot(rand_t, rand_inf, label='random compressed temporal', color='y', lw=2, alpha=0.6)
    # d3 = digitize_me(rand_t, rand_inf)
    # plt.plot(d3[0], d3[1], label='random compressed temporal', color='y', lw=2, alpha=0.6)
    ## vertical lines to show compression
    if plot:
        max_infected = max(max(temp_inf), max(opt_inf), max(rand_inf)) + 10
        plt.vlines(temporal_network.get_time_network_map().keys(), ymin=0, ymax=max_infected/3, ls=':', color='c', lw=1, alpha=0.8)
        plt.vlines(opt_net.get_time_network_map().keys(), ymin=2*max_infected/3, ymax=max_infected, ls='-', color='m', lw=1, alpha=0.8)
        plt.vlines(rand_net.get_time_network_map().keys(), ymin=max_infected/3, ymax=2*max_infected/3, ls='--', color='y', lw=1, alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('Number nodes infected')
        plt.xticks(list(temporal_network.get_time_network_map().keys())[::4])
        plt.legend()
        # plt.show()
        plt.subplot(212)
        # plt.figure('error')
    total_optimal_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(opt_inf))), 2)
    total_random_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(rand_inf))), 2)
    if plot:
        plt.plot(opt_t, -np.array(temp_inf)+np.array(opt_inf), label=f'error from optimal: {total_optimal_error}', color='m', ls='--')
        plt.plot(rand_t, -np.array(temp_inf)+np.array(rand_inf), label=f'error from random: {total_random_error}', color='y', ls='--')
        plt.xlabel('time')
        plt.ylabel('error from fully temporal')
        plt.legend()
        # plt.show()
    return total_optimal_error, total_random_error

def run_multiple(temporal_network, t_interval, beta, number_layers, levels, iters, rand_only=False, temp_inf=None):
    if rand_only:
        rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
        total_random_error = round(np.sum(np.abs(-np.array(temp_inf) + np.array(rand_inf))), 2)
        return total_random_error
    else:
        temp_t, temp_inf, temp_net = run_temporal(temporal_network, t_interval, beta, number_layers, levels, iters)
        rand_t, rand_inf, rand_net = run_random(temporal_network, t_interval, beta, number_layers, levels, iters)
        opt_t, opt_inf, opt_net = run_optimal(temporal_network, t_interval, beta, number_layers, levels, iters)
        total_optimal_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(opt_inf))), 2)
        total_random_error = round(np.sum(np.abs(-np.array(temp_inf)+np.array(rand_inf))), 2)
        return total_optimal_error, total_random_error, temp_inf

t_interval = 3
beta = .002
number_layers = 30

# temporal_network = random_temporal_network_mix(number_layers, t_interval, beta)
# print(temporal_network.length)
# one_round(temporal_network, t_interval, beta, number_layers, levels=1, iters=temporal_network.length-1, plot=True)
# plt.show()

# for lev in range(5):
temporal_network = random_temporal_network_mix(number_layers, t_interval, beta)
iter_range = temporal_network.length
ensemble_length = 5
random_errors = np.zeros((ensemble_length, iter_range))
optimal_errors = np.zeros((ensemble_length, iter_range))
one_round(temporal_network, t_interval, beta, number_layers, levels=1, iters=15, plot=True)
plt.show()
lev = 1
for i in range(iter_range):
    print(lev, i)
    _total_opt, total_rand, _temp = run_multiple(temporal_network, t_interval, beta, number_layers, lev, iters=i)
    random_errors[0][i] = total_rand
    optimal_errors[0][i] = _total_opt
    for t in range(1, ensemble_length):
        # TODO make this so it doesn't run temporal and optimal a hundred times
        total_rand = run_multiple(temporal_network, t_interval, beta, number_layers, levels=lev, iters=i, rand_only=True, temp_inf=_temp)
        random_errors[t][i] = total_rand
        optimal_errors[t][i] = _total_opt

plt.plot(np.mean(random_errors, axis=0), color='y')
std_random = np.sqrt(np.var(random_errors, axis=0))
above_r = np.mean(random_errors, axis=0) + std_random
below_r = np.mean(random_errors, axis=0) - std_random
plt.fill_between(np.arange(iter_range), below_r, above_r, color='y', alpha=0.4)
plt.plot(np.mean(optimal_errors, axis=0), color='m')
std_opt = np.sqrt(np.var(optimal_errors, axis=0))
above_o = np.mean(optimal_errors, axis=0) + std_opt
below_o = np.mean(optimal_errors, axis=0) - std_opt
plt.fill_between(np.arange(iter_range), below_o, above_o, color='m', alpha=0.4)
# plt.xticks(list(int(np.arange(0, iter_range+1))))
plt.xlabel('Iterations')
plt.ylabel('Error from fully temporal')
plt.show()

print(random_errors)


