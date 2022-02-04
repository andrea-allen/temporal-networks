import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import deterministic
import poc_compression
import network_objects

### Creating the networks:
N = 30
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


def one_round(temporal_network, t_interval, beta, number_layers, levels, iters, plot=False):
    optimal_network = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters)
    random_compressed = network_objects.Compressor.compress(temporal_network, level=levels, iterations=iters, optimal=False)

    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=number_layers*t_interval,
                            networks=temporal_network.get_time_network_map())
    solution_t_temporal, solution_p = model.solve_model()
    temporal_solution = np.sum(solution_p, axis=0)

    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=number_layers*t_interval,
                            networks=optimal_network.get_time_network_map())
    solution_t_compressed, solution_p = model.solve_model()
    compressed_solution = np.sum(solution_p, axis=0)

    model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=number_layers*t_interval,
                            networks=random_compressed.get_time_network_map())
    solution_t_random, solution_p = model.solve_model()
    random_solution = np.sum(solution_p, axis=0)

    ##########
    if plot:
        plt.figure('results')
        plt.subplot(211)
        plt.plot(solution_t_temporal, temporal_solution, label='fully temporal', color='c', lw=3)
    d = digitize_me(solution_t_temporal, temporal_solution)
    # plt.plot(d[0], d[1], label='fully temporal', color='c', lw=3)
    if plot:
        plt.plot(solution_t_compressed, compressed_solution, label='optimal compressed temporal', color='m', lw=2, alpha=0.6)
    d2 = digitize_me(solution_t_compressed, compressed_solution)
    # plt.plot(d2[0], d2[1], label='optimal compressed temporal', color='m', lw=2, alpha=0.6)
    if plot:
        plt.plot(solution_t_random, random_solution, label='random compressed temporal', color='y', lw=2, alpha=0.6)
    d3 = digitize_me(solution_t_random, random_solution)
    # plt.plot(d3[0], d3[1], label='random compressed temporal', color='y', lw=2, alpha=0.6)
    ## vertical lines to show compression
    if plot:
        max_infected = max(max(temporal_solution), max(compressed_solution), max(random_solution)) + 10
        plt.vlines(temporal_network.get_time_network_map().keys(), ymin=0, ymax=max_infected/3, ls=':', color='c', lw=1, alpha=0.8)
        plt.vlines(optimal_network.get_time_network_map().keys(), ymin=2*max_infected/3, ymax=max_infected, ls='-', color='m', lw=1, alpha=0.8)
        plt.vlines(random_compressed.get_time_network_map().keys(), ymin=max_infected/3, ymax=2*max_infected/3, ls='--', color='y', lw=1, alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('Number nodes infected')
        plt.xticks(list(temporal_network.get_time_network_map().keys())[::4])
        plt.legend()
        # plt.show()
        plt.subplot(212)
        # plt.figure('error')
    total_optimal_error = round(np.sum(np.abs(-np.array(d[1])+np.array(d2[1]))), 2)
    total_random_error = round(np.sum(np.abs(-np.array(d[1])+np.array(d3[1]))), 2)
    if plot:
        plt.plot(d[0], -np.array(d[1])+np.array(d2[1]), label=f'error from optimal: {total_optimal_error}', color='m', ls='--')
        plt.plot(d[0], -np.array(d[1])+np.array(d3[1]), label=f'error from random: {total_random_error}', color='y', ls='--')
        plt.xlabel('time')
        plt.ylabel('error from fully temporal')
        plt.legend()
        # plt.show()
    return total_optimal_error, total_random_error

t_interval = 4
beta = .005
number_layers = 20

temporal_network = random_temporal_network_mix(number_layers, t_interval, beta)
one_round(temporal_network, t_interval, beta, number_layers, levels=1, iters=20, plot=True)
## levels: number of pairs to squash:
## iterations: rounds of doing that
plt.show()

iter_range = 21
ensemble_length = 5
random_errors = np.zeros((ensemble_length, iter_range))
optimal_errors = np.zeros((ensemble_length, iter_range))

# for lev in range(5):
temporal_network = random_temporal_network_mix(number_layers, t_interval, beta)
lev = 1
for i in range(iter_range):
    print(lev, i)
    for t in range(ensemble_length):
        total_opt, total_rand = one_round(temporal_network, t_interval, beta, number_layers, levels=lev, iters=i)
        random_errors[t][i] = total_rand
        optimal_errors[t][i] = total_opt

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
plt.xlabel('Iterations')
plt.ylabel('Error from fully temporal')
plt.show()

print(random_errors)


