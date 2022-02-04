import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import deterministic
import poc_compression
import network_objects

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

fig, ax = plt.subplots(3, 3)
nx.draw(G1, ax=ax[0, 0], node_size=2, width=0.5, node_color='red')
nx.draw(G2, ax=ax[0, 1], node_size=2, width=0.5, node_color='orange')
nx.draw(G3, ax=ax[0, 2], node_size=2, width=0.5, node_color='purple')
nx.draw(G4, ax=ax[1, 0], node_size=2, width=0.5, node_color='green')
nx.draw(G5, ax=ax[1, 1], node_size=2, width=0.5, node_color='blue')
nx.draw(G6, ax=ax[1, 2], node_size=2, width=0.5, node_color='gold')
nx.draw(G7, ax=ax[2, 0], node_size=2, width=0.5, node_color='pink')
nx.draw(G8, ax=ax[2, 1], node_size=2, width=0.5, node_color='cyan')
nx.draw(G9, ax=ax[2, 2], node_size=2, width=0.5, node_color='forestgreen')
plt.show()

t_interval = 10
beta = .005
temporal_network = network_objects.TemporalNetwork([network_objects.Layer(0, 1*t_interval, beta, A1),
                                                    network_objects.Layer(10, 2*t_interval, beta, A2),
                                                    network_objects.Layer(20, 3*t_interval, beta, A3),
                                                    network_objects.Layer(30, 4*t_interval, beta, A4),
                                                    network_objects.Layer(40, 5*t_interval, beta, A5),
                                                    network_objects.Layer(50, 6*t_interval, beta, A6),
                                                    network_objects.Layer(60, 7*t_interval, beta, A7),
                                                    network_objects.Layer(70, 8*t_interval, beta, A8),
                                                    network_objects.Layer(80, 9*t_interval, beta, A9),
                                                    network_objects.Layer(90, 10*t_interval, beta, A10)
                                                    ])

optimal_network = network_objects.Compressor.compress(temporal_network, level=1, iterations=3)
random_compressed = network_objects.Compressor.compress(temporal_network, level=1, iterations=3, optimal=False)

model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=10*t_interval,
                        networks=temporal_network.get_time_network_map())
solution_t_temporal, solution_p = model.solve_model()
temporal_solution = np.sum(solution_p, axis=0)

model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=10*t_interval,
                        networks=optimal_network.get_time_network_map())
solution_t_compressed, solution_p = model.solve_model()
compressed_solution = np.sum(solution_p, axis=0)

model = deterministic.TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1 / N), end_time=10*t_interval,
                        networks=random_compressed.get_time_network_map())
solution_t_random, solution_p = model.solve_model()
random_solution = np.sum(solution_p, axis=0)


## TO THINK ABOUT FOR TOMORROW: HOW BEST TO HANDLE THE TIME SERIES DIFFERENTIAL
# do something with binning:
def digitize_me(time_vector, infected):
    digitized = np.digitize(np.array(time_vector), np.linspace(0,100,30))
    bin_means = [np.array(time_vector)[digitized == i].mean() for i in range(1, len(np.linspace(0,100,30)))]
    bins_infected = [np.array(infected)[digitized == i].mean() for i in range(1, len(np.linspace(0,100,30)))]
    return bin_means, bins_infected



##########
plt.plot(solution_t_temporal, temporal_solution, label='fully temporal', color='c', lw=3)
d = digitize_me(solution_t_temporal, temporal_solution)
plt.plot(d[0], d[1], label='fully temporal', color='c', lw=3)
plt.plot(solution_t_compressed, compressed_solution, label='optimal compressed temporal', color='m', lw=2, alpha=0.6)
d2 = digitize_me(solution_t_compressed, compressed_solution)
plt.plot(d2[0], d2[1], label='optimal compressed temporal', color='m', lw=2, alpha=0.6)
plt.plot(solution_t_random, random_solution, label='random compressed temporal', color='y', lw=2, alpha=0.6)
d3 = digitize_me(solution_t_random, random_solution)
plt.plot(d3[0], d3[1], label='random compressed temporal', color='y', lw=2, alpha=0.6)
## vertical lines to show compression
plt.vlines(temporal_network.get_time_network_map().keys(), ymin=0, ymax=33, ls=':', color='c', lw=1, alpha=0.8)
plt.vlines(optimal_network.get_time_network_map().keys(), ymin=66, ymax=100, ls='-', color='m', lw=1, alpha=0.8)
plt.vlines(random_compressed.get_time_network_map().keys(), ymin=33, ymax=66, ls='--', color='y', lw=1, alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Number nodes infected')
plt.xticks(list(temporal_network.get_time_network_map().keys()))
plt.legend()
plt.show()

total_optimal_error = np.sum(np.abs(np.array(d[1])-np.array(d2[1])))
total_random_error = np.sum(np.abs(np.array(d[1])-np.array(d3[1])))
plt.plot(d[0], np.array(d[1])-np.array(d2[1]), label=f'error from optimal: {total_optimal_error}')
plt.plot(d[0], np.array(d[1])-np.array(d3[1]), label=f'error from random: {total_random_error}')
plt.xlabel('time')
plt.ylabel('error from fully temporal')
plt.legend()
plt.show()

