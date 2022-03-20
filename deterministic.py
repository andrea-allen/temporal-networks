import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate
import networkx as nx
from scipy import linalg


class SIModel:
    def __init__(self, params, y_init, num_time_steps, approximate=False):
        self.params = params
        self.y_init = y_init
        self.time_series = []
        self.num_time_steps = num_time_steps
        self.approximate = approximate

    def solve_model(self):
        half_time_solution = scipy.integrate.solve_ivp(self.odes_si, t_span=[0, self.num_time_steps / 2],
                                                       y0=self.y_init)
        half_time_prob_vec = [half_time_solution.y[i][-1] for i in range(len(half_time_solution.y))]
        second_half_solution = scipy.integrate.solve_ivp(self.odes_si, t_span=[self.num_time_steps / 2,
                                                                               self.num_time_steps],
                                                         y0=half_time_prob_vec)
        # solution = scipy.integrate.solve_ivp(self.odes_si, t_span=[0, self.num_time_steps], y0=self.y_init)
        solution_t = np.zeros(len(half_time_solution.t) + len(second_half_solution.t))
        solution_ys = np.zeros((len(half_time_solution.y), len(half_time_solution.t) + len(second_half_solution.t)))
        solution_t[:len(half_time_solution.t)] = half_time_solution.t
        solution_t[len(half_time_solution.t):] = second_half_solution.t

        solution_ys[:, :len(half_time_solution.t)] = half_time_solution.y
        solution_ys[:, len(half_time_solution.t):] = second_half_solution.y
        self.time_series.append(solution_t)
        for i in range(len(self.y_init)):
            self.time_series.append(solution_ys[i])
        return self.time_series

    def odes_si(self, t, y):
        A_1 = self.params['A_1']
        A_2 = self.params['A_2']
        beta = self.params['beta']
        alpha = self.params['alpha']
        N = max(len(A_1), len(A_2))

        derivatives = np.zeros(len(y))
        if t < self.num_time_steps / 2:
            A = A_1
        elif t >= self.num_time_steps / 2:
            A = A_2
        for i in range(N):
            derivatives[i] = (1 - y[i]) * beta * np.sum(A[i] @ y) - y[i] * alpha
            if self.approximate:
                derivatives[i] = (1) * beta * np.sum(A[i] @ y) - y[i] * alpha

        return derivatives


def solve_model(model_params, y_init, num_time_steps, approximate):
    ## y_init should be of form [inf_res, rec_res, inf_worker, rec_worker, inf_comm, rec_comm]
    model = SIModel(params=model_params, y_init=y_init, num_time_steps=num_time_steps, approximate=approximate)
    solution_ts = model.solve_model()
    return solution_ts


def plot_model_solution(solution_ts, savefile=None, show=False):
    fig, axs = plt.subplots(2, 1)
    for i in range(1, len(solution_ts)):
        axs[0].plot(solution_ts[0], solution_ts[i], label=i)
    axs[0].set_ylabel('Probability of infection')
    # axs[0].legend()
    axs[1].set_xlabel('Time')
    axs[1].plot(solution_ts[0], np.sum(solution_ts[1:], axis=0))
    axs[1].set_ylabel('Total number infections')

    if show:
        plt.show()


def run_scratchwork():
    adj_matrix_layer_1 = np.array([
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    ])
    adj_matrix_layer_2 = np.array([
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    ])

    adj_matrix_layer_2 = np.zeros((10, 10))

    degree_distribution = [0, 25 / 100, 72 / 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 100, 1 / 100, 1 / 100]
    config_model = nx.generators.configuration_model(
        np.random.choice(np.arange(len(degree_distribution)), p=degree_distribution, size=1000))

    adj_matrix_layer_1 = np.array(nx.adjacency_matrix(config_model).todense())
    er_graph = nx.generators.erdos_renyi_graph(n=len(adj_matrix_layer_1), p=0.005)
    adj_matrix_layer_2 = np.array(nx.adjacency_matrix(er_graph).todense())

    combo_network = nx.Graph()
    combo_network.add_edges_from(config_model.edges())
    combo_network.add_edges_from(er_graph.edges())
    adj_matrix_combo = np.array(nx.adjacency_matrix(combo_network).todense())

    beta = 0.05
    print('Leading eigenvalue of first adjacency matrix:')
    print(max(linalg.eig(adj_matrix_layer_1)[0].real))
    print('Exponential growth would be:')
    eigen_1 = max(linalg.eig(adj_matrix_layer_1)[0].real)
    print([(1 + eigen_1 * beta) ** t for t in range(10)])
    print('Leading eigenvalue of second adjacency matrix:')
    eigen_2 = max(linalg.eig(adj_matrix_layer_2)[0].real)
    print([(1 + eigen_2 * beta) ** t for t in range(10)])
    print(max(linalg.eig(adj_matrix_layer_2)[0].real))

    initial_conditions = np.full(len(adj_matrix_layer_2), 1 / len(adj_matrix_layer_2))
    # initial_conditions = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]
    ## Full model on A1:
    # model_soln = solve_model({'A_1': adj_matrix_layer_1, 'A_2':adj_matrix_layer_1, 'beta':0.2, 'alpha':0},
    #                          y_init=initial_conditions,
    #                          num_time_steps=3)
    # plot_model_solution(model_soln)
    #
    # ## Full model on A2:
    # model_soln = solve_model({'A_1': adj_matrix_layer_2, 'A_2':adj_matrix_layer_2, 'beta':0.2, 'alpha':0},
    #                          y_init=initial_conditions,
    #                          num_time_steps=3)
    # plot_model_solution(model_soln)

    ## Full model switch A1 to A2
    ## Correct way:
    model_soln_ode = solve_model({'A_1': adj_matrix_layer_1, 'A_2': adj_matrix_layer_2, 'beta': beta, 'alpha': 0},
                                 y_init=initial_conditions,
                                 num_time_steps=15, approximate=False)
    plot_model_solution(model_soln_ode)

    model_soln_ode_aggregate = solve_model(
        {'A_1': adj_matrix_combo / 2, 'A_2': adj_matrix_combo / 2, 'beta': beta, 'alpha': 0},
        y_init=initial_conditions,
        num_time_steps=15, approximate=False)
    plot_model_solution(model_soln_ode_aggregate)

    ## Approxmiate p(i) way:
    model_soln_approx = solve_model({'A_1': adj_matrix_layer_1, 'A_2': adj_matrix_layer_2, 'beta': beta, 'alpha': 0},
                                    y_init=initial_conditions,
                                    num_time_steps=15, approximate=True)
    plot_model_solution(model_soln_approx)
    model_soln_approx_aggregate = solve_model(
        {'A_1': adj_matrix_combo / 2, 'A_2': adj_matrix_combo / 2, 'beta': beta, 'alpha': 0},
        y_init=initial_conditions,
        num_time_steps=15, approximate=True)
    plot_model_solution(model_soln_approx_aggregate)

    plt.figure('compare')
    plt.plot(model_soln_ode[0], np.sum(model_soln_ode[1:], axis=0), label='Correct ODE')
    plt.plot(model_soln_ode_aggregate[0], np.sum(model_soln_ode_aggregate[1:], axis=0), label='Correct ODE, aggregate')
    plt.plot(model_soln_approx[0], np.sum(model_soln_approx[1:], axis=0), label='Approx ODE')
    P_0 = initial_conditions
    predicted_num_temporal = np.sum(
        scipy.linalg.expm(beta * adj_matrix_layer_2 * 15 / 2).dot(
            scipy.linalg.expm(beta * adj_matrix_layer_1 * 15 / 2).dot(P_0)))
    predicted_num_aggregate = np.sum(
        scipy.linalg.expm(((beta * (adj_matrix_layer_1 + adj_matrix_layer_2)) / 2) * 15).dot(P_0))
    plt.scatter(15, predicted_num_temporal, label='matexp prediction temporal')
    plt.scatter(15, predicted_num_aggregate, label='matexp prediction aggregate')
    plt.plot(np.arange(1, 15), [0.5 * (1 + eigen_2 * beta) ** t for t in range(1, 15)], label='eigen approx')
    plt.plot(model_soln_approx_aggregate[0], np.sum(model_soln_approx_aggregate[1:], axis=0),
             label='Approx ODE, aggregate')
    plt.legend()
    # plt.plot(model_soln_approx[0], np.full(len(model_soln_approx[0]), len(adj_matrix_layer_1)), label='Number of nodes', ls='--', color='black')
    plt.xlabel('Time')
    plt.ylabel('Total number infections')
    plt.show()

    ## Model aggregated A1 and A2

    ## To try next: Maybe switch exponential growth rates halfway for the eigen approximation


class TemporalSIModel:
    def __init__(self, params, y_init, end_time, networks, approximate=False):
        self.params = params
        self.y_init = y_init
        self.y_current = self.y_init
        self.result = []
        self.end_time = end_time
        self.start_time = 0
        self.approximate = approximate
        self.networks = networks ## dictionary of switch times to adjacency matrices
        self.switch_times = sorted(list(self.networks.keys()))
        self.current_switch_time_index = 0
        self.current_switch_time = self.switch_times[0]
        self.N = len(y_init)

    def odes_si(self, y, t):
        beta = self.params['beta']
        derivatives = np.zeros(len(y))
        A = self.networks[self.current_switch_time]
        for i in range(self.N):
            derivatives[i] = (1 - y[i]) * beta * np.sum(A[i] @ y)
            if self.approximate:
                derivatives[i] = (1) * beta * np.sum(A[i] @ y)
        return derivatives

    def solve_model(self, total_time_steps=300):
        time_series_result = []
        node_probabilities = [[] for n in range(self.N)]
        steps_per_interval = max(int(np.round(total_time_steps/len(self.switch_times), 0)),20)
        # steps_per_interval = 5
        # print(steps_per_interval)
        for switchtime in self.switch_times:
            self.current_switch_time = switchtime
            # switchtime_solution = scipy.integrate.solve_ivp(self.odes_si, t_span=[self.start_time,
            #                                                                       switchtime],
            #                                                 y0=self.y_current)
            switchtime_solution = scipy.integrate.odeint(self.odes_si,
                                                         y0=self.y_current,
                                                         t=np.linspace(self.start_time,
                                                                                  switchtime, steps_per_interval)
                                                            )
            # time_series_result.extend(switchtime_solution.t)
            time_series_result.extend(np.linspace(self.start_time, switchtime, steps_per_interval))
            for i in range(self.N):
                node_probabilities[i].extend(list(switchtime_solution[:, i]))
            # new_initial_p_vec = [switchtime_solution.y[i][-1] for i in range(len(switchtime_solution.y))] # for ivp code
            new_initial_p_vec = [switchtime_solution[:,i][-1] for i in range(self.N)]
            self.start_time = switchtime
            self.y_current = new_initial_p_vec

        return time_series_result, np.array(node_probabilities)

def temporal_model_base_case():
    print('run')
    N=100
    beta = .05
    G1 = nx.generators.erdos_renyi_graph(n=N, p=.05)
    G2 = nx.generators.erdos_renyi_graph(n=N, p=.1)
    G3 = nx.generators.cycle_graph(n=N)
    G4 = nx.generators.erdos_renyi_graph(n=N, p=.05)
    A1 = np.array(nx.adjacency_matrix(G1).todense())
    A2 = np.array(nx.adjacency_matrix(G2).todense())
    A3 = np.array(nx.adjacency_matrix(G3).todense())
    A4 = np.array(nx.adjacency_matrix(G4).todense())
    model = TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1/N), end_time=10, networks={3: A1, 5: A2, 7: A3, 10: A4})
    solution_t, solution_p = model.solve_model()
    infections_timeseries = np.sum(solution_p, axis=0)
    plt.plot(solution_t, infections_timeseries, label='fully temporal')
    model = TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1/N), end_time=10, networks={5: (A1+A2)/2, 7: A3, 10:A4})
    solution_t, solution_p = model.solve_model()
    infections_timeseries = np.sum(solution_p, axis=0)
    plt.plot(solution_t, infections_timeseries, label='1 pair')
    model = TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1/N), end_time=10, networks={5: (A1+A2)/2, 10: (A3+A4)/2})
    solution_t, solution_p = model.solve_model()
    infections_timeseries = np.sum(solution_p, axis=0)
    plt.plot(solution_t, infections_timeseries, label='2 pairs')
    model = TemporalSIModel(params={'beta': beta}, y_init=np.full(N, 1/N), end_time=10, networks={10: (A1+A2+A3+A4)/4})
    solution_t, solution_p = model.solve_model()
    infections_timeseries = np.sum(solution_p, axis=0)
    plt.plot(solution_t, infections_timeseries, label='fully aggregated')
    plt.legend()
    plt.show()

# temporal_model_base_case()



