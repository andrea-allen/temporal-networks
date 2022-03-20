import numpy as np
import time
import scipy.linalg as sl


class TemporalNetwork:
    """
    Holds adjacency matrices for each layer and the time windows for each layer
    """

    def __init__(self, layers):
        self.layers = layers
        self.length = len(layers)

    def get_ordered_pairs(self):
        pairs = list([(self.layers[i], self.layers[i + 1]) for i in range(0, len(self.layers) - 1)])
        return pairs

    def get_time_network_map(self):
        return {layer.end_time: layer.A for layer in self.layers}

    def equals(self, another_net):
        if self.length != another_net.length:
            return False
        else:
            for i in range(self.length):
                if not self.layers[i].equals(another_net.layers[i]):
                    return False
        return True


class Layer:
    """
    One layer of a temporal network. Can be expressed as a network object, adjacency matrix,
    and equipped with a start time t, end time t, and beta transmissibility
    """

    def __init__(self, start_time, end_time, beta, A):
        self.start_time = start_time
        self.end_time = end_time
        self.A = A
        self.N = len(self.A)
        self.beta = beta
        self.duration = self.end_time - self.start_time
        self.dd_normalized = self.set_dd_dist()

    def scaled_matrix(self):
        return self.beta * (self.end_time - self.start_time) * self.A

    def equals(self, another_layer):
        return self.start_time == another_layer.start_time and self.end_time == another_layer.end_time \
               and self.beta == another_layer.beta and self.duration == another_layer.duration \
               and np.array_equal(self.A, another_layer.A)

    def set_dd_dist(self):
        dd = np.array([np.sum(self.A[i]) / self.N for i in range(self.N)])
        if np.sum(dd)==0:
            return dd
        dd = dd / np.sum(dd)
        return dd


class Compressor:
    @staticmethod
    def compress(temporal_net, iterations=1, how='even', error_type='combined'):
        """
        Takes an ordered list of pairs and returns the compressed versions. Sole layers are returned as-is.
        :param error_type: 'terminal' or 'halftime' or 'combined' (default, best)
        :param how: 'optimal' or 'even'
        :param iterations: how many rounds of compression to perform
        :param temporal_net: TemporalNetwork object
        :return: Ordered list of layer pairs or singles
        """
        to_compress = int(iterations)
        if how.lower() == 'even':
            desired_num_layers = temporal_net.length - to_compress
            new_networks = TemporalNetwork(Compressor._even_compression(temporal_net, desired_num_layers))
            return new_networks
        current_net = temporal_net
        total_chosen_error = 0
        for r in range(iterations):
            new_net, chosen_error = Compressor._compress_round(current_net, how, error_type)
            current_net = new_net
            total_chosen_error += chosen_error
        return current_net, total_chosen_error

    @staticmethod
    def _compress_round(temporal_net, how, error_type):
        if how.lower() == 'optimal':
            new_networks, chosen_error = Compressor._optimal_compression(temporal_net, error_type)
            return TemporalNetwork(new_networks), chosen_error
        elif how.lower() == 'random':
            new_networks = Compressor._random_compression(temporal_net)
            return TemporalNetwork(new_networks)

    @staticmethod
    def _optimal_compression(temporal_net, error_type='terminal'):
        all_pairs = temporal_net.get_ordered_pairs()
        epsilons = Compressor.pairwise_epsilon(all_pairs, error_type)
        best_keys = [key for (key, val) in epsilons.items() if val == min(epsilons.values())]
        best_key = best_keys[0]
        # print(f"selecting best key {best_key} with epsilon {min(epsilons.values())}")
        chosen_error = min(epsilons.values())
        # print(f"num best keys was {len(best_keys)}")
        pairs = all_pairs
        new_networks = []
        layers_getting_compressed = [pairs[best_key][0]]
        the_layers = temporal_net.layers
        for id, layer in enumerate(temporal_net.layers):
            if the_layers[id] in layers_getting_compressed:
                new_networks.append(Compressor.aggregate(the_layers[id], the_layers[id+1]))
            elif the_layers[id-1] not in layers_getting_compressed:
                new_networks.append(the_layers[id])
        return new_networks, chosen_error

    @staticmethod
    def _random_compression(temporal_net):
        random_idx = np.random.randint(0, temporal_net.length - 1)
        new_networks = []
        for idx, layer in enumerate(temporal_net.layers):
            if idx == random_idx:
                new_networks.append(Compressor.aggregate(temporal_net.layers[idx], temporal_net.layers[idx + 1]))
            if idx - 1 == random_idx:
                pass
            elif idx != random_idx:
                new_networks.append(temporal_net.layers[idx])
        return new_networks

    @staticmethod
    def _even_compression(temporal_net, desired_num_layers):
        new_networks = []
        layers_boundaries = list([int(i) for i in np.linspace(0, temporal_net.length, desired_num_layers + 1)])
        # 0 20 40 60 for a 60 layer network
        idx = 0
        for b in range(len(layers_boundaries) - 1):
            current_aggregate = temporal_net.layers[layers_boundaries[b]]
            idx += 1
            while idx < layers_boundaries[b + 1]:
                current_aggregate = Compressor.aggregate(current_aggregate, temporal_net.layers[idx])
                idx += 1
            new_networks.append(current_aggregate)
        return new_networks

    @staticmethod
    def pairwise_epsilon(pairs, error_type='terminal'):
        epsilon_values = {}
        for idx, pair in enumerate(pairs):
            try:
                epsilon = Compressor.epsilon(pair[0], pair[1], error_type)
                epsilon_values[idx] = epsilon
            except TypeError:
                pass  # if pair is a single
        return epsilon_values

    @staticmethod
    def aggregate(layer, other):
        total_duration = other.end_time - layer.start_time
        new_adjacency_matrix = ((layer.end_time - layer.start_time) * layer.A
                                + (other.end_time - other.start_time) * other.A) / total_duration
        return Layer(layer.start_time, other.end_time, layer.beta, new_adjacency_matrix)

    @staticmethod
    def epsilon(layer, other, error_type='terminal'):
        if error_type.lower() == 'combined':
            A = layer.scaled_matrix()
            B = other.scaled_matrix()

            #######
            BA = B @ A
            AB = A @ B
            error_terminal = (1 / 2) * (BA - AB) \
                    + (1 / 12) * ((B @ BA) + (A @ AB)
                                  + (A @ (B @ B)) + (B @ (A @ A))) \
                    - (1 / 6) * ((B @ AB) + (A @ BA))
            P0 = layer.dd_normalized

            e_terminal = np.sum(np.abs(error_terminal).dot(P0))

            agg_mat = (layer.duration * layer.A + other.duration*other.A)/(layer.duration+other.duration)
            # full_agg = sl.expm(layer.beta * layer.duration * agg_mat) #FULL MATRIX
            #### APPROXIMATIONS
            # A_scaled = layer.beta * layer.duration * layer.A
            approx_A_temp = A + (A@A)/2 + (A@A@A)/6 \
                            # + (A_scaled@A_scaled@A_scaled@A_scaled)/24 \
                            # + (A_scaled@A_scaled@A_scaled@A_scaled@A_scaled)/(5*24)
            agg_scaled = layer.beta * layer.duration * agg_mat
            approx_Agg = agg_scaled + (agg_scaled@agg_scaled)/2 + (agg_scaled@agg_scaled@agg_scaled)/6 \
                         # + (agg_scaled@agg_scaled@agg_scaled@agg_scaled)/24 \
                         # + (agg_scaled@agg_scaled@agg_scaled@agg_scaled@agg_scaled)/(5*24)
            D = approx_A_temp - approx_Agg

            ######
            # D = (full_A_temp - full_agg) #FULL MATRIX
            error = D
            e_halftime = np.sum(np.abs(error).dot(P0))
            if np.isnan(e_terminal) or np.isnan(e_halftime):
                print('STOP')
            if e_halftime > 0 and e_terminal==0:
                print(f'halftime: {e_halftime}, terminal: {e_terminal}')
            return (e_halftime + e_terminal) * (layer.duration+other.duration)

        elif error_type.lower()=='terminal':
            A = layer.scaled_matrix()
            B = other.scaled_matrix()

            #######
            BA = B @ A
            AB = A @ B
            error = (1 / 2) * (BA - AB) \
                    + (1 / 12) * ((B @ BA) + (A @ AB)
                                  + (A @ (B @ B)) + (B @ (A @ A))) \
                    - (1 / 6) * ((B @ AB) + (A @ BA))
            ########
        elif error_type.lower()=='halftime':
            # Using full matexp
            # full_A_temp = sl.expm(layer.beta * layer.duration * layer.A) #FULL MATRIX
            agg_mat = (layer.duration * layer.A + other.duration*other.A)/(layer.duration+other.duration)
            # full_agg = sl.expm(layer.beta * layer.duration * agg_mat) #FULL MATRIX
            #### APPROXIMATIONS
            A_scaled = layer.beta * layer.duration * layer.A
            approx_A_temp = A_scaled + (A_scaled@A_scaled)/2 + (A_scaled@A_scaled@A_scaled)/6 \
                            # + (A_scaled@A_scaled@A_scaled@A_scaled)/24 \
                            # + (A_scaled@A_scaled@A_scaled@A_scaled@A_scaled)/(5*24)
            agg_scaled = layer.beta * layer.duration * agg_mat
            approx_Agg = agg_scaled + (agg_scaled@agg_scaled)/2 + (agg_scaled@agg_scaled@agg_scaled)/6 \
                         # + (agg_scaled@agg_scaled@agg_scaled@agg_scaled)/24 \
                         # + (agg_scaled@agg_scaled@agg_scaled@agg_scaled@agg_scaled)/(5*24)
            D = approx_A_temp - approx_Agg

            ######
            # D = (full_A_temp - full_agg) #FULL MATRIX
            error = D
            #############

        P0 = layer.dd_normalized

        total_infect_diff = np.sum(np.abs(error).dot(P0))
        total_infect_diff = total_infect_diff * (layer.duration+other.duration)

        return total_infect_diff
