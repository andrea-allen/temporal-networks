import numpy as np


class TemporalNetwork:
    """
    Holds adjacency matrices for each layer and the time windows for each layer
    """

    def __init__(self, layers):
        self.layers = layers
        self.length = len(layers)

    def get_ordered_pairs(self, shift=False):
        pairs = list([(self.layers[i], self.layers[i + 1]) for i in range(0, len(self.layers) - 1, 2)])
        if self.length % 2 == 0:
            return pairs
        else:
            if not shift:
                pairs.append(self.layers[-1])
                return pairs
            if shift:
                pairs = [self.layers[0]]
                pairs.extend(list([(self.layers[i], self.layers[i + 1]) for i in range(1, len(self.layers) - 1, 2)]))
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
        self.beta = beta
        self.duration = self.end_time - self.start_time

    def scaled_matrix(self):
        return self.beta * (self.end_time - self.start_time) * self.A

    def equals(self, another_layer):
        return self.start_time == another_layer.start_time and self.end_time == another_layer.end_time \
               and self.beta == another_layer.beta and self.duration == another_layer.duration \
               and np.array_equal(self.A, another_layer.A)


class Compressor:
    @staticmethod
    def compress(temporal_net, level, optimal=True, iterations=1):
        """
        Takes an ordered list of pairs and returns the compressed versions. Sole layers are returned as-is.
        :param iterations: how many rounds of compression to perform
        :param optimal: Use optimal compression algorithm, otherwise random
        :param temporal_net: TemporalNetwork object
        :param level: int, number of pairs to compress
        :return: Ordered list of layer pairs or singles
        """
        current_net = temporal_net
        for r in range(iterations):
            new_net = Compressor._compress_round(current_net, level, optimal)
            current_net = new_net
        return current_net

    @staticmethod
    def _compress_round(temporal_net, level, optimal):
        if optimal:
            new_networks = Compressor._optimal_compression(temporal_net, level)
            return TemporalNetwork(new_networks)
        else:
            new_networks = Compressor._random_compression(temporal_net, level)
            return TemporalNetwork(new_networks)

    @staticmethod
    def _optimal_compression(temporal_net, level):
        if level==1:
            nonshift_epsilons = Compressor.pairwise_epsilon(temporal_net.get_ordered_pairs(shift=0))
            shift_epsilons = Compressor.pairwise_epsilon(temporal_net.get_ordered_pairs(shift=1))
            best_key_nonshift = min(nonshift_epsilons, key=nonshift_epsilons.get)
            best_key_shift = min(shift_epsilons, key=shift_epsilons.get)
            if shift_epsilons[best_key_shift] < nonshift_epsilons[best_key_nonshift]:
                best_key = best_key_shift
                pairs = temporal_net.get_ordered_pairs(shift=1)
            else:
                best_key = best_key_nonshift
                pairs = temporal_net.get_ordered_pairs(shift=0)

            new_networks = []
            pairs_to_compress = []
            pairs_to_compress.append(pairs[best_key])
            for pair in pairs:
                if pair in pairs_to_compress:
                    new_networks.append(Compressor.aggregate(pair[0], pair[1]))
                else:
                    try:
                        new_networks.extend([pair[0], pair[1]])
                    except TypeError:
                        new_networks.extend([pair])
            return new_networks

        #### shifting, for higher level
        # combine these later
        nonshift_epsilons = Compressor.pairwise_epsilon(temporal_net.get_ordered_pairs(shift=0))
        shift_epsilons = Compressor.pairwise_epsilon(temporal_net.get_ordered_pairs(shift=1))
        if sum(nonshift_epsilons.values()) > sum(shift_epsilons.values()):
            epsilon_values = shift_epsilons
            pairs = temporal_net.get_ordered_pairs(shift=1)
        else:
            epsilon_values = nonshift_epsilons
            pairs = temporal_net.get_ordered_pairs(shift=0)
        new_networks = []
        pairs_to_compress = []
        while len(pairs_to_compress) < level and len(epsilon_values) > 0:
            best_key = min(epsilon_values, key=epsilon_values.get)
            pairs_to_compress.append(pairs[best_key])
            epsilon_values.pop(best_key)
        for pair in pairs:
            if pair in pairs_to_compress:
                new_networks.append(Compressor.aggregate(pair[0], pair[1]))
            else:
                try:
                    new_networks.extend([pair[0], pair[1]])
                except TypeError:
                    new_networks.extend([pair])
        return new_networks

    @staticmethod
    def _random_compression(temporal_net, level):
        # TODO: level might be required to be 1
        random_idx = np.random.randint(0, temporal_net.length-1)
        # random_pair_idxs = np.random.choice(list(np.arange(temporal_net.length-1)), size=level, replace=False)
        new_networks = []
        for idx, layer in enumerate(temporal_net.layers):
            if idx == random_idx:
                new_networks.append(Compressor.aggregate(temporal_net.layers[idx], temporal_net.layers[idx+1]))
            if idx-1 == random_idx:
                pass
            elif idx != random_idx:
                new_networks.append(temporal_net.layers[idx])
        return new_networks

    @staticmethod
    def pairwise_epsilon(pairs):
        epsilon_values = {}
        for idx, pair in enumerate(pairs):
            try:
                epsilon = Compressor.epsilon(pair[0], pair[1])
                epsilon_values[idx] = epsilon
            except TypeError:
                pass  # if pair is a single
        return epsilon_values


    @staticmethod
    def aggregate(layer, other):
        total_duration = other.end_time - layer.start_time
        new_adjacency_matrix = ((layer.end_time-layer.start_time)*layer.A
                                + (other.end_time-other.start_time)*other.A) / total_duration
        return Layer(layer.start_time, other.end_time, layer.beta, new_adjacency_matrix)

    @staticmethod
    def epsilon(layer, other):
        A = layer.scaled_matrix()
        B = other.scaled_matrix()
        N = len(A)
        P0 = np.full(N, 1 / N)
        Z = B + A + (1 / 2) * (B @ A - A @ B) + (1 / 12) * (B @ B @ A + A @ A @ B + A @ B @ B + B @ A @ A) - (1 / 6) * (
                B @ A @ B + A @ B @ A)
        error = Z - (A + B)
        total_infect_diff = np.sum(np.abs(error).dot(P0))
        return total_infect_diff
