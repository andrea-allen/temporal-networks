import unittest
from network_objects import *
from network_objects import Compressor


class TemporalNetworkTests(unittest.TestCase):

    def test_get_ordered_pairs_even(self):
        temporal_net = TemporalNetwork(['a', 'b', 'c', 'd', 'e', 'f'])
        pairs = temporal_net.get_ordered_pairs()
        self.assertEqual(pairs[0], ('a', 'b'))
        self.assertEqual(pairs[1], ('c', 'd'))
        self.assertEqual(pairs[2], ('e', 'f'))
        self.assertEqual(len(pairs), 3)

    def test_get_ordered_pairs_odd(self):
        temporal_net = TemporalNetwork(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        pairs = temporal_net.get_ordered_pairs()
        self.assertEqual(pairs[0], ('a', 'b'))
        self.assertEqual(pairs[1], ('c', 'd'))
        self.assertEqual(pairs[2], ('e', 'f'))
        self.assertEqual(pairs[3], 'g')
        self.assertEqual(len(pairs), 4)

    def test_get_ordered_pairs_shift(self):
        temporal_net = TemporalNetwork(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        pairs = temporal_net.get_ordered_pairs(shift=True)
        self.assertEqual(pairs[0], 'a')
        self.assertEqual(pairs[1], ('b', 'c'))
        self.assertEqual(pairs[2], ('d', 'e'))
        self.assertEqual(pairs[3], ('f', 'g'))
        self.assertEqual(len(pairs), 4)

    def test_equals(self):
        # SETUP
        A = Layer(1, 2, .5,
                  np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]]))
        B = Layer(2, 3, .5,
                  np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]]))
        temporal_net = TemporalNetwork([A, A, B, B])
        another_net = TemporalNetwork([A, B, A, B])

        # ACT / ASSERT
        self.assertTrue(temporal_net.equals(temporal_net))
        self.assertTrue(another_net.equals(another_net))
        self.assertFalse(another_net.equals(temporal_net))
        self.assertFalse(temporal_net.equals(another_net))

    def test_time_network_map(self):
        # SETUP
        A = Layer(1, 2, .5,
                  np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]]))
        B = Layer(2, 3, .5,
                  np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]]))
        C = Layer(3, 4, .5,
                  np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]]))
        temporal_net = TemporalNetwork([A, B, C])
        expected_map = {2: A.A, 3: B.A, 4: C.A}
        # ACT / ASSERT
        self.assertEqual(expected_map, temporal_net.get_time_network_map())




class CompressorTests(unittest.TestCase):
    def test_compress_basecase(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
        A = Layer(1, 2, .5,
                  array_one)
        A2 = Layer(2, 3, .5, array_one)
        B = Layer(3, 4, .5, array_two)
        B2 = Layer(4, 5, .5, array_two)
        temporal_net = TemporalNetwork([A, A2, B, B2])
        # ACT
        compressed_net = Compressor.compress(temporal_net, level=2)
        # ASSERT
        self.assertTrue(TemporalNetwork([Layer(1, 3, .5, array_one), Layer(3, 5, .5, array_two)]).equals(compressed_net))

    def test_compress(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
        A = Layer(1, 2, .5, array_one)
        A2 = Layer(2, 3, .5, array_one)
        A3 = Layer(5, 6, .5, array_one)
        A4 = Layer(7, 8, .5, array_one)
        A5 = Layer(8, 9, .5, array_one)
        B = Layer(3, 4, .5,array_two)
        B2 = Layer(4, 5, .5,array_two)
        B3 = Layer(6, 7, .5,array_two)
        temporal_net = TemporalNetwork([A, A2, B, B2, A3, B3, A4, A5])
        combined_layer = Layer(1,3, .5, (A.A+B.A)/2)
        # ACT
        compressed_net = Compressor.compress(temporal_net, level=3)
        # ASSERT
        self.assertTrue(TemporalNetwork([Layer(1, 3, .5, array_one),
                                         Layer(3, 5, .5, array_two),
                                         A3, B3, Layer(7, 9, .5, array_one)]).equals(compressed_net))
        self.assertFalse(TemporalNetwork([A, combined_layer, combined_layer, combined_layer, A]).equals(compressed_net))

    def test_compress_odd(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])
        A = Layer(1, 2, .5, array_one)
        A2 = Layer(2, 3, .5, array_one)
        A3 = Layer(5, 6, .5, array_one)
        A4 = Layer(7, 8, .5, array_one)
        A5 = Layer(8, 9, .5, array_one)
        B = Layer(3, 4, .5, array_two)
        B2 = Layer(4, 5, .5, array_two)
        B3 = Layer(6, 7, .5, array_two)
        B4 = Layer(9, 10, .5, array_two)
        temporal_net = TemporalNetwork([A, A2, B, B2, A3, B3, A4, A5, B4])
        # ACT
        compressed_net = Compressor.compress(temporal_net, level=3)
        # ASSERT
        self.assertTrue(TemporalNetwork([Layer(1,3,.5,array_one),
                                         Layer(3,5,.5,array_two),
                                         A3,
                                         B3,
                                         Layer(7,9, .5, array_one),
                                         B4]).equals(compressed_net))

    def test_compress_odd_shift(self):
        # SETUP
        array_one = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        array_two = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]])

        temporal_net = TemporalNetwork([Layer(1, 2, .5, array_two),
                                        Layer(2,3, .5,array_one),
                                        Layer(3,4, .5,array_one),
                                        Layer(4,5, .5, array_two),
                                        Layer(5,6, .5, array_two),
                                        Layer(6,7, .5,array_one),
                                        Layer(7,8, .5, array_two),
                                        Layer(8,9, .5,array_one),
                                        Layer(9,10, .5,array_one)])

        expected_compressed_net = TemporalNetwork([Layer(1, 2, .5, array_two),
                                                   Layer(2, 4, .5, array_one),
                                                   Layer(4, 6, .5, array_two),
                                                   Layer(6, 7, .5, array_one),
                                                   Layer(7, 8, .5, array_two),
                                                   Layer(8, 10, .5, array_one)])
        # ACT
        compressed_net = Compressor.compress(temporal_net, level=3)
        # ASSERT
        self.assertTrue(expected_compressed_net.equals(compressed_net))

        # ACT
        compressed_net = Compressor.compress(temporal_net, level=4)
        # ASSERT
        self.assertTrue(TemporalNetwork([
            Layer(1, 2, .5, array_two),
            Layer(2, 4, .5, array_one),
            Layer(4, 6, .5, array_two),
            Layer(6, 8, .5, (array_one+array_two)/2),
            Layer(8,10,.5, array_one)])
                        .equals(compressed_net))

    def test_compress_twice(self):
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        A = Layer(1, 2, 0.8,
                  some_array)
        B = Layer(2, 3, 0.8,
                  some_array)
        C = Layer(3, 4, 0.8,
                  other_array)
        D = Layer(4, 5, 0.8,
                  other_array)
        E = Layer(5, 6, 0.8,
                  some_array)
        F = Layer(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        compressed_net_once = Compressor.compress(temporal_net, level=2)
        # ASSERT
        should_be_network = TemporalNetwork([Layer(1, 3, 0.8, some_array),
                                             Layer(3,5, 0.8, other_array),
                                             E, F])
        alt_should_be = TemporalNetwork([A, B,
                                        Layer(3,5, 0.8, other_array),
                                        Layer(5,7, 0.8, some_array)])
        alt_should_be_2 = TemporalNetwork([Layer(1, 3, 0.8, some_array),
                                         C, D,
                                         Layer(5, 7, 0.8, some_array)])
        self.assertTrue(should_be_network.equals(compressed_net_once) or alt_should_be.equals(compressed_net_once)
                        or alt_should_be_2.equals(compressed_net_once))

        # ACT (compress again)
        compressed_net_twice = Compressor.compress(compressed_net_once, level=2)

        # ASSERT
        if compressed_net_once.equals(should_be_network):
            self.assertTrue(compressed_net_twice.equals(TemporalNetwork([Layer(1,5, 0.8, (some_array+other_array)/2),
                                                                         Layer(5,6,.8,some_array)])))
        if compressed_net_once.equals(alt_should_be):
            self.assertTrue(compressed_net_twice.equals(TemporalNetwork([Layer(1,3, 0.8, some_array),
                                                                         Layer(3, 7,.8,(some_array+other_array)/2)])))
        if compressed_net_once.equals(alt_should_be_2):
            self.assertTrue(compressed_net_twice.equals(TemporalNetwork([Layer(1,4, 0.8, (2*some_array+other_array)/3),
                                                                         Layer(4,7,.8,(2*some_array+other_array)/3)])))

    def test_compress_iteratively(self):
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        A = Layer(1, 2, 0.8,
                  some_array)
        B = Layer(2, 3, 0.8,
                  some_array)
        C = Layer(3, 4, 0.8,
                  other_array)
        D = Layer(4, 5, 0.8,
                  other_array)
        E = Layer(5, 6, 0.8,
                  some_array)
        F = Layer(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        compressed_net_twice = Compressor.compress(temporal_net, level=2, optimal=True, iterations=2)

        # ASSERT
        self.assertTrue(compressed_net_twice.equals(TemporalNetwork([Layer(1, 4, 0.8, (2 * some_array + other_array) / 3),
                                                         Layer(4, 7, .8, (2 * some_array + other_array) / 3)])))




    def test_epsilon_trivial(self):
        A = Layer(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Layer(2, 3, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))

        error = Compressor.epsilon(A, B)
        self.assertEqual(0, error)

    def test_epsilon_nontrivial(self):
        A = Layer(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Layer(2, 3, 0.8,
                  np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]]))

        error = Compressor.epsilon(A, B)
        self.assertNotEqual(0, error)

    def test_aggregate(self):
        A = Layer(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Layer(2, 3, 0.8,
                  np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]]))
        aggregated = Compressor.aggregate(A, B)
        self.assertTrue(aggregated.equals(Layer(A.start_time,
                                                B.end_time,
                                                0.8,
                                                (A.A + B.A)/2)))

    def test_aggregate_by_duration(self):
        A = Layer(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        B = Layer(2, 4, 0.8,
                  np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]]))
        aggregated = Compressor.aggregate(A, B)
        self.assertTrue(aggregated.equals(Layer(A.start_time,
                                                B.end_time,
                                                0.8,
                                                (2*B.A + A.A)/3)))

        # Base case unequal duration aggregation:
        A_longer = Layer(2, 7, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        aggregated_base = Compressor.aggregate(A, A_longer)
        self.assertTrue(aggregated_base.equals(Layer(1,7,0.8,
                                                np.array([[0, 1, 0, 1],
                                                          [1, 0, 1, 0],
                                                          [0, 1, 0, 0],
                                                          [1, 0, 0, 0]])
                                                )))
        self.assertTrue(aggregated_base.equals(Layer(1,7,0.8, (A.A + 5*A_longer.A)/6)))

    def test_time_ordering(self):
        # SETUP
        A = Layer(1, 2, 0.8,
                  np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]]))
        same_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        B = Layer(2, 3, 0.8,
                  same_array)
        C = Layer(3, 4, 0.8,
                  same_array)
        D = Layer(4, 5, 0.8,
                  same_array)
        temporal_net = TemporalNetwork([A, B, C, D])

        # ACT
        compressed_net = Compressor.compress(temporal_net, level=2)
        should_be_network = TemporalNetwork([Layer(1, 3, 0.8, (A.A+B.A)/2),
                                             Layer(3, 5, 0.8, same_array)])
        should_not_be = TemporalNetwork([Layer(1, 2, 0.8, (A.A+B.A)/2),
                                        Layer(3, 4, 0.8, same_array)])

        # ASSERT
        self.assertTrue(should_be_network.equals(compressed_net))
        self.assertFalse(should_not_be.equals(compressed_net))
        # SECOND ACT (level=1)
        compressed_net = Compressor.compress(temporal_net, level=1)
        could_be_network_1 = TemporalNetwork([A,
                                             Layer(2, 3, 0.8, same_array),
                                             D])
        could_be_network_2 = TemporalNetwork([A,B,
                                             Layer(3, 5, 0.8, same_array)])
        # ASSERT
        self.assertTrue(could_be_network_1.equals(compressed_net) or could_be_network_2.equals(compressed_net))

    def test_total_aggregate(self):
        # Test that compressing at level 1, length layers iterations, gives the total sum normalized
        # SETUP
        some_array = np.array([[0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
        other_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 0],
                            [0, 2, 0, 0],
                            [1, 0, 0, 0]])
        third_array = np.array([[0, 0, 0, 1],
                            [0, 0, 2, 2],
                            [0, 2, 0, 0],
                            [1, 2, 0, 0]])
        A = Layer(1, 2, 0.8,
                  some_array)
        B = Layer(2, 3, 0.8,
                  other_array)
        C = Layer(3, 4, 0.8,
                  third_array)
        D = Layer(4, 5, 0.8,
                  other_array)
        E = Layer(5, 6, 0.8,
                  some_array)
        F = Layer(6, 7, 0.8,
                  some_array)
        temporal_net = TemporalNetwork([A, B, C, D, E, F])

        # ACT
        # optimal
        compressed_net = Compressor.compress(temporal_net, level=1, optimal=True, iterations=5)
        # random
        random_net = Compressor.compress(temporal_net, level=1, optimal=False, iterations=5)
        random_net2 = Compressor.compress(temporal_net, level=1, optimal=False, iterations=5)
        random_net3 = Compressor.compress(temporal_net, level=1, optimal=False, iterations=5)

        # ASSERT
        self.assertTrue(np.array_equal(compressed_net.layers[0].A, (some_array+other_array+third_array+other_array
                                                                    +some_array+some_array)/6))
        self.assertTrue(np.array_equal(random_net.layers[0].A, (some_array + other_array + third_array + other_array
                                                                    + some_array + some_array) / 6))
        self.assertTrue(np.array_equal(random_net2.layers[0].A, (some_array + other_array + third_array + other_array
                                                                + some_array + some_array) / 6))
        self.assertTrue(np.array_equal(random_net3.layers[0].A, (some_array + other_array + third_array + other_array
                                                                + some_array + some_array) / 6))
