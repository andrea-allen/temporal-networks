import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import itertools
import scipy
from matplotlib import rc
import sample_sir_sims
import socio_patterns_ex

if __name__ == '__main__':
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    print('Temporal Networks - Andrea Allen, 2020')
    # sample_sir_sims.identity_experiment()
    # This one works:
    socio_patterns_ex.error_over_time(10, '../tij_SFHH.dat_')
    # socio_patterns_ex.error_over_time(30, 'tij_InVS.dat')
    # socio_patterns_ex.error_over_time_hospital(30, 'detailed_list_of_contacts_Hospital.dat_')
    # THIS one works:
    socio_patterns_ex.random_graph_error(10, None)
    # socio_patterns_ex.other_random_graph_error(30, None)
    plt.show()
    # sample_sir_sims.compare(1000, .05, 0) #determine a beta value per timestep