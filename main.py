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
    socio_patterns_ex.run_one_layer()
    # sample_sir_sims.compare(1000, .05, 0) #determine a beta value per timestep