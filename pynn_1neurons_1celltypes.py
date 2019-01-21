from pyNN.brian import *
from pyNN.random import (NumpyRNG,RandomDistribution)
#brian.list_standard_models()
refractory_period = RandomDistribution('uniform', [2.0, 3.0], rng=NumpyRNG(seed=4242))
ctx_parameters={'cm': 0.25, 'tau_m': 20.0, 'v_rest': -60, 'v_thresh': -50, \
'tau_refrac': refractory_period,'v_reset': -60, 'v_spike': -50.0, 'a': 1.0, \
'b': 0.005, 'tau_w': 600, 'delta_T': 2.5,'tau_syn_E': 5.0, 'e_rev_E': 0.0, \
'tau_syn_I': 10.0, 'e_rev_I': -80 }
tc_parameters = ctx_parameters.copy()
tc_parameters.update({'a': 20.0, 'b': 0.0})

thalamocortical_type = EIF_cond_exp_isfa_ista(**tc_parameters)
cortical_type = EIF_cond_exp_isfa_ista(**ctx_parameters)