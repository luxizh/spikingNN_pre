from pyNN.brian import *
from pyNN.random import (NumpyRNG,RandomDistribution)

setup(timestep=0.1,min_delay=2.0) 
#always call setup at 

'''
cell types
'''
#brian.list_standard_models()
#refractory_period = RandomDistribution('uniform', [2.0, 3.0], rng=NumpyRNG(seed=4242))
#Brian does not support heterogenerous refractory periods with CustomRefractoriness
ctx_parameters={'cm': 0.25, 'tau_m': 20.0, 'v_rest': -60, 'v_thresh': -50, \
'tau_refrac': 3.0,'v_reset': -60, 'v_spike': -50.0, 'a': 1.0, \
'b': 0.005, 'tau_w': 600, 'delta_T': 2.5,'tau_syn_E': 5.0, 'e_rev_E': 0.0, \
'tau_syn_I': 10.0, 'e_rev_I': -80 }
tc_parameters = ctx_parameters.copy()
tc_parameters.update({'a': 20.0, 'b': 0.0})

thalamocortical_type = EIF_cond_exp_isfa_ista(**tc_parameters)
cortical_type = EIF_cond_exp_isfa_ista(**ctx_parameters)

# names and default
IF_cond_exp.get_parameter_names()
print(IF_cond_exp.default_parameters)

'''
populations
'''
#tec_cells=create(thalamocortical_type,n=100)
#tc_cells = Population(100, thalamocortical_type)
#ctx_cells = Population(500, cortical_type)

from pyNN.space import Grid2D, RandomStructure, Sphere
tc_cells = Population(100, thalamocortical_type,
                      structure=RandomStructure(boundary=Sphere(radius=200.0)),
                      initial_values={'v': -70.0},
                      label="Thalamocortical neurons")
from pyNN.random import RandomDistribution
v_init = RandomDistribution('uniform', (-70.0, -60.0))
ctx_cells = Population(500, cortical_type,
                       structure=Grid2D(dx=10.0, dy=10.0),
                       initial_values={'v': v_init},
                       label="Cortical neurons")

'''
view
'''
id = ctx_cells[47]           # the 48th neuron in a Population
view = ctx_cells[:80]        # the first eighty neurons
view = ctx_cells[::2]        # every second neuron
view = ctx_cells[45, 91, 7]  # a specific set of neurons
view = ctx_cells.sample(50, rng=NumpyRNG(seed=6538))  # select 50 neurons at random
print(view.parent.label)
print(view.mask)

'''
assemblies
'''
all_cells = tc_cells + ctx_cells
#all_cells = Assembly(tc_cells, ctx_cells)
cells_for_plotting = tc_cells[:10] + ctx_cells[:50]
#Individual populations within an Assembly may be accessed via their labels
print(all_cells.get_population("Thalamocortical neurons"))

for p in all_cells.populations:
   print("%-23s %4d %s" % (p.label, p.size, p.celltype.__class__.__name__))

'''
Inspecting and modifying parameter values and initial conditions
'''
#get()
print(ctx_cells.get('tau_m'))
print(all_cells[0:10].get('v_reset'))
print(ctx_cells.get('tau_refrac'))
print(ctx_cells.get(['tau_m', 'cm']))

#set()
ctx_cells.set(a=2.0, b=0.2)
ctx_cells.initialize(v=RandomDistribution('normal', (-65.0, 2.0)),w=0.0)
print(ctx_cells.celltype.default_initial_values)

'''
Injecting current into neurons
'''
#inject_into()
pulse = DCSource(amplitude=0.5, start=20.0, stop=80.0)
pulse.inject_into(tc_cells)

#inject()
import numpy
times = numpy.arange(0.0, 100.0, 1.0)
amplitudes = 0.1*numpy.sin(times*numpy.pi/100.0)
sine_wave = StepCurrentSource(times=times, amplitudes=amplitudes)
ctx_cells[80:90].inject(sine_wave)

'''
Recording variables and retrieving recorded data
'''
print(ctx_cells.celltype.recordable)
all_cells.record('spikes')
ctx_cells.sample(10).record(('v', 'w')) #, sampling_interval=0.2)

t = run(0.2)
data_block = all_cells.get_data()
'''
from neo.io import NeoHdf5IO
h5file = NeoHdf5IO("my_data.h5")
ctx_cells.write_data(h5file)
h5file.close()
#h5py is not available
'''

'''
Working with individual neurons
'''
print(tc_cells[47])
#parent--a reference to the parent population
a_cell = tc_cells[47]
print(a_cell.parent.label)
#Population.id_to_index(),
print(tc_cells.id_to_index(a_cell))
#set_parameters()
print(a_cell.tau_m)
a_cell.set_parameters(tau_m=10.0, cm=0.5)
print(a_cell.tau_m)
print(a_cell.cm)
