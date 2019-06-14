import pyNN.brian as sim 
import matplotlib.pyplot as plt

sim.setup(timestep=0.1,min_delay=2.0) 
ifcell=sim.create(sim.IF_cond_exp,{'i_offset':0.11,'tau_refrac':3.0,'v_thresh':-51.0},n=1)

pulse = sim.DCSource(amplitude=0.5, start=20.0, stop=80.0)
pulse.inject_into(ifcell[3:7])

ifcell.record('v')
sim.run(100.0)

data = ifcell.get_data()
sim.end()

for segment in data.segments[0]:
    vm = segment.analogsignals[0]
    plt.plot(vm.times, vm)
#plt.legend(loc="upper left")
plt.xlabel("Time (%s)" % vm.times.units._dimensionality)
plt.ylabel("Membrane potential (%s)" % vm.units._dimensionality)

plt.show()