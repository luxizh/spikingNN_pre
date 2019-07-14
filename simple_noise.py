import pyNN.brian as sim
import numpy as np
import random
from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.utility.plotting as pplt
import matplotlib.pyplot as plt

trylabel=1

#def parameters
noiseWeight=5

__delay__ = 0.250 # (ms) 
tauPlus = 8 #20 # 15 # 16.8 from literature
tauMinus = 8 #20 # 30 # 33.7 from literature
aPlus = 0.500  #tum 0.016 #9 #3 #0.5 # 0.03 from literature
aMinus = 0.2500 #255 #tum 0.012 #2.55 #2.55 #05 #0.5 # 0.0255 (=0.03*0.85) from literature 
wMax = 5 #1 # G: 0.15 1
wMaxInit = 1.00#0.1#0.100
wMin = 0
nbIter = 5
testWeightFactor = 1#0.05177
x = 3 # no supervision for x first traj presentations
y = 0# for inside testing of traj to see if it has been learned /!\ stdp not disabled

input_len=30
input_class=3
input_size=input_len*input_class
output_size=3
inhibWeight = -5
stimWeight = 20

v_co=5

cell_params_lif = {'cm': 1,#70
                   'i_offset': 0.0,
                   'tau_m': 20.0,#20
                   'tau_refrac': 10.0,#2 more that t inhibit#10
                   'tau_syn_E': 2.0,#2
                   'tau_syn_I': 10.0,#5
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -55.0
                   }

def generate_data(label):
    spikesTrain=[]
    organisedData = {}
    for i in range(input_class):
        for j in range(input_len):
            neuid=(i,j)
            organisedData[neuid]=[]
    for i in range(input_len):
        neuid=(label,i)
        organisedData[neuid].append(i*v_co)
#        if neuid not in organisedData:
#            organisedData[neuid]=[i*v_co]
#        else:
#            organisedData[neuid].append(i*v_co)
    for i in range(input_class):
        for j in range(input_len):
            neuid=(i,j)
            organisedData[neuid].sort()
            spikesTrain.append(organisedData[neuid])
    runTime = int(max(max(spikesTrain)))
    sim.setup(timestep=1)
    
    noise=sim.Population(input_size,sim.SpikeSourcePoisson(),label='noise')

    
    noise.record(['spikes'])#noise
    
    sim.run(runTime)
    neonoise= noise.get_data(["spikes"])
    spikesnoise = neonoise.segments[0].spiketrains#noise
    sim.end()
    for i in range(input_size):
        for noisespike in spikesnoise[i]:
            spikesTrain[i].append(noisespike)
            spikesTrain[i].sort()
    return spikesTrain
'''    
    for neuronSpikes in organisedData.values():
        neuronSpikes.sort()
        spikesTrain.append(neuronSpikes)
'''
    

def train(label,untrained_weights):
    organisedStim = {}
    labelSpikes = []
    spikeTimes = generate_data(label)

    for i in range(output_size):
        labelSpikes.append([])
    labelSpikes[label] = [int(max(max(spikeTimes)))+1]

    
    if untrained_weights == None:
        untrained_weights = RandomDistribution('uniform', low=wMin, high=wMaxInit).next(input_size*output_size)
        #untrained_weights = RandomDistribution('normal_clipped', mu=0.1, sigma=0.05, low=wMin, high=wMaxInit).next(input_size*output_size)
        untrained_weights = np.around(untrained_weights, 3)
        #saveWeights(untrained_weights, 'untrained_weightssupmodel1traj')
        print ("init!")
    
    #untrained_weights=weight_list

    print "length untrained_weights :", len(untrained_weights)

    if len(untrained_weights)>input_size:
        training_weights = [[0 for j in range(output_size)] for i in range(input_size)] #np array? size 1024x25
        k=0
        #for i in untrained_weights:
        #    training_weights[i[0]][i[1]]=i[2]
        for i in range(input_size):
            for j in range(output_size):
                training_weights[i][j] = untrained_weights[k]
                k += 1
    else:
        training_weights = untrained_weights

    connections = []
    for n_pre in range(input_size): # len(untrained_weights) = input_size
        for n_post in range(output_size): # len(untrained_weight[0]) = output_size; 0 or any n_pre
            connections.append((n_pre, n_post, training_weights[n_pre][n_post], __delay__)) #index
    runTime = int(max(max(spikeTimes)))+100
    #####################
    sim.setup(timestep=1)
    #def populations
    #noise=sim.Population(input_size,sim.SpikeSourcePoisson(),label='noise')
    layer1=sim.Population(input_size,sim.SpikeSourceArray, {'spike_times': spikeTimes},label='inputspikes')
    layer2=sim.Population(output_size,sim.IF_curr_exp,cellparams=cell_params_lif,label='outputspikes')
    supsignal=sim.Population(output_size,sim.SpikeSourceArray, {'spike_times': labelSpikes},label='supersignal')

    #def learning rule
    stdp = sim.STDPMechanism(
                            #weight=untrained_weights,
                            #weight=0.02,  # this is the initial value of the weight
                            #delay="0.2 + 0.01*d",
                            timing_dependence=sim.SpikePairRule(tau_plus=tauPlus, tau_minus=tauMinus,A_plus=aPlus, A_minus=aMinus),
                            #weight_dependence=sim.MultiplicativeWeightDependence(w_min=wMin, w_max=wMax),
                            weight_dependence=sim.AdditiveWeightDependence(w_min=wMin, w_max=wMax),
                            dendritic_delay_fraction=0)
    #def projections

    #noise_proj=sim.Projection(noise,layer1,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=noiseWeight, delay=0))

    stdp_proj = sim.Projection(layer1, layer2, sim.FromListConnector(connections), synapse_type=stdp)
    inhibitory_connections = sim.Projection(layer2, layer2, sim.AllToAllConnector(allow_self_connections=False), 
                                            synapse_type=sim.StaticSynapse(weight=inhibWeight, delay=__delay__), 
                                            receptor_type='inhibitory')
    stim_proj = sim.Projection(supsignal, layer2, sim.OneToOneConnector(), 
                                synapse_type=sim.StaticSynapse(weight=stimWeight, delay=__delay__))

    #noise.record(['spikes'])#noise
    
    layer1.record(['spikes'])

    layer2.record(['v','spikes'])
    supsignal.record(['spikes'])
    sim.run(runTime)

    print("Weights:{}".format(stdp_proj.get('weight', 'list')))

    weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)]
    neo = layer2.get_data(["spikes", "v"])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]
    neostim = supsignal.get_data(["spikes"])
    print(label)
    spikestim = neostim.segments[0].spiketrains
    neoinput= layer1.get_data(["spikes"])
    spikesinput = neoinput.segments[0].spiketrains

    #neonoise= noise.get_data(["spikes"])
    #spikesnoise = neonoise.segments[0].spiketrains#noise

    plt.close('all')
    pplt.Figure(
    pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0,runTime)),
    #pplt.Panel(spikesnoise, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),#noise
    pplt.Panel(spikesinput, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
    pplt.Panel(spikestim, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
    pplt.Panel(spikes, xticks=True, xlabel="Time (ms)", yticks=True, markersize=2, xlim=(0,runTime)),
    title="Training"+str(label),
    annotations="Training"+str(label)
                ).save('plot_noise/'+str(trylabel)+str(label)+'_training.png')
    #plt.hist(weight_list[1], bins=100)
    #plt.show()
    plt.close('all')
    print(wMax)
    '''
    plt.hist([weight_list[1][0:input_size], weight_list[1][input_size:input_size*2], weight_list[1][input_size*2:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'], range=(0, wMax))
    plt.title('weight distribution')
    plt.xlabel('Weight value')
    plt.ylabel('Weight count')
    '''
    #plt.show()
    #plt.show()
                
    sim.end()
    for i in weight_list[0]:
        #training_weights[int(i[0])][int(i[1])]=float(i[2])
        weight_list[1][int(i[0])*output_size+int(i[1])]=i[2]
    return weight_list[1]


def test(spikeTimes, trained_weights,label):

    #spikeTimes = extractSpikes(sample)
    runTime = int(max(max(spikeTimes)))+100

    ##########################################

    sim.setup(timestep=1)
    
    pre_pop = sim.Population(input_size, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(output_size, sim.IF_curr_exp , cell_params_lif, label="post_pop")
    '''
    if len(untrained_weights)>input_size:
        training_weights = [[0 for j in range(output_size)] for i in range(input_size)] #np array? size 1024x25
        k=0
        for i in untrained_weights:
            training_weights[i[0]][i[1]]=i[2]
    '''
    if len(trained_weights) > input_size:
        weigths = [[0 for j in range(output_size)] for i in range(input_size)] #np array? size 1024x25
        k=0
        for i in range(input_size):
            for j in range(output_size):
                weigths[i][j] = trained_weights[k]
                k += 1
    else:
        weigths = trained_weights
    
    connections = []
    
    #k = 0
    for n_pre in range(input_size): # len(untrained_weights) = input_size
        for n_post in range(output_size): # len(untrained_weight[0]) = output_size; 0 or any n_pre
            #connections.append((n_pre, n_post, weigths[n_pre][n_post]*(wMax), __delay__))
            connections.append((n_pre, n_post, weigths[n_pre][n_post]*(wMax)/max(trained_weights), __delay__)) #
            #k += 1

    prepost_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections), synapse_type=sim.StaticSynapse(), receptor_type='excitatory') # no more learning !!
    #inhib_proj = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight, delay=__delay__), receptor_type='inhibitory')
    # no more lateral inhib

    post_pop.record(['v', 'spikes'])
    sim.run(runTime)

    neo = post_pop.get_data(['v', 'spikes'])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]
    f1=pplt.Figure(
    # plot voltage 
    pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0, runTime+100)),
    # raster plot
    pplt.Panel(spikes, xlabel="Time (ms)", xticks=True, yticks=True, markersize=2, xlim=(0, runTime+100)),
    title='Test with label ' + str(label),
    annotations='Test with label ' + str(label)
                )
    f1.save('plot_noise/'+str(trylabel)+str(label)+'_test.png')
    f1.fig.texts=[]
    print("Weights:{}".format(prepost_proj.get('weight', 'list')))

    weight_list = [prepost_proj.get('weight', 'list'), prepost_proj.get('weight', format='list', with_address=False)]
    #predict_label=
    sim.end()
    return spikes


#==============main================

weight_list=None
#weight_list=np.load("trainedweight"+str(trylabel)+".npy")
for i in range(5):
    #spikeTimes=generate_data(1)
    weight_list=train(label=1,untrained_weights=weight_list)
np.save("trainedweight"+str(trylabel)+".npy",weight_list)
'''
for i in range(1):
    label=random.randint(0,2)
    #label=0
    weight_list=train(label=label,untrained_weights=weight_list)
    #weight_list=weight_list[0]
    #print weight_list

#import pickle
np.save("trainedweight"+str(trylabel)+".npy",weight_list)


weight_list=np.load("trainedweight"+str(trylabel)+".npy")
print("training finish!")
print (max(weight_list))

#spikeTimes=generate_data(0)
#spikes=test(spikeTimes,weight_list,0)


for i in range(3):
    spikeTimes=generate_data(i)
    print i
    spikes=test(spikeTimes,weight_list,i)
    print(i,spikes)
'''



