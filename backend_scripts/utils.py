import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy.stats import ttest_1samp
import pickle
import matplotlib
import os
import multiprocessing as mp


def saveload(opt, name, variblelist):
    name = name + '.pickle'
    if opt == 'save':
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var


def loaddata(name):
    with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
        var = pickle.load(f)
        print('Data Loaded: {}'.format(name))
        f.close()
        return var


def savedata(name, variblelist):
    name = name + '.pickle'
    with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(variblelist, f)
        print('Data Saved')
        f.close()

def plot_dgr(dgr,scl, pltidx, patype):
    plt.subplot(pltidx)
    dgidx = [2 * scl - 1, 9 * scl - 1, 16 * scl - 1, 22 * scl - 1, 24 * scl - 1, 26 * scl - 1]
    mdg = np.mean(dgr, axis=0)
    sdg = np.std(dgr, axis=0)
    index = []
    for i in range(patype):
        index.append('S {}'.format(dgidx[i]+1))
    df = pd.DataFrame({'Dgr':mdg},index=index)
    df2 = pd.DataFrame({'Dgr':sdg},index=index)
    ax = df.plot.bar(rot=0, ax=plt.gca(), yerr=df2 / dgr.shape[0], color='k')
    plt.axhline(y=mdg[0], color='g', linestyle='--')
    if patype == 1:
        chnc = 100/49
    else:
        chnc = 100/6
    plt.axhline(y=chnc, color='r', linestyle='--')
    plt.title('Visit Ratio (%)')
    tv,pv = ttest_1samp(dgr, chnc, axis=0)
    for i,p in enumerate(ax.patches):
        if pv[i] < 0.001:
            ax.text(p.get_x(),  p.get_height()*1.05, '***', size=15)
        elif pv[i] < 0.01:
            ax.text(p.get_x(),  p.get_height()*1.05, '**', size=15)
        elif pv[i] < 0.05:
            ax.text(p.get_x(),  p.get_height()*1.05, '*', size=15)


def find_cue(c):
    c = c.reshape(len(c),-1)[:,0]
    if np.sum(c) > 0:
        cue = np.argmax(c)+1
    else: cue = 0
    return cue

def save_rdyn(rdyn, mtype,t,startpos,cue, rfr):
    rfr = tf.cast(rfr,dtype=tf.float32)
    if '{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue)) in rdyn:
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue))].append(rfr.numpy()[0])
    else:
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue))] = []
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue))].append(rfr.numpy()[0])


def get_default_hp(task, platform='laptop'):
    if task == '1pa':
        epochs = 9
        trsess = 5
        evsess = None
    else :
        epochs = None
        trsess = 20
        evsess = 2

    hp = {
        # Environment parameters
        'mazesize': 1.6,  # meters
        'task': task,  # type of task determiens number of training, evaluation sessions
        'tstep': 100,  # ms each step taken is every 100 ms
        'time': 3600,  # seconds total trial time
        'probetime': 60,  # seconds total probe time
        'render': False,  # dont plot real time movement
        'epochs': epochs,  # only for single displaced location task
        'trsess': trsess,  # number of training trials
        'evsess': evsess,  # number of evaluation trials
        'platform': platform,  # laptop, gpu or server with multiple processors
        'taua': 250,  # reward decay time
        'taub': 120,  # reward rise time
        'Rval': 4,  # magnitude of reward disbursed

        # input parameters
        'npc': 7,  # number of place cells across vertical and horizontal axes
        'cuescl': 3,  # gain of cue
        'cuesize': 18,  # size of cue

        # hidden parameters
        'nhid': 8192,  # number of hidden units for A2C
        'hidact': 'phia',  # activation function for A2C hidden layer
        'sparsity': 3,  # threshold for ReLU activation function

        # actor parameters:
        'nact': 40,  # number of actor units
        'actact': 'relu',  # activation of actor units
        'alat': True,  # use lateral connectivity for ring attractor dynamics
        'actns': 0.25,  # exploratory noise for actor
        'maxspeed': 0.03,  # a0 scaling factor for veloctiy
        'actorw-': -1,  # inhibitory scale for lateral connectivity
        'actorw+': 1,  # excitatory scale for lateral connectivity
        'actorpsi': 20,  # lateral connectivity spread
        'tau': 150,  # membrane time constant for all cells
        'ncri': 1,  # number of critic

        # reservoir parameters
        'ract': 'tanh',  # reservoir activation function
        'recact': 'tanh',  # reservoir recurrent activiation function
        'chaos': 1.5,  # chaos gain lambda
        'cp': [1, 0.1],  # connection probability - input to reservoir & within reservoir
        'resns': 0.025,  # white noise in reservoir
        'recwinscl': 1,  # reservoir input weight scale
        'nrnn': 1024,  # number of rnn units

        # learning parameters
        'taug': 10000,  # reward discount factor gamme for A2C
        'eulerm': 1,  # euler approximation for TD error 1 - forward, 0 - backward

        # motor controller parameters
        'omitg':0.025,  # threshold to omit L2(goal) to suppress motor controller
        'mcbeta': 4,  # motor controller beta
        'xylr': 0.00015,  # learning rate of self position coordinate network
        'recallbeta': 1,  # recall beta within symbolic memory

        # others
        'savevar': False,  # individual run variables
        'savefig': True,  # save output figure
        'savegenvar': False,  # save compiled variables latency, visit ratio
        'modeltype': None,

    }

    if hp['platform'] == 'laptop':
        matplotlib.use('Qt5Agg')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        hp['cpucount'] = 1
    elif hp['platform'] == 'server':
        matplotlib.use('Qt5Agg')
        hp['cpucount'] = mp.cpu_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif hp['platform'] == 'gpu':
        #print(tf.config.list_physical_devices('GPU'))
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        ngpu = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        #matplotlib.use('Agg')
        hp['cpucount'] = ngpu
    return hp


