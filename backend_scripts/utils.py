import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy.stats import ttest_1samp
import pickle
import matplotlib
import os


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
    plt.title('Time Spent at Correct Location (%)')
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
        'tstep': 100,  # ms
        'time': 3600,  # seconds
        'render': False,
        'epochs': epochs,
        'trsess': trsess,
        'evsess': evsess,
        'platform': platform,
        'taua': 250,
        'taub': 120,
        'npa': 6,
        'Rval': 4,

        # input parameters
        'npc': 7,
        'cuescl': 3,
        'cuesize': 18,
        'workmem': False,

        # hidden parameters
        'nhid': 8192,
        'hidact': 'phia',
        'sparsity': 3,

        # actor parameters:
        'nact': 40,
        'actact': 'relu',
        'alat': True,
        'actns': 0.25,
        'maxspeed': 0.03,
        'actorw-': -1,
        'actorw+': 1,
        'actorpsi': 20,
        'tau': 150,
        'ncri': 1,

        # reservoir parameters
        'ract': 'tanh',
        'recact': 'tanh',
        'chaos': 1.5,
        'cp': [1, 0.1],
        'resns': 0.025,
        'recwinscl': 1,
        'nrnn': 1024,

        # learning parameters
        'taug': 10000,
        'eulerm': 1,  # euler approximation for TD error 1 - forward, 0 - backward

        # motor controller parameters
        'omitg':0.025,
        'mcbeta': 4,
        'xylr': 0.00015,
        'recallbeta': 1,

        # others
        'savevar': False,
        'savefig': True,
        'savegenvar': False,
        'modeltype': None,

    }

    if hp['platform'] == 'laptop':
        matplotlib.use('Qt5Agg')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        hp['cpucount'] = 1
    elif hp['platform'] == 'server':
        matplotlib.use('Qt5Agg')
        hp['cpucount'] = 5 #mp.cpu_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif hp['platform'] == 'gpu':
        #print(tf.config.list_physical_devices('GPU'))
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        ngpu = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        #matplotlib.use('Agg')
        hp['cpucount'] = ngpu
    return hp


