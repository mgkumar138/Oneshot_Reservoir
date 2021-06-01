import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy.stats import ttest_1samp
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib
import os
import multiprocessing as mp

def savefigformats(imname, fig=None):
    data_path = 'C:\\Users\\Razer\\PycharmProjects\\multiple_pa_td\\Data&Fig\\'
    #data_path = 'D:/TD_HL/1pa/Data/'
    if fig is None:
        fig = [plt.gcf()]
    for f in fig:
        f.savefig('{}.png'.format(data_path + imname))
        f.savefig('{}.pdf'.format(data_path+imname))
        f.savefig('{}.svg'.format(data_path + imname))
        f.savefig('{}.eps'.format(data_path + imname))

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


def plot_dynamics(alldyn, N=1000,probelen=600, aidx=0):
    from sklearn.decomposition import PCA
    f, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax = ax.reshape(-1)
    lst = list(alldyn[aidx].keys())
    lst = [lst[i:i + 6] for i in range(0, len(lst), 6)]
    pca = PCA(2)
    for i, ls in enumerate(lst):
        sessact = np.zeros([6,probelen,N])

        for c in range(6):
            idx = ls[c]
            n = int(idx[-1]) - 1
            if n == 6:
                n = 0
            elif n == 7:
                n = 5
            sessact[n] = np.array(alldyn[aidx][idx])
        if i == 0:
            transact = pca.fit_transform(sessact.reshape(-1,N))
        else:
            transact = pca.transform(sessact.reshape(-1,N))
        transact = transact.reshape([6,probelen,2])

        if i == 4:
            cuecol = ['darkgreen', 'dimgray', 'gold', 'tab:orange', 'tab:blue', 'deeppink']
            cuenum = [7,2,3,4,5,8]
        elif i == 5:
            cuecol = ['green', 'gray', 'yellow', 'orange', 'blue', 'red']
            cuenum = [11,12,13,14,15,16]
        else:
            cuecol = ['tab:green', 'dimgray', 'gold', 'tab:orange', 'tab:blue', 'tab:red']
            cuenum = [1,2,3,4,5,6]

        ax[i].set_title(ls[0][:6])
        ax[i].set_xlabel('PCA 1')
        ax[i].set_ylabel('PCA 2')
        for c in range(6):
            ax[i].plot(transact[c,:,0],transact[c,:,1],color=cuecol[c])
            ax[i].scatter(transact[c,0,0],transact[c,0,1],marker='X',zorder=9,color=cuecol[c])
            ax[i].scatter(transact[c,-1, 0], transact[c,-1, 1], marker='s',zorder=9,color=cuecol[c])
            ax[i].text(transact[c,-1, 0], transact[c,-1, 1], 'C{}'.format(cuenum[c]),zorder=10)
    f.tight_layout()
    #f.savefig('./Fig/fig_pca_{}.png'.format(hp['exptname']))
    return pca


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
        'time': 600,  # seconds
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
        'hidact': 'relu',
        'sparsity': 0,
        'K': None,

        # actor parameters:
        'nact': 40,
        'actact': 'relu',
        'alat': True,
        'actns': 0.25,
        'qtau': 150,
        'maxspeed': 0.03,
        'actorw-': -1,
        'actorw+': 1,
        'actorpsi': 20,

        # critic parameters
        'crins': 0.0001,
        'tau': 150,
        'vscale': 1,
        'ncri': 1,
        'criact': 'relu',

        # reservoir parameters
        'resact': 'relu',
        'resrecact': 'tanh',
        'chaos': 1.5,
        'cp': [1, 0.1],
        'resns': 0.025,
        'recwinscl': 1,

        # learning parameters
        'lr': 0.0005,
        'taug': 2000,
        'eulerm': 1,  # euler approximation for TD error 1 - forward, 0 - backward

        # others
        'savevar': False,
        'savefig': True,
        'saveweight': False,
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


