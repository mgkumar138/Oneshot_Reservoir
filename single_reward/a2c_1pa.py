# os.chdir('../')
#sys.path.insert(0, os.path.abspath('../../'))
import sys
sys.path.append("../")
# import importlib
# importlib.import_module("‪C:\\Users\\Razer\\PycharmProjects\\Res_1shot\\backend_script\\")
import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend_scripts.maze_env import Navex
from backend_scripts.utils import get_default_hp
import multiprocessing as mp
from functools import partial
import tensorflow as tf
from backend_scripts.model import BackpropAgent
from backend_scripts.utils import saveload


def singlepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)
    trsess = hp['trsess']  # number of training trials per cue, location
    epochs = hp['epochs'] # number of epochs to go through all cue-location combination

    # store performance
    totlat = np.zeros([btstp, epochs, trsess])
    totdgr = np.zeros([btstp, epochs])
    totpath = np.zeros([btstp,epochs, 601,2 ])

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(main_singleloc_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpath[b], mdlw = x[b]

    totlat[totlat == 0] = np.nan
    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=12)
    plt.subplot(331)
    plt.ylabel('Latency (s)')
    totlat *=hp['tstep']/1000
    plt.title('Latency per learning trial, change target every {} trials'.format(trsess))
    plt.errorbar(x=np.arange(epochs*trsess),y=np.mean(totlat,axis=0).reshape(-1),yerr=np.std(totlat,axis=0).reshape(-1)/btstp)
    plt.scatter(np.arange(epochs * trsess), np.mean(totlat,axis=0).reshape(-1),marker='*',color='k')
    for i in range(epochs):
        plt.axvline(i*trsess,color='r')

    plt.subplot(334)
    plt.title('Visit Ratio per Epoch')
    dgrmean = np.mean(totdgr, axis=0)
    dgrstd = np.std(totdgr, axis=0)
    plt.errorbar(x=np.arange(epochs), y=dgrmean, yerr=dgrstd/btstp)
    plt.plot(dgrmean, 'k', linewidth=3)

    mvpath = totpath[0]
    midx = np.linspace(0,epochs-1,3,dtype=int)
    for i in range(3):
        plt.subplot(3,3,i+7)
        plt.title('Probe trial {}'.format(midx[i]))
        plt.plot(mvpath[midx[i],:-1,0],mvpath[midx[i],:-1,1],'k')
        rloc = mvpath[midx[i],-1]
        plt.axis([-0.8,0.8,-0.8,0.8])
        plt.gca().set_aspect('equal', adjustable='box')
        circle = plt.Circle(rloc, 0.03, color='r')
        plt.gcf().gca().add_artist(circle)

    plt.tight_layout()
    plt.show()

    if hp['savefig']:
        plt.savefig('../Fig/fig_{}.png'.format(exptname))
    if hp['savegenvar']:
        saveload('save', '../Data/genvars_{}_b{}_{}'.format(exptname, btstp, dt.time()),   [totlat, totdgr, totpath])

    return totlat, totdgr, mvpath, mdlw


def main_singleloc_expt(hp,b):

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Navex(hp)

    trsess = hp['trsess']
    epochs = hp['epochs']

    # Create nonrewarded probe trial index

    # store performance
    lat = np.zeros([epochs, trsess])
    dgr = np.zeros([epochs])
    totpath = np.zeros([epochs, env.normax + 1, 2])

    # Start experiment
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = BackpropAgent(hp=hp,env=env)
    mdlw = None

    # start training
    for e in range(epochs):

        env.make(noreward=[hp['trsess']])
        rlocs = env.rlocs
        print('All Rlocs in Epoch {}: {}'.format(e, rlocs))
        lat[e], dgr[e], mdlw, totpath[e] = run_1rloc_expt(b, env, hp, agent, trsess, useweight=mdlw)

    # if hp['savevar']:
    #     saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()), [totpath, lat, dgr])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, totpath, mdlw


def run_1rloc_expt(b, env, hp, agent, sessions, useweight=None, noreward=None):
    lat = np.zeros(sessions)
    dgr = []

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        agent.memory.clear()

        while not done:
            if env.rendercall:
                env.render()

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            allstate, rfr, rho, value, actsel, action = agent.act(state=state, cue_r_fb=cue)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            if reward <= 0 and done:
                reward = -1  # if reward location not reached, penalise agent
            elif reward > 0:
                reward = hp['Rval']  # once reward location reached, terminate trial
                done = True

            agent.memory.store(state=allstate, action=actsel,reward=reward)

            # if t in env.nort:
            #     save_rdyn(alldyn[5], mtype, t, env.startpos, env.cue, h)

            if done:
                if t not in env.nort:
                    agent.replay()
                break

        if env.probe:
            dgr = env.dgr
            mvpath = np.concatenate([np.array(env.tracks[:env.normax]),env.rloc[None,:]],axis=0)
        else:
            lat[t] = env.i

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('{} | D {:4.3f} | st {} | S {} | Dgr {} | g {}'.format(
                    t, ds4r,  env.startpos[0], env.i // (1000 // env.tstep), env.dgr, env.rloc))

    mdlw = agent.model.get_weights()

    return lat, dgr, mdlw, mvpath


if __name__ == '__main__':

    hp = get_default_hp(task='1pa',platform='laptop')

    hp['btstp'] = 1
    hp['savefig'] = True
    hp['savegenvar'] = False

    ''' model parameters '''
    hp['nhid'] = 8192  # number of hidden units ~ Expansion ratio = nhid/67
    hp['hidact'] = 'phia'  # phiA, phiB, relu, etc
    hp['sparsity'] = 3  # Threshold
    hp['taug'] = 10000    # TD error time constant

    ''' Other Model parameters '''
    hp['lr'] = 0.000035
    hp['actalpha'] = 1/4  # to smoothen action taken by agent
    hp['maxspeed'] = 0.07  # step size per 100ms
    hp['entbeta'] = -0.001
    hp['valalpha'] = 0.5

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '1pa_hid_a2c_{}n_{}ra_{}lr_{}tg_b{}_{}'.format(
        hp['nhid'], hp['hidact'], hp['lr'], hp['taug'], hp['btstp'], dt.monotonic())

    totlat, totdgr, mvpath, mdlw = singlepa_script(hp)