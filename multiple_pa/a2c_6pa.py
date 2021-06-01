import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import saveload, get_default_hp, find_cue, plot_dgr
import time as dt
from backend_scripts.maze_env import Maze
import multiprocessing as mp
from functools import partial
import tensorflow as tf
import pandas as pd


def run_a2cagent_multiplepa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    if mtype == 'train':
        mvpath = np.zeros((3, 6, env.normax, 2))
    else:
        mvpath = np.zeros((6, env.normax, 2))
    lat = np.zeros(sessions)
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions * len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        agent.memory.clear()

        if t % len(env.rlocs) == 0:
            sesslat = []

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

            if done:
                if t not in env.nort:
                    agent.replay()
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            sesslat.append(np.nan)
            if mtype == 'train':
                sid = np.argmax(np.array(noreward) == (t // 6) + 1)
                mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
            else:
                mvpath[env.idx] = env.tracks[:env.normax]

            if mtype == 'npa':
                if (find_cue(env.cue) == np.array([7, 8])).any():
                    dgr.append(env.dgr)
            else:
                dgr.append(env.dgr)
        else:
            sesslat.append(env.i)

        if (t + 1) % 6 == 0:
            lat[((t + 1) // 6) - 1] = np.mean(sesslat)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | D {:4.3f} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep),ds4r, env.startpos[0],np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1], env.sessr))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        sesspi = np.array(dgr) > (100/6)
        sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        sesspi = np.sum(np.array(dgr) > (100/6))
        dgr = np.mean(dgr)

    mdlw = agent.model.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


def multiplepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totlat = np.zeros([btstp, (hp['trsess'] + hp['evsess'] * 3)])
    totdgr = np.zeros([btstp, 6])
    totpi = np.zeros_like(totdgr)
    diffw = np.zeros([btstp, 2, 3])  # bt, number of layers, modelcopy
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(setup_a2cagent_multiplepa_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], diffw[b], mvpath, allw, alldyn = x[b]
        #totlat[b], totdgr[b], totpi[b], diffw[b], mvpath, allw, alldyn, agent = control_multiplepa_expt(hp,b)

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(331)
    plt.title('Latency')
    plt.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')
    #plt.plot(np.mean(totlat,axis=0),linewidth=3)

    plot_dgr(totdgr, scl, 332, 6)

    plt.subplot(333)
    df = pd.DataFrame(np.mean(diffw[:,-2:], axis=0), columns=['OPA', 'NPA', 'NM'], index=['Critic', 'Actor'])
    ds = pd.DataFrame(np.std(diffw[:,-2:], axis=0), columns=['OPA', 'NPA', 'NM'], index=['Critic', 'Actor'])
    df.plot.bar(rot=0, ax=plt.gca(), yerr=ds)
    plt.title(np.round(np.mean(totpi,axis=0),1))

    env = Maze(hp)

    col = ['b', 'g', 'r', 'y', 'm', 'k']
    for i,m in enumerate(['train','train','train','opa', 'npa', 'nm']):

        plt.subplot(3, 3, i+4)
        plt.title('{}'.format(m))
        env.make(m)
        k = mvpath[i]
        for pt in range(len(mvpath[2])):
            plt.plot(np.array(k[pt])[:, 0], np.array(k[pt])[:, 1], col[pt], alpha=0.5)
            circle = plt.Circle(env.rlocs[pt], env.rrad, color=col[pt])
            plt.gcf().gca().add_artist(circle)
        plt.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('square')

    print(exptname)

    plt.tight_layout()

    if hp['savefig']:
        plt.savefig('../Fig/fig_{}.png'.format(exptname))
    if hp['savegenvar']:
        saveload('save', '../Data/genvars_{}_b{}_{}'.format(exptname, btstp, dt.time()),
                 [totlat, totdgr, totpi])

    return totlat, totdgr, totpi, diffw, mvpath, allw, alldyn


def setup_a2cagent_multiplepa_expt(hp,b):
    from backend_scripts.model import BackpropAgent
    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)

    trsess = hp['trsess']
    evsess = hp['evsess']

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # store performance
    lat = np.zeros(trsess + evsess * 3)
    dgr = np.zeros(6)
    diffw = np.zeros([2, 3])
    pi = np.zeros_like(dgr)

    # Start experiment
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    res = {}
    alldyn = [rdyn,qdyn,cdyn,tdyn, wtrk, res]
    mvpath = np.zeros([6, 6, env.normax, 2])
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = BackpropAgent(hp=hp,env=env)

    # Start Training
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_a2cagent_multiplepa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_a2cagent_multiplepa_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_a2cagent_multiplepa_expt(b,'npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_a2cagent_multiplepa_expt(b, 'nm', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    for i, k in enumerate([opaw, npaw, nmw]):
        for j in np.arange(-2,0):
            diffw[j, i] = np.sum(abs(k[j] - trw[j])) / np.size(k[j])

    allw = [trw, opaw, npaw, nmw]

    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi, diffw])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, diffw, mvpath, allw, alldyn


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')

    hp['btstp'] = 1
    hp['savefig'] = True
    hp['savevar'] = False
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

    hp['exptname'] = '6pa_hid_a2c_{}n_{}ra_{}lr_{}tg_b{}_{}'.format(
        hp['nhid'], hp['hidact'], hp['lr'], hp['taug'], hp['btstp'], dt.monotonic())

    totlat, totdgr, totpi, diffw, mvpath, allw, alldyn = multiplepa_script(hp)
