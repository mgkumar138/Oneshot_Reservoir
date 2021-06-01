import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import saveload, get_default_hp, find_cue, plot_dgr
import time as dt
from backend_scripts.maze_env import MultiplePA_Maze
import multiprocessing as mp
from functools import partial
import tensorflow as tf


def run_a2cagent_multiplepa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    if mtype!='train':
        mvpath = np.zeros((12,env.normax+1,2))
    else:
        mvpath = None
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
            if mtype != 'train':
                mvpath[env.idx] = np.concatenate([np.array(env.tracks)[:env.normax], env.rloc[None, :]], axis=0)
            dgr.append(env.dgr)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | gl {} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), env.rloc, env.startpos[0],  np.round(dgr, 1)))

            # Session information
            if (t + 1) % env.totr == 0:
                print('################## {} Session {}/{}, PI {} ################'.format(
                    mtype, (t + 1) // env.totr, sessions, env.sessr))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        sesspi = np.array(dgr) > (100 / 6) * 2
        sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        sesspi = np.sum(np.array(dgr) > (100 / 12) * 2)
        dgr = np.mean(dgr)

    mdlw = agent.model.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return mvpath, mdlw, dgr, sesspi


def multiplepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totdgr = np.zeros([btstp, 4])
    totpi = np.zeros_like(totdgr)
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(setup_a2cagent_multiplepa_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totdgr[b], totpi[b], mvpath = x[b]
        #totlat[b], totdgr[b], totpi[b], diffw[b], mvpath, allw, alldyn, agent = control_multiplepa_expt(hp,b)

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)

    plot_dgr(totdgr, scl, 131, 4)
    plot_dgr(totpi, scl, 132, 4)

    import matplotlib.cm as cm
    col = cm.rainbow(np.linspace(0, 1, 12))
    plt.subplot(1, 3, 3)
    k = mvpath
    for pt in range(12):
        plt.plot(np.array(k[pt])[:-1, 0], np.array(k[pt])[:-1, 1], color=col[pt], alpha=0.5, linewidth=1, zorder=1)
        plt.scatter(np.array(k[pt])[0, 0], np.array(k[pt])[0, 1], color=col[pt], alpha=0.5, linewidth=1, zorder=1, marker='o',s=50)
        plt.scatter(np.array(k[pt])[-2, 0], np.array(k[pt])[-2, 1], color=col[pt], alpha=0.5, linewidth=1, zorder=2, marker='s', s=50)

        circle = plt.Circle(k[pt][-1], 0.03, color=col[pt], zorder=9)
        plt.gcf().gca().add_artist(circle)
        circle2 = plt.Circle(k[pt][-1], 0.03, color='k', fill=False, zorder=10)
        plt.gcf().gca().add_artist(circle2)

    plt.axis((-0.8, 0.8, -0.8, 0.8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('square')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.tight_layout()

    print(exptname)

    if hp['savefig']:
        plt.savefig('../Fig/fig_{}.png'.format(exptname))
    if hp['savegenvar']:
        saveload('save', '../Data/genvars_{}_b{}_{}'.format(exptname, btstp, dt.time()),
                 [ totdgr, totpi])

    return totdgr, totpi, mvpath


def setup_a2cagent_multiplepa_expt(hp,b):
    from backend_scripts.model import BackpropAgent
    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = MultiplePA_Maze(hp)

    trsess = hp['trsess']
    evsess = int(trsess*.1)
    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # store performance
    dgr = np.zeros(4)
    pi = np.zeros_like(dgr)

    # Start experiment
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    res = {}
    alldyn = [rdyn,qdyn,cdyn,tdyn, wtrk, res]
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = BackpropAgent(hp=hp,env=env)

    # Start Training
    _, trw, dgr[:3], pi[:3] = run_a2cagent_multiplepa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    mvpath,  npa1w, dgr[3], pi[3] = run_a2cagent_multiplepa_expt(b,'12npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [mvpath, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return dgr, pi, mvpath


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

    hp['exptname'] = '12pa_hid_a2c_{}n_{}ra_{}lr_{}tg_b{}_{}'.format(
        hp['nhid'], hp['hidact'], hp['lr'], hp['taug'], hp['btstp'], dt.monotonic())

    totdgr, totpi, mvpath = multiplepa_script(hp)
