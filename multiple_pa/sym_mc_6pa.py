import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import find_cue, saveload, plot_dgr
import time as dt
from backend_scripts.maze_env import Maze
from backend_scripts.utils import get_default_hp
import multiprocessing as mp
from functools import partial
import tensorflow as tf
from backend_scripts.model import Foster_MC_Agent


def multiplepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totlat = np.zeros([btstp, (hp['trsess'] + hp['evsess'] * 3)])
    totdgr = np.zeros([btstp, 6])
    totpi = np.zeros_like(totdgr)
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(control_multiplepa_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], mvpath = x[b]
        #totlat[b], totdgr[b], totpi[b], diffw[b], mvpath, allw, alldyn, agent = control_multiplepa_expt(hp,b)

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(331)
    plt.title('Latency')
    plt.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')
    #plt.plot(np.mean(totlat,axis=0),linewidth=3)

    plot_dgr(totdgr, scl, 332, 6)

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

    return totlat, totdgr, totpi, mvpath


def control_multiplepa_expt(hp,b):

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)

    trsess = hp['trsess']
    evsess = int(trsess*.1)

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # store performance
    lat = np.zeros(trsess + evsess * 3)
    dgr = np.zeros(6)
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
    agent = Foster_MC_Agent(hp=hp,env=env)

    # Start Training
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_control_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_control_multiple_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_control_multiple_expt(b,'npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_control_multiple_expt(b, 'nm', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath


def run_control_multiple_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    if mtype=='train':
        mvpath = np.zeros((3,6,env.normax,2))
    else:
        mvpath = np.zeros((6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    # if mtype=='nm':
    #     agent.pc.flip_pcs()

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*6):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.agent_reset()
        cpc = agent.pcstate
        xy = agent.xystate

        if t%6==0:
            sesslat = []

        while not done:
            if env.rendercall:
                env.render()

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
            if t in env.nort or t in env.noct:
                plastic = False
            else:
                plastic = True

            # plasticity using Forward euler
            if hp['eulerm'] == 1:
                xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, xy=xy, cpc=cpc)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, = agent.act(state=state, cue_r_fb=cue)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            agent.store(xy=xy, cue_r_fb=state_cue, R=reward, done=done, plastic=plastic)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, xy=xy, cpc=cpc)

            if mtype == 'npa' and t not in env.nort and b == 0:
                if np.argmax(cue) == 6 or np.argmax(cue) == 7:
                    print(agent.goal.numpy()[0], tf.norm(agent.goal[0], ord=2).numpy())

            if done:
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
            print('T {} | C {} | S {} | D {:4.3f} | st {} | goal {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep),ds4r, env.startpos[0], agent.goal, np.round(dgr,1)))

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


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')

    hp['trsess'] = 20
    hp['evsess'] = 2
    hp['cuescl'] = 3
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 1
    hp['time'] = 3600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Model parameters '''
    hp['xylr'] = 0.00015  # 0.00015
    hp['eulerm'] = 1

    hp['mcbeta'] = 4  # 4
    hp['omitg'] = 0.025
    hp['storebeta'] = 1
    hp['recallbeta'] = 1

    hp['Rval'] = 4
    hp['taua'] = 250
    hp['cuescl'] = 3

    # First 30seconds: place cell activity & action update switched off, sensory cue given
    # After 30seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = False

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '6pa_sym_xy_{}t_{}om_{}taua_{}xy_b{}_{}'.format(
        hp['time'], hp['omitg'],
        hp['taua'],hp['xylr'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath = multiplepa_script(hp)
