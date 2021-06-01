import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import saveload
import time as dt
from backend_scripts.maze_env import Navex
from backend_scripts.utils import get_default_hp
import multiprocessing as mp
from functools import partial

from backend_scripts.model import Res_MC_Agent


def singlepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)
    trsess = hp['trsess']  # number of training trials per cue, location
    epochs = hp['epochs'] # number of epochs to go through all cue-location combination

    # store performance
    totlat = np.zeros([btstp, epochs, trsess])
    totdgr = np.zeros([btstp, epochs])
    totw = np.zeros([btstp, 49,2])
    totpath = np.zeros([btstp,epochs, 601,2 ])

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(main_singleloc_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpath[b], totw[b] = x[b]

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

    plt.subplot(335)
    plt.title('X')
    plt.imshow(totw[0,:49, 0].reshape(7, 7))
    plt.colorbar()

    plt.subplot(336)
    plt.title('Y')
    plt.imshow(totw[0,:49, 1].reshape(7, 7))
    plt.colorbar()

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
        saveload('save', '../Data/genvars_{}_b{}_{}'.format(exptname, btstp, dt.time()),   [totlat, totdgr, totpath, totw])

    return totlat, totdgr, mvpath, totw


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
    #tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = Res_MC_Agent(hp=hp,env=env)
    mdlw = None

    # start training
    for e in range(epochs):

        env.make(noreward=[hp['trsess']])
        rlocs = env.rlocs
        print('All Rlocs in Epoch {}: {}'.format(e, rlocs))
        lat[e], dgr[e], mdlw, totpath[e] = run_1rloc_expt(b, env, hp, agent, trsess, useweight=mdlw)

    # if hp['savevar']:
    #     saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()), [totpath,mdlw[3], lat, dgr])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, totpath, mdlw[3]


def run_1rloc_expt(b, env, hp, agent, sessions, useweight=None, noreward=None):
    lat = np.zeros(sessions)
    dgr = []

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.agent_reset()
        cpc = agent.pcstate
        xy = agent.xystate
        h = agent.mstate
        mstate = agent.mstate
        g = agent.gstate

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
                xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic,
                                 R=reward, xy=xy, cpc=cpc, h=h,g=g,mstate=mstate)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, h, mstate, g = agent.act(state=state, cue_r_fb=cue, mstate=mstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, R=reward, xy=xy, cpc=cpc, h=h, g=g,
                                 mstate=mstate)

            # if t in env.nort:
            #     save_rdyn(alldyn[5], mtype, t, env.startpos, env.cue, h)

            if done:
                if reward == 0 and plastic:
                    agent.model.layers[-2].set_weights([agent.model.layers[-2].get_weights()[0] * 0])
                break

        if env.probe:
            dgr = env.dgr
            mvpath = np.concatenate([np.array(env.tracks[:env.normax]),env.rloc[None,:]],axis=0)
        else:
            lat[t] = env.i

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('{} | D {:4.3f} | st {} | S {} | Dgr {} | Memory {} | g {}'.format(
                    t, ds4r,  env.startpos[0], env.i // (1000 // env.tstep), env.dgr, agent.goal, env.rloc))

    mdlw = agent.model.get_weights()

    return lat, dgr, mdlw, mvpath


if __name__ == '__main__':

    hp = get_default_hp(task='1pa',platform='server')

    hp['epochs'] = 9
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 30
    hp['trsess'] = 5
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savegenvar'] = True

    ''' Model parameters '''
    hp['xylr'] = 0.00015
    hp['eulerm'] = 1
    hp['stochlearn'] = False

    hp['mcbeta'] = 4  # 4
    hp['omitg'] = 0.15

    hp['lr'] = 0.0005
    hp['nrnn'] = 1024
    hp['ract'] = 'tanh'
    hp['recact'] = 'tanh'
    hp['chaos'] = 1.5
    hp['recwinscl'] = 1
    hp['cp'] = [1,0.1]
    hp['resns'] = 0.025

    hp['Rval'] = 4
    hp['taua'] = 250
    hp['cuescl'] = 3

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '1pa_res_xy_{}sl_{}smc_reset_{}om_{}bm_{}n_{}tau_{}taua_{}xy_{}lr_{}dt_b{}_{}'.format(
        hp['stochlearn'],    hp['usesmc'], hp['omitg'], hp['mcbeta'],
        hp['nrnn'], hp['tau'], hp['taua'],hp['xylr'],
        hp['lr'],  hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, mvpath, mdlw = singlepa_script(hp)