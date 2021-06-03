import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import time as dt
from backend_scripts.maze_env import Navex, Maze, MultiplePA_Maze
import multiprocessing as mp
from functools import partial
import tensorflow as tf
from backend_scripts.model import BackpropAgent, Res_MC_Agent, Foster_MC_Agent
from backend_scripts.utils import saveload, find_cue, save_rdyn, plot_dgr
from copy import deepcopy


''' 12NPA scripts '''

def npapa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totdgr = np.zeros([btstp, 4])
    totpi = np.zeros_like(totdgr)
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    if hp['agenttype'] == 'a2c':
        x = pool.map(partial(a2c_12pa_expt, hp), np.arange(btstp))
    elif hp['agenttype'] == 'sym':
        x = pool.map(partial(sym_12pa_expt, hp), np.arange(btstp))
    else:
        x = pool.map(partial(res_12pa_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totdgr[b], totpi[b], mvpath, agentmemory, allw = x[b]
        #totlat[b], totdgr[b], totpi[b], diffw[b], mvpath, allw, alldyn, agent = control_multiplepa_expt(hp,b)

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)

    plot_dgr(totdgr, scl, 231, 4)
    plot_dgr(totpi, scl, 232, 4)

    if hp['agenttype'] != 'a2c':
        if agentmemory:
            plt.subplot(234)
            plt.imshow(agentmemory[1],aspect='auto')
            plt.title('Memory')
            plt.colorbar()

        if allw:
            plt.subplot(235)
            plt.imshow(allw[1][:,0].reshape(7,7),aspect='auto')
            plt.title('X')
            plt.colorbar()
            plt.subplot(236)
            plt.imshow(allw[1][:,1].reshape(7,7),aspect='auto')
            plt.title('Y')
            plt.colorbar()

    import matplotlib.cm as cm
    col = cm.rainbow(np.linspace(0, 1, 12))
    plt.subplot(2, 3, 3)
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


def res_12pa_expt(hp,b):

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
    agent = Res_MC_Agent(hp=hp,env=env)

    # Start Training
    _, trw, dgr[:3], pi[:3] = run_res_12pa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    mvpath,  npa1w, dgr[3], pi[3] = run_res_12pa_expt(b,'12npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return dgr, pi, mvpath, _, [trw[-1], npa1w[-1]]

def run_res_12pa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    if mtype!='train':
        mvpath = np.zeros((12,env.normax+1,2))
    else:
        mvpath = None
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)
        # if b == 0:
        #     print(agent.model.get_weights())

    for t in range(sessions*env.totr):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.agent_reset()
        cpc = agent.pcstate
        xy = agent.xystate
        h = tf.zeros_like(agent.mstate)
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

            xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic,
                             R=reward, xy=xy, cpc=cpc, h=h,g=g,mstate=mstate)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, h, mstate, g = agent.act(state=state, cue_r_fb=cue, mstate=mstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            if done:
                if reward == 0 and plastic:
                    agent.model.layers[-2].set_weights([agent.model.layers[-2].get_weights()[0] * 0])
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            if mtype != 'train':
                mvpath[env.idx] = np.concatenate([np.array(env.tracks)[:env.normax], env.rloc[None,:]],axis=0)
            dgr.append(env.dgr)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | gl {} | st {} | goal {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep),env.rloc, env.startpos[0], agent.goal, np.round(dgr,1)))

            # Session information
            if (t + 1) % env.totr == 0:
                print('################## {} Session {}/{}, PI {} ################'.format(
                    mtype, (t + 1) // env.totr, sessions, env.sessr))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        sesspi = np.array(dgr) > (100/6)*2
        sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        sesspi = np.sum(np.array(dgr) > (100/12)*2)
        dgr = np.mean(dgr)

    mdlw = agent.model.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return mvpath, mdlw, dgr, sesspi


def a2c_12pa_expt(hp,b):
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
    _, trw, dgr[:3], pi[:3] = run_a2c_12pa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    mvpath,  npa1w, dgr[3], pi[3] = run_a2c_12pa_expt(b,'12npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [mvpath, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return dgr, pi, mvpath, _, [trw, npa1w]

def run_a2c_12pa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
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


def sym_12pa_expt(hp,b):

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
    agent = Foster_MC_Agent(hp=hp,env=env)

    # Start Training
    _, trw, dgr[:3], pi[:3], memt = run_sym_12pa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    mvpath,  npa1w, dgr[3], pi[3], mem1 = run_sym_12pa_expt(b,'12npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [mvpath, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return dgr, pi, mvpath, [memt, mem1], [trw[0], npa1w[0]]


def run_sym_12pa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    if mtype!='train':
        mvpath = np.zeros((12,env.normax+1,2))
    else:
        mvpath = None
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)
        agent.memory = np.zeros_like(agent.memory)
        if b == 0:
            print(np.max(agent.model.get_weights()), np.min(agent.model.get_weights()))
            #print(agent.memory)
            #print(agent.model.get_weights())

    for t in range(sessions*env.totr):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.agent_reset()
        cpc = agent.pcstate
        xy = agent.xystate

        while not done:
            if env.rendercall:
                env.render()

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
            if t in env.nort or t in env.noct:
                plastic = False
            else:
                plastic = True

            xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, xy=xy, cpc=cpc)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, = agent.act(state=state, cue_r_fb=cue)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            agent.store(xy=xy, cue_r_fb=state_cue, R=reward, done=done, plastic=plastic)

            if done:
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            if mtype != 'train':
                mvpath[env.idx] = np.concatenate([np.array(env.tracks)[:env.normax], env.rloc[None,:]],axis=0)
            dgr.append(env.dgr)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | gl {} | st {} | goal {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep),env.rloc, env.startpos[0], agent.goal, np.round(dgr,1)))

            # Session information
            if (t + 1) % env.totr == 0:
                print('################## {} Session {}/{}, PI {} ################'.format(
                    mtype, (t + 1) // env.totr, sessions, env.sessr))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        sesspi = np.array(dgr) > (100/6)*2
        sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        sesspi = np.sum(np.array(dgr) > (100/12)*2)
        dgr = np.mean(dgr)

    mdlw = agent.model.get_weights()
    print(np.max(agent.model.get_weights()), np.min(agent.model.get_weights()))

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return mvpath, mdlw, dgr, sesspi, deepcopy(agent.memory)



''' multiple pa scripts '''


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

    if hp['agenttype'] == 'a2c':
        x = pool.map(partial(a2c_multiplepa_expt, hp), np.arange(btstp))
    elif hp['agenttype'] == 'sym':
        x = pool.map(partial(sym_multiplepa_expt, hp), np.arange(btstp))
    else:
        x = pool.map(partial(res_multiplepa_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], mvpath = x[b]

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(331)
    plt.title('Latency')
    plt.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')

    plot_dgr(totdgr, scl, 332, 6)

    env = Maze(hp)

    col = ['b', 'g', 'r', 'y', 'm', 'k']
    mlegend = ['PS1','PS2','PS3','OPA','2NPA','6NPA']
    for i,m in enumerate(['train','train','train','opa', '2npa', '6npa']):

        plt.subplot(3, 3, i+4)
        plt.title(mlegend[i])
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


def res_multiplepa_expt(hp,b):

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
    gdyn = {}

    alldyn = [rdyn,gdyn]
    mvpath = np.zeros([6, 6, env.normax, 2])
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = Res_MC_Agent(hp=hp,env=env)

    # Start Training
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_res_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_res_multiple_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_res_multiple_expt(b,'2npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_res_multiple_expt(b, '6npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, gdyn, mvpath, lat, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath


def run_res_multiple_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    if mtype=='train':
        mvpath = np.zeros((3,6,env.normax,2))
    else:
        mvpath = np.zeros((6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*6):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.agent_reset()
        cpc = agent.pcstate
        xy = agent.xystate
        h = tf.zeros_like(agent.mstate)
        mstate = agent.mstate
        g = agent.gstate

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

            xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic,
                             R=reward, xy=xy, cpc=cpc, h=h,g=g,mstate=mstate)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, h, mstate, g = agent.act(state=state, cue_r_fb=cue, mstate=mstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            if t in env.nort:
                save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, h)
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, g)

            if done:
                if reward == 0 and plastic:
                    agent.model.layers[-2].set_weights([agent.model.layers[-2].get_weights()[0] * 0])
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            sesslat.append(np.nan)
            if mtype == 'train':
                sid = np.argmax(np.array(noreward) == (t // 6) + 1)
                mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
            else:
                mvpath[env.idx] = env.tracks[:env.normax]

            if mtype == '2npa':
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


def a2c_multiplepa_expt(hp,b):
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
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_a2c_multiplepa_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_a2c_multiplepa_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_a2c_multiplepa_expt(b,'2npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_a2c_multiplepa_expt(b, '6npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath

def run_a2c_multiplepa_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
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

            if mtype == '2npa':
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


def sym_multiplepa_expt(hp,b):

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
    lat[:trsess], mvpath[:3], trw, dgr[:3], pi[:3] = run_sym_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    # Start Evaluation
    lat[trsess:trsess + evsess], mvpath[3], opaw, dgr[3], pi[3] = run_sym_multiple_expt(b,'opa', env, hp,agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess:trsess + evsess * 2], mvpath[4],  npaw, dgr[4], pi[4] = run_sym_multiple_expt(b,'2npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])
    lat[trsess + evsess * 2:], mvpath[5], nmw, dgr[5], pi[5] = run_sym_multiple_expt(b, '6npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath


def run_sym_multiple_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    if mtype=='train':
        mvpath = np.zeros((3,6,env.normax,2))
    else:
        mvpath = np.zeros((6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

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

            xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, xy=xy, cpc=cpc)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, = agent.act(state=state, cue_r_fb=cue)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            agent.store(xy=xy, cue_r_fb=state_cue, R=reward, done=done, plastic=plastic)

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

            if mtype == '2npa':
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



''' single PA scripts '''

def singlepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)
    trsess = hp['trsess']  # number of training trials per cue, location
    epochs = hp['epochs'] # number of epochs to go through all cue-location combination

    # store performance
    totlat = np.zeros([btstp, epochs, trsess])
    totdgr = np.zeros([btstp, epochs])
    totpath = np.zeros([btstp,epochs, 601,2])

    pool = mp.Pool(processes=hp['cpucount'])

    if hp['agenttype'] == 'a2c':
        x = pool.map(partial(a2c_singleloc_expt, hp), np.arange(btstp))
    elif hp['agenttype'] == 'sym':
        x = pool.map(partial(sym_singleloc_expt, hp), np.arange(btstp))
    else:
        x = pool.map(partial(res_singleloc_expt, hp), np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpath[b], mdlw = x[b]

    totlat[totlat == 0] = np.nan
    plt.figure(figsize=(15, 6))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=12)
    plt.subplot(241)
    plt.ylabel('Latency (s)')
    totlat *=hp['tstep']/1000
    plt.title('Latency per learning trial, change target every {} trials'.format(trsess))
    plt.errorbar(x=np.arange(epochs*trsess),y=np.mean(totlat,axis=0).reshape(-1),
                 yerr=np.std(totlat,axis=0).reshape(-1)/btstp, marker='*', color='k')

    plt.subplot(242)
    plt.title('Visit Ratio per Epoch')
    dgrmean = np.mean(totdgr, axis=0)
    dgrstd = np.std(totdgr, axis=0)
    plt.errorbar(x=np.arange(epochs), y=dgrmean, yerr=dgrstd/btstp)
    plt.plot(dgrmean, 'k', linewidth=3)

    if hp['agenttype'] != 'a2c':
        plt.subplot(243)
        plt.title('X')
        im = plt.imshow(mdlw[:49, 0].reshape(7, 7))
        plt.colorbar(im,fraction=0.046, pad=0.04)

        plt.subplot(244)
        plt.title('Y')
        im = plt.imshow(mdlw[:49, 1].reshape(7, 7))
        plt.colorbar(im,fraction=0.046, pad=0.04)

    mvpath = totpath[0]
    midx = np.linspace(0,epochs-1,3,dtype=int)
    for i in range(3):
        plt.subplot(2,4,i+5)
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


def a2c_singleloc_expt(hp,b):

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
        lat[e], dgr[e], mdlw, totpath[e] = run_a2c_1rloc_expt(b, env, hp, agent, trsess, useweight=mdlw)

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, totpath, mdlw


def run_a2c_1rloc_expt(b, env, hp, agent, sessions, useweight=None, noreward=None):
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


def res_singleloc_expt(hp,b):

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
        lat[e], dgr[e], mdlw, totpath[e] = run_res_1rloc_expt(b, env, hp, agent, trsess, useweight=mdlw)

    # if hp['savevar']:
    #     saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()), [totpath,mdlw[3], lat, dgr])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, totpath, mdlw[3]


def run_res_1rloc_expt(b, env, hp, agent, sessions, useweight=None, noreward=None):
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

            xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic,
                             R=reward, xy=xy, cpc=cpc, h=h,g=g,mstate=mstate)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, h, mstate, g = agent.act(state=state, cue_r_fb=cue, mstate=mstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

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


def sym_singleloc_expt(hp,b):

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
    agent = Foster_MC_Agent(hp=hp,env=env)
    mdlw = None

    # start training
    for e in range(epochs):

        env.make(noreward=[hp['trsess']])
        rlocs = env.rlocs
        print('All Rlocs in Epoch {}: {}'.format(e, rlocs))
        lat[e], dgr[e], mdlw, totpath[e] = run_sym_1rloc_expt(b, env, hp, agent, trsess, useweight=mdlw)

    # if hp['savevar']:
    #     saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()), [totpath, lat, dgr])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, totpath, mdlw[-1]


def run_sym_1rloc_expt(b, env, hp, agent, sessions, useweight=None, noreward=None):
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

        while not done:
            if env.rendercall:
                env.render()

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
            if t in env.nort or t in env.noct:
                plastic = False
            else:
                plastic = True

            xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, xy=xy, cpc=cpc)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            state_cue, cpc, qhat, _, = agent.act(state=state, cue_r_fb=cue)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            agent.store(xy=xy, cue_r_fb=state_cue, R=reward, done=done, plastic=plastic)

            # if t in env.nort:
            #     save_rdyn(alldyn[5], mtype, t, env.startpos, env.cue, h)

            if done:
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