import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import npapa_script

'''
def multiplepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    #totlat = np.zeros([btstp, (hp['trsess'] + hp['evsess'] * 3)])
    totdgr = np.zeros([btstp, 4])
    totpi = np.zeros_like(totdgr)
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(control_multiplepa_expt, hp), np.arange(btstp))

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


def control_multiplepa_expt(hp,b):

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
    _, trw, dgr[:3], pi[:3] = run_control_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    #mvpath,  npa1w, dgr[3], pi[3] = run_control_multiple_expt(b, 'npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    mvpath,  npa1w, dgr[3], pi[3] = run_control_multiple_expt(b,'12npa', env, hp, agent, alldyn, evsess, trw, noreward=[nonrp[0]])

    # Summarise weight change of layers
    if hp['savevar']:
        saveload('save', '../Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, dgr, pi])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return dgr, pi, mvpath


def run_control_multiple_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    if mtype!='train':
        mvpath = np.zeros((12,env.normax+1,2))
    else:
        mvpath = None
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    # if mtype=='nm':
    #     agent.pc.flip_pcs()

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
        #trackr = []

        # plt.figure()
        # plt.title('chaos:{}'.format(hp['chaos'])+' ns:0.025')
        # plt.imshow(np.vstack(trackr).T, aspect='auto')
        # plt.colorbar()
        # plt.show()

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
            #trackr.append(h)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(qhat)

            # Use action on environment, ds4r: distance from reward
            state, _, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                xy = agent.learn(s1=state, cue_r1_fb=cue, plasticity=plastic, R=reward, xy=xy, cpc=cpc, h=h, g=g,
                                 mstate=mstate)

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
'''

if __name__ == '__main__':

    hp = get_default_hp(task='12pa',platform='server')
    hp['agenttype'] = 'res'
    hp['btstp'] = 30
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    hp['stochlearn'] = False
    hp['lr'] = 0.0005
    hp['nrnn'] = 1024

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_xy_{}sl_{}t_{}om_{}ch_{}n_{}tau_{}taua_{}xy_{}lr_b{}_{}'.format(
        hp['task'],hp['agenttype'], hp['stochlearn'],hp['time'],hp['omitg'], hp['chaos'],
        hp['nrnn'], hp['tau'], hp['taua'],hp['xylr'],  hp['lr'],  hp['btstp'],dt.monotonic())

    totdgr, totpi, mvpath = npapa_script(hp)

