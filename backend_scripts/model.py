import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Res_MC_Agent:
    def __init__(self, hp, env):
        ''' environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.lr = hp['lr']
        self.mcbeta = hp['mcbeta']
        self.omitg = hp['omitg']
        self.alpha = hp['tstep']/hp['tau']

        self.xylr = hp['xylr']
        self.stochlearn = hp['stochlearn']
        self.npc = hp['npc']
        self.nact = hp['nact']

        self.xystate = tf.zeros([1, 2])
        self.pcstate = tf.zeros([1, hp['npc'] * hp['npc']])

        ''' memory '''
        self.resns = hp['resns']
        self.nrnn = hp['nrnn']
        self.phat = 0
        self.gstate = tf.zeros([1,2])
        self.mstate = tf.random.normal(shape=[1, hp['nrnn']], mean=0, stddev=np.sqrt(1 / self.alpha) * self.resns)
        self.pastpre = 0
        self.verr = []

        ''' Setup model: Place cell --> Action cells '''

        self.pc = place_cells(hp)
        self.model = Res_Model(hp)
        self.ac = action_cells(hp)

    def act(self, state, cue_r_fb, mstate):
        cpc = tf.cast(self.pc.sense(state),dtype=tf.float32)

        state_cue_fb = tf.cast(tf.concat([cpc, cue_r_fb],axis=0)[None, :],dtype=tf.float32)  # combine all inputs

        h, x, g, xy = self.model(state_cue_fb, mstate)
        self.goal = g

        ''' move to goal using motor controller: MC '''
        qhat = motor_controller(goal=self.goal, xy=xy, ac=self.ac, beta=self.mcbeta, omitg=self.omitg)

        return state_cue_fb, cpc, qhat, xy, h, x, g

    def learn(self, s1, cue_r1_fb, R, xy, cpc,h,g, mstate, plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, _, xy2, _, _, _ = self.act(s1, cue_r1_fb, mstate)

        if plasticity:
            self.pastpre = tf.cast((1-self.alpha) * self.pastpre + self.alpha * cpc,dtype=tf.float32)
            tdxy = (-self.env.dtxy[None,:] + xy2-xy)
            exy = tf.matmul(self.pastpre,tdxy,transpose_a=True)
            dwxy = self.tstep * self.xylr * exy
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwxy])

            if R > 0:
                if self.stochlearn:
                    gns = g + tf.cast(tf.random.normal(mean=0, stddev=np.sqrt(1 / self.alpha) * 0.25, shape=g.shape), dtype=tf.float32)

                    P = -tf.reduce_sum((xy2 - gns) ** 2)
                    self.phat = (1 - self.alpha) * self.phat + self.alpha * P
                    if P > self.phat:
                        M = 1
                    else:
                        M = 0
                    self.gstate = (1 - self.alpha) * self.gstate + self.alpha * gns
                    eg = tf.matmul(h, (gns - self.gstate), transpose_a=True)
                    dwg = self.lr * eg * M
                    self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwg])
                else:
                    eg = tf.matmul(h, (xy2 - g), transpose_a=True)
                    dwg = self.tstep * self.lr * eg * R
                    self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwg])

        return xy2

    def agent_reset(self):
        self.phat = 0
        self.pastpre = 0
        self.gstate = tf.zeros([1,2])
        self.mstate = tf.random.normal(shape=[1, self.nrnn], mean=0, stddev=
        tf.random.normal(shape=[1, self.nrnn], mean=0, stddev=np.sqrt(1 / self.alpha) * self.resns))
        self.xystate = tf.zeros([1, 2])
        self.pcstate = tf.zeros([1, self.npc**2])

class Res_Model(tf.keras.Model):
    def __init__(self, hp):
        super(Res_Model, self).__init__()
        rnncell = RNN_Cell(hp=hp, ninput=67)
        self.res = tf.keras.layers.RNN(rnncell,return_state=True, return_sequences=False, time_major=False,
                                        stateful=False, name='reservoir')

        self.goal = tf.keras.layers.Dense(units=2, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='xygoal')

        self.xy = tf.keras.layers.Dense(units=2, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='currpos')

    def call(self, inputs, states):
        pc = inputs[:,:49]
        h, x = self.res(inputs[None,:], initial_state=states)
        g = self.goal(h)
        xy = self.xy(pc)
        return h, x, g, xy

class RNN_Cell(tf.keras.layers.Layer):
    def __init__(self, hp, ninput):
        self.units = hp['nrnn']
        self.state_size = hp['nrnn']
        super(RNN_Cell, self).__init__()

        self.nrnn = hp['nrnn']
        self.ralpha = hp['tstep'] / hp['tau']
        self.recns = np.sqrt(1 / self.ralpha) * hp['resns']
        self.resact = choose_activation(hp['ract'], hp)
        self.resrecact = choose_activation(hp['recact'], hp)
        self.cp = hp['cp']
        self.recwinscl = hp['recwinscl']

        ''' win weight init'''
        winconn = np.random.uniform(-self.recwinscl, self.recwinscl, (ninput, self.nrnn))  # uniform dist [-1,1]
        winprob = np.random.choice([0, 1], (ninput, self.nrnn), p=[1 - self.cp[0], self.cp[0]])
        self.w_in = np.multiply(winconn, winprob)

        ''' wrec weight init '''
        connex = np.random.normal(0, np.sqrt(1 / (self.cp[1] * self.nrnn)), size=(self.nrnn, self.nrnn))
        prob = np.random.choice([0, 1], (self.nrnn, self.nrnn), p=[1 - self.cp[1], self.cp[1]])
        w_rec = hp['chaos'] * np.multiply(connex, prob)  # initialise random network with connection probability
        w_rec *= (np.eye(self.nrnn) == 0)  # remove self recurrence
        self.w_rec = w_rec

    def build(self, input_shape):
        self.win = self.add_weight(shape=(input_shape[-1], self.nrnn),
                                   initializer=tf.constant_initializer(self.w_in),
                                   name='win')
        self.wrec = self.add_weight(
            shape=(self.nrnn, self.nrnn),
            initializer=tf.constant_initializer(self.w_rec),
            name='wrec')
        self.built = True

    def call(self, inputs, states):
        I = tf.matmul(inputs, self.win)
        rjt = tf.matmul(self.resrecact(states[0]), self.wrec)
        sigmat = tf.random.normal(shape=(1, self.nrnn), mean=0, stddev=self.recns)

        xit = (1 - self.ralpha) * states[0] + self.ralpha * (I + rjt + sigmat)
        rit = self.resact(xit)
        return rit, xit

def motor_controller(goal, xy, ac, q=0, beta=4, omitg=0.025):
    if tf.norm(goal[0], ord=2) > omitg:
        dircomp = tf.cast(goal - xy, dtype=tf.float32)
        qk = tf.matmul(dircomp, ac.aj)
        sattw = tf.nn.softmax(beta * qk)
    else:
        sattw = tf.zeros([1,ac.nact])
    qns = sattw + q
    return qns


class place_cells():
    def __init__(self, hp):
        self.sigcoeff = 2  # larger coeff makes distribution sharper
        self.npc = hp['npc']  # vpcn * hpcn  # square maze
        self.au = hp['mazesize']
        hori = np.linspace(-self.au / 2, self.au / 2, self.npc)
        vert = np.linspace(-self.au / 2, self.au / 2, self.npc)
        self.pcdev = hori[1] - hori[0]  # distance between each place cell

        self.pcs = np.zeros([self.npc * self.npc, 2])
        i = 0
        for x in hori[::-1]:
            for y in vert:
                self.pcs[i] = np.array([y, x])
                i += 1

    def sense(self, s):
        ''' to convert coordinate s to place cell activity '''
        norm = np.sum((s - self.pcs) ** 2, axis=1)
        pcact = np.exp(-norm / (self.sigcoeff * self.pcdev ** 2))
        return pcact

    def check_pc(self, showpc='n'):
        ''' to show place cell distribution on Maze '''
        if showpc == 'y':
            plt.figure()
            plt.scatter(self.pcs[:, 0], self.pcs[:, 1], s=20, c='r')
            plt.axis((-self.au / 2, self.au / 2, -self.au / 2, self.au / 2))
            for i in range(self.npc):
                circ = plt.Circle(self.pcs[i], self.pcdev, color='g', fill=False)
                plt.gcf().gca().add_artist(circ)
            plt.show()


class action_cells():
    def __init__(self, hp):
        self.nact = hp['nact']
        self.alat = hp['alat']
        self.tstep = hp['tstep']
        self.astep = hp['maxspeed'] * self.tstep
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        self.aj = tf.cast(self.astep * np.array([np.sin(thetaj), np.cos(thetaj)]), dtype=tf.float32)
        self.qalpha = self.tstep / hp['tau']
        self.qstate = tf.zeros((1, self.nact))  # initialise actor units to 0
        self.ns = np.sqrt(1 / self.qalpha) * hp['actns']

        wminus = hp['actorw-']  # -1
        wplus = hp['actorw+']  # 1
        psi = hp['actorpsi']  # 20
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        thetadiff = np.tile(thetaj[None, :], (self.nact, 1)) - np.tile(thetaj[:, None], (1, self.nact))
        f = np.exp(psi * np.cos(thetadiff))
        f = f - f * np.eye(self.nact)
        norm = np.sum(f, axis=0)[0]
        self.wlat = tf.cast((wminus/self.nact) + wplus * f / norm,dtype=tf.float32)
        self.actact = choose_activation(hp['actact'],hp)

    def reset(self):
        self.qstate = tf.zeros((1, self.nact)) # reset actor units to 0

    def move(self, q):
        Y = q + tf.random.normal(mean=0, stddev=self.ns, shape=(1, self.nact), dtype=tf.float32)
        if self.alat:
            Y += tf.matmul(self.actact(self.qstate),self.wlat)
        self.qstate = (1 - self.qalpha) * self.qstate + self.qalpha * Y
        rho = self.actact(self.qstate)
        at = tf.matmul(self.aj, rho, transpose_b=True).numpy()[:, 0]/self.nact

        # movedist = np.linalg.norm(at,2)*1000/self.tstep  # m/s
        # self.maxactor.append(movedist)
        return at, rho


def choose_activation(actname,hp=None):
    if actname == 'sigm':
        act = tf.sigmoid
    elif actname == 'tanh':
        act = tf.tanh
    elif actname == 'relu':
        act = tf.nn.relu
    elif actname == 'softplus':
        act = tf.nn.softplus
    elif actname == 'elu':
        act = tf.nn.elu
    elif actname == 'leakyrelu':
        act = tf.nn.leaky_relu
    elif actname == 'phia':
        act = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=hp['sparsity'])
    elif actname == 'softmax':
        act = tf.nn.softmax
    else:
        def no_activation(x):
            return x
        act = no_activation
    return act


class BackpropAgent:
    def __init__(self, hp, env):
        ''' environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.taug = hp['taug']
        self.beg = (1 - (self.tstep / self.taug))  # taug for euler backward approximation
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.rstate = tf.zeros([1,hp['nhid']])
        self.action = np.zeros(2)
        self.actalpha = hp['actalpha']
        self.alpha = hp['tstep']/hp['tau']

        ''' critic parameters '''
        self.ncri = hp['ncri']
        self.vstate = tf.zeros([1, self.ncri])
        self.eulerm = hp['eulerm']
        self.maxcritic = 0
        self.loss = 0

        ''' Setup model: Place cell --> Action cells '''
        self.pc = place_cells(hp)
        self.model = BackpropModel(hp)
        self.ac = action_cells(hp)
        self.memory = Memory()
        self.opt = tf.optimizers.RMSprop(learning_rate=self.lr)
        self.be = hp['betaent']
        self.bv = hp['betaval']

    def act(self, state, cue_r_fb):
        s = self.pc.sense(state)  # convert coordinate info to place cell activity
        state_cue_fb = np.concatenate([s, cue_r_fb])  # combine all inputs

        if self.env.done:
            # silence all inputs after trial ends
            state_cue_fb = np.zeros_like(state_cue_fb)

        ''' Predict next action '''
        r, q, c = self.model(tf.cast(state_cue_fb[None, :], dtype=tf.float32))

        # stochastic discrete action selection
        action_prob_dist = tf.nn.softmax(q)
        actsel = np.random.choice(range(self.nact), p=action_prob_dist.numpy()[0])
        actdir = self.ac.aj[:,actsel]/self.tstep # constant speed 0.03
        self.action = (1-self.actalpha)*self.action + self.actalpha*actdir.numpy()

        return state_cue_fb, r, q, c, actsel, self.action

    def replay(self):
        discount_reward = self.discount_normalise_rewards(self.memory.rewards)

        with tf.GradientTape() as tape:
            policy_loss, value_loss, total_loss = self.compute_loss(self.memory, discount_reward)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        self.tderr = tf.reshape(total_loss, (1, 1))

        return policy_loss, value_loss, total_loss

    def discount_normalise_rewards(self, rewards):
        discounted_rewards = []
        cumulative = 0
        for reward in rewards[::-1]:
            cumulative = reward + self.beg * cumulative
            discounted_rewards.append(cumulative)
        discounted_rewards.reverse()

        return discounted_rewards

    def compute_loss(self, memory, discounted_rewards):
        _, logit, values = self.model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))

        # Advantage = Discounted R - V(s) = TD error
        advantage = tf.convert_to_tensor(np.array(discounted_rewards), dtype=tf.float32) - values[:,0]

        value_loss = advantage**2

        # compute actor policy loss
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=np.array(memory.actions))
        policy_loss = neg_log_prob * tf.stop_gradient(advantage)

        # compute entropy & add negative to prevent faster convergence of actions & better initial exploration
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=tf.nn.softmax(logit))

        # merge all losses to train network tgt
        comb_loss = tf.reduce_mean((self.bv * value_loss + policy_loss + self.be * entropy))
        self.loss = comb_loss

        return policy_loss, value_loss, comb_loss

    def cri_reset(self):
        self.vstate = tf.zeros([1, self.ncri])


class BackpropModel(tf.keras.Model):
    def __init__(self, hp):
        super(BackpropModel, self).__init__()
        self.nact = hp['nact']
        self.ncri = hp['ncri']
        self.nhid = hp['nhid']
        self.npc = hp['npc']
        self.hidact = hp['hidact']

        self.hidden = tf.keras.layers.Dense(units=self.nhid,
                                                 activation=choose_activation(self.hidact, hp),
                                                 use_bias=False, name='hidden',
                                                 kernel_initializer=
                                                 tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None))

        self.critic = tf.keras.layers.Dense(units=self.ncri, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='critic')
        self.actor = tf.keras.layers.Dense(units=self.nact, activation='linear',
                                           use_bias=False, kernel_initializer='zeros', name='actor')

    def call(self, inputs):
        r = self.hidden(inputs)
        c = self.critic(r)
        q = self.actor(r)
        return r, q, c


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.memstates = []

    def store(self, state, action, reward, memorystate=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.memstates.append(memorystate)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.memstates = []


class Foster_MC_Agent:
    def __init__(self, hp, env):
        ''' environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.mcbeta = hp['mcbeta']
        self.omitg = hp['omitg']
        self.alpha = hp['tstep']/hp['tau']
        self.xylr = hp['xylr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.xystate = tf.zeros([1, 2])
        self.pcstate = tf.zeros([1, hp['npc']* hp['npc']])

        ''' memory '''
        self.memory = np.zeros([18,49+18+2])
        self.pastpre = 0
        self.recallbeta = hp['recallbeta']

        ''' Setup model: Place cell --> Action cells '''
        self.pc = place_cells(hp)
        self.model = Foster_Model(hp)
        self.ac = action_cells(hp)

    def act(self, state, cue_r_fb):
        cpc = tf.cast(self.pc.sense(state),dtype=tf.float32)

        state_cue_fb = tf.cast(tf.concat([cpc, cue_r_fb],axis=0)[None, :],dtype=tf.float32)  # combine all inputs

        xy = self.model(cpc[None, :])

        self.goal = self.recall(cue_r_fb=state_cue_fb)

        ''' move to goal using motor controller: MC '''
        qhat = motor_controller(goal=self.goal, xy=xy, ac=self.ac, beta=self.mcbeta, omitg=self.omitg)

        return state_cue_fb, cpc, qhat, xy

    def store(self,xy, cue_r_fb,R, done, plastic):
        if done and plastic:
            memidx = np.argmax(cue_r_fb)-49
            if R>0:
                self.memory[memidx] = np.concatenate([cue_r_fb,xy],axis=1)
            else:
                self.memory[memidx] = 0

    def recall(self,cue_r_fb):
        # attention mechanism to query memory to retrieve goal coord
        qk = tf.matmul(cue_r_fb, self.memory[:, :-2], transpose_b=True)
        At = tf.nn.softmax(self.recallbeta * qk)
        gt = tf.matmul(At, self.memory[:, -2:])  # goalxy
        return tf.cast(gt,dtype=tf.float32)

    def learn(self, s1, cue_r1_fb, xy, cpc,plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, _, xy2 = self.act(s1, cue_r1_fb)

        if plasticity:
            self.pastpre = tf.cast((1-self.alpha) * self.pastpre + self.alpha * cpc,dtype=tf.float32)
            tdxy = (-self.env.dtxy[None,:] + xy2-xy)
            exy = tf.matmul(self.pastpre,tdxy,transpose_a=True)
            dwxy = self.tstep * self.xylr * exy
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwxy])

        return xy2

    def agent_reset(self):
        self.pastpre = 0
        self.xystate = tf.zeros([1, 2])
        self.pcstate = tf.zeros([1, self.npc**2])


class Foster_Model(tf.keras.Model):
    def __init__(self, hp):
        super(Foster_Model, self).__init__()
        self.actor = tf.keras.layers.Dense(units=40, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='actor')
        self.critic = tf.keras.layers.Dense(units=1, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='critic')

        self.xy = tf.keras.layers.Dense(units=2, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='currpos')

    def call(self, inputs):
        pc = inputs
        xy = self.xy(pc)
        return xy

