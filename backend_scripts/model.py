import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Res_MC_Agent:
    def __init__(self, hp, env):
        ''' Reservoir agent to learn self position and target location '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.lr = hp['lr']  # reservoir learning rate
        self.mcbeta = hp['mcbeta']  # motor controller beta
        self.omitg = hp['omitg']  # threshold to omit goal coordinate
        self.omite = hp['omite']
        self.alpha = hp['tstep']/hp['tau']  # alpha value

        self.xylr = hp['xylr']  # self position coordiante learning rate
        self.stochlearn = hp['stochlearn']  # sparse learning algorithm
        self.npc = hp['npc']  # number of place cells
        self.nact = hp['nact']  # number of actor cells

        self.xystate = tf.zeros([1, 2])  # initialise self position states
        self.pcstate = tf.zeros([1, hp['npc'] * hp['npc']])

        ''' memory '''
        self.resns = hp['resns']  # noise in reservoir
        self.nrnn = hp['nrnn'] # number of reservoir cells
        self.phat = 0  # initialise performance metric
        self.gstate = tf.zeros([1,3])  # initialise goal state
        self.mstate = tf.random.normal(shape=[1, hp['nrnn']],
                                       mean=0, stddev=np.sqrt(1 / self.alpha) * self.resns)  # initialie reservoir state
        self.pastpre = 0

        ''' Setup model: Place cell, Reservoir, Action cell '''
        self.pc = place_cells(hp)
        self.model = Res_Model(hp)
        self.ac = action_cells(hp)
        self.usesmc = hp['usesmc']
        if self.usesmc == 'neural':
            self.mc = tf.keras.models.load_model('../motor_controller/eps_motor_controller_2h_0.75omg_1024_2021-06-04')

    def act(self, state, cue_r_fb, mstate):
        '''given state and cue, make action'''
        cpc = tf.cast(self.pc.sense(state),dtype=tf.float32)  # convert 2D state into place cell activity

        state_cue_fb = tf.cast(tf.concat([cpc, cue_r_fb],axis=0)[None, :],dtype=tf.float32)  # combine all inputs

        h, x, g, xy = self.model(state_cue_fb, mstate)  # goal and self position prediction by network model
        self.goal = g

        ''' move to goal using motor controller: MC '''
        if self.usesmc == 'goal':
            # use symbolic motor controller with ||goal||^2 > omega
            qhat = motor_controller(goal=self.goal, xy=xy, ac=self.ac, beta=self.mcbeta, omitg=self.omitg)
        elif self.usesmc == 'confi':
            # use symbolic motor controller with confi > omega
            qhat = eps_motor_controller(goal=self.goal, xy=xy, ac=self.ac, beta=self.mcbeta, omite=self.omite)
        elif self.usesmc == 'neural':
            # use neural motor controller
            qhat = self.mc(tf.concat([self.goal, xy], axis=1))
        else:
            print('No motor controller selected! Choose: goal or confi or neural')

        return state_cue_fb, cpc, qhat, xy, h, x, g

    def learn(self, s1, cue_r1_fb, R, xy, cpc,h,g, mstate, plasticity=True):
        ''' Hebbian update rule '''
        _, _, _, xy2, _, _, _ = self.act(s1, cue_r1_fb, mstate)  # estimate new position after making action

        if plasticity:  # switch off plasticity during probe
            # low pass place cell filter
            self.pastpre = tf.cast((1-self.alpha) * self.pastpre + self.alpha * cpc,dtype=tf.float32)
            tdxy = (-self.env.dtxy[None,:] + xy2-xy)  # TD error for self position
            exy = tf.matmul(self.pastpre,tdxy,transpose_a=True)  # compute trace
            dwxy = self.xylr * exy  # learning rate * trace
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwxy]) # update weight

            if R > 0: # to increase computational speed, perform trace computation only when reward is positive
                if self.stochlearn:
                    ''' 4 factor exploratory hebbian rule '''
                    # add noise to goal
                    gns = g + tf.cast(tf.random.normal(mean=0, stddev=np.sqrt(1 / self.alpha) * 0.25, shape=g.shape), dtype=tf.float32)
                    self.gstate = (1 - self.alpha) * self.gstate + self.alpha * gns  # los pass filtered place cell
                    target = tf.concat([xy2, tf.ones([1, 1])], axis=1)
                    P = -tf.reduce_sum((target - gns) ** 2) # compute performance metric
                    self.phat = (1 - self.alpha) * self.phat + self.alpha * P  # low pass filtered performance

                    if P > self.phat:  # modulatory factor
                        M = 1
                    else:
                        M = 0

                    eg = tf.matmul(h, (gns - self.gstate), transpose_a=True) # trace pre x (post - lowpass)
                    dwg = self.lr * eg * M * (R !=0)  # trace * modulatory * reward presence
                    self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwg])  # update weights
                else:
                    ''' perceptron rule '''
                    target = tf.concat([xy2, tf.ones([1,1])],axis=1)
                    eg = tf.matmul(h, (target - g), transpose_a=True)  # trace: pre * (target - goal prediction)
                    dwg = self.tstep * self.lr * eg * R  # trace * reward
                    self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwg])

        return xy2

    def agent_reset(self):
        ''' reset agent states every new trial '''
        self.phat = 0
        self.pastpre = 0
        self.gstate = tf.zeros([1,3])
        self.mstate = tf.random.normal(shape=[1, self.nrnn], mean=0, stddev=
        tf.random.normal(shape=[1, self.nrnn], mean=0, stddev=np.sqrt(1 / self.alpha) * self.resns))
        self.xystate = tf.zeros([1, 2])
        self.pcstate = tf.zeros([1, self.npc**2])

class Res_Model(tf.keras.Model):
    def __init__(self, hp):
        ''' reservoir & goal + self position cordinate prediction '''
        super(Res_Model, self).__init__()
        rnncell = RNN_Cell(hp=hp, ninput=67)
        self.res = tf.keras.layers.RNN(rnncell,return_state=True, return_sequences=False, time_major=False,
                                        stateful=False, name='reservoir')

        self.goal = tf.keras.layers.Dense(units=3, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='xygoal')

        self.xy = tf.keras.layers.Dense(units=2, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='currpos')

    def call(self, inputs, states):
        pc = inputs[:,:49]  # get place cell activity from input
        h, x = self.res(inputs[None,:], initial_state=states)
        g = self.goal(h)  # goal estimate
        xy = self.xy(pc)  # pass place cell activity to self position network
        return h, x, g, xy

class RNN_Cell(tf.keras.layers.Layer):
    def __init__(self, hp, ninput):
        ''' reservoir definition '''
        super(RNN_Cell, self).__init__()

        self.state_size = hp['nrnn']
        self.nrnn = hp['nrnn']  # number of units
        self.ralpha = hp['tstep'] / hp['tau']  # time constant
        self.recns = np.sqrt(1 / self.ralpha) * hp['resns']  # white noise
        self.resact = choose_activation(hp['ract'], hp)  # reservoir activation function
        self.resrecact = choose_activation(hp['recact'], hp)  # recurrent activation function
        self.cp = hp['cp']  # connection probability
        self.recwinscl = hp['recwinscl']  # input weight scale

        ''' input weight'''
        winconn = np.random.uniform(-self.recwinscl, self.recwinscl, (ninput, self.nrnn))  # uniform dist [-1,1]
        winprob = np.random.choice([0, 1], (ninput, self.nrnn), p=[1 - self.cp[0], self.cp[0]])
        self.w_in = np.multiply(winconn, winprob) # cater to different input connection probabilities

        ''' recurrent weight '''
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
        I = tf.matmul(inputs, self.win)  # get input current
        rjt = tf.matmul(self.resrecact(states[0]), self.wrec)  # get past recurrent activity
        sigmat = tf.random.normal(shape=(1, self.nrnn), mean=0, stddev=self.recns)  # white noise

        xit = (1 - self.ralpha) * states[0] + self.ralpha * (I + rjt + sigmat)  # new membrane potential
        rit = self.resact(xit)  # reservoir activity
        return rit, xit

def motor_controller(goal, xy, ac, beta=4, omitg=0.025):
    ''' motor controller to decide direction to move with current position and goal location '''
    if tf.norm(goal[0,:2], ord=2) > omitg:  # omit goal if goal is less than threshold
        dircomp = tf.cast(goal[:,:2] - xy, dtype=tf.float32)  # vector subtraction
        qk = tf.matmul(dircomp, ac.aj)  # choose action closest to direction to move
        qhat = tf.nn.softmax(beta * qk)  # scale action with beta and get probability of action
    else:
        qhat = tf.zeros([1,ac.nact])  # if goal below threshold, no action selected by motor controller
    return qhat

def eps_motor_controller(goal, xy, ac, beta=4, omite=0.75):
    ''' motor controller to decide direction to move with current position and goal location '''
    if goal[0,-1] > omite:  # omit goal if goal is less than threshold
        dircomp = tf.cast(goal[:,:2] - xy, dtype=tf.float32)  # vector subtraction
        qk = tf.matmul(dircomp, ac.aj)  # choose action closest to direction to move
        qhat = tf.nn.softmax(beta * qk)  # scale action with beta and get probability of action
    else:
        qhat = tf.zeros([1,ac.nact])  # if goal below threshold, no action selected by motor controller
    return qhat


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
        self.nact = hp['nact']  # number of action units
        self.alat = hp['alat']  # to use lateral connectivity
        self.tstep = hp['tstep']
        self.astep = hp['maxspeed'] * self.tstep  # maxstep size a0
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        self.aj = tf.cast(self.astep * np.array([np.sin(thetaj), np.cos(thetaj)]), dtype=tf.float32)
        self.qalpha = self.tstep / hp['tau']  # actor time constant
        self.qstate = tf.zeros((1, self.nact))  # initialise actor units to 0
        self.ns = np.sqrt(1 / self.qalpha) * hp['actns']  # white noise for exploration

        wminus = hp['actorw-']  # -1
        wplus = hp['actorw+']  # 1
        psi = hp['actorpsi']  # 20
        thetadiff = np.tile(thetaj[None, :], (self.nact, 1)) - np.tile(thetaj[:, None], (1, self.nact))
        f = np.exp(psi * np.cos(thetadiff))
        f = f - f * np.eye(self.nact)
        norm = np.sum(f, axis=0)[0]
        self.wlat = tf.cast((wminus/self.nact) + wplus * f / norm,dtype=tf.float32)  # lateral connectivity matrix
        self.actact = choose_activation(hp['actact'],hp)  # actor activation function

    def reset(self):
        self.qstate = tf.zeros((1, self.nact)) # reset actor units to 0

    def move(self, q):
        Y = q + tf.random.normal(mean=0, stddev=self.ns, shape=(1, self.nact), dtype=tf.float32)  # add white noise
        if self.alat:
            Y += tf.matmul(self.actact(self.qstate),self.wlat) # use lateral connectivity
        self.qstate = (1 - self.qalpha) * self.qstate + self.qalpha * Y  # new membrane potential
        rho = self.actact(self.qstate)  # new actor activity
        at = tf.matmul(self.aj, rho, transpose_b=True).numpy()[:, 0]/self.nact  # choose direction of movement
        return at, rho


def choose_activation(actname,hp=None):
    ''' range of activation functions to use'''
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
        self.gamma = hp['gamma'] # gamma reward discount factor
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.action = np.zeros(2)
        self.alphaa2c = hp['alphaa2c']   # smooth action taken
        self.alpha = hp['tstep']/hp['tau']

        ''' critic parameters '''
        self.ncri = hp['ncri']
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

        ''' Predict next action '''
        r, q, c = self.model(tf.cast(state_cue_fb[None, :], dtype=tf.float32))  # model prediction

        # stochastic discrete action selection
        action_prob_dist = tf.nn.softmax(q)
        actsel = np.random.choice(range(self.nact), p=action_prob_dist.numpy()[0])  # choose 1 action based on probability
        actdir = self.ac.aj[:,actsel]/self.tstep  # select 1 out of 40 possible direction of movement
        self.action = (1-self.alphaa2c)*self.action + self.alphaa2c*actdir.numpy()  # smoothen action trajectory

        return state_cue_fb, r, q, c, actsel, self.action

    def replay(self):
        discount_reward = self.discount_normalise_rewards(self.memory.rewards)  # compute discounted rewards

        with tf.GradientTape() as tape:
            policy_loss, value_loss, total_loss = self.compute_loss(self.memory, discount_reward)

        grads = tape.gradient(total_loss, self.model.trainable_weights)  # compute gradients
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))  # apply graidents with optimiser
        self.tderr = tf.reshape(total_loss, (1, 1))

        return policy_loss, value_loss, total_loss

    def discount_normalise_rewards(self, rewards):
        discounted_rewards = []
        cumulative = 0
        for reward in rewards[::-1]:
            cumulative = reward + self.gamma * cumulative  # discounted reward with gamma
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

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Foster_MC_Agent:
    def __init__(self, hp, env):
        ''' Symbolic agent to learn multiple goals equivalent to Foster et al. (2000) '''
        self.env = env
        self.tstep = hp['tstep']
        self.hp = hp

        ''' agent parameters '''
        self.mcbeta = hp['mcbeta']  # motor controller beta
        self.omitg = hp['omitg']  # omit goal threshold
        self.omite = hp['omite']
        self.alpha = hp['tstep']/hp['tau']
        self.xylr = hp['xylr']  # self position learning rate
        self.npc = hp['npc']  # number of place cells
        self.nact = hp['nact']  # number of action cells
        self.xystate = tf.zeros([1, 2])  # initialise self position network
        self.pcstate = tf.zeros([1, hp['npc']* hp['npc']])

        ''' memory '''
        self.memory = np.zeros([18,49+18+2+1])  # 2D episodic memory matrix to store place cell, cue, goal coordinate
        self.pastpre = 0
        self.recallbeta = hp['recallbeta']  # recall beta
        self.tempgoal = 0
        self.goal = tf.zeros([1,3])

        ''' Setup model: Place cell, symbolic, Action cells '''
        self.pc = place_cells(hp)
        self.model = Foster_Model()
        self.ac = action_cells(hp)
        self.usesmc = hp['usesmc']
        if self.usesmc == 'neural':
            self.mc = tf.keras.models.load_model('../motor_controller/eps_motor_controller_2h_0.75omg_1024_2021-06-04')

    def act(self, state, cue_r_fb):
        cpc = tf.cast(self.pc.sense(state),dtype=tf.float32)

        state_cue_fb = tf.cast(tf.concat([cpc, cue_r_fb],axis=0)[None, :],dtype=tf.float32)  # combine all inputs

        xy = self.model(cpc[None, :])  # get self position estimate

        self.tempgoal = self.recall(state_cue=state_cue_fb)

        ''' move to goal using motor controller: MC '''
        if self.usesmc == 'goal':
            # use symbolic motor controller with ||goal||^2 > omega
            qhat = motor_controller(goal=self.goal, xy=xy, ac=self.ac, beta=self.mcbeta, omitg=self.omitg)
        elif self.usesmc == 'confi':
            # use symbolic motor controller with confi > omega
            qhat = eps_motor_controller(goal=self.goal, xy=xy, ac=self.ac, beta=self.mcbeta, omite=self.omite)
        elif self.usesmc == 'neural':
            # use neural motor controller
            qhat = self.mc(tf.concat([self.goal, xy], axis=1))
        else:
            print('No motor controller selected! Choose: goal or confi or neural')

        return state_cue_fb, cpc, qhat, xy

    def store(self,xy, cue_r_fb,R, done, plastic):
        ''' store current position as goal if reward is positive at cue indexed row '''
        if done and plastic:
            memidx = np.argmax(cue_r_fb)-49
            if R>0:
                self.memory[memidx] = np.concatenate([cue_r_fb,xy, np.array([[1]])],axis=1)
            else:
                self.memory[memidx] = 0

    def recall(self,state_cue):
        ''' attention mechanism to query memory to retrieve goal coord'''
        qk = tf.matmul(state_cue, self.memory[:, :67], transpose_b=True)  # use current position & cue to query memory
        At = tf.nn.softmax(self.recallbeta * qk)  # attention weight
        gt = tf.matmul(At, self.memory[:, -3:])  # goalxy
        return tf.cast(gt,dtype=tf.float32)

    def learn(self, s1, cue_r1_fb, xy, cpc,plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, _, xy2 = self.act(s1, cue_r1_fb)  # get new curent position after taking action

        self.goal = (1 - self.alpha) * self.goal + self.alpha * self.tempgoal

        if plasticity:
            ''' learn self position during training trials '''
            self.pastpre = tf.cast((1-self.alpha) * self.pastpre + self.alpha * cpc,dtype=tf.float32)
            tdxy = (-self.env.dtxy[None,:] + xy2-xy)
            exy = tf.matmul(self.pastpre,tdxy,transpose_a=True)
            dwxy = self.tstep * (self.hp['xylr']/100) * exy
            self.model.layers[-1].set_weights([tf.clip_by_value(self.model.layers[-1].get_weights()[0] + dwxy,-1,1)])

        return xy2

    def agent_reset(self):
        self.pastpre = 0
        self.xystate = tf.zeros([1, 2])
        self.pcstate = tf.zeros([1, self.npc**2])
        self.goal = tf.zeros([1, 3])
        self.tempgoal = 0


class Foster_Model(tf.keras.Model):
    def __init__(self):
        super(Foster_Model, self).__init__()

        self.xy = tf.keras.layers.Dense(units=2, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='currpos')

    def call(self, inputs):
        xy = self.xy(inputs)
        return xy

