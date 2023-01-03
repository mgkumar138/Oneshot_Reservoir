import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, hp):

        ''' learn Tse et al., 2007 OPA and NPA tasks '''
        self.hp=hp
        self.tstep = hp['tstep'] # simulation time step
        self.maxstep = hp['time']*(1000 // self.tstep) # max training time
        self.normax = hp['probetime'] * (1000 // self.tstep)  # Non-rewarded probe trial max time 60s
        self.au = 1.6  # maze size
        self.rrad = 0.03  # reward location radius
        self.testrad = 0.03  # test location radius
        self.stay = False  # when agent reaches reward location, agent position is fixed
        self.rendercall = hp['render']  # plot realtime movement of agent
        self.bounpen = 0.01  # bounce back from boundary
        self.punish = 0  # no punishment
        self.Rval = hp['Rval']  # reward value when agent reaches reward location
        self.dtxy = np.zeros(2)  # true self-motion

        ''' Define Reward location '''
        ncues = hp['cuesize']  # size of cue vector
        sclf = hp['cuescl']  # gain for cue
        self.smell = np.eye(ncues) * sclf  # sensory cue to be passed to agent
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])  # number of reward locations in maze

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        if self.rendercall:
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

    def make(self, mtype='train', nocue=None, noreward=None):
        # make maze environment with reward locations and respective sensory cues
        self.mtype = mtype
        if mtype =='train' or mtype == 'opa':
            self.rlocs = np.array([self.holoc[8],self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35],self.holoc[40]])
            self.cues = self.smell[:6]
            self.totr = 6

        elif mtype == '2npa':
            self.rlocs = np.array(
                [self.holoc[1], self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35], self.holoc[47]])
            self.cues = np.concatenate([self.smell[6][None, :], self.smell[1:5], self.smell[7][None, :]], axis=0)
            self.totr = 6

        elif mtype == '6npa':
            self.rlocs = np.array(
                [self.holoc[2], self.holoc[19], self.holoc[23], self.holoc[28], self.holoc[32], self.holoc[46]])
            self.cues = self.smell[10:16]
            self.totr = 6

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*6, i*6)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*6, i*6))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%6 == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(6, 6, replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%6]
        self.rloc = self.rlocs[self.idx]  # reward location at current trial
        self.cue = self.cues[self.idx]  # cue at current trial
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)  # random start position
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []  # track trajectory
        self.tracks.append(self.x)  # include start location
        self.t = trial
        self.cordig = 0  # visit correct location
        self.totdig = 0  # visit a reward location
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(6))
        self.mask.remove(self.idx)
        self.d2r = np.zeros(self.totr)
        self.hitbound = False
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0
        cue = self.cue

        if self.stay:
            # stay at reward location if reached target
            at = np.zeros_like(at)
        xt1 = self.x + at  # update new location

        ax = np.concatenate([(-self.au / 2 < xt1), (self.au / 2 > xt1)]) # -xy,+xy
        if np.sum(ax)<4:
            self.hitbound = True
            R = self.punish
            if np.argmin(ax)>1: # if hit right or top, bounce back by 0.01
                xt1 -=at
                xt1 += self.bounpen*(ax[2:]-1)
            elif np.argmin(ax)<=1: # if hit left or bottom, bounce back by 0.01
                xt1 -=at
                xt1 -= self.bounpen*(ax[:2]-1)
        else:
            self.hitbound = False

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1
            for orl in self.rlocs[self.mask]:
                if np.linalg.norm(orl-xt1,2)<self.testrad:
                    self.totdig += 1

            if self.i == self.normax:
                self.done = True
                if self.totr == 6:
                    # visit ratio to correct target compared to other targets
                    self.dgr = 100 * self.cordig / (self.totdig + 1e-5)
                else:
                    # visit ratio at correct target over total time
                    self.dgr = np.round(100 * self.cordig / (self.normax), 5)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            # training trial
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, start reward disbursement
                cue = self.cue
                R = self.Rval
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if reward > 0:
                cue = self.cue
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2)  # eucledian distance from reward location
        self.dtxy = xt1 - self.x
        self.x = xt1

        return self.x, cue, reward, self.done, distr

    def render(self):
        if len(self.tracks)>1:
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
        plt.show()
        plt.pause(0.001)


def randpos(au):
    # to randomly chose a start position
    stpos = (au/2)*np.concatenate([np.eye(2),-1*np.eye(2)],axis=0)
    idx = np.random.choice(4,1, replace=True) # east, north, west, south
    randst = stpos[idx]
    return randst.reshape(-1), idx


class run_Rstep():
    def __init__(self,hp):
        '''continuous reward function'''
        self.rat = 0
        self.rbt = 0
        self.rt = 0
        self.taua = hp['taua']
        self.taub = hp['taub']
        self.tstep = hp['tstep']
        self.Rval = hp['Rval']
        self.totR = 0
        self.fullR = (1 - 1e-4) * self.Rval/self.tstep # Stop trial after 99.99% of reward disbursed
        self.count = False
        self.countR = 0

    def convR(self,rat, rbt):
        rat = (1 - (self.tstep / self.taua)) * rat
        rbt = (1 - (self.tstep / self.taub)) * rbt
        rt = (rat - rbt) / (self.taua - self.taub)
        return rat, rbt, rt

    def step(self,R):
        if R>0 and self.count is False:
            self.fullR = (1-1e-4)*R/self.tstep
            self.count = True
        self.rat += R
        self.rbt += R
        self.rat, self.rbt, self.rt = self.convR(self.rat, self.rbt)
        self.totR += self.rt
        done = False
        if self.totR>=self.fullR: # end after fullR reached or max 3 seconds
            done = True
        #     print(self.countR)
        # if self.count: self.countR +=1
        return self.rt, done


class Navex:
    def __init__(self,hp):
        ''' Learning single displaced locations '''
        self.hp = hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # Train max time, 1hr
        self.normax = hp['probetime']  * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        self.au = 1.6
        self.rrad = 0.03
        self.bounpen = 0.01
        self.testrad = 0.03
        self.stay = False
        self.rendercall = hp['render']
        self.Rval = hp['Rval']
        self.dtxy = np.zeros(2)

        ''' Define Reward location '''
        ncues = hp['cuesize']
        holes = np.linspace((-self.au/2)+0.2,(self.au/2)-0.2,7) # each reward location is 20 cm apart
        sclf = hp['cuescl'] # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])

        ''' sequence '''
        self.locseq = np.random.choice(np.arange(49),49,replace=False)
        self.loci = 0

        ''' create dig sites '''
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        if self.rendercall:
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

    def make(self, mtype=1, nocue=None, noreward=None):
        self.mtype = mtype
        assert isinstance(mtype,int)

        rlocsidx = self.locseq[self.loci]

        self.rlocs = []
        for r in range(mtype):
            self.rlocs.append(self.holoc[rlocsidx])
            #self.rlocs.append(self.holoc[24]) (0,0) coordinate
        self.rlocs = np.array(self.rlocs)
        self.cues = np.tile(self.smell[0],(len(self.smell),1))  # np.zeros_like(self.smell)
        self.loci += 1

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*len(self.rlocs), i*len(self.rlocs))) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*len(self.rlocs), i*len(self.rlocs)))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%len(self.rlocs) == 0: # reset order of cues presented after NR trials
            self.ridx = np.random.choice(np.arange(len(self.rlocs)), len(self.rlocs), replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%len(self.rlocs)]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(len(self.rlocs)))
        self.mask.remove(self.idx)
        self.hitbound = False
        if trial in self.nort:
            self.probe = True
        else:
            self.probe = False
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        cue = self.cue

        if self.stay:
            # stay at reward location if reached target
            at = np.zeros_like(at)
        xt1 = self.x + at  # update new location

        ax = np.concatenate([(-self.au / 2 < xt1), (self.au / 2 > xt1)]) # -xy,+xy
        if np.sum(ax)<4:
            self.hitbound = True
            if np.argmin(ax)>1: # if hit right or top, bounce back by 0.01
                xt1 -=at
                xt1 += self.bounpen*(ax[2:]-1)
            elif np.argmin(ax)<=1: # if hit left or bottom, bounce back by 0.01
                xt1 -=at
                xt1 -= self.bounpen*(ax[:2]-1)
        else:
            self.hitbound = False

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            # time spent = location within 0.1m near reward location with no overlap of other locations
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1

            if self.mtype>1:
                for orl in self.rlocs[self.mask]:
                    if np.linalg.norm(orl-xt1,2)<self.testrad:
                        self.totdig += 1

            if self.i == self.normax:
                self.done = True
                if self.mtype == 1:
                    self.dgr = np.round(100 * self.cordig / self.normax, 5)
                else:
                    self.dgr = 100 * self.cordig / (self.totdig + 1e-10)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, r=1 at first instance
                cue = self.cue
                R = self.Rval
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2) # eucledian distance away from reward location
        self.dtxy = xt1 - self.x
        self.x = xt1

        return self.x, cue, reward, self.done, distr

    def render(self):
        if len(self.tracks)>1:
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
        plt.show()
        plt.pause(0.001)


class MultiplePA_Maze:
    def __init__(self, hp):

        ''' Learn 12 NPAs after learn 6 PAs from Tse et al. (2007) '''
        self.hp=hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # max training time
        self.normax = hp['probetime']  * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        self.au = 1.6
        self.rrad = 0.03
        self.testrad = 0.03
        self.stay = False
        self.rendercall = hp['render']
        self.bounpen = 0.01
        self.punish = 0  # no punishment
        self.Rval = hp['Rval']
        self.dtxy = np.zeros(2)

        ''' Define Reward location '''
        ncues = hp['cuesize']
        sclf = hp['cuescl']  # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])

        self.loci = 0
        self.opaloc = np.array([8, 13, 18, 30, 35, 40])  # to exclude OPA locations from new NPA
        self.npaloc = np.arange(49)
        self.npaloc = np.array(list(set(self.npaloc)-set(self.opaloc)))
        # choose 12 NPA locations randomly for every new agent instantiation
        self.npaloc = np.random.choice(self.npaloc, 36, replace=False)[:36].reshape(3,12)

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        if self.rendercall:
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

    def make(self, mtype='train', nocue=None, noreward=None):
        self.mtype = mtype

        if mtype =='train':
            self.rlocs = np.array(
                [self.holoc[8], self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35], self.holoc[40]])
            self.totr = 6
            self.cues = self.smell[:6]

        elif mtype == '12npa':
            rlocidx = self.npaloc[self.loci]
            self.rlocs = []
            for r in rlocidx:
                self.rlocs.append(self.holoc[r])
            self.rlocs = np.array(self.rlocs)
            self.cues = self.smell[6:]
            self.totr = 12
            self.loci += 1

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*self.totr, i*self.totr)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*self.totr, i*self.totr))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%self.totr == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(self.totr, self.totr, replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%self.totr]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(self.totr))
        self.mask.remove(self.idx)
        self.d2r = np.zeros(self.totr)
        self.hitbound = False
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        cue = self.cue

        if self.stay:
            # stay at reward location if reached target
            at = np.zeros_like(at)
        xt1 = self.x + at  # update new location

        ax = np.concatenate([(-self.au / 2 < xt1), (self.au / 2 > xt1)]) # -xy,+xy
        if np.sum(ax)<4:
            self.hitbound = True
            R = self.punish
            if np.argmin(ax)>1: # if hit right or top, bounce back by 0.01
                xt1 -=at
                xt1 += self.bounpen*(ax[2:]-1)
            elif np.argmin(ax)<=1: # if hit left or bottom, bounce back by 0.01
                xt1 -=at
                xt1 -= self.bounpen*(ax[:2]-1)
        else:
            self.hitbound = False

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1

            for orl in self.rlocs[self.mask]:
                if np.linalg.norm(orl-xt1,2)<self.testrad:
                    self.totdig += 1

            if self.i == self.normax:
                self.done = True
                # visit ratio to correct target compared to other targets
                self.dgr = 100 * self.cordig / (self.totdig + 1e-5)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:

            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, r=1 at first instance
                cue = self.cue
                R = self.Rval
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if reward > 0:
                cue = self.cue
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2)  # eucledian distance from reward location
        self.dtxy = xt1 - self.x
        self.x = xt1

        return self.x, cue, reward, self.done, distr

    def render(self):
        if len(self.tracks)>1:
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
        plt.show()
        plt.pause(0.001)