import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import multiplepa_script


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='server')
    hp['agenttype'] = 'a2c'
    hp['btstp'] = 9
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
    hp['betaent'] = -0.001
    hp['betaval'] = 0.5

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_hid_{}_{}n_{}ra_{}lr_{}tg_b{}_{}'.format(
        hp['task'],hp['agenttype'], hp['nhid'], hp['hidact'], hp['lr'], hp['taug'], hp['btstp'], dt.monotonic())

    totlat, totdgr, totpi, mvpath = multiplepa_script(hp)
