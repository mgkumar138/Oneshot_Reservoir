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

    ''' agent parameters '''
    hp['lr'] = 0.000035
    hp['maxspeed'] = 0.07  # step size per 100ms

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_hid_{}_{}n_{}ra_{}lr_{}g_b{}_{}'.format(
        hp['task'],hp['agenttype'], hp['nhid'], hp['hidact'], hp['lr'], hp['gamma'], hp['btstp'], dt.monotonic())

    totlat, totdgr, totpi, mvpath = multiplepa_script(hp)
