import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import multiplepa_script

if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')
    hp['agenttype'] = 'res'
    hp['btstp'] = 1
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Agent parameters '''
    hp['stochlearn'] = True
    hp['lr'] = 0.0005  # 0.0005
    hp['nrnn'] = 1024
    hp['taua'] = 2500  # increase reward duration
    hp['usesmc'] = 'neural'  # False = use neural motor controller

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_xy_{}sl_{}smc_{}t_{}om_{}ch_{}n_{}tau_{}taua_{}xy_{}lr_b{}_{}'.format(
        hp['task'],hp['agenttype'], hp['stochlearn'],hp['usesmc'], hp['time'],hp['omitg'], hp['chaos'],
        hp['nrnn'], hp['tau'], hp['taua'],hp['xylr'],  hp['lr'],  hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath = multiplepa_script(hp)

