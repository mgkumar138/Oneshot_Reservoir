import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import singlepa_script


if __name__ == '__main__':

    hp = get_default_hp(task='1pa',platform='laptop')
    hp['agenttype'] = 'sym'
    hp['btstp'] = 1
    hp['savefig'] = True
    hp['savegenvar'] = False

    ''' Agent parameters '''
    hp['recallbeta'] = 1
    hp['usesmc'] = 'epsilon'

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_xy_{}smc_{}t_{}om_{}taua_{}xy_b{}_{}'.format(
        hp['task'], hp['agenttype'],hp['usesmc'],hp['time'], hp['omitg'], hp['taua'],hp['xylr'],hp['btstp'],dt.monotonic())

    totlat, totdgr, mvpath, mdlw = singlepa_script(hp)