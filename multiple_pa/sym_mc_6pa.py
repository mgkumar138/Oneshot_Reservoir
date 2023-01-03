import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import multiplepa_script


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')
    hp['agenttype'] = 'sym'
    hp['btstp'] = 1
    hp['tstep'] = 100
    hp['actns'] = 0.25
    hp['actorw+'] = 1
    hp['actorw-'] = -1
    hp['taua'] = 250
    hp['taub'] = 120
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Agent parameters '''
    hp['recallbeta'] = 1
    hp['usesmc'] = 'confi'

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}+_{}-_{}_{}_xy_{}smc_{}xy_{}dt_b{}_{}'.format(
        hp['actorw+'],  hp['actorw-'],    hp['task'], hp['agenttype'],hp['usesmc'],hp['xylr'],hp['tstep'], hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath = multiplepa_script(hp)
